# Script to extract text from images using PaddleOCR
# Includes OpenCV preprocessing, folder processing, and parallel processing.
import cv2
from PIL import Image
import numpy as np
import os
import argparse
import platform
from tqdm import tqdm
import logging
import multiprocessing
import warnings
import csv
import re
import math
from collections import defaultdict

# Filter out ccache warnings at the script's start (before PaddleOCR import)
warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")

from paddleocr import PaddleOCR, draw_ocr

__version__ = "1.2.0" # Script version (English comments, logging, and docstrings)

# --- Constants for Labeling ---
RE_JUMIN_NUMBER_FULL = r"\d{6}\s*-\s*\d{7}" # RRN with hyphen
RE_JUMIN_NUMBER_CLEAN = r"\d{13}" # RRN without hyphen (13 digits)
RE_KOREAN_NAME = r"^[가-힣]+$" # Korean name pattern
RE_DATE_YYYY_MM_DD = r"(\d{4})\s*[\.,년]?\s*(\d{1,2})\s*[\.,월]?\s*(\d{1,2})\s*[\.일]?" # Date YYYY.MM.DD

# Keywords for address identification (note spaces for some tokens)
KW_ADDRESS_TOKENS = ["특별시", "광역시", "도", "시 ", "군 ", "구 ", "읍 ", "면 ", "동 ", "리 ", "로 ", "길 ", "아파트", "빌라", " 번지", " 호"]
KW_DOC_TITLES = ["주민등록증", "운전면허증", "공무원증"] # Common ID document titles
# Suffixes for authority names (e.g., 청장, 시장)
KW_AUTHORITY_SUFFIXES = ["청장", "시장", "군수", "구청장", "경찰서장", "지방경찰청장", "위원회위원장"]
KW_AUTHORITY_KEYWORDS = ["발급기관", "기관명"] # Explicit keywords for issuing authority
KW_REGION_NAMES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주"] # Major Korean region names
KW_REGION_SUFFIXES = ["특별시", "광역시", "특별자치시", "도", "특별자치도"] # Suffixes for region names
# --- End Constants ---


# --- Logger Setup ---
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """Sets up the basic logger."""
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    else: # If handlers exist, just set their level
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
            if isinstance(handler, logging.StreamHandler):
                 logger.propagate = False # Avoid duplicate logs if root logger also has a handler

# Global variables for worker processes
_worker_csv_lock = None
_worker_csv_file_path = None
_worker_log_level_env_var = 'OCR_LOG_LEVEL' # Default environment variable name for worker log level

def worker_initializer_func(lock, csv_path, log_level_env_name):
    """Initialization function called when each worker process starts."""
    global _worker_csv_lock, _worker_csv_file_path, _worker_log_level_env_var

    _worker_csv_lock = lock
    _worker_csv_file_path = csv_path
    _worker_log_level_env_var = log_level_env_name

    warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")
    log_level_str = os.environ.get(_worker_log_level_env_var, 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level) # Setup logging for this worker
    logger.debug(f"Worker {os.getpid()}: Initialized (Lock, CSV path set, Logging level: {log_level_str}).")


def preprocess_image_for_ocr(image_path):
    """Preprocesses an image to improve OCR accuracy."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return None

        # 1. Deskewing (Horizontal alignment)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):  # Consider near-horizontal lines
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                # Adjust angle if it's out of the -45 to 45 degree range to prevent large rotations
                if median_angle > 45: median_angle -= 90
                elif median_angle < -45: median_angle += 90

                if abs(median_angle) > 0.5: # Only rotate if the angle is significant
                    logger.debug(f"Deskewing image: {os.path.basename(image_path)}, Angle: {median_angle:.2f}°")
                    height, width = img.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (width, height),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Update grayscale image after rotation
                else:
                    logger.debug(f"No significant skew detected for: {os.path.basename(image_path)}, Detected angle: {median_angle:.2f}°")
        else:
            logger.debug(f"No lines detected for deskewing: {os.path.basename(image_path)}")

        # 2. Contrast Enhancement (CLAHE) and Denoising (Median Blur)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)
        logger.debug(f"Preprocessing complete (Deskew, CLAHE, Median Blur): {os.path.basename(image_path)}")
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV error during preprocessing (File: {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"Exception during preprocessing (File: {os.path.basename(image_path)}): {e}")
        return None


def extract_text_from_image_worker(ocr_engine_params, image_data, filename_for_log=""):
    """Extracts text from the given image data using PaddleOCR. (For worker processes)"""
    ocr_engine = None
    try:
        logger.debug(f"Worker {os.getpid()}: Initializing PaddleOCR engine... (File: {filename_for_log}, Params: {ocr_engine_params})")
        ocr_engine = PaddleOCR(**ocr_engine_params)
        logger.debug(f"Worker {os.getpid()}: PaddleOCR engine initialized. (File: {filename_for_log})")

        logger.debug(f"Worker {os.getpid()}: Starting OCR for: {filename_for_log}")
        result = ocr_engine.ocr(image_data, cls=True) # cls=True enables text angle classification
        logger.debug(f"Worker {os.getpid()}: OCR complete for: {filename_for_log}")

        extracted_items = []
        if result and result[0] is not None: # result is a list of lists, one for each detected text line
            for line_info in result[0]:
                text, confidence = line_info[1] # text and confidence score
                bounding_box = line_info[0] # [top-left, top-right, bottom-right, bottom-left]
                try:
                    # Ensure bounding box coordinates are floats
                    float_box = [[float(p[0]), float(p[1])] for p in bounding_box]
                    extracted_items.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": float_box
                    })
                except (ValueError, TypeError):
                     logger.warning(f"Invalid bounding box data format: {bounding_box} (Text: {text})")
                     extracted_items.append({ # Still append text if box is invalid
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": None
                    })
        return extracted_items
    except Exception as e:
        # Log detailed error if debug level is enabled
        logger.error(f"Worker {os.getpid()}: Error during OCR processing (File: {filename_for_log}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
        return None # Return None on error
    finally:
        if ocr_engine: # Release PaddleOCR engine resources
            del ocr_engine
            logger.debug(f"Worker {os.getpid()}: PaddleOCR engine deleted (File: {filename_for_log})")

# --- Bounding Box Utility Functions ---
def get_box_coordinates(box):
    """Validates and returns bounding box coordinates if valid."""
    if box and len(box) == 4 and all(len(pt) == 2 for pt in box):
        return box
    return None

def get_box_center(box):
    """Calculates the center coordinates of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        x_center = sum(pt[0] for pt in coords) / 4
        y_center = sum(pt[1] for pt in coords) / 4
        return x_center, y_center
    return None, None

def get_box_top(box):
    """Gets the top-most y-coordinate of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        return min(pt[1] for pt in coords)
    return float('inf') # Return infinity if box is invalid

def get_box_bottom(box):
    """Gets the bottom-most y-coordinate of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        return max(pt[1] for pt in coords)
    return float('-inf') # Return negative infinity if box is invalid

def get_box_left(box):
    """Gets the left-most x-coordinate of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        return min(pt[0] for pt in coords)
    return float('inf')

def get_box_right(box):
    """Gets the right-most x-coordinate of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        return max(pt[0] for pt in coords)
    return float('-inf')

def get_box_height(box):
    """Calculates the height of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        min_y = min(pt[1] for pt in coords)
        max_y = max(pt[1] for pt in coords)
        return max_y - min_y
    return 0

def get_box_width(box):
    """Calculates the width of a bounding box."""
    coords = get_box_coordinates(box)
    if coords:
        min_x = min(pt[0] for pt in coords)
        max_x = max(pt[0] for pt in coords)
        return max_x - min_x
    return 0
# --- End Bounding Box Utility Functions ---

def reorder_korean_address(address_text):
    """Reorders Korean address formats to a more standard sequence (e.g., City/Province first)."""
    # Patterns to find City/Province names appearing after street names or building numbers
    address_patterns = [
        # e.g., "법원로11길 서울특별시송파구" -> "서울특별시송파구 법원로11길"
        r'([가-힣0-9]+(?:로|길)[0-9]*)\s+([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)[가-힣]*(?:시|군|구)?)',
        # e.g., "101동 서울특별시강남구" -> "서울특별시강남구 101동"
        r'([0-9]+[A-Za-z]?(?:동|호|층|번지)?)\s+([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)[가-힣]*(?:시|군|구)?)'
    ]
    for pattern in address_patterns:
        match = re.search(pattern, address_text)
        if match:
            part1, part2 = match.groups()
            # Reorder: Group2 (City/Province) + Group1 (Street/Number)
            reordered = re.sub(pattern, r'\2 \1', address_text)
            logger.debug(f"Reordering address: '{address_text}' -> '{reordered}'")
            address_text = reordered
    # Fix spacing around '동' (dong/building) and '호' (ho/unit)
    address_text = re.sub(r'([0-9]+)\s+([A-Za-z])\s+동', r'\1\2동', address_text) # e.g., "B 동" -> "B동"
    address_text = re.sub(r'([A-Za-z])\s+동', r'\1동', address_text)
    return address_text

def label_text_item_initial(extracted_item, image_width, image_height):
    """Assigns an initial label to an extracted text item based on patterns and heuristics."""
    text = extracted_item['text']
    box = extracted_item['bounding_box']
    x_center, y_center = get_box_center(box)

    if x_center is None or y_center is None: # Should not happen if box is validated before
        logger.warning(f"Attempting initial labeling with invalid bounding box: {box} (Text: {text})")
        return "기타" # "Other"

    # 1. Resident Registration Number (RRN)
    if re.fullmatch(RE_JUMIN_NUMBER_FULL, text) or \
       re.fullmatch(RE_JUMIN_NUMBER_CLEAN, text.replace("-","").replace(" ","")):
        cleaned_text = text.replace("-","").replace(" ","")
        if len(cleaned_text) == 13 and cleaned_text.isdigit():
            extracted_item['text'] = f"{cleaned_text[:6]}-{cleaned_text[6:]}" # Standardize format
        return "주민등록번호" # "Resident Registration Number"

    # 2. Name
    cleaned_text_for_name = text.replace(" ","")
    if 2 <= len(cleaned_text_for_name) <= 5 and re.fullmatch(RE_KOREAN_NAME, cleaned_text_for_name):
        # Names are typically in a certain region of an ID card
        if image_height > 0 and (image_height * 0.1 < y_center < image_height * 0.5) and len(cleaned_text_for_name) <= 4:
             return "이름" # "Name"

    # 3. Date Candidate (YYYY.MM.DD format)
    date_match = re.search(RE_DATE_YYYY_MM_DD, text)
    if date_match:
        try:
            year, month, day = map(int, date_match.groups())
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31: # Basic date validation
                 extracted_item['normalized_date'] = f"{year}.{month:02d}.{day:02d}"
                 return "날짜_후보" # "Date_Candidate"
        except ValueError:
            logger.debug(f"Error converting date parts to int: {text}")


    # 3.1 Date Part Candidate (YYYY, MM, or DD)
    if text.isdigit():
        val = int(text)
        if len(text) == 4 and 1900 <= val <= 2100: # Year
            extracted_item['date_part_type'] = 'year'
            extracted_item['value'] = val
            return "날짜_부분" # "Date_Part"
        elif 1 <= len(text) <= 2: # Month or Day
            if 1 <= val <= 12: # Possible month
                extracted_item['date_part_type'] = 'month'
                extracted_item['value'] = val
                return "날짜_부분" # "Date_Part"
            if 1 <= val <= 31: # Possible day
                # If already labeled as month, this might be a day, or an ambiguous month/day
                extracted_item['date_part_type'] = 'day' if 'date_part_type' not in extracted_item or extracted_item['date_part_type'] != 'month' else 'month_or_day'
                extracted_item['value'] = val
                return "날짜_부분" # "Date_Part"


    # 4. Issuing Authority Seal/Stamp Candidate (typically at the bottom)
    if any(text.endswith(suffix) for suffix in KW_AUTHORITY_SUFFIXES) or \
       any(keyword in text for keyword in KW_AUTHORITY_KEYWORDS):
        if image_height > 0 and y_center > image_height * 0.7: # Lower 30% of the image
            return "발급기관_직인" # "IssuingAuthority_Seal"

    # 5. Issuing Authority Region Candidate (considered with the seal)
    cleaned_text_for_region = text.replace(" ","")
    is_region = any(keyword in cleaned_text_for_region for keyword in KW_REGION_NAMES) or \
                  any(cleaned_text_for_region.endswith(suffix) for suffix in KW_REGION_SUFFIXES)
    if is_region:
        # Region names can be part of the authority name, less strict on position initially
        if image_height > 0 and y_center > image_height * 0.6: # Lower 40%
             return "발급기관_지역" # "IssuingAuthority_Region"

    # 6. Address (relatively long text, contains specific keywords)
    is_address = any(keyword.strip() in text and len(text) > len(keyword.strip()) + 2 for keyword in KW_ADDRESS_TOKENS)
    if is_address or (len(text) > 15 and ("로" in text or "길" in text or "동" in text or "아파트" in text)): # Heuristic for street addresses
        if image_height > 0 and (image_height * 0.25 < y_center < image_height * 0.85): # Addresses can appear in a wide area
            return "주소" # "Address"

    # 7. Document Name (top of the image, specific keywords)
    if any(title in text for title in KW_DOC_TITLES):
        if image_height > 0 and y_center < image_height * 0.3: # Upper 30%
            box_h = get_box_height(box)
            if box_h > image_height * 0.025: # Text height is somewhat significant
                return "문서명" # "DocumentName"

    return "기타" # "Other"

# --- Refined Labeling Helper Functions ---
def _collect_label_candidates(labeled_items):
    """Categorizes initially labeled items into candidate groups."""
    candidates = {
        "seal": [], "region": [], "date_full": [], "date_part_year": [], "date_part_month": [], "date_part_day": [],
        "address": [], "name": [], "jumin": [], "doc_name": [], "other": []
    }
    for item in labeled_items: # Iterate through the original list of labeled items
        label = item['label']
        item_copy = item.copy() # Work with a copy to avoid modifying the original item in this stage
        if label == '발급기관_직인': candidates["seal"].append(item_copy)
        elif label == '발급기관_지역': candidates["region"].append(item_copy)
        elif label == '날짜_후보': candidates["date_full"].append(item_copy)
        elif label == '날짜_부분':
            part_type = item_copy.get('date_part_type')
            if part_type == 'year': candidates["date_part_year"].append(item_copy)
            elif part_type == 'month': candidates["date_part_month"].append(item_copy)
            elif part_type == 'day': candidates["date_part_day"].append(item_copy)
            else: candidates["other"].append(item_copy) # Ambiguous date parts go to "Other"
        elif label == '주소': candidates["address"].append(item_copy)
        elif label == '이름': candidates["name"].append(item_copy)
        elif label == '주민등록번호': candidates["jumin"].append(item_copy)
        elif label == '문서명': candidates["doc_name"].append(item_copy)
        else: candidates["other"].append(item_copy) # Default to "Other"
    return candidates

def _find_original_item_index(item_to_find, list_of_items):
    """Finds the index of an item in a list based on content, not object identity."""
    for idx, current_item in enumerate(list_of_items):
        if (current_item['text'] == item_to_find['text'] and
            current_item['bounding_box'] == item_to_find['bounding_box'] and
            current_item.get('label') == item_to_find.get('label')): # Compare key attributes
            return idx
    return -1 # Not found

def _determine_final_authority(seal_candidates, region_candidates, labeled_items, processed_indices):
    """Combines and determines the final issuing authority text."""
    primary_auth_candidates = [item for item in seal_candidates if any(keyword in item['text'] for keyword in KW_AUTHORITY_KEYWORDS) or any(item['text'].endswith(suffix) for suffix in KW_AUTHORITY_SUFFIXES)]
    
    if not primary_auth_candidates:
        primary_auth_candidates = seal_candidates
        if not primary_auth_candidates and region_candidates:
            region_candidates.sort(key=lambda x: (get_box_bottom(x['bounding_box']) if x['bounding_box'] else float('inf')), reverse=True)
            if region_candidates:
                chosen_region = region_candidates[0]
                final_authority_item = chosen_region.copy()
                final_authority_item['label'] = "발급기관"
                original_idx = _find_original_item_index(chosen_region, labeled_items)
                if original_idx != -1: processed_indices.add(original_idx)
                else: logger.warning(f"Could not find original item for standalone region authority: {chosen_region.get('text')}")
                logger.debug(f"Final Issuing Authority (Region only): {final_authority_item['text']}")
                return final_authority_item, processed_indices
            return None, processed_indices

    if not primary_auth_candidates: return None, processed_indices

    primary_auth_candidates.sort(key=lambda x: (get_box_bottom(x['bounding_box']) if x['bounding_box'] else float('-inf')), reverse=True)
    primary_authority_item = primary_auth_candidates[0]
    
    final_authority_text = primary_authority_item['text']
    # Start with the primary authority's bounding box coordinates
    all_coords_for_authority_bb = []
    if primary_authority_item['bounding_box']:
        all_coords_for_authority_bb.extend(primary_authority_item['bounding_box'])


    best_region_item = None
    if region_candidates and primary_authority_item['bounding_box']:
        seal_cx, seal_cy = get_box_center(primary_authority_item['bounding_box'])
        min_distance = float('inf')
        for region_item_cand in region_candidates:
            # Ensure we are not trying to combine an item with itself if it was in both lists
            if region_item_cand['text'] == primary_authority_item['text'] and region_item_cand['bounding_box'] == primary_authority_item['bounding_box']:
                continue

            if region_item_cand['bounding_box']:
                reg_cx, reg_cy = get_box_center(region_item_cand['bounding_box'])
                dist = ((seal_cy - reg_cy)**2) + ((seal_cx - reg_cx - get_box_width(region_item_cand['bounding_box']))**2 if seal_cx > reg_cx else (seal_cx - reg_cx)**2)
                if abs(seal_cy - reg_cy) < get_box_height(primary_authority_item['bounding_box']) * 1.5:
                    if dist < min_distance:
                        min_distance = dist
                        best_region_item = region_item_cand
    
    if best_region_item:
        if get_box_left(best_region_item['bounding_box']) < get_box_left(primary_authority_item['bounding_box']):
            final_authority_text = f"{best_region_item['text']} {primary_authority_item['text']}"
        else:
            final_authority_text = f"{primary_authority_item['text']} {best_region_item['text']}"
        
        if best_region_item['bounding_box']: all_coords_for_authority_bb.extend(best_region_item['bounding_box'])
        original_region_idx = _find_original_item_index(best_region_item, labeled_items)
        if original_region_idx != -1: processed_indices.add(original_region_idx)
        else: logger.warning(f"Could not find original item for authority region: {best_region_item.get('text')}")


    final_authority_obj = primary_authority_item.copy()
    final_authority_obj['label'] = "발급기관"
    final_authority_obj['text'] = final_authority_text

    if all_coords_for_authority_bb:
        min_x_agg = min(p[0] for p in all_coords_for_authority_bb)
        min_y_agg = min(p[1] for p in all_coords_for_authority_bb)
        max_x_agg = max(p[0] for p in all_coords_for_authority_bb)
        max_y_agg = max(p[1] for p in all_coords_for_authority_bb)
        final_authority_obj['bounding_box'] = [[min_x_agg, min_y_agg], [max_x_agg, min_y_agg], [max_x_agg, max_y_agg], [min_x_agg, max_y_agg]]
    
    original_primary_auth_idx = _find_original_item_index(primary_authority_item, labeled_items)
    if original_primary_auth_idx != -1: processed_indices.add(original_primary_auth_idx)
    else: logger.warning(f"Could not find original item for primary authority: {primary_authority_item.get('text')}")
    
    logger.debug(f"Final Issuing Authority: {final_authority_text}")
    return final_authority_obj, processed_indices


def _determine_final_issue_date(date_full_candidates, year_candidates, month_candidates, day_candidates, final_authority_item, labeled_items, processed_indices, image_width):
    """Determines the final issue date, prioritizing full dates, then combining parts."""
    if not final_authority_item or not final_authority_item['bounding_box']:
        if date_full_candidates: # If no authority, pick the lowest full date on the page
            date_full_candidates.sort(key=lambda it: get_box_bottom(it['bounding_box']), reverse=True)
            chosen_date_item = date_full_candidates[0]
            final_date = chosen_date_item.copy()
            final_date['label'] = "발급일자"
            final_date['text'] = final_date.get('normalized_date', final_date['text'])
            original_idx = _find_original_item_index(chosen_date_item, labeled_items)
            if original_idx != -1: processed_indices.add(original_idx)
            logger.debug(f"Final Issue Date (no authority, lowest full date): {final_date['text']}")
            return final_date, processed_indices
        return None, processed_indices

    auth_top_y = get_box_top(final_authority_item['bounding_box'])
    auth_center_x = get_box_center(final_authority_item['bounding_box'])[0]
    auth_height = get_box_height(final_authority_item['bounding_box'])

    # 1. Use full date candidates if available and suitably positioned
    best_full_date_item = None
    if date_full_candidates:
        min_dist_to_auth = float('inf')
        for date_item_cand in date_full_candidates:
            if not date_item_cand['bounding_box']: continue
            date_bottom_y = get_box_bottom(date_item_cand['bounding_box'])
            date_center_x = get_box_center(date_item_cand['bounding_box'])[0]
            
            if date_bottom_y < auth_top_y : # Date should be above the authority
                vertical_gap = auth_top_y - date_bottom_y
                horizontal_gap = abs(date_center_x - auth_center_x)
                if vertical_gap < auth_height * 2.5 and horizontal_gap < image_width * 0.3:
                    current_dist = vertical_gap + horizontal_gap * 0.5 
                    if current_dist < min_dist_to_auth:
                        min_dist_to_auth = current_dist
                        best_full_date_item = date_item_cand
        
        if best_full_date_item:
            final_date = best_full_date_item.copy()
            final_date['label'] = "발급일자"
            final_date['text'] = final_date.get('normalized_date', final_date['text'])
            original_idx = _find_original_item_index(best_full_date_item, labeled_items)
            if original_idx != -1: processed_indices.add(original_idx)
            logger.debug(f"Final Issue Date (Full Date Candidate): {final_date['text']}")
            return final_date, processed_indices

    # 2. Combine date parts (Year, Month, Day)
    # Filter parts that are positioned reasonably relative to the authority
    rel_years = [yc for yc in year_candidates if yc['bounding_box'] and get_box_bottom(yc['bounding_box']) < auth_top_y + auth_height * 0.5] # Slightly below top is OK
    rel_months = [mc for mc in month_candidates if mc['bounding_box'] and get_box_bottom(mc['bounding_box']) < auth_top_y + auth_height * 0.5]
    rel_days = [dc for dc in day_candidates if dc['bounding_box'] and get_box_bottom(dc['bounding_box']) < auth_top_y + auth_height * 0.5]


    if rel_years and rel_months and rel_days:
        best_ymd_combo = None
        min_ymd_metric = float('inf') # Metric to minimize (e.g., span + y-variance)

        for y_item in rel_years:
            for m_item in rel_months:
                # Ensure month is different from year if they came from same initial 'month_or_day'
                if y_item.get('value') == m_item.get('value') and y_item['text'] == m_item['text'] and y_item['bounding_box'] == m_item['bounding_box']:
                    continue
                for d_item in rel_days:
                    if (y_item.get('value') == d_item.get('value') and y_item['text'] == d_item['text'] and y_item['bounding_box'] == d_item['bounding_box']) or \
                       (m_item.get('value') == d_item.get('value') and m_item['text'] == d_item['text'] and m_item['bounding_box'] == d_item['bounding_box']):
                        continue
                    
                    current_combo = [y_item, m_item, d_item]
                    # Sort by x-coordinate to check for Y M D order
                    current_combo.sort(key=lambda p: get_box_left(p['bounding_box']))
                    
                    # Check if sorted order matches Y, M, D types (simplified)
                    if not (current_combo[0].get('date_part_type') == 'year' and \
                            current_combo[1].get('date_part_type') == 'month' and \
                            current_combo[2].get('date_part_type') == 'day'):
                        continue # Not a valid Y M D sequence by x-position

                    y_coords = [get_box_center(p['bounding_box'])[1] for p in current_combo]
                    y_variance = max(y_coords) - min(y_coords)
                    
                    # Prefer items on a similar horizontal line
                    if y_variance < get_box_height(current_combo[0]['bounding_box']) * 1.5: # Height of year part
                        min_x_val = get_box_left(current_combo[0]['bounding_box'])
                        max_x_val = get_box_right(current_combo[2]['bounding_box'])
                        current_span = max_x_val - min_x_val
                        
                        # Metric: smaller span and smaller y_variance are better
                        current_metric = current_span + y_variance * 2 
                        if current_metric < min_ymd_metric:
                            min_ymd_metric = current_metric
                            best_ymd_combo = (current_combo[0], current_combo[1], current_combo[2]) # Store the x-sorted combo

        if best_ymd_combo:
            y_final, m_final, d_final = best_ymd_combo
            combined_date_text = f"{y_final['value']}.{m_final['value']:02d}.{d_final['value']:02d}"
            
            all_coords_ymd = []
            for part_item in [y_final, m_final, d_final]:
                if part_item['bounding_box']: all_coords_ymd.extend(part_item['bounding_box'])
            
            combined_bb_ymd = None
            if all_coords_ymd:
                min_x = min(c[0] for c in all_coords_ymd); min_y = min(c[1] for c in all_coords_ymd)
                max_x = max(c[0] for c in all_coords_ymd); max_y = max(c[1] for c in all_coords_ymd)
                combined_bb_ymd = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

            final_date_obj = {
                'text': combined_date_text, 'label': '발급일자',
                'confidence': min(p['confidence'] for p in [y_final, m_final, d_final]),
                'bounding_box': combined_bb_ymd
            }
            for part_item in [y_final, m_final, d_final]:
                original_idx = _find_original_item_index(part_item, labeled_items)
                if original_idx != -1: processed_indices.add(original_idx)
            logger.debug(f"Final Issue Date (Combined Parts): {final_date_obj['text']}")
            return final_date_obj, processed_indices
        
    return None, processed_indices


def _determine_final_address(address_candidates, labeled_items, processed_indices, image_width):
    """Combines address segments and determines the final address."""
    valid_address_items = []
    for item_cand in address_candidates:
        original_idx = _find_original_item_index(item_cand, labeled_items)
        if original_idx != -1 and original_idx not in processed_indices:
            valid_address_items.append(item_cand) # Use the original item if found, or the copy
        elif original_idx == -1 : # If copy was not found in original (should not happen with current _collect_label_candidates)
             valid_address_items.append(item_cand)


    if not valid_address_items:
        return [], processed_indices

    valid_address_items.sort(key=lambda it: get_box_top(it['bounding_box']))
    
    merged_addresses_groups = [] # List of lists (groups of items to be merged)
    current_merge_group = []

    for i, item_to_merge in enumerate(valid_address_items):
        if not item_to_merge['bounding_box']: continue
        if not current_merge_group:
            current_merge_group.append(item_to_merge)
        else:
            prev_item_in_group = current_merge_group[-1]
            vertical_distance = get_box_top(item_to_merge['bounding_box']) - get_box_bottom(prev_item_in_group['bounding_box'])
            avg_char_height = get_box_height(prev_item_in_group['bounding_box']) or 10

            prev_left, prev_right = get_box_left(prev_item_in_group['bounding_box']), get_box_right(prev_item_in_group['bounding_box'])
            curr_left, curr_right = get_box_left(item_to_merge['bounding_box']), get_box_right(item_to_merge['bounding_box'])
            horizontal_overlap = max(0, min(prev_right, curr_right) - max(prev_left, curr_left))
            horizontal_ok = horizontal_overlap > - (image_width * 0.05) or abs(curr_left - prev_left) < image_width * 0.25 # Allow small negative overlap (gap) or similar start

            if vertical_distance < avg_char_height * 2.0 and horizontal_ok: # Slightly more lenient merge condition
                current_merge_group.append(item_to_merge)
            else:
                merged_addresses_groups.append(list(current_merge_group))
                current_merge_group = [item_to_merge]
    
    if current_merge_group:
        merged_addresses_groups.append(list(current_merge_group))

    final_address_objects_list = []
    for group_of_items in merged_addresses_groups:
        if not group_of_items: continue
        group_of_items.sort(key=lambda x: (get_box_top(x['bounding_box']), get_box_left(x['bounding_box'])))
        
        group_texts_list = [g_item['text'] for g_item in group_of_items]
        combined_text_for_group = reorder_korean_address(" ".join(group_texts_list).strip())
        
        all_coords_for_group_bb = []
        min_conf_for_group = 1.0
        for g_item_in_group in group_of_items:
            if g_item_in_group['bounding_box']: all_coords_for_group_bb.extend(g_item_in_group['bounding_box'])
            min_conf_for_group = min(min_conf_for_group, g_item_in_group['confidence'])
            original_idx = _find_original_item_index(g_item_in_group, labeled_items)
            if original_idx != -1: processed_indices.add(original_idx)

        bb_for_group = None
        if all_coords_for_group_bb:
            min_x = min(c[0] for c in all_coords_for_group_bb); min_y = min(c[1] for c in all_coords_for_group_bb)
            max_x = max(c[0] for c in all_coords_for_group_bb); max_y = max(c[1] for c in all_coords_for_group_bb)
            bb_for_group = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        
        if combined_text_for_group:
            final_address_objects_list.append({
                'text': combined_text_for_group, 'label': '주소',
                'confidence': min_conf_for_group, 'bounding_box': bb_for_group
            })
            logger.debug(f"Final Address (merged): {combined_text_for_group}")
            
    return final_address_objects_list, processed_indices


def _select_best_single_item(candidates, label_name, labeled_items, processed_indices, 
                             sort_key_func=None, filter_func=None, reverse_sort=True):
    """Selects the best single item from candidates based on sorting and filtering."""
    if not candidates: return None, processed_indices
    
    valid_candidates = []
    for c_item in candidates: # c_item is a copy from _collect_label_candidates
        original_idx = _find_original_item_index(c_item, labeled_items)
        if original_idx != -1 and original_idx not in processed_indices:
            valid_candidates.append(c_item) # Add the copy to valid_candidates
        elif original_idx == -1: # Should ideally not happen if c_item was indeed from labeled_items
            logger.warning(f"_select_best_single_item: Candidate item '{c_item.get('text')}' not found in original labeled_items. Skipping.")

    if filter_func:
        valid_candidates = [c for c in valid_candidates if filter_func(c)]
    
    if not valid_candidates: return None, processed_indices
    
    if sort_key_func:
        valid_candidates.sort(key=sort_key_func, reverse=reverse_sort)
    
    best_item_from_candidates = valid_candidates[0] # This is a copy
    final_item_object = best_item_from_candidates.copy() # Work with another copy for the final result
    final_item_object['label'] = label_name
    
    # Mark the original item corresponding to best_item_from_candidates as processed
    original_idx_of_best = _find_original_item_index(best_item_from_candidates, labeled_items)
    if original_idx_of_best != -1:
        processed_indices.add(original_idx_of_best)
    else:
        logger.warning(f"Could not find original item for {label_name}: {best_item_from_candidates.get('text')}. This may lead to data duplication or mislabeling.")

    logger.debug(f"Final {label_name}: {final_item_object['text']}")
    return final_item_object, processed_indices
# --- End Refined Labeling Helper Functions ---


def refine_and_finalize_labels(labeled_items, image_width, image_height):
    """Refines initial labels and finalizes them by considering relationships between text items."""
    final_results = []
    processed_indices = set() # Stores indices of items from 'labeled_items' that have been processed

    candidates = _collect_label_candidates(labeled_items) # Candidates are copies

    # Name (highest on the page, among candidates)
    final_name, processed_indices = _select_best_single_item(
        candidates["name"], "이름", labeled_items, processed_indices,
        sort_key_func=lambda x: (get_box_top(x['bounding_box']) if x['bounding_box'] else float('inf')), 
        reverse_sort=False # Ascending sort (lower y-value is higher on page)
    )
    if final_name: final_results.append(final_name)

    # Resident Registration Number (highest confidence)
    final_jumin, processed_indices = _select_best_single_item(
        candidates["jumin"], "주민등록번호", labeled_items, processed_indices,
        sort_key_func=lambda x: x['confidence'],
        reverse_sort=True # Descending sort (higher confidence first)
    )
    if final_jumin: final_results.append(final_jumin)
    
    # Document Name (highest confidence, then closest to center-top)
    final_doc_name, processed_indices = _select_best_single_item(
        candidates["doc_name"], "문서명", labeled_items, processed_indices,
        sort_key_func=lambda x: (x['confidence'], -abs(get_box_center(x['bounding_box'])[0] - image_width/2) if x['bounding_box'] and image_width > 0 else 0),
        reverse_sort=True
    )
    if final_doc_name: final_results.append(final_doc_name)

    # Issuing Authority
    final_authority, processed_indices = _determine_final_authority(
        candidates["seal"], candidates["region"], labeled_items, processed_indices
    )
    if final_authority: final_results.append(final_authority)

    # Issue Date (based on authority's position)
    final_issue_date, processed_indices = _determine_final_issue_date(
        candidates["date_full"], candidates["date_part_year"], candidates["date_part_month"], candidates["date_part_day"],
        final_authority, labeled_items, processed_indices, image_width
    )
    if final_issue_date: final_results.append(final_issue_date)

    # Address
    final_address_list, processed_indices = _determine_final_address(
        candidates["address"], labeled_items, processed_indices, image_width
    )
    final_results.extend(final_address_list)
    
    # Handle "Other" items
    # Add items initially labeled as "Other" if they haven't been processed as part of another category
    for other_item_cand in candidates["other"]: # These are copies
        original_idx = _find_original_item_index(other_item_cand, labeled_items)
        if original_idx != -1 and original_idx not in processed_indices:
            final_item = other_item_cand.copy() # It's already a copy, but for clarity
            final_item['label'] = '기타'
            final_results.append(final_item)
            processed_indices.add(original_idx)
            
    # Add any remaining unprocessed items from the original list as "Other"
    for i, original_item_from_list in enumerate(labeled_items):
        if i not in processed_indices:
            item_copy_for_other = original_item_from_list.copy()
            item_copy_for_other['label'] = '기타'
            final_results.append(item_copy_for_other)
            # processed_indices.add(i) # Not strictly necessary to add here as it's the last loop

    return final_results


def get_system_font_path():
    """Returns a path to a default system font suitable for the OS (checks existence)."""
    system = platform.system()
    font_path = None
    if system == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf" # Malgun Gothic
    elif system == "Darwin": # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc" # Apple SD Gothic Neo
        if not os.path.exists(font_path): # Fallback
            font_path = "/Library/Fonts/AppleGothic.ttf"
    elif system == "Linux":
        # Common Korean font paths on Linux
        common_linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",      # Nanum Gothic (common)
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf", # Noto Sans CJK KR
            "/usr/share/fonts/korean/NanumGothic.ttf",             # Another possible Nanum path
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # DejaVu (often default, but not Korean)
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break
    if font_path and os.path.exists(font_path):
        logger.debug(f"OS ({system}) system font found: {font_path}")
        return font_path
    logger.debug(f"OS ({system}) system font not found at expected paths (Path checked: {font_path}). Visualization might use default.")
    return None

def determine_font_for_visualization():
    """Determines the font path to use for visualization, logging the choice."""
    font_path_to_use = get_system_font_path()
    if font_path_to_use:
        logger.info(f"Using system-detected font for text visualization: {font_path_to_use}")
        return font_path_to_use

    logger.info("System font not found. Checking for local fonts in 'fonts' directory.")
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd() # Fallback if __file__ is not defined

    local_korean_font_candidates = [
        os.path.join(script_dir, 'fonts', 'malgun.ttf'),    # e.g., Malgun Gothic
        os.path.join(script_dir, 'fonts', 'NanumGothic.ttf') # e.g., Nanum Gothic
    ]
    for local_font in local_korean_font_candidates:
        if os.path.exists(local_font):
            logger.info(f"Using local font: {local_font}")
            return local_font

    logger.warning("No specific Korean font found for visualization. PaddleOCR's internal default will be used (may not support Korean well).")
    return None # Let PaddleOCR use its default


def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename,
                       preprocessed_img=None, show_image_flag=False, font_path_to_use=None):
    """Displays and saves the OCR results overlaid on the image."""
    if not extracted_data:
        logger.info(f"{original_filename}: No OCR results to visualize.")
        return

    try:
        if preprocessed_img is not None:
            # Convert grayscale to RGB if necessary for display
            image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB) if len(preprocessed_img.shape) == 2 else preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else: # Load original image if no preprocessed one is provided
            image = Image.open(original_image_path).convert('RGB')

        # Prepare data for draw_ocr: boxes, texts, scores
        boxes = [item['bounding_box'] for item in extracted_data if item.get('bounding_box')]
        # Include the determined label in the displayed text
        txts = [f"{item.get('label', 'N/A')}: {item['text']}" for item in extracted_data if item.get('bounding_box')]
        scores = [item['confidence'] for item in extracted_data if item.get('bounding_box')]

        if not boxes: # No valid bounding boxes to draw
            logger.info(f"{original_filename}: No bounding boxes to visualize.")
            return

        # Use PaddleOCR's utility to draw results
        im_show = draw_ocr(np.array(image), boxes, txts, scores, font_path=font_path_to_use)
        im_show_pil = Image.fromarray(im_show) # Convert back to PIL Image

        # Save the visualized image
        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr_result{ext}"
        output_image_path = os.path.join(output_dir, output_image_filename)
        im_show_pil.save(output_image_path)
        logger.debug(f"OCR result visualization saved to: {output_image_path}")

        if show_image_flag: # Optionally display the image
            logger.info(f"Displaying OCR result for {original_filename}.")
            im_show_pil.show()
    except FileNotFoundError:
        logger.error(f"Original image file not found for visualization: '{original_image_path}'.")
    except Exception as e:
        logger.error(f"Exception during OCR result visualization (File: {original_filename}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))


def process_single_image_task(task_args_tuple):
    """Worker function to process a single image: preprocess, OCR, label, and save results."""
    global _worker_csv_lock, _worker_csv_file_path # Use global lock and path set by initializer
    (current_image_path, filename, ocr_engine_params, output_dir,
     skip_preprocessing, show_image, font_path) = task_args_tuple

    status_message = f"Error processing {filename}" # Default status
    try:
        logger.debug(f"Worker {os.getpid()}: Starting processing for: {filename}")
        ocr_input_image_data = current_image_path # Path or NumPy array
        processed_image_for_display = None # For visualization if preprocessed
        image_width, image_height = 0, 0
        try: # Get image dimensions for labeling heuristics
            with Image.open(current_image_path) as img_pil: image_width, image_height = img_pil.size
        except Exception as e: logger.error(f"Worker {os.getpid()}: Error reading image dimensions ({filename}): {e}")

        if not skip_preprocessing:
            processed_img_np_array = preprocess_image_for_ocr(current_image_path)
            if processed_img_np_array is not None:
                ocr_input_image_data = processed_img_np_array
                processed_image_for_display = processed_img_np_array # Save for display
            else: logger.warning(f"Worker {os.getpid()}: Preprocessing failed for {filename}; attempting OCR on original.")

        # Step 1: Extract text
        extracted_text_items = extract_text_from_image_worker(ocr_engine_params, ocr_input_image_data, filename_for_log=filename)

        if extracted_text_items:
            # Step 2: Initial labeling
            initially_labeled_items_list = []
            for item_dict in extracted_text_items:
                # Perform initial labeling only if image dimensions are known
                initial_label = label_text_item_initial(item_dict, image_width, image_height) if image_width > 0 and image_height > 0 else "기타"
                item_dict['label'] = initial_label # Add initial label to the item dictionary
                initially_labeled_items_list.append(item_dict)
            
            # Step 3: Refine and finalize labels
            final_labeled_data_list = refine_and_finalize_labels(initially_labeled_items_list, image_width, image_height)

            # Step 4: Prepare data for CSV
            csv_rows_to_write_list = []
            for final_item in final_labeled_data_list:
                csv_rows_to_write_list.append([
                    filename, final_item.get('label', '기타'), final_item.get('text', '').replace('"', '""'), # Escape quotes for CSV
                    round(final_item.get('confidence', 0.0), 4), str(final_item.get('bounding_box', 'N/A'))
                ])
            
            # Write to CSV (synchronized)
            if csv_rows_to_write_list and _worker_csv_lock and _worker_csv_file_path:
                with _worker_csv_lock:
                    try:
                        with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                            csv_writer_obj = csv.writer(csvfile)
                            csv_writer_obj.writerows(csv_rows_to_write_list)
                        logger.info(f"Worker {os.getpid()}: Appended text from {filename} to {os.path.basename(_worker_csv_file_path)}.")
                    except IOError as e: logger.error(f"Worker {os.getpid()}: CSV write error for {filename}: {e}")
            
            # Step 5: Visualize results (using final labels)
            display_ocr_result(current_image_path, final_labeled_data_list, output_dir, filename,
                               preprocessed_img=processed_image_for_display,
                               show_image_flag=show_image, font_path_to_use=font_path)
            status_message = f"{filename} processed successfully."
        else: # No text extracted
            logger.info(f"Worker {os.getpid()}: No text extracted from {filename}.")
            status_message = f"No text found in {filename}."
            # Optionally, write a "no text" row to CSV
            if _worker_csv_lock and _worker_csv_file_path:
                 with _worker_csv_lock:
                     try:
                         with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                             csv.writer(csvfile).writerow([filename, "N/A", "No text extracted", 0.0, "N/A"])
                     except IOError as e: logger.error(f"Worker {os.getpid()}: CSV write error for {filename} (no text): {e}")
        return status_message
    except Exception as e: # Catch-all for unexpected errors in the worker
        logger.error(f"Worker {os.getpid()}: Unhandled exception in process_single_image_task (File: {filename}): {e}", exc_info=True)
        # Attempt to write an error row to CSV
        if _worker_csv_lock and _worker_csv_file_path:
             with _worker_csv_lock:
                 try:
                     with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                         csv.writer(csvfile).writerow([filename, "Error", str(e), 0.0, "N/A"])
                 except IOError as io_err: logger.error(f"Worker {os.getpid()}: CSV write error for {filename} (task error): {io_err}")
        return status_message # Return the default error status


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR script to extract text from images in a folder.",
        formatter_class=argparse.RawTextHelpFormatter # Allows for newlines in help text
    )
    parser.add_argument("input_dir", nargs='?', default='input',
                        help="Path to the folder containing images for text extraction.\n(Default: 'input')")
    parser.add_argument("--output_dir", default='output',
                        help="Path to the folder where OCR result images and text files will be saved.\n(Default: 'output')")
    parser.add_argument("--lang", default='korean',
                        help="Language to use for OCR (e.g., 'korean', 'en', 'ch_sim').\n(Default: 'korean')")
    parser.add_argument("--show_image", action='store_true',
                        help="Display each processed image with OCR results on screen.")
    parser.add_argument("--no_preprocess", action='store_true',
                        help="Skip the image preprocessing step.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}',
                        help="Show script's version number and exit.")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug level logging for more detailed output.")
    parser.add_argument('--use_gpu', action='store_true',
                        help="Attempt to use GPU for OCR processing if available.\n(Requires NVIDIA GPU and CUDA environment)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes for parallel processing.\n(Default: Number of CPU cores)")
    return parser.parse_args()

def prepare_directories(input_dir_path, output_dir_path):
    """Checks input directory and creates output directory if it doesn't exist."""
    if not os.path.isdir(input_dir_path):
        logger.error(f"Input directory '{input_dir_path}' does not exist or is not a directory. Exiting.")
        exit(1)
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            logger.info(f"Output directory created: {output_dir_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory '{output_dir_path}': {e}")
            exit(1)

def create_final_summary_csv(labeled_results_csv_path, final_output_csv_path):
    """Reads the intermediate 'ocr_labeled_text.csv' and creates a final summary 'ocr_text.csv'."""
    logger.info(f"Reading '{labeled_results_csv_path}' to create final summary CSV ('{os.path.basename(final_output_csv_path)}')...")
    # Structure for final CSV: one row per image file
    per_file_data_map = defaultdict(lambda: {
        "문서명": "", "이름": "", "주민등록번호": "", "주소": [], # Address can be multi-line
        "발급일자": "", "발급기관": "", "기타": [] # "Other" items
    })

    try:
        with open(labeled_results_csv_path, 'r', newline='', encoding='utf-8-sig') as infile_csv:
            csv_reader = csv.reader(infile_csv)
            header_row = next(csv_reader) # Skip header
            # Expected header: ["Image Filename", "Label", "Extracted Text", "Confidence", "Bounding Box (str)"]
            try:
                filename_col_idx = header_row.index('Image Filename')
                label_col_idx = header_row.index('Label')
                text_col_idx = header_row.index('Extracted Text')
            except ValueError:
                logger.error(f"Header mismatch in '{labeled_results_csv_path}'. Expected 'Image Filename', 'Label', 'Extracted Text'.")
                return

            for row_data in csv_reader:
                if len(row_data) <= max(filename_col_idx, label_col_idx, text_col_idx):
                    logger.warning(f"Skipping malformed row in intermediate CSV: {row_data}")
                    continue
                
                image_filename = row_data[filename_col_idx]
                label_type = row_data[label_col_idx]
                extracted_text = row_data[text_col_idx]

                if label_type == "N/A" or label_type == "Error": # Skip rows indicating no text or errors
                    continue

                current_file_entry = per_file_data_map[image_filename]
                if label_type == "기타": # "Other"
                    if extracted_text: current_file_entry[label_type].append(extracted_text)
                elif label_type == "주소": # "Address" - collect all parts
                    if extracted_text: current_file_entry[label_type].append(extracted_text)
                elif label_type in current_file_entry: # For single-value fields
                    # Use the first valid text found for these labels
                    if not current_file_entry[label_type] and extracted_text:
                        current_file_entry[label_type] = extracted_text
                    elif current_file_entry[label_type] and extracted_text:
                        # Log if multiple values are found for a supposedly single-value field
                        logger.debug(f"File '{image_filename}', Label '{label_type}': Existing value '{current_file_entry[label_type]}' -> New value '{extracted_text}'. Keeping first value or apply update logic if needed.")
                else: # Should not happen if all labels are predefined
                    logger.warning(f"Unknown label '{label_type}' encountered during final CSV creation (File: {image_filename}, Text: {extracted_text}). Treating as 'Other'.")
                    if extracted_text: current_file_entry["기타"].append(extracted_text)

    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {labeled_results_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading '{labeled_results_csv_path}': {e}", exc_info=True)
        return

    # Write the final summarized CSV
    final_csv_header = ["Image Filename", "문서명", "이름", "주민등록번호", "주소", "발급일자", "발급기관", "기타"]
    try:
        with open(final_output_csv_path, 'w', newline='', encoding='utf-8-sig') as outfile_csv:
            csv_writer_obj = csv.writer(outfile_csv)
            csv_writer_obj.writerow(final_csv_header)

            for image_filename, data_dict in sorted(per_file_data_map.items()): # Sort by filename
                # Combine address parts
                address_combined_text = ""
                if data_dict["주소"]:
                    # Remove duplicates while preserving order (if important, otherwise set is fine)
                    unique_address_parts = []
                    for addr_part in data_dict["주소"]:
                        clean_addr_part = addr_part.strip()
                        if clean_addr_part and clean_addr_part not in unique_address_parts:
                            unique_address_parts.append(clean_addr_part)
                    address_combined_text = " ".join(unique_address_parts)
                    # address_combined_text = reorder_korean_address(address_combined_text) # Already reordered in refine step

                # Combine "Other" items, remove duplicates, sort, and join
                unique_other_texts_list = sorted(list(set(s.strip() for s in data_dict["기타"] if s and s.strip())))
                other_text_combined = "; ".join(unique_other_texts_list)
                
                row_to_write = [
                    image_filename, data_dict["문서명"], data_dict["이름"], data_dict["주민등록번호"],
                    address_combined_text, data_dict["발급일자"], data_dict["발급기관"], other_text_combined
                ]
                csv_writer_obj.writerow(row_to_write)
        logger.info(f"Final summary CSV file created: {final_output_csv_path}")
    except IOError as e:
        logger.error(f"IOError writing to '{final_output_csv_path}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating final summary CSV: {e}", exc_info=True)


def main():
    """Main execution logic of the OCR script."""
    # Set multiprocessing start method for non-Windows systems if not already set
    # 'spawn' is generally safer than 'fork' on macOS and some Linux setups.
    try:
        if platform.system() != "Windows" and multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
            logger.debug("Multiprocessing start method set to 'spawn'.")
    except RuntimeError: # Can't set start method if already started
        logger.debug("Multiprocessing start method already set.")
    except AttributeError: # For older Python versions that might not have get_start_method
        logger.debug("multiprocessing.get_start_method(allow_none=True) not available.")
    
    if platform.system() == "Windows": # Required for multiprocessing on Windows when freezing script
        multiprocessing.freeze_support()

    args = parse_arguments()

    # Setup main process logging
    log_level_main = logging.DEBUG if args.debug else logging.INFO
    # Set environment variable for worker process logging level
    os.environ[_worker_log_level_env_var] = logging.getLevelName(log_level_main)
    setup_logging(log_level_main)

    logger.info(f"OCR Script Version: {__version__}")
    logger.info(f"Command-line arguments: {args}")

    prepare_directories(args.input_dir, args.output_dir)

    supported_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_filenames_list = [
        f_name for f_name in os.listdir(args.input_dir) if f_name.lower().endswith(supported_image_extensions)
    ]

    if not image_filenames_list:
        logger.warning(f"No supported image files found in input directory: '{args.input_dir}'. Exiting.")
        exit(0)

    # Determine number of worker processes
    num_workers_to_use = args.num_workers if args.num_workers is not None and args.num_workers > 0 else os.cpu_count()
    if num_workers_to_use is None: # Fallback if os.cpu_count() returns None
        num_workers_to_use = 1
        logger.warning("Could not detect CPU core count. Using 1 worker process.")
    logger.info(f"Number of worker processes to use: {num_workers_to_use}")
    logger.info(f"Total images to process: {len(image_filenames_list)}")

    # PaddleOCR engine parameters
    ocr_engine_configuration = {
        'use_angle_cls': True, # Enable text angle classification
        'lang': args.lang,
        'use_gpu': args.use_gpu,
        'show_log': False # Control PaddleOCR's internal logging via script's logger
    }
    if '+' in args.lang: # e.g., 'korean+en' for multiple languages
        logger.info(f"Detected composite language setting: '{args.lang}'. Ensure PaddleOCR supports this combination.")
    
    font_path_for_viz = determine_font_for_visualization() # Determine font for drawing results
    
    # Path for the intermediate CSV file (detailed, per-text-item results)
    intermediate_csv_file_path = os.path.join(args.output_dir, "ocr_labeled_text.csv")
    
    # Multiprocessing Manager for shared Lock object
    # Must be created before the Pool if the lock is passed via initargs
    mp_manager = multiprocessing.Manager()
    csv_access_lock = mp_manager.Lock() # Lock for synchronized access to the CSV file

    # Header for the intermediate CSV file
    csv_header_intermediate_fields = ["Image Filename", "Label", "Extracted Text", "Confidence", "Bounding Box (str)"]
    try: # Initialize the intermediate CSV file with its header
        with open(intermediate_csv_file_path, 'w', newline='', encoding='utf-8-sig') as csv_f:
            csv_writer = csv.writer(csv_f)
            csv_writer.writerow(csv_header_intermediate_fields)
        logger.info(f"Intermediate CSV file '{intermediate_csv_file_path}' initialized with header.")
    except IOError as e:
        logger.error(f"Error initializing intermediate CSV file '{intermediate_csv_file_path}': {e}. Exiting.")
        exit(1)

    # Prepare list of arguments for each task to be processed by the Pool
    tasks_for_pool = []
    for img_filename in image_filenames_list:
        full_image_path = os.path.join(args.input_dir, img_filename)
        tasks_for_pool.append((
            full_image_path, img_filename, ocr_engine_configuration, args.output_dir,
            args.no_preprocess, args.show_image, font_path_for_viz
        ))
    
    processing_pool = None
    try:
        # Create a Pool of worker processes
        processing_pool = multiprocessing.Pool(
            processes=num_workers_to_use,
            initializer=worker_initializer_func, # Function to run at start of each worker
            initargs=(csv_access_lock, intermediate_csv_file_path, _worker_log_level_env_var) # Args for initializer
        )
        logger.info("Starting parallel image processing...")

        # Use imap_unordered for progress tracking with tqdm as tasks complete
        # This processes tasks in parallel and yields results as they finish.
        for task_status_msg in tqdm(processing_pool.imap_unordered(process_single_image_task, tasks_for_pool),
                                  total=len(tasks_for_pool), desc="Overall Image Processing"):
            if task_status_msg: # Log status message from worker
                logger.debug(f"Task result received: {task_status_msg}")

        logger.info("All tasks submitted to pool and result iteration complete.")

    except Exception as e:
        logger.error(f"Unexpected error in main process during parallel processing: {e}", exc_info=True)
    finally:
        if processing_pool: # Ensure pool is properly closed and joined
            logger.info("Closing worker pool...")
            processing_pool.close() # Prevent new tasks from being submitted
            logger.info("Pool.close() called. Waiting for worker processes to terminate...")
            processing_pool.join() # Wait for all worker processes to finish
            logger.info("Pool.join() called. All worker processes have terminated.")

    logger.info(f"Processing of {len(image_filenames_list)} image files complete.")
    logger.info(f"Detailed extracted text information saved in: '{intermediate_csv_file_path}'")
    
    # Create the final summary CSV file from the intermediate results
    final_summary_csv_file_path = os.path.join(args.output_dir, "ocr_text.csv")
    create_final_summary_csv(intermediate_csv_file_path, final_summary_csv_file_path)
    
    logger.info(f"Script execution finished. All results saved in '{args.output_dir}' folder.")

if __name__ == "__main__":
    main()
