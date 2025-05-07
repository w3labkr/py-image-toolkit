# PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 스크립트 (OpenCV 전처리 및 폴더/병렬 처리 추가)
import cv2 # OpenCV 라이브러리 임포트
from PIL import Image
import numpy as np # NumPy 임포트 (OpenCV 이미지 처리에 사용)
import os # 파일 및 디렉토리 관리를 위해 os 모듈 임포트
import argparse # 명령줄 인자 처리를 위해 argparse 모듈 임포트
import platform # 운영체제 감지를 위해 platform 모듈 임포트
from tqdm import tqdm # 진행률 표시를 위해 tqdm 임포트
import logging # 로깅 모듈 임포트
import multiprocessing # 병렬 처리를 위해 multiprocessing 임포트
import warnings # 경고 메시지 제어를 위해 warnings 모듈 임포트
import csv # CSV 파일 처리를 위해 csv 모듈 임포트
import re # 텍스트 라벨링을 위한 정규 표현식 모듈
import math # 좌표 거리 계산을 위해 math 모듈 임포트
from collections import defaultdict # 기본값을 가진 딕셔너리 사용

# 스크립트 시작 시점에 ccache 경고 필터 설정 (PaddleOCR 임포트 전에 적용되도록)
warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")

from paddleocr import PaddleOCR, draw_ocr

__version__ = "1.1.2" # 스크립트 버전 정보 (_select_best_single_item reverse 인자 오류 수정)

# --- Constants for Labeling ---
RE_JUMIN_NUMBER_FULL = r"\d{6}\s*-\s*\d{7}"
RE_JUMIN_NUMBER_CLEAN = r"\d{13}"
RE_KOREAN_NAME = r"^[가-힣]+$"
RE_DATE_YYYY_MM_DD = r"(\d{4})\s*[\.,년]?\s*(\d{1,2})\s*[\.,월]?\s*(\d{1,2})\s*[\.일]?"

KW_ADDRESS_TOKENS = ["특별시", "광역시", "도", "시 ", "군 ", "구 ", "읍 ", "면 ", "동 ", "리 ", "로 ", "길 ", "아파트", "빌라", " 번지", " 호"] # 공백 포함 주의
KW_DOC_TITLES = ["주민등록증", "운전면허증", "공무원증"]
KW_AUTHORITY_SUFFIXES = ["청장", "시장", "군수", "구청장", "경찰서장", "지방경찰청장", "위원회위원장"] # 위원회위원장 추가
KW_AUTHORITY_KEYWORDS = ["발급기관", "기관명"] # 발급기관 명시 키워드
KW_REGION_NAMES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주"]
KW_REGION_SUFFIXES = ["특별시", "광역시", "특별자치시", "도", "특별자치도"]
# --- End Constants ---


# --- 로거 설정 ---
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """기본 로거를 설정합니다."""
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    else:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
            if isinstance(handler, logging.StreamHandler):
                 logger.propagate = False

# 작업자 프로세스 내에서 사용할 전역 변수
_worker_csv_lock = None
_worker_csv_file_path = None
_worker_log_level_env_var = 'OCR_LOG_LEVEL' # 기본값

def worker_initializer_func(lock, csv_path, log_level_env_name):
    """각 작업자 프로세스 시작 시 호출될 초기화 함수입니다."""
    global _worker_csv_lock, _worker_csv_file_path, _worker_log_level_env_var

    _worker_csv_lock = lock
    _worker_csv_file_path = csv_path
    _worker_log_level_env_var = log_level_env_name

    warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")
    log_level_str = os.environ.get(_worker_log_level_env_var, 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level)
    logger.debug(f"작업자 {os.getpid()}: 초기화 완료 (Lock, CSV 경로 설정, 로깅 레벨: {log_level_str}).")


def preprocess_image_for_ocr(image_path):
    """OCR 정확도 향상을 위해 이미지를 전처리합니다."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지를 불러올 수 없습니다: {image_path}")
            return None

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                if median_angle > 45: median_angle -= 90
                elif median_angle < -45: median_angle += 90

                if abs(median_angle) > 0.5:
                    logger.debug(f"이미지 회전 보정: {os.path.basename(image_path)}, 각도: {median_angle:.2f}°")
                    height, width = img.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (width, height),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    logger.debug(f"이미지 회전이 필요하지 않음: {os.path.basename(image_path)}, 감지된 각도: {median_angle:.2f}°")
        else:
            logger.debug(f"직선을 감지할 수 없어 회전 보정을 건너뜁니다: {os.path.basename(image_path)}")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)
        logger.debug(f"전처리 완료 (수평 보정, CLAHE, Median Blur): {os.path.basename(image_path)}")
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV 오류 (전처리, 파일: {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"예외 발생 (전처리, 파일: {os.path.basename(image_path)}): {e}")
        return None


def extract_text_from_image_worker(ocr_engine_params, image_data, filename_for_log=""):
    """주어진 이미지 데이터에서 텍스트를 추출합니다. (작업자 프로세스용)"""
    ocr_engine = None
    try:
        logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 초기화 중... (파일: {filename_for_log}, 파라미터: {ocr_engine_params})")
        ocr_engine = PaddleOCR(**ocr_engine_params)
        logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 초기화 완료. (파일: {filename_for_log})")

        logger.debug(f"작업자 {os.getpid()}: OCR 시작: {filename_for_log}")
        result = ocr_engine.ocr(image_data, cls=True)
        logger.debug(f"작업자 {os.getpid()}: OCR 완료: {filename_for_log}")

        extracted_items = []
        if result and result[0] is not None:
            for line_info in result[0]:
                text, confidence = line_info[1]
                bounding_box = line_info[0]
                try:
                    float_box = [[float(p[0]), float(p[1])] for p in bounding_box]
                    extracted_items.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": float_box
                    })
                except (ValueError, TypeError):
                     logger.warning(f"잘못된 바운딩 박스 데이터 형식: {bounding_box} (텍스트: {text})")
                     extracted_items.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": None
                    })
        return extracted_items
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: OCR 처리 중 오류 (파일: {filename_for_log}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
        return None
    finally:
        if ocr_engine:
            del ocr_engine
            logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 삭제됨 (파일: {filename_for_log})")

# --- 바운딩 박스 유틸리티 함수 ---
def get_box_coordinates(box):
    if box and len(box) == 4 and all(len(pt) == 2 for pt in box):
        return box
    return None

def get_box_center(box):
    coords = get_box_coordinates(box)
    if coords:
        x_center = sum(pt[0] for pt in coords) / 4
        y_center = sum(pt[1] for pt in coords) / 4
        return x_center, y_center
    return None, None

def get_box_top(box):
    coords = get_box_coordinates(box)
    if coords:
        return min(pt[1] for pt in coords)
    return float('inf')

def get_box_bottom(box):
    coords = get_box_coordinates(box)
    if coords:
        return max(pt[1] for pt in coords)
    return float('-inf')

def get_box_left(box):
    coords = get_box_coordinates(box)
    if coords:
        return min(pt[0] for pt in coords)
    return float('inf')

def get_box_right(box):
    coords = get_box_coordinates(box)
    if coords:
        return max(pt[0] for pt in coords)
    return float('-inf')

def get_box_height(box):
    coords = get_box_coordinates(box)
    if coords:
        min_y = min(pt[1] for pt in coords)
        max_y = max(pt[1] for pt in coords)
        return max_y - min_y
    return 0

def get_box_width(box):
    coords = get_box_coordinates(box)
    if coords:
        min_x = min(pt[0] for pt in coords)
        max_x = max(pt[0] for pt in coords)
        return max_x - min_x
    return 0
# --- 바운딩 박스 유틸리티 함수 끝 ---

def reorder_korean_address(address_text):
    """한국식 주소 형식을 올바른 순서로 재정렬합니다."""
    address_patterns = [
        r'([가-힣0-9]+(?:로|길)[0-9]*)\s+([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)[가-힣]*(?:시|군|구)?)',
        r'([0-9]+[A-Za-z]?(?:동|호|층|번지)?)\s+([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)[가-힣]*(?:시|군|구)?)'
    ]
    for pattern in address_patterns:
        match = re.search(pattern, address_text)
        if match:
            part1, part2 = match.groups()
            reordered = re.sub(pattern, r'\2 \1', address_text)
            logger.debug(f"주소 순서 재정렬: '{address_text}' -> '{reordered}'")
            address_text = reordered
    address_text = re.sub(r'([0-9]+)\s+([A-Za-z])\s+동', r'\1\2동', address_text)
    address_text = re.sub(r'([A-Za-z])\s+동', r'\1동', address_text)
    return address_text

def label_text_item_initial(extracted_item, image_width, image_height):
    """추출된 텍스트 항목에 초기 라벨을 할당합니다."""
    text = extracted_item['text']
    box = extracted_item['bounding_box']
    x_center, y_center = get_box_center(box)

    if x_center is None or y_center is None:
        logger.warning(f"유효하지 않은 바운딩 박스로 초기 라벨링 시도: {box} (텍스트: {text})")
        return "기타"

    # 1. 주민등록번호
    # 정규식 검사 후, 13자리 숫자인지, 그리고 실제 유효한 주민번호 패턴인지 추가 검증 가능 (여기서는 형식만)
    if re.fullmatch(RE_JUMIN_NUMBER_FULL, text) or \
       re.fullmatch(RE_JUMIN_NUMBER_CLEAN, text.replace("-","").replace(" ","")):
        cleaned_text = text.replace("-","").replace(" ","")
        if len(cleaned_text) == 13 and cleaned_text.isdigit(): # 13자리 숫자인 경우에만 하이픈 추가
            extracted_item['text'] = f"{cleaned_text[:6]}-{cleaned_text[6:]}"
        return "주민등록번호"

    # 2. 이름
    cleaned_text_for_name = text.replace(" ","")
    if 2 <= len(cleaned_text_for_name) <= 5 and re.fullmatch(RE_KOREAN_NAME, cleaned_text_for_name):
        # 이름은 보통 이미지의 특정 영역에 위치 (예: 주민등록번호 근처 또는 상단)
        if image_height > 0 and (image_height * 0.1 < y_center < image_height * 0.5) and len(cleaned_text_for_name) <= 4: # 위치 조건 완화 및 일반화
             return "이름"

    # 3. 날짜 후보 (YYYY.MM.DD 형식)
    date_match = re.search(RE_DATE_YYYY_MM_DD, text)
    if date_match:
        try:
            year, month, day = map(int, date_match.groups())
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31: # 유효한 날짜 범위 검사
                 extracted_item['normalized_date'] = f"{year}.{month:02d}.{day:02d}"
                 return "날짜_후보"
        except ValueError:
            logger.debug(f"날짜 형식 변환 중 오류: {text}")


    # 3.1 날짜 부분 후보 (YYYY, MM, DD)
    if text.isdigit():
        val = int(text)
        if len(text) == 4 and 1900 <= val <= 2100: # 연도
            extracted_item['date_part_type'] = 'year'
            extracted_item['value'] = val
            return "날짜_부분"
        elif 1 <= len(text) <= 2: # 월 또는 일
            if 1 <= val <= 12: # 월 가능성
                extracted_item['date_part_type'] = 'month'
                extracted_item['value'] = val
                return "날짜_부분"
            if 1 <= val <= 31: # 일 가능성 (월 후보가 아니거나, 월/일 둘 다 가능할 때)
                # 만약 이미 'month'로 분류되었다면, 중복을 피하기 위해 다른 조건을 둘 수 있음
                # 여기서는 단순하게 둘 다 '날짜_부분'으로 하고, refine 단계에서 구분
                extracted_item['date_part_type'] = 'day' if 'date_part_type' not in extracted_item or extracted_item['date_part_type'] != 'month' else 'month_or_day'
                extracted_item['value'] = val
                return "날짜_부분"


    # 4. 발급기관 직인 후보 (보통 이미지 하단에 위치)
    if any(text.endswith(suffix) for suffix in KW_AUTHORITY_SUFFIXES) or \
       any(keyword in text for keyword in KW_AUTHORITY_KEYWORDS):
        if image_height > 0 and y_center > image_height * 0.7: # 하단 30%
            return "발급기관_직인" # "발급기관_후보" 등으로 변경하여 refine에서 처리

    # 5. 발급기관 지역 후보 (직인과 함께 고려됨)
    cleaned_text_for_region = text.replace(" ","")
    is_region = any(keyword in cleaned_text_for_region for keyword in KW_REGION_NAMES) or \
                  any(cleaned_text_for_region.endswith(suffix) for suffix in KW_REGION_SUFFIXES)
    if is_region:
        # 지역명은 발급기관명의 일부로 사용될 수 있으므로, 위치 제약을 덜 엄격하게
        if image_height > 0 and y_center > image_height * 0.6: # 하단 40%
             return "발급기관_지역"

    # 6. 주소 (상대적으로 긴 텍스트, 특정 키워드 포함)
    is_address = any(keyword.strip() in text and len(text) > len(keyword.strip()) + 2 for keyword in KW_ADDRESS_TOKENS) # 키워드 외 추가 텍스트가 있는지
    if is_address or (len(text) > 15 and ("로" in text or "길" in text or "동" in text or "아파트" in text)): # 도로명 주소 패턴 추가
        if image_height > 0 and (image_height * 0.25 < y_center < image_height * 0.85): # 주소는 넓은 영역에 나타날 수 있음
            return "주소"

    # 7. 문서명 (이미지 상단, 특정 키워드)
    if any(title in text for title in KW_DOC_TITLES):
        if image_height > 0 and y_center < image_height * 0.3: # 상단 30%
            # 글자 크기(높이)가 이미지 높이의 일정 비율 이상인 경우 추가 고려 가능
            box_h = get_box_height(box)
            if box_h > image_height * 0.025: # 예: 이미지 높이의 2.5% 이상
                return "문서명"

    return "기타"

# --- Refined Labeling Helper Functions ---
def _collect_label_candidates(labeled_items):
    """초기 라벨링된 항목들을 유형별 후보군으로 분류합니다."""
    candidates = {
        "seal": [], "region": [], "date_full": [], "date_part_year": [], "date_part_month": [], "date_part_day": [],
        "address": [], "name": [], "jumin": [], "doc_name": [], "other": []
    }
    for item in labeled_items:
        label = item['label']
        item_copy = item.copy() # 항상 복사본 사용
        if label == '발급기관_직인': candidates["seal"].append(item_copy)
        elif label == '발급기관_지역': candidates["region"].append(item_copy)
        elif label == '날짜_후보': candidates["date_full"].append(item_copy)
        elif label == '날짜_부분':
            part_type = item_copy.get('date_part_type')
            if part_type == 'year': candidates["date_part_year"].append(item_copy)
            elif part_type == 'month': candidates["date_part_month"].append(item_copy)
            elif part_type == 'day': candidates["date_part_day"].append(item_copy)
            else: candidates["other"].append(item_copy) # 불명확한 날짜 부분은 기타로
        elif label == '주소': candidates["address"].append(item_copy)
        elif label == '이름': candidates["name"].append(item_copy)
        elif label == '주민등록번호': candidates["jumin"].append(item_copy)
        elif label == '문서명': candidates["doc_name"].append(item_copy)
        else: candidates["other"].append(item_copy)
    return candidates

def _determine_final_authority(seal_candidates, region_candidates, labeled_items, processed_indices):
    """발급기관 텍스트를 조합하고 최종 결정합니다."""
    # 1. 명시적 "발급기관" 키워드 또는 직인 접미사가 있는 후보 우선
    primary_authority_candidates = [item for item in seal_candidates if any(keyword in item['text'] for keyword in KW_AUTHORITY_KEYWORDS) or any(item['text'].endswith(suffix) for suffix in KW_AUTHORITY_SUFFIXES)]
    
    if not primary_authority_candidates: # 명시적 후보가 없으면, '발급기관_직인'으로 라벨링된 모든 후보 사용
        primary_authority_candidates = seal_candidates
        if not primary_authority_candidates and region_candidates: # 직인 없고 지역만 있으면 지역을 발급기관으로 간주 (예: "서울특별시")
             # 지역 후보 중 가장 신뢰도 높거나 적절한 위치의 것을 선택
            region_candidates.sort(key=lambda x: (get_box_bottom(x['bounding_box']) if x['bounding_box'] else float('inf')), reverse=True)
            if region_candidates:
                chosen_region = region_candidates[0]
                final_authority_item = chosen_region.copy()
                final_authority_item['label'] = "발급기관"
                try: processed_indices.add(labeled_items.index(chosen_region))
                except ValueError: logger.warning(f"단독 지역 발급기관 항목 원본 찾지 못함: {chosen_region}")
                logger.debug(f"최종 발급기관 (지역 단독): {final_authority_item['text']}")
                return final_authority_item, processed_indices
            return None, processed_indices # 지역 후보도 없으면 포기

    if not primary_authority_candidates: return None, processed_indices


    # 가장 아래쪽에 있는 후보를 주 발급기관으로 선택 (직인이 보통 가장 아래에 있으므로)
    primary_authority_candidates.sort(key=lambda x: (get_box_bottom(x['bounding_box']) if x['bounding_box'] else float('-inf')), reverse=True)
    primary_authority_item = primary_authority_candidates[0]
    
    final_authority_text = primary_authority_item['text']
    combined_bb_coords = list(primary_authority_item['bounding_box']) if primary_authority_item['bounding_box'] else []

    # 주 발급기관과 가까운 지역명 후보 찾기
    best_region_item = None
    if region_candidates and primary_authority_item['bounding_box']:
        seal_cx, seal_cy = get_box_center(primary_authority_item['bounding_box'])
        min_distance = float('inf')
        for region_item in region_candidates:
            if region_item['bounding_box'] and region_item != primary_authority_item: # 자기 자신 제외
                reg_cx, reg_cy = get_box_center(region_item['bounding_box'])
                # 거리 계산: y좌표 차이의 제곱 + x좌표 차이의 제곱 (유클리드 거리 제곱)
                # 지역명이 직인 왼쪽에 오는 경우가 많으므로 x좌표 차이에 약간의 패널티
                dist = ((seal_cy - reg_cy)**2) + ((seal_cx - reg_cx - get_box_width(region_item['bounding_box']))**2 if seal_cx > reg_cx else (seal_cx - reg_cx)**2)
                
                # 지역명이 직인과 같은 줄에 있거나 약간 위에 있는 경우 선호
                if abs(seal_cy - reg_cy) < get_box_height(primary_authority_item['bounding_box']) * 1.5: # 높이 1.5배 이내
                    if dist < min_distance:
                        min_distance = dist
                        best_region_item = region_item
    
    if best_region_item:
        # 지역명과 직인 텍스트 순서 결정 (보통 지역명 + 직위명)
        if get_box_left(best_region_item['bounding_box']) < get_box_left(primary_authority_item['bounding_box']):
            final_authority_text = f"{best_region_item['text']} {primary_authority_item['text']}"
        else: # 드문 경우지만, 순서가 반대일 수 있음
            final_authority_text = f"{primary_authority_item['text']} {best_region_item['text']}"
        
        if best_region_item['bounding_box']: combined_bb_coords.extend(list(best_region_item['bounding_box']))
        try: processed_indices.add(labeled_items.index(best_region_item))
        except ValueError: logger.warning(f"발급기관 지역 항목 원본 찾지 못함: {best_region_item}")

    final_authority_item_obj = primary_authority_item.copy()
    final_authority_item_obj['label'] = "발급기관"
    final_authority_item_obj['text'] = final_authority_text

    if combined_bb_coords: # 여러 박스를 합친 경우 전체를 감싸는 새 바운딩 박스 계산
        min_x_agg = min(p[0] for box_part in combined_bb_coords for p in (box_part if isinstance(box_part[0], list) else [box_part]))
        min_y_agg = min(p[1] for box_part in combined_bb_coords for p in (box_part if isinstance(box_part[0], list) else [box_part]))
        max_x_agg = max(p[0] for box_part in combined_bb_coords for p in (box_part if isinstance(box_part[0], list) else [box_part]))
        max_y_agg = max(p[1] for box_part in combined_bb_coords for p in (box_part if isinstance(box_part[0], list) else [box_part]))
        final_authority_item_obj['bounding_box'] = [[min_x_agg, min_y_agg], [max_x_agg, min_y_agg], [max_x_agg, max_y_agg], [min_x_agg, max_y_agg]]
    
    try: processed_indices.add(labeled_items.index(primary_authority_item))
    except ValueError: logger.warning(f"주 발급기관 항목 원본 찾지 못함: {primary_authority_item}")
    
    logger.debug(f"최종 발급기관: {final_authority_text}")
    return final_authority_item_obj, processed_indices


def _determine_final_issue_date(date_full_candidates, year_candidates, month_candidates, day_candidates, final_authority_item, labeled_items, processed_indices, image_width):
    """발급일자를 결정합니다."""
    if not final_authority_item or not final_authority_item['bounding_box']:
        # 발급기관 정보가 없으면, 이미지 하단 중앙 부근의 날짜를 발급일자로 시도
        # (이 로직은 추가 개선 필요)
        if date_full_candidates:
            date_full_candidates.sort(key=lambda it: get_box_bottom(it['bounding_box']), reverse=True) # 가장 아래쪽
            chosen_date = date_full_candidates[0].copy()
            chosen_date['label'] = "발급일자"
            chosen_date['text'] = chosen_date.get('normalized_date', chosen_date['text'])
            try: processed_indices.add(labeled_items.index(date_full_candidates[0]))
            except ValueError: logger.warning(f"발급기관 없는 발급일자 후보 원본 찾지 못함")
            logger.debug(f"최종 발급일자 (발급기관 없음, 전체 후보 중 최하단): {chosen_date['text']}")
            return chosen_date, processed_indices
        return None, processed_indices


    auth_top_y = get_box_top(final_authority_item['bounding_box'])
    auth_center_x = get_box_center(final_authority_item['bounding_box'])[0]
    auth_height = get_box_height(final_authority_item['bounding_box'])

    # 1. 완전한 날짜 후보 ("날짜_후보") 사용
    best_full_date_item = None
    if date_full_candidates:
        min_dist_to_auth = float('inf')
        for date_item in date_full_candidates:
            if not date_item['bounding_box']: continue
            date_bottom_y = get_box_bottom(date_item['bounding_box'])
            date_center_x = get_box_center(date_item['bounding_box'])[0]
            
            # 발급기관 바로 위에 있고 (수직 간격), 수평으로도 가까운 날짜 선택
            if date_bottom_y < auth_top_y :
                vertical_gap = auth_top_y - date_bottom_y
                horizontal_gap = abs(date_center_x - auth_center_x)
                # 수직 간격은 발급기관 높이의 2배 이내, 수평 간격은 이미지 너비의 30% 이내
                if vertical_gap < auth_height * 2.5 and horizontal_gap < image_width * 0.3:
                    # 발급기관과의 거리를 기준으로 가장 가까운 것 선택 (수직 우선)
                    current_dist = vertical_gap + horizontal_gap * 0.5 
                    if current_dist < min_dist_to_auth:
                        min_dist_to_auth = current_dist
                        best_full_date_item = date_item
        
        if best_full_date_item:
            final_date = best_full_date_item.copy()
            final_date['label'] = "발급일자"
            final_date['text'] = final_date.get('normalized_date', final_date['text'])
            try: processed_indices.add(labeled_items.index(best_full_date_item))
            except ValueError: logger.warning(f"발급일자 후보 항목 원본 찾지 못함: {best_full_date_item}")
            logger.debug(f"최종 발급일자 (날짜_후보): {final_date['text']}")
            return final_date, processed_indices

    # 2. 부분 날짜 조합 시도 (YYYY, MM, DD)
    # 발급기관 위에 있는 부분 날짜들만 필터링
    relevant_years = [yc for yc in year_candidates if yc['bounding_box'] and get_box_bottom(yc['bounding_box']) < auth_top_y]
    relevant_months = [mc for mc in month_candidates if mc['bounding_box'] and get_box_bottom(mc['bounding_box']) < auth_top_y]
    relevant_days = [dc for dc in day_candidates if dc['bounding_box'] and get_box_bottom(dc['bounding_box']) < auth_top_y]

    # 발급기관 위에 날짜 부분이 없으면 발급기관 근처의 모든 날짜 부분 사용
    if not (relevant_years and relevant_months and relevant_days):
        logger.debug("발급기관 위에 충분한 날짜 부분이 없어 모든 날짜 부분을 고려합니다.")
        relevant_years = year_candidates
        relevant_months = month_candidates
        relevant_days = day_candidates

    if relevant_years and relevant_months and relevant_days:
        # Y, M, D를 x 좌표 기준으로 정렬
        all_parts = sorted(relevant_years + relevant_months + relevant_days, key=lambda p: get_box_left(p['bounding_box']))
        
        # 연속된 Y, M, D 찾기 (가장 왼쪽부터 시작하는 3개 조합)
        # 실제로는 더 많은 조합과 검증이 필요함
        # 여기서는 가장 일반적인 Y M D 순서를 가정하고, x좌표가 가까운 3개를 찾으려 시도
        
        best_ymd_combo = None
        min_ymd_span = float('inf')

        # 모든 가능한 연도, 월, 일 조합 시도
        for y in relevant_years:
            for m in relevant_months:
                for d in relevant_days:
                    # 세 항목이 시각적으로 그룹을 이루는지 (y좌표 유사)
                    y_coords = [get_box_center(p['bounding_box'])[1] for p in [y, m, d]]
                    if max(y_coords) - min(y_coords) < get_box_height(y['bounding_box']) * 2: # y좌표 차이 허용 범위 확대
                        # x축 전체 길이 (span)
                        x_coords = [get_box_center(p['bounding_box'])[0] for p in [y, m, d]]
                        min_x = min(get_box_left(p['bounding_box']) for p in [y, m, d])
                        max_x = max(get_box_right(p['bounding_box']) for p in [y, m, d])
                        current_span = max_x - min_x
                        
                        # 가장 작은 span을 가진 조합 선택 (가장 가까이 있는 항목들)
                        if current_span < min_ymd_span:
                            min_ymd_span = current_span
                            best_ymd_combo = (y, m, d)

        if best_ymd_combo:
            y, m, d = best_ymd_combo
            combined_date_text = f"{y['value']}.{m['value']:02d}.{d['value']:02d}"
            
            all_coords_ymd = []
            for part_item in [y, m, d]:
                if part_item['bounding_box']: 
                    all_coords_ymd.extend(part_item['bounding_box'])
            
            combined_bb_ymd = None
            if all_coords_ymd:
                min_x = min(c[0] for c in all_coords_ymd); min_y = min(c[1] for c in all_coords_ymd)
                max_x = max(c[0] for c in all_coords_ymd); max_y = max(c[1] for c in all_coords_ymd)
                combined_bb_ymd = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

            final_date = {
                'text': combined_date_text, 'label': '발급일자',
                'confidence': min(p['confidence'] for p in [y, m, d]),
                'bounding_box': combined_bb_ymd
            }
            for part_item in [y, m, d]:
                try: processed_indices.add(labeled_items.index(part_item))
                except ValueError: logger.warning(f"날짜 부분 항목 원본 찾지 못함: {part_item}")
            logger.debug(f"최종 발급일자 (부분 조합): {final_date['text']}")
            return final_date, processed_indices
        
    return None, processed_indices


def _determine_final_address(address_candidates, labeled_items, processed_indices, image_width):
    """주소 항목들을 조합하고 최종 결정합니다."""
    # 아직 처리되지 않은 주소 후보만 필터링
    valid_address_items = [item for item in address_candidates if labeled_items.index(item) not in processed_indices]
    if not valid_address_items:
        return [], processed_indices

    # Y 좌표 기준으로 정렬 후, 가까운 것들 병합 (기존 로직과 유사하게 가되, refine_labels 밖으로 분리)
    valid_address_items.sort(key=lambda it: get_box_top(it['bounding_box']))
    
    merged_addresses = []
    current_merge_group = []

    for i, item in enumerate(valid_address_items):
        if not item['bounding_box']: continue
        if not current_merge_group:
            current_merge_group.append(item)
        else:
            prev_item = current_merge_group[-1]
            vertical_distance = get_box_top(item['bounding_box']) - get_box_bottom(prev_item['bounding_box'])
            avg_char_height = get_box_height(prev_item['bounding_box']) or 10 # 0 방지

            # 수평 겹침/근접성 확인
            prev_left, prev_right = get_box_left(prev_item['bounding_box']), get_box_right(prev_item['bounding_box'])
            curr_left, curr_right = get_box_left(item['bounding_box']), get_box_right(item['bounding_box'])
            horizontal_overlap = max(0, min(prev_right, curr_right) - max(prev_left, curr_left))
            horizontal_ok = horizontal_overlap > 0 or abs(curr_left - prev_left) < image_width * 0.2 # x축 차이 20% 이내

            if vertical_distance < avg_char_height * 1.8 and horizontal_ok: # 병합 조건 살짝 완화
                current_merge_group.append(item)
            else:
                merged_addresses.append(list(current_merge_group))
                current_merge_group = [item]
    
    if current_merge_group: # 마지막 그룹 추가
        merged_addresses.append(list(current_merge_group))

    final_address_objects = []
    for group in merged_addresses:
        if not group: continue
        # 그룹 내 텍스트 조합 (y좌표, x좌표 순으로 정렬 후)
        group.sort(key=lambda x: (get_box_top(x['bounding_box']), get_box_left(x['bounding_box'])))
        
        group_texts = [g_item['text'] for g_item in group]
        combined_text = reorder_korean_address(" ".join(group_texts).strip())
        
        all_coords_group = []
        min_conf_group = 1.0
        for g_item in group:
            if g_item['bounding_box']: all_coords_group.extend(g_item['bounding_box'])
            min_conf_group = min(min_conf_group, g_item['confidence'])
            try: processed_indices.add(labeled_items.index(g_item))
            except ValueError: logger.warning(f"주소 병합 중 원본 항목 찾지 못함: {g_item}")

        bb_group = None
        if all_coords_group:
            min_x = min(c[0] for c in all_coords_group); min_y = min(c[1] for c in all_coords_group)
            max_x = max(c[0] for c in all_coords_group); max_y = max(c[1] for c in all_coords_group)
            bb_group = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        
        if combined_text:
            final_address_objects.append({
                'text': combined_text, 'label': '주소',
                'confidence': min_conf_group, 'bounding_box': bb_group
            })
            logger.debug(f"최종 주소 (병합됨): {combined_text}")
            
    return final_address_objects, processed_indices


def _select_best_single_item(candidates, label_name, labeled_items, processed_indices, 
                             sort_key_func=None, filter_func=None, reverse_sort=True): # reverse_sort 매개변수 추가
    """단일 항목을 선택하는 일반화된 로직. 필터링, 정렬 방향 제어 기능 추가."""
    if not candidates: return None, processed_indices
    
    # 원본 candidates 리스트에서 아직 처리되지 않은 항목만 필터링
    # labeled_items.index(c) 호출이 반복적으로 일어나는 것을 피하기 위해,
    # candidates의 각 항목에 원본 인덱스를 미리 저장해두는 방식도 고려 가능.
    # 여기서는 간결성을 위해 현재 방식 유지.
    valid_candidates = []
    for c_item in candidates:
        try:
            original_idx = labeled_items.index(c_item)
            if original_idx not in processed_indices:
                valid_candidates.append(c_item)
        except ValueError:
            # candidates 리스트에 있는 항목이 labeled_items에 없는 경우는 이론상 발생하면 안됨 (copy()를 사용하므로 객체는 다를 수 있음)
            # 만약 발생한다면, candidates 생성 로직 또는 labeled_items 전달 방식 점검 필요
            # 여기서는 일단 해당 후보를 건너뜀
            logger.warning(f"_select_best_single_item: 후보 항목 {c_item.get('text', '')}을 원본 리스트에서 찾을 수 없어 건너뜁니다.")
            continue
            
    if filter_func:
        valid_candidates = [c for c in valid_candidates if filter_func(c)]
    
    if not valid_candidates: return None, processed_indices
    
    if sort_key_func:
        valid_candidates.sort(key=sort_key_func, reverse=reverse_sort) # reverse_sort 매개변수 사용
    
    best_item_original = valid_candidates[0]
    best_item_copy = best_item_original.copy()
    best_item_copy['label'] = label_name
    
    try:
        # 원본 리스트에서 best_item_original과 동일한 객체를 찾아 인덱스 추가
        # (주의: copy()로 생성된 후보이므로, 원본 객체를 찾아야 함)
        # 이 부분은 candidates가 labeled_items의 부분집합이라는 가정 하에 동작.
        # 만약 candidates의 항목이 labeled_items의 복사본이라면, 값 기반 비교나 다른 식별자 필요.
        # 현재 _collect_label_candidates에서 .copy()를 사용하므로, 값 기반으로 원본을 찾아야 할 수 있음.
        # 가장 간단한 방법은 labeled_items를 순회하며 값(텍스트, 바운딩박스 등)이 같은 원본 항목을 찾는 것.
        # 여기서는 일단 객체 참조가 유지된다고 가정하고 진행 (만약 문제가 지속되면 이 부분 수정 필요)
        
        # labeled_items에서 best_item_original (복사되기 전의 원본 객체)을 찾아야 함.
        # candidates는 labeled_items의 원소를 복사한 것이므로, best_item_original은 labeled_items에 있어야 함.
        # 다만, _collect_label_candidates에서 item.copy()를 하므로,
        # candidates의 항목은 labeled_items의 항목과 다른 객체임.
        # 따라서, 값 기반으로 찾아야 함.
        
        found_original_in_labeled_items = False
        for idx, orig_item_in_list in enumerate(labeled_items):
            # 값 기반 비교 (텍스트, 바운딩 박스, 초기 라벨 등 주요 속성 비교)
            if (orig_item_in_list['text'] == best_item_original['text'] and
                orig_item_in_list['bounding_box'] == best_item_original['bounding_box'] and
                orig_item_in_list.get('label') == best_item_original.get('label')): # 초기 라벨도 비교
                processed_indices.add(idx)
                found_original_in_labeled_items = True
                break
        if not found_original_in_labeled_items:
             logger.warning(f"{label_name}으로 선택된 항목의 원본을 labeled_items에서 찾지 못했습니다: {best_item_original.get('text','')}")


    except ValueError: # 이 예외는 발생하지 않아야 함 (위의 로직으로 처리)
        logger.warning(f"{label_name} 후보 항목 원본 찾지 못함 (ValueError): {best_item_original}")

    logger.debug(f"최종 {label_name}: {best_item_copy['text']}")
    return best_item_copy, processed_indices

# --- End Refined Labeling Helper Functions ---


def refine_and_finalize_labels(labeled_items, image_width, image_height): # 이미지 크기 인자 추가
    """초기 라벨링된 결과를 바탕으로 관계를 파악하여 최종 라벨을 결정하고 조합합니다."""
    final_results = []
    processed_indices = set()

    candidates = _collect_label_candidates(labeled_items)

    # 이름 (가장 위쪽 우선)
    final_name, processed_indices = _select_best_single_item(
        candidates["name"], "이름", labeled_items, processed_indices,
        sort_key_func=lambda x: (get_box_top(x['bounding_box']) if x['bounding_box'] else float('inf')), 
        reverse_sort=False # 오름차순 (낮은 y값이 위)
    )
    if final_name: final_results.append(final_name)

    # 주민등록번호 (신뢰도 가장 높은 것)
    final_jumin, processed_indices = _select_best_single_item(
        candidates["jumin"], "주민등록번호", labeled_items, processed_indices,
        sort_key_func=lambda x: x['confidence'],
        reverse_sort=True # 내림차순 (높은 신뢰도 우선)
    )
    if final_jumin: final_results.append(final_jumin)
    
    # 문서명 (신뢰도, 중앙 정렬 우선)
    final_doc_name, processed_indices = _select_best_single_item(
        candidates["doc_name"], "문서명", labeled_items, processed_indices,
        sort_key_func=lambda x: (x['confidence'], -abs(get_box_center(x['bounding_box'])[0] - image_width/2) if x['bounding_box'] and image_width > 0 else 0),
        reverse_sort=True # 신뢰도는 내림차순, 중앙 정렬 점수는 값이 클수록 좋으므로 (음수화 후 내림차순 = 절대값 작은것 우선)
    )
    if final_doc_name: final_results.append(final_doc_name)

    # 발급기관
    final_authority, processed_indices = _determine_final_authority(
        candidates["seal"], candidates["region"], labeled_items, processed_indices
    )
    if final_authority: final_results.append(final_authority)

    # 발급일자 (발급기관 위치 기반)
    final_issue_date, processed_indices = _determine_final_issue_date(
        candidates["date_full"], candidates["date_part_year"], candidates["date_part_month"], candidates["date_part_day"],
        final_authority, labeled_items, processed_indices, image_width
    )
    if final_issue_date: final_results.append(final_issue_date)

    # 주소
    final_address_list, processed_indices = _determine_final_address(
        candidates["address"], labeled_items, processed_indices, image_width
    )
    final_results.extend(final_address_list)
    
    # 기타 항목 처리
    # 1. 초기부터 '기타'였던 항목들 중 아직 처리 안된 것
    for item_candidate in candidates["other"]:
        # item_candidate는 복사본이므로, 원본 labeled_items에서 해당 항목을 찾아야 함
        original_item_idx = -1
        for idx, li_item in enumerate(labeled_items):
            if (li_item['text'] == item_candidate['text'] and 
                li_item['bounding_box'] == item_candidate['bounding_box'] and
                li_item.get('label') == item_candidate.get('label')): # 초기 라벨까지 비교
                original_item_idx = idx
                break
        
        if original_item_idx != -1 and original_item_idx not in processed_indices:
            item_copy = item_candidate.copy(); item_copy['label'] = '기타' # 최종 라벨 '기타'로 확정
            final_results.append(item_copy)
            processed_indices.add(original_item_idx)
            
    # 2. 초기 라벨이 있었으나 최종 선택되지 않은 모든 나머지 항목들
    for i, original_item in enumerate(labeled_items):
        if i not in processed_indices:
            item_copy = original_item.copy()
            item_copy['label'] = '기타' # 최종적으로 '기타'로 확정
            final_results.append(item_copy)
            # processed_indices.add(i) # 이미 루프의 마지막 단계이므로 추가할 필요 없음

    return final_results


def get_system_font_path():
    """운영 체제에 맞는 기본 시스템 폰트 경로를 반환합니다 (존재 여부 확인 포함)."""
    system = platform.system()
    font_path = None
    if system == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif system == "Darwin": # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        if not os.path.exists(font_path):
            font_path = "/Library/Fonts/AppleGothic.ttf"
    elif system == "Linux":
        common_linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # 영문 기본
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
            "/usr/share/fonts/korean/NanumGothic.ttf",
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break
    if font_path and os.path.exists(font_path):
        logger.debug(f"OS ({system}) 시스템 폰트 확인: {font_path}")
        return font_path
    logger.debug(f"OS ({system}) 시스템 폰트를 찾지 못했습니다 (경로: {font_path}). 한글 폰트가 없으면 시각화 시 깨질 수 있습니다.")
    return None # PaddleOCR 기본 폰트 사용 유도 (영문 위주)

def determine_font_for_visualization():
    """시각화에 사용할 폰트 경로를 결정하고 관련 정보를 로깅합니다."""
    font_path_to_use = get_system_font_path()
    if font_path_to_use:
        logger.info(f"텍스트 시각화를 위해 시스템 자동 감지 폰트 사용: {font_path_to_use}")
        return font_path_to_use

    logger.info("시스템 폰트를 찾지 못하여 로컬 'fonts' 폴더 내 폰트를 확인합니다.")
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()

    local_korean_font_candidates = [
        os.path.join(script_dir, 'fonts', 'malgun.ttf'),
        os.path.join(script_dir, 'fonts', 'NanumGothic.ttf')
    ]
    for local_font in local_korean_font_candidates:
        if os.path.exists(local_font):
            logger.info(f"로컬 폰트 사용: {local_font}")
            return local_font

    logger.warning("시각화를 위한 특정 한글 폰트를 찾지 못했습니다. PaddleOCR 내부 기본 폰트가 사용될 수 있으며, 한글이 깨질 수 있습니다.")
    return None


def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename,
                       preprocessed_img=None, show_image_flag=False, font_path_to_use=None):
    """OCR 결과를 이미지 위에 표시하고 저장합니다."""
    if not extracted_data:
        logger.info(f"{original_filename}: 시각화할 OCR 결과가 없습니다.")
        return

    try:
        if preprocessed_img is not None:
            image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB) if len(preprocessed_img.shape) == 2 else preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else:
            image = Image.open(original_image_path).convert('RGB')

        boxes = [item['bounding_box'] for item in extracted_data if item.get('bounding_box')]
        txts = [f"{item.get('label', 'N/A')}: {item['text']}" for item in extracted_data if item.get('bounding_box')]
        scores = [item['confidence'] for item in extracted_data if item.get('bounding_box')]

        if not boxes:
            logger.info(f"{original_filename}: 시각화할 바운딩 박스가 없습니다.")
            return

        im_show = draw_ocr(np.array(image), boxes, txts, scores, font_path=font_path_to_use)
        im_show_pil = Image.fromarray(im_show)

        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr_result{ext}"
        output_image_path = os.path.join(output_dir, output_image_filename)
        im_show_pil.save(output_image_path)
        logger.debug(f"OCR 결과 시각화 이미지가 {output_image_path}에 저장되었습니다.")

        if show_image_flag:
            logger.info(f"{original_filename} 결과 이미지를 표시합니다.")
            im_show_pil.show()
    except FileNotFoundError:
        logger.error(f"원본 이미지 파일 '{original_image_path}'을 찾을 수 없습니다 (결과 표시 중).")
    except Exception as e:
        logger.error(f"OCR 결과 표시/저장 중 예외 발생 (파일: {original_filename}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))


def process_single_image_task(task_args_tuple):
    """단일 이미지 처리 작업을 수행하는 함수 (multiprocessing.Pool의 작업자용)."""
    global _worker_csv_lock, _worker_csv_file_path
    (current_image_path, filename, ocr_engine_params, output_dir,
     skip_preprocessing, show_image, font_path) = task_args_tuple

    status_message = f"{filename} 처리 중 오류 발생"
    try:
        logger.debug(f"작업자 {os.getpid()}: 처리 시작: {filename}")
        ocr_input_data = current_image_path
        processed_image_for_display = None
        image_width, image_height = 0, 0
        try:
            with Image.open(current_image_path) as img_pil: image_width, image_height = img_pil.size
        except Exception as e: logger.error(f"작업자 {os.getpid()}: 이미지 크기 읽기 오류 ({filename}): {e}")

        if not skip_preprocessing:
            processed_img_data = preprocess_image_for_ocr(current_image_path)
            if processed_img_data is not None:
                ocr_input_data = processed_img_data
                processed_image_for_display = processed_img_data
            else: logger.warning(f"작업자 {os.getpid()}: {filename}: 외부 전처리에 실패하여 원본 이미지로 OCR 시도.")

        extracted_items = extract_text_from_image_worker(ocr_engine_params, ocr_input_data, filename_for_log=filename)

        if extracted_items:
            initially_labeled_items = []
            for item in extracted_items:
                initial_label = label_text_item_initial(item, image_width, image_height) if image_width > 0 and image_height > 0 else "기타"
                item['label'] = initial_label
                initially_labeled_items.append(item)
            
            final_labeled_results = refine_and_finalize_labels(initially_labeled_items, image_width, image_height)

            # 주소 항목 통합
            address_items = [item for item in final_labeled_results if item.get('label') == '주소']
            non_address_items = [item for item in final_labeled_results if item.get('label') != '주소']
            
            csv_rows_to_write = []
            
            # 주소가 있는 경우 통합
            if address_items:
                # 주소 텍스트 추출
                address_texts = [item.get('text', '') for item in address_items if item.get('text')]
                
                # 중복 제거 및 정렬
                unique_addresses = []
                for addr in address_texts:
                    if addr and addr.strip() and addr.strip() not in unique_addresses:
                        unique_addresses.append(addr.strip())
                
                # 주소 항목 통합
                combined_address = " ".join(unique_addresses)
                
                # 한국식 주소 형식 재정렬
                combined_address = reorder_korean_address(combined_address)
                
                # 첫 번째 주소 항목을 수정하여 통합된 주소로 변경
                first_address_item = address_items[0].copy()
                first_address_item['text'] = combined_address
                
                # 통합된 주소 항목 추가
                csv_rows_to_write.append([
                    filename, first_address_item.get('label', '주소'), first_address_item.get('text', '').replace('"', '""'),
                    round(first_address_item.get('confidence', 0.0), 4), str(first_address_item.get('bounding_box', 'N/A'))
                ])
            
            # 비주소 항목 추가
            for item in non_address_items:
                csv_rows_to_write.append([
                    filename, item.get('label', '기타'), item.get('text', '').replace('"', '""'),
                    round(item.get('confidence', 0.0), 4), str(item.get('bounding_box', 'N/A'))
                ])
            
            if csv_rows_to_write and _worker_csv_lock and _worker_csv_file_path:
                with _worker_csv_lock:
                    try:
                        with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                            csv.writer(csvfile).writerows(csv_rows_to_write)
                        logger.info(f"작업자 {os.getpid()}: {filename}의 텍스트를 {os.path.basename(_worker_csv_file_path)}에 추가.")
                    except IOError as e: logger.error(f"작업자 {os.getpid()}: CSV 쓰기 오류 ({filename}): {e}")
            
            display_ocr_result(current_image_path, final_labeled_results, output_dir, filename,
                               preprocessed_img=processed_image_for_display,
                               show_image_flag=show_image, font_path_to_use=font_path)
            status_message = f"{filename} 처리 성공"
        else:
            logger.info(f"작업자 {os.getpid()}: {filename}에서 텍스트를 추출하지 못했습니다.")
            status_message = f"{filename} 텍스트 없음"
            if _worker_csv_lock and _worker_csv_file_path:
                 with _worker_csv_lock:
                     try:
                         with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                             csv.writer(csvfile).writerow([filename, "N/A", "No text extracted", 0.0, "N/A"])
                     except IOError as e: logger.error(f"작업자 {os.getpid()}: CSV 쓰기 오류 ({filename}): {e}")
        return status_message
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: process_single_image_task 오류 (파일: {filename}): {e}", exc_info=True)
        if _worker_csv_lock and _worker_csv_file_path:
             with _worker_csv_lock:
                 try:
                     with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                         csv.writer(csvfile).writerow([filename, "오류", str(e), 0.0, "N/A"])
                 except IOError as io_err: logger.error(f"작업자 {os.getpid()}: CSV 쓰기 오류 ({filename}): {io_err}")
        return status_message


def parse_arguments():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", nargs='?', default='input', help="텍스트를 추출할 이미지가 포함된 폴더 경로 (기본값: 'input')")
    parser.add_argument("--output_dir", default='output', help="OCR 결과 이미지 및 텍스트 파일을 저장할 폴더 경로 (기본값: 'output')")
    parser.add_argument("--lang", default='korean', help="OCR 언어 (예: korean, en, ch_sim) (기본값: 'korean')")
    parser.add_argument("--show_image", action='store_true', help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="스크립트 버전을 표시하고 종료합니다.")
    parser.add_argument('--debug', action='store_true', help="디버그 레벨 로깅을 활성화합니다.")
    parser.add_argument('--use_gpu', action='store_true', help="GPU 사용 (NVIDIA GPU 및 CUDA 환경 필요)")
    parser.add_argument("--num_workers", type=int, default=None, help="병렬 처리 작업자 수 (기본값: 시스템 CPU 코어 수)")
    return parser.parse_args()

def prepare_directories(input_dir, output_dir):
    """입력 및 출력 디렉토리를 확인하고, 출력 디렉토리가 없으면 생성합니다."""
    if not os.path.isdir(input_dir):
        logger.error(f"입력 디렉토리 '{input_dir}'가 없거나 디렉토리가 아닙니다. 종료합니다.")
        exit(1)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"출력 디렉토리 생성: {output_dir}")
        except OSError as e:
            logger.error(f"출력 디렉토리 '{output_dir}' 생성 실패: {e}")
            exit(1)

def create_final_summary_csv(labeled_csv_path, output_csv_path):
    """ ocr_labeled_text.csv 파일을 읽어 최종 요약 CSV 파일 (ocr_text.csv)을 생성합니다. """
    logger.info(f"'{labeled_csv_path}' 파일을 읽어 최종 요약 CSV ('{os.path.basename(output_csv_path)}') 생성을 시작합니다...")
    file_data_map = defaultdict(lambda: {"문서명": "", "이름": "", "주민등록번호": "", "주소": [], "발급일자": "", "발급기관": "", "기타": []})

    try:
        with open(labeled_csv_path, 'r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            try:
                filename_idx = header.index('Image Filename')
                label_idx = header.index('Label')
                text_idx = header.index('Extracted Text')
            except ValueError:
                logger.error(f"'{labeled_csv_path}' 헤더 오류. 'Image Filename', 'Label', 'Extracted Text' 열 필요.")
                return

            for row in reader:
                if len(row) <= max(filename_idx, label_idx, text_idx):
                    logger.warning(f"잘못된 형식의 행 발견 (건너뜀): {row}")
                    continue
                image_filename, label, text = row[filename_idx], row[label_idx], row[text_idx]
                if label == "N/A" or label == "오류": continue

                if label == "기타":
                    if text: file_data_map[image_filename][label].append(text)
                elif label == "주소":
                    # 주소는 리스트로 수집하여 나중에 통합
                    if text: file_data_map[image_filename][label].append(text)
                elif label in file_data_map[image_filename]:
                    # 해당 라벨에 아직 값이 없거나, 현재 텍스트가 더 유의미하다고 판단될 경우 (예: 더 길거나, 신뢰도 높거나 - 현재는 첫 값 우선)
                    if not file_data_map[image_filename][label] and text:
                        file_data_map[image_filename][label] = text
                    elif file_data_map[image_filename][label] and text: # 이미 값이 있는 경우
                         logger.debug(f"파일 '{image_filename}', 라벨 '{label}': 기존 값 '{file_data_map[image_filename][label]}' -> 새 값 '{text}'. 기존 값 유지 또는 필요시 업데이트 로직 추가.")
                else: # 정의되지 않은 라벨 (이론상 refine_and_finalize_labels에서 모두 처리되어야 함)
                    logger.warning(f"최종 요약 CSV 생성 중 알 수 없는 라벨 '{label}' (파일: {image_filename}, 텍스트: {text}). '기타'로 처리.")
                    if text: file_data_map[image_filename]["기타"].append(text)
    except FileNotFoundError: logger.error(f"입력 CSV 파일을 찾을 수 없습니다: {labeled_csv_path}"); return
    except Exception as e: logger.error(f"'{labeled_csv_path}' 파일 읽기 중 오류: {e}", exc_info=True); return

    output_header = ["Image Filename", "문서명", "이름", "주민등록번호", "주소", "발급일자", "발급기관", "기타"]
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(output_header)
            for image_filename, data in sorted(file_data_map.items()): # 파일명 순으로 정렬
                # 주소 항목 통합
                combined_address = ""
                if data["주소"]:
                    # 중복 제거 및 정렬
                    unique_addresses = []
                    for addr in data["주소"]:
                        if addr and addr.strip() and addr.strip() not in unique_addresses:
                            unique_addresses.append(addr.strip())
                    
                    # 주소 항목 통합 (공백으로 구분)
                    combined_address = " ".join(unique_addresses)
                    # 한국식 주소 형식 재정렬 (필요한 경우)
                    combined_address = reorder_korean_address(combined_address)
                
                # 기타 항목 중복 제거 및 정렬 후 병합
                unique_other_texts = sorted(list(set(s.strip() for s in data["기타"] if s and s.strip()))) # 공백만 있는 문자열 제외
                other_text_combined = "; ".join(unique_other_texts)
                writer.writerow([image_filename, data["문서명"], data["이름"], data["주민등록번호"], combined_address, data["발급일자"], data["발급기관"], other_text_combined])
        logger.info(f"최종 요약 CSV 파일 생성 완료: {output_csv_path}")
    except IOError as e: logger.error(f"'{output_csv_path}' 파일 쓰기 중 오류: {e}")
    except Exception as e: logger.error(f"최종 요약 CSV 생성 중 예기치 않은 오류: {e}", exc_info=True)


def main():
    """스크립트의 메인 실행 로직입니다."""
    try:
        if platform.system() != "Windows" and multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: logger.debug("멀티프로세싱 시작 방식 이미 설정됨.")
    except AttributeError: logger.debug("get_start_method(allow_none=True) 사용 불가.")
    if platform.system() == "Windows": multiprocessing.freeze_support()

    args = parse_arguments()
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_level_env_name = 'OCR_WORKER_LOG_LEVEL'
    os.environ[log_level_env_name] = logging.getLevelName(log_level)
    setup_logging(log_level)

    logger.info(f"OCR 스크립트 버전: {__version__}")
    logger.info(f"명령줄 인자: {args}")
    prepare_directories(args.input_dir, args.output_dir)

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_filenames = [f for f in os.listdir(args.input_dir) if f.lower().endswith(supported_extensions)]
    if not image_filenames:
        logger.warning(f"입력 디렉토리 '{args.input_dir}'에서 지원되는 이미지 파일을 찾을 수 없습니다."); exit(0)

    num_workers = args.num_workers if args.num_workers is not None and args.num_workers > 0 else os.cpu_count()
    if num_workers is None: num_workers = 1; logger.warning("CPU 코어 수 감지 불가, 작업자 수 1로 설정.")
    logger.info(f"사용할 작업자 프로세스 수: {num_workers}")
    logger.info(f"총 {len(image_filenames)}개의 이미지 파일을 처리합니다.")

    ocr_params = {'use_angle_cls': True, 'lang': args.lang, 'use_gpu': args.use_gpu, 'show_log': False}
    if '+' in args.lang: logger.info(f"복합 언어 설정 '{args.lang}' 감지.")
    
    font_path = determine_font_for_visualization()
    intermediate_csv_path = os.path.join(args.output_dir, "ocr_labeled_text.csv")
    
    manager = multiprocessing.Manager()
    csv_lock = manager.Lock()
    csv_header_intermediate = ["Image Filename", "Label", "Extracted Text", "Confidence", "Bounding Box (str)"]
    try:
        with open(intermediate_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csv.writer(csvfile).writerow(csv_header_intermediate)
        logger.info(f"중간 CSV 파일 '{intermediate_csv_path}' 초기화 완료.")
    except IOError as e: logger.error(f"중간 CSV 파일 초기화 오류 ({intermediate_csv_path}): {e}"); exit(1)

    tasks = [(os.path.join(args.input_dir, fn), fn, ocr_params, args.output_dir, args.no_preprocess, args.show_image, font_path) for fn in image_filenames]
    
    pool = None
    try:
        pool = multiprocessing.Pool(processes=num_workers, initializer=worker_initializer_func, initargs=(csv_lock, intermediate_csv_path, log_level_env_name))
        logger.info("병렬 이미지 처리 시작...")
        for res_status in tqdm(pool.imap_unordered(process_single_image_task, tasks), total=len(tasks), desc="전체 이미지 처리 중"):
            if res_status: logger.debug(f"작업 결과 수신: {res_status}")
    except Exception as e: logger.error(f"병렬 처리 중 주 프로세스 오류: {e}", exc_info=True)
    finally:
        if pool:
            logger.info("풀 종료 시작..."); pool.close(); logger.info("Pool.close() 완료. 작업자 종료 대기 중..."); pool.join(); logger.info("Pool.join() 완료.")

    logger.info(f"총 {len(image_filenames)}개 이미지 처리 완료. 개별 추출 정보는 '{intermediate_csv_path}'에 저장됨.")
    
    final_summary_csv_path = os.path.join(args.output_dir, "ocr_text.csv")
    create_final_summary_csv(intermediate_csv_path, final_summary_csv_path)
    logger.info(f"스크립트 실행 완료. 모든 결과는 '{args.output_dir}' 폴더에 저장됨.")

if __name__ == "__main__":
    main()
