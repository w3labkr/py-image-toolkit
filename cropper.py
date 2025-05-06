# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
import logging
import time
import json # Added for configuration file processing
import concurrent.futures # Using ProcessPoolExecutor
from PIL import Image, UnidentifiedImageError
from typing import Tuple, List, Optional, Dict, Any, Union

# Attempt to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        logging.info("tqdm library not installed, progress bar will be omitted. (pip install tqdm)")
        return iterable

__version__ = "1.6.3" # Version update

# --- Default Configuration Values ---
DEFAULT_CONFIG = {
    "output_dir": "output_final",
    "method": "largest",
    "reference": "eye",
    "ratio": None,
    "confidence": 0.6,
    "nms": 0.3,
    "overwrite": True,
    "output_format": None,
    "jpeg_quality": 95,
    "min_face_width": 30,
    "min_face_height": 30,
    "padding_percent": 5.0,
    "rule": "both",
    "workers": os.cpu_count(),
    "verbose": False,
    "dry_run": False
}

# DNN Model Information
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")  # Global variable for model file path

# Output Filename Suffixes
OUTPUT_SUFFIX_THIRDS = '_thirds'
OUTPUT_SUFFIX_GOLDEN = '_golden'
# --- End of Configuration Values ---

# --- Logging Setup ---
# Basic logging handler setup (for the main process)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
# Remove existing handlers (prevent duplicate logging)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO) # Default level INFO

# --- Utility Functions ---

def download_model(url: str, file_path: str) -> bool:
    """Downloads the model file from the specified URL."""
    if not os.path.exists(file_path):
        logging.info(f"Downloading model file... ({os.path.basename(file_path)})")
        try:
            urllib.request.urlretrieve(url, file_path)
            logging.info("Download complete.")
            return True
        except Exception as e:
            logging.error(f"Model file download failed: {e}")
            logging.error(f"       Please manually download from the following URL and save as '{file_path}': {url}")
            return False
    else:
        # This log is useful only in the main process
        if os.getpid() == main_process_pid:
             logging.info(f"Model file '{os.path.basename(file_path)}' already exists.")
        return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    """Converts a ratio string (e.g., '16:9', '1.0', 'None') to a float."""
    if ratio_str is None or str(ratio_str).lower() == 'none':
        return None
    try:
        ratio_str = str(ratio_str)
        if ':' in ratio_str:
            w_str, h_str = ratio_str.split(':')
            w, h = float(w_str), float(h_str)
            if h <= 0 or w <= 0:
                logging.warning(f"Ratio width or height cannot be zero or less: '{ratio_str}'. Using original ratio.")
                return None
            return w / h
        else:
            ratio = float(ratio_str)
            if ratio <= 0:
                logging.warning(f"Ratio must be greater than zero: '{ratio_str}'. Using original ratio.")
                return None
            return ratio
    except ValueError:
        logging.warning(f"Invalid ratio string format: '{ratio_str}'. Using original ratio.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing ratio ('{ratio_str}'): {e}")
        return None

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logging.info(f"Configuration file loaded successfully: {config_path}")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Configuration file parsing error ({config_path}): {e}")
        return {}
    except Exception as e:
        logging.error(f"Error loading configuration file ({config_path}): {e}")
        return {}

# --- Core Logic Functions ---

def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray, min_w: int, min_h: int) -> List[Dict[str, Any]]:
    """
    Detects faces using the preloaded DNN model (YuNet).
    Excludes faces smaller than the specified minimum size (min_w, min_h).
    """
    detected_subjects = []
    if image is None or image.size == 0:
        logging.warning("Input image for face detection is empty.")
        return []

    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        logging.warning(f"Invalid image size ({img_w}x{img_h}), skipping face detection.")
        return []

    try:
        detector.setInputSize((img_w, img_h))
        # Perform face detection - this might be where errors occur
        faces = detector.detect(image)

        if faces[1] is not None:
            for idx, face_info in enumerate(faces[1]):
                x, y, w, h = map(int, face_info[:4])
                if w < min_w or h < min_h:
                    logging.debug(f"Face ID {idx} ({w}x{h}) ignored as it's smaller than the minimum size ({min_w}x{min_h}).")
                    continue

                r_eye_x, r_eye_y = face_info[4:6]
                l_eye_x, l_eye_y = face_info[6:8]
                confidence = face_info[14]

                x = max(0, x); y = max(0, y)
                w = min(img_w - x, w); h = min(img_h - y, h)

                if w > 0 and h > 0:
                    bbox_center = (x + w // 2, y + h // 2)
                    eye_center = bbox_center
                    # Use eye center if landmarks are valid
                    if r_eye_x > 0 and r_eye_y > 0 and l_eye_x > 0 and l_eye_y > 0:
                        ecx = int(round((r_eye_x + l_eye_x) / 2))
                        ecy = int(round((r_eye_y + l_eye_y) / 2))
                        ecx = max(0, min(img_w - 1, ecx)); ecy = max(0, min(img_h - 1, ecy))
                        eye_center = (ecx, ecy)
                    else:
                        logging.debug(f"Eye landmarks for face ID {idx} are invalid, using BBox center.")

                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e:
        # Enhanced logging for OpenCV errors
        logging.error(f"OpenCV error during face detection (image size: {img_w}x{img_h}): {e}")
        return [] # Return empty list on error
    except Exception as e:
        logging.error(f"Unexpected problem during DNN face detection: {e}")
        return []
    return detected_subjects


def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'eye') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """Selects the main subject from the list of detected subjects and returns its bbox and reference point."""
    if not subjects:
        logging.debug("Select main subject: No subjects detected.")
        return None

    img_h, img_w = img_shape
    best_subject = None

    try:
        if len(subjects) == 1:
            best_subject = subjects[0]
        elif method == 'largest':
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
        elif method == 'center':
            img_center = (img_w / 2, img_h / 2)
            best_subject = min(subjects, key=lambda s: math.dist(s['bbox_center'], img_center))
        else: # Default to largest if method is unknown
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        logging.debug(f"Main subject selected (Method: {method}, Reference: {reference_point_type}). BBox: {best_subject['bbox']}")
        return best_subject['bbox'], ref_center

    except Exception as e:
        logging.error(f"Error selecting main subject: {e}")
        return None


def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """Returns a list of intersection points based on the composition rule (thirds or golden)."""
    points = []
    if width <= 0 or height <= 0:
        logging.warning(f"Cannot calculate rule points for invalid size ({width}x{height}).")
        return []

    try:
        if rule_type == 'thirds':
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden':
            phi_inv = (math.sqrt(5) - 1) / 2 # Inverse golden ratio
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            points = [(w, h) for w in lines_w for h in lines_h]
        else:
            logging.warning(f"Unknown composition rule '{rule_type}'. Using image center.")
            points = [(width / 2, height / 2)]
        # Round points to nearest integer
        return [(int(round(px)), int(round(py))) for px, py in points]
    except Exception as e:
        logging.error(f"Error calculating rule points (Rule: {rule_type}): {e}")
        return []


def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    """Calculates the optimal crop area (x1, y1, x2, y2) based on reference point, rule points, and aspect ratio."""
    height, width = img_shape
    if height <= 0 or width <= 0:
        logging.warning(f"Cannot crop image with zero or negative height ({height}) or width ({width}).")
        return None
    if not rule_points:
        logging.warning("No rule points provided, skipping crop calculation.")
        return None

    cx, cy = subject_center # Center of the subject (e.g., eye center)

    try:
        # Determine aspect ratio to use
        if height > 0:
            aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        else:
             logging.warning("Image height is zero, cannot calculate original aspect ratio. Cannot crop.")
             return None

        if aspect_ratio <= 0:
            logging.warning(f"Invalid target aspect ratio ({aspect_ratio}). Cannot calculate crop.")
            return None

        # Find the rule point closest to the subject center
        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point # This point will be the center of the crop

        # Calculate the maximum possible crop dimensions centered on the target point
        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)

        if max_w <= 0 or max_h <= 0:
             logging.debug("Target point is on the image boundary, cannot create a valid crop.")
             return None

        # Determine final crop dimensions based on the limiting dimension (width or height) and target ratio
        crop_h_from_w = max_w / aspect_ratio # Height if width is max_w
        crop_w_from_h = max_h * aspect_ratio # Width if height is max_h

        # Choose the smaller crop that fits within max_w and max_h
        if crop_h_from_w <= max_h + 1e-6: # Allow for floating point inaccuracies
            final_w, final_h = max_w, crop_h_from_w
        else:
            final_w, final_h = crop_w_from_h, max_h

        # Calculate crop coordinates (top-left and bottom-right)
        x1 = target_x - final_w / 2
        y1 = target_y - final_h / 2
        x2 = x1 + final_w
        y2 = y1 + final_h

        # Ensure coordinates are within image bounds and are integers
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        # Check if the resulting crop area is valid
        if x1 >= x2 or y1 >= y2:
            logging.warning("Calculated crop area has zero size.")
            return None

        logging.debug(f"Calculated crop area: ({x1}, {y1}) - ({x2}, {y2})")
        return x1, y1, x2, y2

    except Exception as e:
        logging.error(f"Error calculating optimal crop: {e}")
        return None


def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    """Applies padding to the calculated crop area and adjusts to stay within image boundaries."""
    x1, y1, x2, y2 = crop_coords
    img_h, img_w = img_shape
    crop_w = x2 - x1
    crop_h = y2 - y1

    if padding_percent <= 0:
        return crop_coords # No padding needed

    # Calculate padding amounts for each side
    pad_x = int(round(crop_w * padding_percent / 100 / 2))
    pad_y = int(round(crop_h * padding_percent / 100 / 2))

    # Apply padding
    new_x1 = x1 - pad_x
    new_y1 = y1 - pad_y
    new_x2 = x2 + pad_x
    new_y2 = y2 + pad_y

    # Clamp coordinates to image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)

    # Check if padding resulted in an invalid area
    if new_x1 >= new_x2 or new_y1 >= new_y2:
        logging.warning("Crop area size became zero after applying padding. Padding will not be applied.")
        return crop_coords # Return original coords

    logging.debug(f"Padded crop area ({padding_percent}%): ({new_x1}, {new_y1}) - ({new_x2}, {new_y2})")
    return new_x1, new_y1, new_x2, new_y2


# --- Main Processing Function ---

def process_image(image_path: str, output_dir: str, detector: cv2.FaceDetectorYN, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single image: detects faces, calculates crops, and saves results
    or logs actions in Dry Run mode. Returns a dictionary with status and messages.
    (Modified to accept Dict instead of Namespace)
    """
    filename = os.path.basename(image_path)
    logging.debug(f"Processing started: {filename}")
    start_time = time.time()
    is_dry_run = args_dict.get('dry_run', False)
    status = {'filename': filename, 'success': False, 'saved_files': 0, 'message': '', 'dry_run': is_dry_run}
    exif_data = None
    original_ext = os.path.splitext(image_path)[1].lower()

    try:
        # Open image with Pillow to handle various formats and read EXIF
        with Image.open(image_path) as pil_img:
            try:
                exif_data = pil_img.info.get('exif')
                if exif_data:
                    logging.debug(f"{filename}: EXIF data found.")
            except Exception as exif_err:
                logging.warning(f"{filename}: Error processing EXIF data: {exif_err}. Saving without EXIF.")
                exif_data = None

            # Convert to RGB for OpenCV compatibility
            pil_img_rgb = pil_img.convert('RGB')
            # Convert Pillow image to OpenCV format (BGR)
            img = np.array(pil_img_rgb)[:, :, ::-1].copy()

    except FileNotFoundError:
        status['message'] = "Image file not found."
        logging.error(f"{filename}: {status['message']}")
        return status
    except UnidentifiedImageError:
        status['message'] = "Cannot open image file or unsupported format."
        logging.error(f"{filename}: {status['message']}")
        return status
    except Exception as e:
        status['message'] = f"Error loading image: {e}"
        logging.error(f"{filename}: {status['message']}")
        return status

    img_h, img_w = img.shape[:2]
    if img_h <= 0 or img_w <= 0:
        status['message'] = f"Invalid image dimensions ({img_w}x{img_h})."
        logging.warning(f"{filename}: {status['message']}")
        return status

    # Detect faces
    detected_faces = detect_faces_dnn(detector, img, args_dict['min_face_width'], args_dict['min_face_height'])
    if not detected_faces:
        status['message'] = "No valid faces detected (considering minimum size)."
        logging.info(f"{filename}: {status['message']}") # Info level as it's a common case
        return status

    # Select main subject
    selection_result = select_main_subject(detected_faces, (img_h, img_w), args_dict['method'], args_dict['reference'])
    if not selection_result:
         status['message'] = "Failed to select main subject."
         logging.warning(f"{filename}: {status['message']}")
         return status
    subj_bbox, ref_center = selection_result

    # Setup output filename and extension
    base = os.path.splitext(filename)[0]
    target_ratio_str = str(args_dict['ratio']) if args_dict['ratio'] is not None else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}" # Filename safe ratio string
    ref_str = f"_ref{args_dict['reference']}"
    output_format = args_dict['output_format'].lower() if args_dict['output_format'] else None
    if output_format:
        output_ext = f".{output_format.lstrip('.')}"
        # Validate if Pillow supports the requested format
        if output_ext[1:] not in Image.registered_extensions():
             logging.warning(f"{filename}: Unsupported output format '{args_dict['output_format']}'. Using original format '{original_ext}'.")
             output_ext = original_ext
    else:
        output_ext = original_ext # Keep original extension if format not specified

    # Determine composition rules to apply
    rules_to_apply = []
    if args_dict['rule'] == 'thirds' or args_dict['rule'] == 'both':
        rules_to_apply.append(('thirds', OUTPUT_SUFFIX_THIRDS))
    if args_dict['rule'] == 'golden' or args_dict['rule'] == 'both':
        rules_to_apply.append(('golden', OUTPUT_SUFFIX_GOLDEN))

    if not rules_to_apply:
        status['message'] = f"No composition rule selected to apply ('{args_dict['rule']}')."
        logging.warning(f"{filename}: {status['message']}")
        return status

    saved_count = 0
    crop_errors = []
    # Crop and save/Dry Run for each applicable rule
    for rule_name, suffix in rules_to_apply:
        rule_points = get_rule_points(img_w, img_h, rule_name)
        target_ratio_float = parse_aspect_ratio(args_dict['ratio'])
        crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, target_ratio_float)

        if crop_coords:
            padded_coords = apply_padding(crop_coords, (img_h, img_w), args_dict['padding_percent'])
            x1, y1, x2, y2 = padded_coords

            if x1 >= x2 or y1 >= y2:
                msg = f"Rule '{rule_name}': Final crop area size is zero."
                logging.warning(f"{filename}: {msg}")
                crop_errors.append(msg)
                continue # Skip this rule

            out_filename = f"{base}{suffix}{ratio_str}{ref_str}{output_ext}"
            out_path = os.path.join(output_dir, out_filename)

            # Check for overwrite condition
            if not args_dict['overwrite'] and os.path.exists(out_path) and not is_dry_run:
                logging.info(f"{filename}: File exists and overwrite disabled - skipping: {out_filename}")
                continue

            if is_dry_run:
                logging.info(f"[DRY RUN] {filename}: Would save '{out_filename}' (Rule: {rule_name}, Area: {x1},{y1}-{x2},{y2})")
                saved_count += 1
            else:
                # Perform actual cropping and saving
                try:
                    cropped_img_bgr = img[y1:y2, x1:x2] # Crop using OpenCV (BGR)
                    if cropped_img_bgr.size == 0:
                         raise ValueError("Cropped image size is zero.")

                    # Convert cropped image back to RGB for Pillow saving
                    cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                    pil_cropped_img = Image.fromarray(cropped_img_rgb)

                    # Prepare save options (EXIF, quality)
                    save_options = {}
                    if exif_data and isinstance(exif_data, bytes):
                        save_options['exif'] = exif_data
                    if output_ext.lower() in ['.jpg', '.jpeg']:
                        save_options['quality'] = args_dict['jpeg_quality']
                        save_options['optimize'] = True
                        save_options['progressive'] = True
                    # Add other format options here if needed (e.g., PNG compression)

                    pil_cropped_img.save(out_path, **save_options)
                    logging.debug(f"{filename}: Saved successfully - {out_filename}")
                    saved_count += 1
                except IOError as e:
                     msg = f"File write error ({out_filename}): {e}"
                     logging.error(f"{filename}: {msg}")
                     crop_errors.append(msg)
                except ValueError as e: # Catch zero-size crop error
                     msg = f"Cropped image processing error ({out_filename}): {e}"
                     logging.error(f"{filename}: {msg}")
                     crop_errors.append(msg)
                except Exception as e:
                    msg = f"Error saving cropped image ({out_filename}): {e}"
                    # Include stack trace in verbose mode
                    logging.error(f"{filename}: {msg}", exc_info=args_dict.get('verbose', False))
                    crop_errors.append(msg)
        else:
             # Crop calculation failed for this rule
             msg = f"Rule '{rule_name}': Failed to generate valid crop area."
             logging.warning(f"{filename}: {msg}")
             crop_errors.append(msg)

    end_time = time.time()
    processing_time = end_time - start_time

    # Final status update for the image
    if saved_count > 0:
        status['success'] = True
        status['saved_files'] = saved_count
        action_verb = "Simulation complete" if is_dry_run else "Processing complete"
        status['message'] = f"{action_verb} ({saved_count} file(s) {'to be saved' if is_dry_run else 'saved'}, {processing_time:.2f}s)."
        if crop_errors:
            status['message'] += f" Some errors occurred: {'; '.join(crop_errors)}"
        logging.info(f"{filename}: {status['message']}")
    elif detected_faces: # Faces detected, but no files saved/simulated
        status['message'] = f"Faces detected, but failed to {'simulate' if is_dry_run else 'crop/save'} valid files ({processing_time:.2f}s). Errors: {'; '.join(crop_errors) if crop_errors else 'Crop area calculation failed'}"
        logging.info(f"{filename}: {status['message']}")
    # If no faces were detected, the message is already set earlier

    return status


# --- Parallel Processing Wrapper ---
def process_image_wrapper(args_tuple):
    """ Wrapper function for concurrent.futures.ProcessPoolExecutor """
    # Unpack arguments needed for each process
    image_path, output_dir, model_path, args_dict = args_tuple
    # Reset logger for each process (optional but recommended)
    process_logger = logging.getLogger()
    if not process_logger.hasHandlers(): # Prevent adding handlers multiple times
        process_handler = logging.StreamHandler()
        process_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        process_handler.setFormatter(process_formatter)
        process_logger.addHandler(process_handler)
        process_logger.setLevel(logging.DEBUG if args_dict.get('verbose') else logging.INFO)

    detector = None # Initialize detector
    try:
        # --- Load detector within each process ---
        logging.debug(f"Process started: Loading model ({os.path.basename(model_path)})...")
        detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0))
        detector.setScoreThreshold(args_dict['confidence'])
        detector.setNMSThreshold(args_dict['nms'])
        logging.debug("Process started: Model loading complete.")
        # --- Detector loading finished ---

        # Call the actual image processing function (pass the loaded detector)
        return process_image(image_path, output_dir, detector, args_dict)

    except Exception as e:
        # Catch exceptions not handled within process_image or during model loading
        filename = os.path.basename(image_path)
        logging.error(f"{filename}: Critical error during processing (wrapper): {e}", exc_info=True)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'message': f"Critical error in wrapper: {e}", 'dry_run': args_dict.get('dry_run', False)}


# --- Command-Line Interface and Execution ---
# Store the main process PID (for preventing duplicate logs)
main_process_pid = os.getpid()

def main():
    """
    Main function to parse arguments, configure settings, and orchestrate image processing.
    Handles single file and directory processing, including parallel execution.
    """
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(
        description="""Automatically crop images based on face detection and composition rules (Rule of Thirds, Golden Ratio).""", # Module docstring in English
        formatter_class=argparse.RawTextHelpFormatter # Preserve help message formatting
    )
    # --- Basic Arguments ---
    parser.add_argument("input_path", help="Path to the image file or directory to process.")
    # --- Shorthand Arguments Added ---
    parser.add_argument("-o", "--output_dir", help="Directory to save results.")
    parser.add_argument("--config", help="Path to a JSON configuration file to load options from.")

    # --- Behavior Control Arguments ---
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, help="Allow overwriting output files.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate processing without saving files.")
    parser.add_argument("-w", "--workers", type=int, help="Number of parallel workers (0 or 1 for sequential processing).")

    # --- Face Detection and Selection Arguments ---
    parser.add_argument("-m", "--method", choices=['largest', 'center'], help="Method to select the main subject.")
    parser.add_argument("--ref", "--reference", dest="reference", choices=['eye', 'box'], help="Reference point type for composition.")
    parser.add_argument("-c", "--confidence", type=float, help="Minimum face detection confidence.")
    parser.add_argument("-n", "--nms", type=float, help="Face detection NMS threshold.")
    parser.add_argument("--min-face-width", type=int, help="Minimum face width to process (pixels).")
    parser.add_argument("--min-face-height", type=int, help="Minimum face height to process (pixels).")

    # --- Cropping and Composition Arguments ---
    parser.add_argument("-r", "--ratio", type=str, help="Target crop aspect ratio (e.g., '16:9', '1.0', 'None').")
    parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], help="Composition rule(s) to apply.")
    parser.add_argument("-p", "--padding-percent", type=float, help="Padding percentage around the crop area (%).")

    # --- Output Format Arguments ---
    parser.add_argument("--output-format", type=str, help="Output image format (e.g., 'jpg', 'png').")
    parser.add_argument("-q", "--jpeg-quality", type=int, choices=range(1, 101), metavar="[1-100]", help="JPEG save quality (1-100).")

    # --- Miscellaneous Arguments ---
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging (DEBUG level).")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    # Initial parse to get config file path if provided
    args = parser.parse_args()

    # --- Load and Merge Settings ---
    config = DEFAULT_CONFIG.copy()
    config_path = args.config
    if config_path:
        loaded_config = load_config(config_path)
        # Merge loaded config into defaults
        for key, value in loaded_config.items():
            if key in config:
                # Handle specific type conversions if necessary
                if key == 'ratio' and value is not None:
                    config[key] = str(value)
                elif key == 'padding_percent' and value is not None:
                    config[key] = float(value)
                else:
                    config[key] = value
            else:
                logging.warning(f"Unknown key '{key}' in configuration file ignored.")

    # Override with command-line arguments (only if they differ from defaults or loaded config)
    cmd_args = vars(args)
    for key, value in cmd_args.items():
        if value is not None: # Only consider args explicitly provided by the user
            # Check if it's a boolean action (store_true/false/BooleanOptional)
            is_bool_action = isinstance(parser.get_default(key), bool) or \
                             any(action.dest == key and isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction))
                                 for action in parser._actions)

            if is_bool_action:
                 # Handle boolean flags correctly
                 if isinstance(value, bool):
                      config[key] = value
                 # Handle cases where action implies bool but value isn't explicitly bool (e.g. --overwrite vs --no-overwrite)
                 elif isinstance(parser._get_action_from_dest(key), (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                      config[key] = value # Assign the parsed value (True/False)
            # For non-boolean args, override if value is different from default OR if loaded from config and cmd differs
            elif value != parser.get_default(key) or (config_path and key in loaded_config and value != loaded_config.get(key)):
                if key == 'ratio' and value is not None:
                     config[key] = str(value)
                elif key == 'padding_percent' and value is not None:
                     config[key] = float(value)
                elif key != 'config': # Don't store the config path itself in the final args
                     config[key] = value

    # Store final settings in a dictionary (for passing to ProcessPoolExecutor)
    final_args_dict = config

    # Set logging level (for the main process)
    if final_args_dict['verbose']:
        logger.setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")
        logging.debug(f"Final effective settings: {final_args_dict}")
    else:
         logger.setLevel(logging.INFO)

    if final_args_dict['dry_run']:
        logging.info("***** Running in Dry Run mode. No files will be saved. *****")

    # --- Final Argument Validation ---
    if final_args_dict['min_face_width'] < 0:
        logging.warning(f"Minimum face width must be >= 0 ({final_args_dict['min_face_width']}). Using default {DEFAULT_CONFIG['min_face_width']}.")
        final_args_dict['min_face_width'] = DEFAULT_CONFIG['min_face_width']
    if final_args_dict['min_face_height'] < 0:
        logging.warning(f"Minimum face height must be >= 0 ({final_args_dict['min_face_height']}). Using default {DEFAULT_CONFIG['min_face_height']}.")
        final_args_dict['min_face_height'] = DEFAULT_CONFIG['min_face_height']
    if final_args_dict['padding_percent'] < 0:
        logging.warning(f"Padding percent must be >= 0 ({final_args_dict['padding_percent']}). Using 0.")
        final_args_dict['padding_percent'] = 0.0
    if final_args_dict['workers'] < 0:
        logging.warning(f"Number of workers must be >= 0 ({final_args_dict['workers']}). Using default {DEFAULT_CONFIG['workers']}.")
        final_args_dict['workers'] = DEFAULT_CONFIG['workers']

    # --- Preparation Steps ---
    # Download DNN model file (only in the main process)
    if not download_model(YUNET_MODEL_URL, YUNET_MODEL_PATH):
        logging.critical("DNN model file is not ready. Aborting processing.")
        return

    # Create output directory (only if not in Dry Run mode)
    if not final_args_dict['dry_run']:
        output_dir = final_args_dict['output_dir']
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logging.info(f"Output directory created: {output_dir}")
            except OSError as e:
                logging.critical(f"Failed to create output directory: {e}")
                return
        elif not os.path.isdir(output_dir):
             logging.critical(f"The specified output path '{output_dir}' is not a directory.")
             return
    else:
         logging.info(f"[DRY RUN] Skipping output directory check/creation: {final_args_dict['output_dir']}")

    # --- Image Processing ---
    input_path = args.input_path # Use the original input path from args
    if os.path.isfile(input_path):
        # Process a single file (no parallel processing needed, load detector in main process)
        logging.info(f"Starting single file processing: {input_path}")
        try:
            # Load detector here for single file case
            detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (0, 0))
            detector.setScoreThreshold(final_args_dict['confidence'])
            detector.setNMSThreshold(final_args_dict['nms'])
            # Call process_image directly with the loaded detector and args dictionary
            result = process_image(input_path, final_args_dict['output_dir'], detector, final_args_dict)
            logging.info(f"Single file processing finished. Result: {result['message']}")
        except cv2.error as e:
             logging.critical(f"Failed to load face detection model (single file): {e}")
        except Exception as e:
             logging.critical(f"Error during single file processing: {e}", exc_info=True)

    elif os.path.isdir(input_path):
        # Process a directory
        logging.info(f"Starting directory processing: {input_path}")
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp') # Added webp
        try:
            all_files = os.listdir(input_path)
            # Filter for supported image files
            image_files = [os.path.join(input_path, f) for f in all_files
                           if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(input_path, f))]
        except OSError as e:
            logging.critical(f"Cannot access input directory: {e}")
            return

        if not image_files:
            logging.info("No image files to process found in the directory.")
            return

        num_workers = final_args_dict['workers'] if final_args_dict['workers'] > 0 else 1
        # Consider system CPU count when using ProcessPoolExecutor
        actual_workers = min(num_workers, os.cpu_count()) if num_workers > 0 else 1
        logging.info(f"Starting processing for {len(image_files)} image file(s) (Workers requested: {num_workers}, Max actual workers: {actual_workers})...")
        total_start_time = time.time()
        results = []
        processed_count = 0
        success_count = 0
        total_saved_files = 0
        failed_files = []

        # Parallel or sequential processing
        if num_workers > 1:
             # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                # Create list of tuples with arguments for each task
                # Pass model_path and settings dict instead of the detector object
                tasks = [(img_path, final_args_dict['output_dir'], YUNET_MODEL_PATH, final_args_dict) for img_path in image_files]
                try:
                    # Map tasks to the wrapper function
                    results_iterator = executor.map(process_image_wrapper, tasks)
                    # Use tqdm for progress bar if available
                    if TQDM_AVAILABLE:
                         results = list(tqdm(results_iterator, total=len(tasks), desc="Processing images"))
                    else:
                         # Log progress periodically if tqdm is not available
                         temp_results = []
                         for i, result in enumerate(results_iterator):
                              temp_results.append(result)
                              if (i + 1) % 50 == 0: # Log every 50 files
                                   logging.info(f"Progress: {i + 1}/{len(tasks)} files processed...")
                         results = temp_results
                except Exception as e:
                     logging.critical(f"Critical error during parallel processing: {e}", exc_info=True)
                     # Note: Individual file errors are handled in the wrapper/process_image
        else:
            # Sequential processing (load detector once in main process)
            logging.info("Running in sequential processing mode.")
            try:
                detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (0, 0))
                detector.setScoreThreshold(final_args_dict['confidence'])
                detector.setNMSThreshold(final_args_dict['nms'])
                # Iterate through files, using tqdm if available
                iterator = tqdm(image_files, desc="Processing images") if TQDM_AVAILABLE else image_files
                for img_path in iterator:
                     # Call process_image directly
                     results.append(process_image(img_path, final_args_dict['output_dir'], detector, final_args_dict))
            except cv2.error as e:
                 logging.critical(f"Failed to load face detection model (sequential): {e}")
                 return # Stop sequential processing if model fails to load
            except Exception as e:
                 logging.critical(f"Error during sequential processing: {e}", exc_info=True)
                 return


        # Aggregate and summarize results
        for result in results:
            processed_count += 1
            if result['success']:
                success_count += 1
                total_saved_files += result['saved_files']
            if not result['success']:
                 # Store filename and message for failed files
                 failed_files.append(f"{result['filename']} ({result['message']})")

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        action_verb = "Simulation" if final_args_dict['dry_run'] else "Processing"

        # Print summary
        logging.info("-" * 30)
        logging.info(f"          Directory {action_verb} Summary          ")
        logging.info("-" * 30)
        logging.info(f"Total files attempted: {processed_count} / {len(image_files)}")
        logging.info(f"Successfully processed files: {success_count}")
        logging.info(f"Total cropped images {'to be saved' if final_args_dict['dry_run'] else 'saved'}: {total_saved_files}")
        logging.info(f"Files with errors or partial failures: {len(failed_files)}")
        if failed_files:
            logging.info("Failure details (check logs for more info):")
            # Limit displayed failed files in summary
            for i, fail_info in enumerate(failed_files):
                 if i < 10:
                    # Show only filename in summary for brevity
                    logging.info(f"  - {fail_info.split(' (')[0]}")
                 elif i == 10:
                    logging.info("  - ... (Check logs for more failed items)")
                    break
        logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
        logging.info("-" * 30)

    else:
        logging.critical(f"Input path not found or invalid: {input_path}")

if __name__ == "__main__":
    # Store main process PID (used to differentiate logging in child processes)
    main_process_pid = os.getpid()
    main()
