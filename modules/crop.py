# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
import logging
import time
import concurrent.futures
from PIL import Image, UnidentifiedImageError, ImageOps
from typing import Tuple, List, Optional, Dict, Any, Union
import tqdm

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
    logger.setLevel(logging.WARNING)

YUNET_MODEL_FILENAME: str = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL: str = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_MODEL_FILENAME}"
OUTPUT_SUFFIX_THIRDS: str = '_thirds'
OUTPUT_SUFFIX_GOLDEN: str = '_golden'
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
MODEL_DIR_NAME: str = "models"

main_process_pid = os.getpid()

class CropSetupError(Exception):
    """Custom exception for critical errors during crop setup."""
    pass

def setup_logging(level: int):
    logger.setLevel(level)

def download_model(model_url: str, file_path: str) -> bool:
    model_dir = os.path.dirname(file_path)
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
        except OSError as e:
            logger.error(f"  -> Error: Failed to create model directory '{os.path.abspath(model_dir)}': {e}")
            return False

    if not os.path.exists(file_path):
        if os.getpid() == main_process_pid:
            logger.info(f"  -> Info: Downloading model file... ({os.path.basename(file_path)}) from {model_url}")
        try:
            if os.getpid() == main_process_pid:
                with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=f"  Downloading {os.path.basename(file_path)}") as t:
                    def reporthook(blocknum, blocksize, totalsize):
                        if totalsize > 0:
                            t.total = totalsize
                        t.update(blocksize)
                    urllib.request.urlretrieve(model_url, file_path, reporthook=reporthook)
            else:
                 urllib.request.urlretrieve(model_url, file_path)

            if os.getpid() == main_process_pid:
                logger.info(f"  -> Info: Download complete. Model saved to {os.path.abspath(file_path)}")
            return True
        except Exception as e:
            logger.error(f"  -> Error: Model file download failed: {e}")
            logger.error(f"       Please manually download from the following URL and save as '{os.path.abspath(file_path)}': {model_url}")
            return False
    return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    if ratio_str is None or str(ratio_str).strip().lower() == 'none':
        return None
    try:
        ratio_str = str(ratio_str).strip()
        if ':' in ratio_str:
            w_str, h_str = ratio_str.split(':')
            w, h = float(w_str), float(h_str)
            if h <= 0 or w <= 0:
                logger.warning(f"  -> Warning: Ratio width or height cannot be zero or less: '{ratio_str}'. Using original ratio.")
                return None
            return w / h
        else:
            ratio = float(ratio_str)
            if ratio <= 0:
                logger.warning(f"  -> Warning: Ratio must be greater than zero: '{ratio_str}'. Using original ratio.")
                return None
            return ratio
    except ValueError:
        logger.warning(f"  -> Warning: Invalid ratio string format: '{ratio_str}'. Using original ratio.")
        return None
    except Exception as e:
        logger.error(f"  -> Error: Unexpected error parsing ratio ('{ratio_str}'): {e}")
        return None

def create_output_directory(output_dir: str, dry_run: bool) -> bool:
    abs_output_dir = os.path.abspath(output_dir)
    if dry_run:
        logger.info(f"  -> Info: [DRY RUN] Skipping output directory check/creation: {abs_output_dir}")
        return True
    if not os.path.exists(abs_output_dir):
        try:
            os.makedirs(abs_output_dir)
            logger.info(f"  -> Info: Created output directory: {abs_output_dir}")
            return True
        except OSError as e:
            logger.critical(f"  -> Critical: Failed to create output directory '{abs_output_dir}': {e}")
            return False
    elif not os.path.isdir(abs_output_dir):
         logger.critical(f"  -> Critical: The specified output path '{abs_output_dir}' is not a directory.")
         return False
    return True

def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray, min_w: int, min_h: int) -> List[Dict[str, Any]]:
    detected_subjects = []
    if image is None or image.size == 0:
        logger.warning("  -> Warning: Input image for face detection is empty.")
        return []
    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        logger.warning(f"  -> Warning: Invalid image size ({img_w}x{img_h}), skipping face detection.")
        return []
    try:
        detector.setInputSize((img_w, img_h))
        faces = detector.detect(image)

        if faces is not None and faces[1] is not None:
            for face_info in faces[1]:
                x, y, w, h = map(int, face_info[:4])
                if w < min_w or h < min_h:
                    continue

                r_eye_x, r_eye_y = face_info[4:6]; l_eye_x, l_eye_y = face_info[6:8]
                confidence = face_info[14]

                x = max(0, x); y = max(0, y)
                w = min(img_w - x, w); h = min(img_h - y, h)

                if w > 0 and h > 0:
                    bbox_center = (x + w // 2, y + h // 2)
                    eye_center = bbox_center

                    if r_eye_x > 0 and r_eye_y > 0 and l_eye_x > 0 and l_eye_y > 0:
                        ecx = int(round((r_eye_x + l_eye_x) / 2)); ecy = int(round((r_eye_y + l_eye_y) / 2))
                        ecx = max(0, min(img_w - 1, ecx)); ecy = max(0, min(img_h - 1, ecy))
                        eye_center = (ecx, ecy)

                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e:
        logger.error(f"  -> Error: OpenCV error during face detection (image size: {img_w}x{img_h}): {e}")
    except Exception as e:
        logger.error(f"  -> Error: Unexpected problem during DNN face detection: {e}", exc_info=logger.level == logging.DEBUG)
    return detected_subjects

def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'box') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    if not subjects: return None
    img_h, img_w = img_shape; best_subject = None
    try:
        if len(subjects) == 1: best_subject = subjects[0]
        elif method == 'largest': best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
        elif method == 'center':
            img_center = (img_w / 2, img_h / 2)
            best_subject = min(subjects, key=lambda s: math.dist(s['bbox_center'], img_center))
        else:
            logger.warning(f"  -> Warning: Unknown selection method '{method}'. Defaulting to 'largest'.")
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        return best_subject['bbox'], ref_center
    except Exception as e:
        logger.error(f"  -> Error: Error selecting main subject: {e}", exc_info=logger.level == logging.DEBUG)
        return None

def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    points = []
    if width <= 0 or height <= 0: logger.warning(f"  -> Warning: Cannot calculate rule points for invalid size ({width}x{height})."); return []
    try:
        if rule_type == 'thirds':
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden':
            phi_inv = (math.sqrt(5) - 1) / 2
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            points = [(w, h) for w in lines_w for h in lines_h]
        else:
            logger.warning(f"  -> Warning: Unknown composition rule '{rule_type}'. Using image center.")
            points = [(width / 2, height / 2)]
        return [(int(round(px)), int(round(py))) for px, py in points]
    except Exception as e:
        logger.error(f"  -> Error: Error calculating rule points (Rule: {rule_type}, Size: {width}x{height}): {e}", exc_info=logger.level == logging.DEBUG)
        return []

def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    height, width = img_shape
    if height <= 0 or width <= 0: logger.warning(f"  -> Warning: Cannot crop image with zero/negative dimensions ({width}x{height})."); return None
    if not rule_points: logger.warning("  -> Warning: No rule points provided, skipping crop calculation."); return None

    cx, cy = subject_center
    try:
        aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        if aspect_ratio <= 0: logger.warning(f"  -> Warning: Invalid target aspect ratio ({aspect_ratio}). Cannot calculate crop."); return None

        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point

        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)
        if max_w <= 0 or max_h <= 0: return None

        crop_h_from_w = max_w / aspect_ratio
        crop_w_from_h = max_h * aspect_ratio

        if crop_h_from_w <= max_h + 1e-6:
            final_w, final_h = max_w, crop_h_from_w
        else:
            final_w, final_h = crop_w_from_h, max_h

        x1 = target_x - final_w / 2; y1 = target_y - final_h / 2
        x2 = x1 + final_w; y2 = y1 + final_h

        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        if x1 >= x2 or y1 >= y2:
            logger.warning(f"  -> Warning: Calculated crop area has zero/negative size ({x1},{y1} to {x2},{y2})."); return None
        return x1, y1, x2, y2
    except Exception as e:
        logger.error(f"  -> Error: Error calculating optimal crop (Subject: {subject_center}, RulePt: {closest_point if 'closest_point' in locals() else 'N/A'}): {e}", exc_info=logger.level == logging.DEBUG)
        return None

def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop_coords; img_h, img_w = img_shape
    crop_w = x2 - x1; crop_h = y2 - y1
    if padding_percent <= 0: return crop_coords

    pad_x = int(round(crop_w * padding_percent / 100 / 2))
    pad_y = int(round(crop_h * padding_percent / 100 / 2))

    new_x1 = max(0, x1 - pad_x); new_y1 = max(0, y1 - pad_y)
    new_x2 = min(img_w, x2 + pad_x); new_y2 = min(img_h, y2 + pad_y)

    if new_x1 >= new_x2 or new_y1 >= new_y2:
        logger.warning(f"  -> Warning: Crop area invalid after {padding_percent}% padding. No padding applied.")
        return crop_coords
    return new_x1, new_y1, new_x2, new_y2

def _load_and_prepare_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[bytes], Optional[str]]:
    filename = os.path.basename(image_path); exif_data = None; original_ext = os.path.splitext(image_path)[1].lower(); img_bgr = None
    try:
        with Image.open(image_path) as pil_img:
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
                exif_data = pil_img.info.get('exif')
            except Exception as exif_err:
                logger.warning(f"  -> Warning: {filename}: Error processing EXIF data: {exif_err}. Proceeding without EXIF.")
                exif_data = None

            pil_img_rgb = pil_img.convert('RGB')
            img_bgr = np.array(pil_img_rgb)[:, :, ::-1].copy()
    except FileNotFoundError: logger.error(f"  -> Error: {filename}: Image file not found."); return None, None, None
    except UnidentifiedImageError: logger.error(f"  -> Error: {filename}: Cannot open or unsupported image format."); return None, None, None
    except Exception as e: logger.error(f"  -> Error: {filename}: Error loading image: {e}", exc_info=logger.level == logging.DEBUG); return None, None, None
    return img_bgr, exif_data, original_ext

def _determine_output_filename(base_filename: str, suffix_str: str, settings: argparse.Namespace, original_ext: str) -> Tuple[str, str]:
    target_ratio_str = str(settings.ratio) if settings.ratio is not None else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"; ref_str = f"_ref{settings.reference}"

    output_format = settings.output_format.lower() if settings.output_format else None
    if output_format:
        output_ext = f".{output_format.lstrip('.')}"
        if output_ext.lower() not in Image.registered_extensions():
             logger.warning(f"  -> Warning: {base_filename}: Unsupported output format '{settings.output_format}'. Using original '{original_ext}'.")
             output_ext = original_ext
    else:
        output_ext = original_ext

    out_filename = f"{base_filename}{suffix_str}{ratio_str}{ref_str}{output_ext}"
    return out_filename, output_ext

def _save_cropped_image(cropped_bgr: np.ndarray, out_path: str, output_ext: str, settings: argparse.Namespace, exif_data: Optional[bytes]) -> bool:
    filename = os.path.basename(out_path)
    try:
        if cropped_bgr.size == 0: raise ValueError("Cropped image data is empty.")

        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_cropped_img = Image.fromarray(cropped_rgb)

        save_options = {}
        if not settings.strip_exif and exif_data and isinstance(exif_data, bytes):
            save_options['exif'] = exif_data

        if output_ext.lower() in ['.jpg', '.jpeg']:
            save_options['quality'] = settings.jpeg_quality
            save_options['optimize'] = True
            save_options['progressive'] = True
        elif output_ext.lower() == '.webp':
            save_options['quality'] = settings.webp_quality

        pil_cropped_img.save(out_path, **save_options)
        return True
    except IOError as e: logger.error(f"  -> Error: {filename}: File write error: {e}"); return False
    except ValueError as e: logger.error(f"  -> Error: {filename}: Cropped image processing error: {e}"); return False
    except Exception as e: logger.error(f"  -> Error: {filename}: Error saving cropped image: {e}", exc_info=settings.verbose); return False

def _process_composition_rule(rule_name: str, rule_suffix: str, img_bgr: np.ndarray, ref_center: Tuple[int, int], current_settings: argparse.Namespace, exif_data: Optional[bytes], original_ext: str) -> Tuple[bool, bool, str]:
    filename = os.path.basename(current_settings.input_path); base_filename = os.path.splitext(filename)[0]
    img_h, img_w = img_bgr.shape[:2]
    saved_actual_file = False; was_skipped_overwrite = False; error_message = ""

    rule_points = get_rule_points(img_w, img_h, rule_name)
    if not rule_points: error_message = f"Rule '{rule_name}': Failed to get rule points."; logger.warning(f"  -> Warning: {filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, current_settings.target_ratio_float)
    if not crop_coords: error_message = f"Rule '{rule_name}': Failed to calculate optimal crop."; logger.warning(f"  -> Warning: {filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    padded_coords = apply_padding(crop_coords, (img_h, img_w), current_settings.padding_percent)
    x1, y1, x2, y2 = padded_coords
    if x1 >= x2 or y1 >= y2: error_message = f"Rule '{rule_name}': Final crop area invalid after padding."; logger.warning(f"  -> Warning: {filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    out_filename, output_ext = _determine_output_filename(base_filename, rule_suffix, current_settings, original_ext)
    out_path = os.path.join(current_settings.output_dir, out_filename)

    if not current_settings.overwrite and os.path.exists(out_path) and not current_settings.dry_run:
        error_message = f"Rule '{rule_name}': Skipped (file exists, overwrite disabled)."
        was_skipped_overwrite = True
        return saved_actual_file, was_skipped_overwrite, error_message

    if current_settings.dry_run:
        saved_actual_file = True
    else:
        cropped_img_bgr = img_bgr[y1:y2, x1:x2]
        save_success = _save_cropped_image(cropped_img_bgr, out_path, output_ext, current_settings, exif_data)
        if save_success: saved_actual_file = True
        else: error_message = f"Rule '{rule_name}': Failed to save file '{out_filename}'."
    return saved_actual_file, was_skipped_overwrite, error_message

def process_image(image_path: str, global_settings: argparse.Namespace, detector: cv2.FaceDetectorYN) -> Dict[str, Any]:
    filename = os.path.basename(image_path)
    start_time = time.time()
    
    current_settings = argparse.Namespace(**vars(global_settings))
    current_settings.input_path = image_path

    status = {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules': 0, 'message': '', 'dry_run': current_settings.dry_run}

    img_bgr, exif_data, original_ext = _load_and_prepare_image(image_path)
    if img_bgr is None: status['message'] = "Failed to load or prepare image."; return status

    img_h, img_w = img_bgr.shape[:2]
    if img_h <= 0 or img_w <= 0: status['message'] = f"Invalid image dimensions ({img_w}x{img_h})."; logger.warning(f"  -> Warning: {filename}: {status['message']}"); return status
    
    detected_faces = detect_faces_dnn(detector, img_bgr, current_settings.min_face_width, current_settings.min_face_height)
    if not detected_faces: status['message'] = "No valid faces detected (considering minimum size)."; return status

    selection_result = select_main_subject(detected_faces, (img_h, img_w), current_settings.method, current_settings.reference)
    if not selection_result: status['message'] = "Failed to select main subject."; return status
    _, ref_center = selection_result

    rules_to_process = []
    if current_settings.rule in ['thirds', 'both']: rules_to_process.append(('thirds', OUTPUT_SUFFIX_THIRDS))
    if current_settings.rule in ['golden', 'both']: rules_to_process.append(('golden', OUTPUT_SUFFIX_GOLDEN))
    if not rules_to_process: status['message'] = f"No composition rule selected ('{current_settings.rule}')."; return status

    saved_rules_count = 0; skipped_overwrite_count_for_image = 0; rule_errors = []
    for rule_name, rule_suffix in rules_to_process:
        saved, skipped_overwrite, error_msg = _process_composition_rule(rule_name, rule_suffix, img_bgr, ref_center, current_settings, exif_data, original_ext)
        if saved: saved_rules_count += 1
        if skipped_overwrite: skipped_overwrite_count_for_image +=1
        if error_msg and not saved and not skipped_overwrite : rule_errors.append(error_msg)

    end_time = time.time(); processing_time = end_time - start_time
    status['skipped_overwrite_rules'] = skipped_overwrite_count_for_image

    if saved_rules_count > 0:
        status['success'] = True; status['saved_files'] = saved_rules_count
        action = "simulated" if current_settings.dry_run else "saved"
        status['message'] = f"Processed ({saved_rules_count} file(s) {action}, {skipped_overwrite_count_for_image} rule(s) skipped by overwrite, {processing_time:.2f}s)."
        if rule_errors: status['message'] += f" Some rules failed: {'; '.join(rule_errors)}"
    elif skipped_overwrite_count_for_image > 0 and not rule_errors :
        status['success'] = False
        status['message'] = f"All applicable rules skipped due to overwrite policy ({processing_time:.2f}s)."
    else:
        status['success'] = False
        err_summary = '; '.join(rule_errors) if rule_errors else "Crop/Save failed or no faces/rules applicable."
        status['message'] = f"Failed to {'simulate' if current_settings.dry_run else 'crop/save'} for all rules ({processing_time:.2f}s). Errors: {err_summary}"
    return status

def process_image_wrapper(args_tuple: Tuple[str, argparse.Namespace]) -> Dict[str, Any]:
    image_path, settings = args_tuple; filename = os.path.basename(image_path); detector = None
    try:
        if not os.path.exists(settings.yunet_model_path):
             if not download_model(YUNET_MODEL_URL, settings.yunet_model_path):
                  raise RuntimeError(f"Model file {settings.yunet_model_path} not found and download failed in worker.")

        detector = cv2.FaceDetectorYN.create(settings.yunet_model_path, "", (0, 0))
        detector.setScoreThreshold(settings.confidence); detector.setNMSThreshold(settings.nms)
        return process_image(image_path, settings, detector)

    except cv2.error as e:
        logger.error(f"  -> Error: {filename}: OpenCV error in worker: {e}", exc_info=settings.verbose)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules':0, 'message': f"OpenCV Error: {e}", 'dry_run': settings.dry_run}
    except Exception as e:
        logger.error(f"  -> Error: {filename}: Critical error in worker: {e}", exc_info=True)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules':0, 'message': f"Critical error in worker: {e}", 'dry_run': settings.dry_run}

def execute_crop_operation(settings: argparse.Namespace) -> int:   
    if settings.yunet_model_path is None:
        settings.yunet_model_path = os.path.join(MODEL_DIR_NAME, YUNET_MODEL_FILENAME)
    settings.yunet_model_path = os.path.abspath(settings.yunet_model_path)

    settings.target_ratio_float = parse_aspect_ratio(settings.ratio)
    
    setup_logging(logging.DEBUG if settings.verbose else logging.INFO)

    logger.info(f"===== Image Cropping Script Started =====")
    if settings.dry_run:
        logger.info("***** Running in Dry Run mode. No files will be saved. *****")

    if not download_model(YUNET_MODEL_URL, settings.yunet_model_path):
        raise CropSetupError(f"DNN model file '{settings.yunet_model_path}' is not available or download failed.")

    if not create_output_directory(settings.output_dir, settings.dry_run):
        raise CropSetupError(f"Could not prepare output directory '{settings.output_dir}'.")

    input_path = settings.input_path
    skipped_scan_items_details = []
    images_with_errors_count = 0

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.warning(f"  -> Warning: Single file '{input_path}' is not a supported image type. Skipping.")
            logger.info(f"===== Image Cropping Script Finished =====")
            return 0

        logger.info(f"  -> Info: Starting single file processing: {os.path.abspath(input_path)}")
        detector = None
        try:
            detector = cv2.FaceDetectorYN.create(settings.yunet_model_path, "", (0, 0))
            detector.setScoreThreshold(settings.confidence); detector.setNMSThreshold(settings.nms)
            result = process_image(input_path, settings, detector)
            
            if result.get('success'):
                logger.info(f"  -> Success: {os.path.basename(input_path)}: {result.get('message', 'Processing finished.')}")
            else: 
                if result.get('skipped_overwrite_rules', 0) > 0 and not result.get('saved_files',0) and "failed" not in result.get('message','').lower() :
                     logger.info(f"  -> Info: {os.path.basename(input_path)}: {result.get('message', 'All rules skipped by overwrite policy.')}")
                else: 
                     logger.warning(f"  -> Warning: {os.path.basename(input_path)}: {result.get('message', 'Processing failed or no faces/rules applicable.')}")
                     images_with_errors_count += 1


        except cv2.error as e:
            logger.critical(f"  -> Critical: Failed to load face detection model (single file): {e}")
            images_with_errors_count += 1
        except Exception as e:
            logger.critical(f"  -> Critical: Error during single file processing: {e}", exc_info=True)
            images_with_errors_count += 1

    elif os.path.isdir(input_path):
        image_files = []
        try:
            all_items = os.listdir(input_path)
            for item_name in all_items:
                item_path = os.path.join(input_path, item_name)
                if os.path.isfile(item_path):
                    if item_name.lower().endswith(SUPPORTED_EXTENSIONS):
                        image_files.append(item_path)
                    else:
                        skipped_scan_items_details.append(f"{item_name} (Unsupported extension)")
                elif os.path.isdir(item_path):
                    skipped_scan_items_details.append(f"{item_name} (Directory)")
                else:
                    skipped_scan_items_details.append(f"{item_name} (Not a file or directory)")

        except OSError as e:
            logger.critical(f"  -> Critical: Cannot access input directory '{os.path.abspath(input_path)}': {e}")
            raise CropSetupError(f"Cannot access input directory '{os.path.abspath(input_path)}': {e}")


        if not image_files:
            logger.info(f"  -> Info: No supported image files ({', '.join(SUPPORTED_EXTENSIONS)}) found in '{os.path.abspath(input_path)}'.")
            if skipped_scan_items_details:
                 logger.info(f"  -> Info: {len(skipped_scan_items_details)} other item(s) were found and skipped during scan.")
            logger.info(f"===== Image Cropping Script Finished =====")
            return 0

        available_cpus = os.cpu_count()
        if available_cpus is None:
            logger.warning("  -> Warning: Could not determine number of CPU cores. Defaulting to 1 worker.")
            available_cpus = 1
        elif available_cpus == 0:
            logger.warning("  -> Warning: os.cpu_count() returned 0. Defaulting to 1 worker.")
            available_cpus = 1

        actual_workers = min(available_cpus, len(image_files))
        actual_workers = max(1, actual_workers)
        is_parallel = actual_workers > 1

        total_start_time = time.time()
        results_list = []

        tasks = [(img_path, settings) for img_path in image_files]
        detector_init_failed_sequentially = False

        if is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                try:
                    future_results = executor.map(process_image_wrapper, tasks)
                    tqdm_extra_kwargs = {}
                    if not settings.verbose: 
                        tqdm_extra_kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                        tqdm_extra_kwargs['ncols'] = 80

                    for result in tqdm.tqdm(future_results, total=len(tasks), desc="Processing images", unit="file", **tqdm_extra_kwargs):
                        results_list.append(result)
                except Exception as e:
                     logger.critical(f"  -> Critical: Error during parallel processing execution: {e}", exc_info=True)
                     images_with_errors_count = len(tasks)
        else: 
            detector = None
            try:
                detector = cv2.FaceDetectorYN.create(settings.yunet_model_path, "", (0, 0))
                detector.setScoreThreshold(settings.confidence); detector.setNMSThreshold(settings.nms)
                tqdm_extra_kwargs = {}
                if not settings.verbose:
                    tqdm_extra_kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                    tqdm_extra_kwargs['ncols'] = 80
                
                for task_args_item in tqdm.tqdm(tasks, desc="Processing images sequentially", unit="file", **tqdm_extra_kwargs):
                     results_list.append(process_image(task_args_item[0], task_args_item[1], detector))
            except cv2.error as e:
                logger.critical(f"  -> Critical: Failed to load face detection model (sequential): {e}")
                detector_init_failed_sequentially = True
                images_with_errors_count = len(tasks)
            except Exception as e:
                logger.critical(f"  -> Critical: Error during sequential processing loop: {e}", exc_info=True)
                images_with_errors_count = len(tasks)


        successful_image_count = 0; total_generated_count = 0; total_skipped_overwrite_count = 0

        if not detector_init_failed_sequentially and not (is_parallel and images_with_errors_count == len(tasks)):
            current_errors = 0
            for res_idx, res in enumerate(results_list):
                if res and isinstance(res, dict):
                    if res.get('success', False):
                        successful_image_count += 1
                    elif not res.get('success', False) and \
                         not (res.get('skipped_overwrite_rules', 0) > 0 and \
                              not res.get('saved_files', 0) and \
                              "failed" not in res.get('message','').lower()): 
                        current_errors +=1
                        logger.warning(f"  -> Warning: {res.get('filename')}: {res.get('message')}") 

                    total_generated_count += res.get('saved_files', 0)
                    total_skipped_overwrite_count += res.get('skipped_overwrite_rules', 0)
                else: 
                     current_errors +=1
                     failed_filename = tasks[res_idx][0] if res_idx < len(tasks) else "Unknown file"
                     logger.error(f"  -> Error: Received invalid result from worker for {os.path.basename(failed_filename)}: {res}")
            images_with_errors_count = current_errors

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time if 'total_start_time' in locals() else 0

    else: 
        if not os.path.exists(input_path):
             err_msg = f"Input path not found: {os.path.abspath(input_path)}"
             logger.critical(f"  -> Critical: {err_msg}")
             raise CropSetupError(err_msg)
        else:
             err_msg = f"Input path is neither a file nor a directory: {os.path.abspath(input_path)}"
             logger.critical(f"  -> Critical: {err_msg}")
             raise CropSetupError(err_msg)

    return images_with_errors_count
