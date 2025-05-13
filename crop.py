# -*- coding: utf-8 -*-
from PIL import Image, UnidentifiedImageError
from typing import Tuple, List, Optional, Dict, Any, Union
import os
import cv2
import numpy as np
import math
import argparse
import time
import sys
import requests

YUNET_MODEL_FILENAME: str = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL: str = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_MODEL_FILENAME}"
OUTPUT_SUFFIX_THIRDS: str = '_thirds'
OUTPUT_SUFFIX_GOLDEN: str = '_golden'
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
MODEL_DIR_NAME: str = "models"

def download_model(model_url: str, file_path: str, is_main_process: bool) -> bool:
    model_dir = os.path.dirname(file_path)
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
        except OSError as e:
            print(f"Failed to create model directory '{os.path.abspath(model_dir)}': {e}")
            return False

    if not os.path.exists(file_path):
        if is_main_process:
            print(f"Downloading model file... ({os.path.basename(file_path)}) from {model_url}")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if is_main_process:
                print(f"Model file downloaded successfully: {os.path.abspath(file_path)}")
            return True
        except Exception as e:
            print(f"Model file download failed: {e}")
            print(f"Please manually download from the following URL and save as '{os.path.abspath(file_path)}': {model_url}")
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
                print(f"Invalid ratio values in '{ratio_str}'. Width and height must be positive. Using original ratio.")
                return None
            return w / h
        else:
            ratio = float(ratio_str)
            if ratio <= 0:
                print(f"Invalid ratio value '{ratio_str}'. Ratio must be positive. Using original ratio.")
                return None
            return ratio
    except ValueError:
        print(f"Invalid ratio string format: '{ratio_str}'. Using original ratio.")
        return None
    except Exception as e:
        print(f"Unexpected error parsing ratio ('{ratio_str}'): {e}")
        return None

def create_output_directory(output_dir: str) -> bool:
    abs_output_dir = os.path.abspath(output_dir)
    if not os.path.exists(abs_output_dir):
        try:
            os.makedirs(abs_output_dir)
            print(f"Created output directory: {abs_output_dir}")
            return True
        except OSError as e:
            print(f"Failed to create output directory '{abs_output_dir}': {e}")
            return False
    elif not os.path.isdir(abs_output_dir):
         print(f"The specified output path '{abs_output_dir}' is not a directory.")
         return False
    return True

def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray, min_w: int, min_h: int) -> List[Dict[str, Any]]:
    detected_subjects = []
    if image is None or image.size == 0:
        print("Input image for face detection is empty.")
        return []
    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        print(f"Invalid image size ({img_w}x{img_h}), skipping face detection.")
        return []
    try:
        detector.setInputSize((img_w, img_h))
        faces = detector.detect(image)

        if faces is not None and faces[1] is not None:
            for face_info in faces[1]:
                x, y, w, h = int(face_info[0]), int(face_info[1]), int(face_info[2]), int(face_info[3])
                if w < min_w or h < min_h:
                    continue

                landmarks = face_info[4:14].reshape(5, 2).astype(int)
                score = float(face_info[14])

                eye_center_x = (landmarks[0][0] + landmarks[1][0]) / 2
                eye_center_y = (landmarks[0][1] + landmarks[1][1]) / 2
                eye_center = (int(round(eye_center_x)), int(round(eye_center_y)))

                bbox_center_x = x + w / 2
                bbox_center_y = y + h / 2
                bbox_center = (int(round(bbox_center_x)), int(round(bbox_center_y)))

                detected_subjects.append({
                    'bbox': (x, y, w, h),
                    'landmarks': landmarks,
                    'score': score,
                    'eye_center': eye_center,
                    'bbox_center': bbox_center
                })
    except cv2.error as e:
        print(f"OpenCV error during face detection (image size: {img_w}x{img_h}): {e}")
    except Exception as e:
        print(f"Unexpected problem during DNN face detection: {e}")
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
            best_subject = min(subjects, key=lambda s: math.dist(s.get('bbox_center', (0,0)), img_center))
        else:
            print(f"Unknown selection method '{method}'. Defaulting to 'largest'.")
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        if best_subject:
            bbox_center_val = best_subject.get('bbox_center', 'N/A')
            eye_center_val = best_subject.get('eye_center', 'N/A')

            is_eye_center_valid = isinstance(eye_center_val, tuple) and len(eye_center_val) == 2
            is_bbox_center_valid = isinstance(bbox_center_val, tuple) and len(bbox_center_val) == 2

            if reference_point_type == 'eye' and is_eye_center_valid:
                ref_center = eye_center_val
            elif is_bbox_center_valid:
                ref_center = bbox_center_val
            else:
                bx, by, bw, bh = best_subject['bbox']
                ref_center = (int(round(bx + bw / 2)), int(round(by + bh / 2)))
        else:
            print("No best subject could be determined.")
            return None

        return best_subject['bbox'], ref_center
    except Exception as e:
        print(f"Error selecting main subject: {e}")
        return None

def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    points = []
    if width <= 0 or height <= 0: print(f"Cannot calculate rule points for invalid size ({width}x{height})."); return []
    try:
        if rule_type == 'thirds':
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden':
            phi_inv = (math.sqrt(5) - 1) / 2
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            points = [(w, h) for w in lines_w for h in lines_h]
        else:
            print(f"Unknown composition rule '{rule_type}'. Using image center.")
            points = [(width / 2, height / 2)]
        return [(int(round(px)), int(round(py))) for px, py in points]
    except Exception as e:
        print(f"Error calculating rule points (Rule: {rule_type}, Size: {width}x{height}): {e}")
        return []

def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    height, width = img_shape
    if height <= 0 or width <= 0: print(f"Cannot crop image with zero/negative dimensions ({width}x{height})."); return None
    if not rule_points: print("No rule points provided, skipping crop calculation."); return None

    cx, cy = subject_center
    try:
        aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        if aspect_ratio <= 0: print(f"Invalid target aspect ratio ({aspect_ratio}). Cannot calculate crop."); return None

        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point

        max_w_for_target = 2 * min(target_x, width - target_x)
        max_h_for_target = 2 * min(target_y, height - target_y)
        if max_w_for_target <= 0 or max_h_for_target <= 0:
            print(f"Cannot create a valid crop centered at rule point ({target_x},{target_y}) as it's too close to image edge or image dimensions are too small.")
            return None

        crop_h_from_w = max_w_for_target / aspect_ratio
        crop_w_from_h = max_h_for_target * aspect_ratio

        if crop_h_from_w <= max_h_for_target + 1e-6:
            final_w, final_h = max_w_for_target, crop_h_from_w
        else:
            final_w, final_h = crop_w_from_h, max_h_for_target

        x1_raw = target_x - final_w / 2.0; y1_raw = target_y - final_h / 2.0
        x2_raw = x1_raw + final_w; y2_raw = y1_raw + final_h

        x1, y1 = max(0, int(round(x1_raw))), max(0, int(round(y1_raw)))
        x2, y2 = min(width, int(round(x2_raw))), min(height, int(round(y2_raw)))

        if x1 >= x2 or y1 >= y2:
            print(f"Calculated crop area has zero/negative size ({x1},{y1}) to ({x2},{y2}). Raw was ({x1_raw:.2f},{y1_raw:.2f}) to ({x2_raw:.2f},{y2_raw:.2f})."); return None
        return x1, y1, x2, y2
    except Exception as e:
        print(f"Error calculating optimal crop (Subject: {subject_center}, RulePt: {closest_point if 'closest_point' in locals() else 'N/A'}): {e}")
        return None

def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop_coords; img_h, img_w = img_shape
    crop_w = x2 - x1; crop_h = y2 - y1
    if padding_percent <= 0: return crop_coords

    pad_x = int(round(crop_w * padding_percent / 100.0 / 2.0))
    pad_y = int(round(crop_h * padding_percent / 100.0 / 2.0))

    new_x1 = max(0, x1 - pad_x); new_y1 = max(0, y1 - pad_y)
    new_x2 = min(img_w, x2 + pad_x); new_y2 = min(img_h, y2 + pad_y)

    if new_x1 >= new_x2 or new_y1 >= new_y2:
        print(f"Crop area invalid after {padding_percent}% padding. Original: ({x1},{y1})-({x2},{y2}). Padded: ({new_x1},{new_y1})-({new_x2},{new_y2}). No padding applied.")
        return crop_coords
    return new_x1, new_y1, new_x2, new_y2

def _load_and_prepare_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    filename = os.path.basename(image_path); original_ext = os.path.splitext(image_path)[1].lower(); img_bgr = None
    try:
        with Image.open(image_path) as pil_img:
            pil_img_rgb = pil_img.convert('RGB')
            img_bgr = np.array(pil_img_rgb)[:, :, ::-1].copy()
    except FileNotFoundError: print(f"{filename}: Image file not found."); return None, None
    except UnidentifiedImageError: print(f"{filename}: Cannot open or unsupported image format."); return None, None
    except Exception as e: print(f"{filename}: Error loading image: {e}"); return None, None
    return img_bgr, original_ext

def _determine_output_filename(base_filename: str, suffix_str: str, settings: argparse.Namespace, original_ext: str) -> Tuple[str, str]:
    target_ratio_str = str(settings.ratio) if settings.ratio is not None else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"; ref_str = f"_ref{settings.reference}"

    output_ext = original_ext

    out_filename = f"{base_filename}{suffix_str}{ratio_str}{ref_str}{output_ext}"
    return out_filename, output_ext

def _save_cropped_image(cropped_bgr: np.ndarray, out_path: str, settings: argparse.Namespace) -> bool:
    filename = os.path.basename(out_path)
    try:
        if cropped_bgr.size == 0: raise ValueError("Cropped image data is empty.")

        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_cropped_img = Image.fromarray(cropped_rgb)

        pil_cropped_img.save(out_path)
        return True
    except IOError as e: print(f"{filename}: File write error: {e}"); return False
    except ValueError as e: print(f"{filename}: Cropped image processing error: {e}"); return False
    except Exception as e: print(f"{filename}: Error saving cropped image: {e}"); return False

def _process_composition_rule(rule_name: str, rule_suffix: str, img_bgr: np.ndarray, ref_center: Tuple[int, int], current_settings: argparse.Namespace, original_ext: str) -> Tuple[bool, bool, str]:
    filename = os.path.basename(current_settings.input_path); base_filename = os.path.splitext(filename)[0]
    img_h, img_w = img_bgr.shape[:2]
    saved_actual_file = False; was_skipped_overwrite = False; error_message = ""

    rule_points = get_rule_points(img_w, img_h, rule_name)
    if not rule_points: error_message = f"Rule '{rule_name}': Failed to get rule points."; print(f"{filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, current_settings.target_ratio_float)
    if not crop_coords: error_message = f"Rule '{rule_name}': Failed to calculate optimal crop."; print(f"{filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    padded_coords = apply_padding(crop_coords, (img_h, img_w), current_settings.padding_percent)
    x1, y1, x2, y2 = padded_coords
    if x1 >= x2 or y1 >= y2: error_message = f"Rule '{rule_name}': Final crop area invalid after padding."; print(f"{filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    out_filename, output_ext = _determine_output_filename(base_filename, rule_suffix, current_settings, original_ext)
    out_path = os.path.join(current_settings.output_dir, out_filename)

    if not current_settings.overwrite and os.path.exists(out_path):
        error_message = f"Rule '{rule_name}': Skipped (file exists, overwrite disabled)."
        was_skipped_overwrite = True
        return saved_actual_file, was_skipped_overwrite, error_message

    cropped_img_bgr = img_bgr[y1:y2, x1:x2]
    save_success = _save_cropped_image(cropped_img_bgr, out_path, current_settings)
    if save_success: saved_actual_file = True
    else: error_message = f"Rule '{rule_name}': Failed to save file '{out_filename}'."
    return saved_actual_file, was_skipped_overwrite, error_message

def process_image(image_path: str, global_settings: argparse.Namespace, detector: cv2.FaceDetectorYN) -> Dict[str, Any]:
    filename = os.path.basename(image_path)
    start_time = time.time()

    current_settings = argparse.Namespace(**vars(global_settings))
    current_settings.input_path = image_path

    status = {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules': 0, 'message': ''}

    img_bgr, original_ext = _load_and_prepare_image(image_path)
    if img_bgr is None: status['message'] = "Failed to load or prepare image."; return status

    img_h, img_w = img_bgr.shape[:2]
    if img_h <= 0 or img_w <= 0: status['message'] = f"Invalid image dimensions ({img_w}x{img_h})."; return status

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
        saved, skipped_overwrite, error_msg = _process_composition_rule(rule_name, rule_suffix, img_bgr, ref_center, current_settings, original_ext)
        if saved: saved_rules_count += 1
        if skipped_overwrite: skipped_overwrite_count_for_image +=1
        if error_msg and not saved and not skipped_overwrite : rule_errors.append(error_msg)

    end_time = time.time(); processing_time = end_time - start_time
    status['skipped_overwrite_rules'] = skipped_overwrite_count_for_image
    status['saved_files'] = saved_rules_count

    if saved_rules_count > 0:
        status['success'] = True
        status['message'] = f"Saved {saved_rules_count} file(s)"
        if skipped_overwrite_count_for_image > 0:
            status['message'] += f", skipped {skipped_overwrite_count_for_image} rule(s) (overwrite)"
        if rule_errors:
             status['message'] += f". Errors: {'; '.join(rule_errors)}"
    elif skipped_overwrite_count_for_image > 0:
        status['success'] = False
        status['message'] = f"Skipped {skipped_overwrite_count_for_image} rule(s) (overwrite policy)"
        if rule_errors:
             status['message'] += f". Other errors: {'; '.join(rule_errors)}"
    else:
        status['success'] = False
        err_summary = '; '.join(rule_errors) if rule_errors else "Crop/Save failed or no faces/rules applicable."
        status['message'] = f"Failed. Errors: {err_summary}"

    status['message'] += f" ({processing_time:.2f}s)."
    return status



def execute_crop_operation(settings: argparse.Namespace) -> Dict[str, int]:
    if settings.yunet_model_path is None:
        settings.yunet_model_path = os.path.join(MODEL_DIR_NAME, YUNET_MODEL_FILENAME)
    settings.yunet_model_path = os.path.abspath(settings.yunet_model_path)

    settings.target_ratio_float = parse_aspect_ratio(settings.ratio)

    if not download_model(YUNET_MODEL_URL, settings.yunet_model_path, True):
        print(f"Error: DNN model file is unavailable or download failed.")
        sys.exit(2)

    if not create_output_directory(settings.output_dir):
        print(f"Error: Cannot prepare output directory.")
        sys.exit(2)

    input_path = settings.input_path
    summary = {'processed': 0, 'saved': 0, 'skipped': 0, 'failed': 0}

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"File '{input_path}' is not a supported image type. Skipping.")
            return summary 

        print(f"Processing image file: {os.path.abspath(input_path)}")
        summary['processed'] = 1
        detector = None
        try:
            detector = cv2.FaceDetectorYN.create(settings.yunet_model_path, "", (0, 0))
            detector.setScoreThreshold(settings.confidence); detector.setNMSThreshold(settings.nms)
            result = process_image(input_path, settings, detector)

            saved_count = result.get('saved_files', 0)
            skipped_count = result.get('skipped_overwrite_rules', 0)
            is_success = result.get('success', False)
            message = result.get('message', "No message provided.")

            summary['saved'] = saved_count
            summary['skipped'] = skipped_count

            if is_success and saved_count > 0:
                print(f"Processed '{input_path}': {message}")
            elif not is_success and saved_count == 0 and skipped_count > 0:
                print(f"Skipped '{input_path}': {message}")
            else:
                print(f"Failed '{input_path}': {message}")
                summary['failed'] = 1
                 
        except cv2.error as e:
            print(f"Failed to load face detection model: {e}")
            summary['failed'] = 1
        except Exception as e:
            print(f"Error during image processing: {e}")
            summary['failed'] = 1
    else:
        if not os.path.exists(input_path):
            err_msg = f"Input path not found: {os.path.abspath(input_path)}"
            print(err_msg)
            sys.exit(2)
        else:
            err_msg = f"Input path is not a file: {os.path.abspath(input_path)}"
            print(err_msg)
            sys.exit(2)

    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Image cropping using face detection (YuNet) and composition rules (thirds, golden ratio).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", help="Path to the image file to process.")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save results. Default: 'output'.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output files. Default: False.")

    parser.add_argument("-m", "--method", choices=['largest', 'center'], default='largest', help="Method to select main subject. Default: largest.")
    parser.add_argument("--ref", "--reference", dest="reference", choices=['eye', 'box'], default='box', help="Reference point for composition. Default: box.")
    parser.add_argument("-c", "--confidence", type=float, default=0.6, help="Min face detection confidence. Default: 0.6.")
    parser.add_argument("-n", "--nms", type=float, default=0.3, help="Face detection NMS threshold. Default: 0.3.")
    parser.add_argument("--min-face-width", type=int, default=30, help="Min face width in pixels. Default: 30.")
    parser.add_argument("--min-face-height", type=int, default=30, help="Min face height in pixels. Default: 30.")

    parser.add_argument("--ratio", type=str, default=None, help="Target crop aspect ratio (e.g., '16:9', '1.0', 'None'). Default: None.")
    parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], default='both', help="Composition rule to use. Default: both.")
    parser.add_argument("--padding-percent", type=float, default=5.0, help="Padding percentage around crop. Default: 5.0.")
    
    parser.add_argument("--yunet-model-path", type=str, default=None, help="Path to the YuNet ONNX model file. If not specified, it defaults to models/face_detection_yunet_2023mar.onnx and will be downloaded if missing.")

    args = parser.parse_args()

    try:
        summary = execute_crop_operation(args) 
        
        total_processed = summary.get('processed', 0)
        total_saved = summary.get('saved', 0) 
        total_skipped_overwrite = summary.get('skipped', 0) 
        total_failed = summary.get('failed', 0) 

        print(f"===== Image Cropping Script Finished =====")
        print(f"Summary: Image Processed={total_processed}, Outputs Saved={total_saved}, Outputs Skipped(Overwrite)={total_skipped_overwrite}, Failed={total_failed}")

        if total_failed > 0:
            print(f"Completed with failure.")
            sys.exit(1)
        elif total_saved > 0:
            print(f"Completed successfully (saved {total_saved} output files).")
            sys.exit(0)
        elif total_skipped_overwrite > 0:
            print(f"Completed, but no new output files were saved due to overwrite policy (skipped {total_skipped_overwrite} outputs).")
            sys.exit(0)
        else:
            print("Completed, but no output files were generated (check logs for details).")
            sys.exit(0)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
