# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
import logging
import time
import json
import concurrent.futures
from PIL import Image, UnidentifiedImageError, ImageOps # ImageOps for EXIF handling
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field # Use dataclass for configuration
import tqdm # Added tqdm for progress bar

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)


# --- Constants ---
__version__ = "1.7.8" # Version update: English comments/docstrings, static script log name

# --- Configuration Dataclass ---
@dataclass
class Config:
    """Holds all configuration settings for the script."""
    input_path: str = "input"
    output_dir: str = "output"
    config_path: Optional[str] = None
    method: str = 'largest' # ['largest', 'center']
    reference: str = 'box' # ['eye', 'box']
    ratio: Optional[str] = None # e.g., '16:9', '1.0', 'None'
    confidence: float = 0.6
    nms: float = 0.3
    overwrite: bool = True
    output_format: Optional[str] = None # e.g., 'jpg', 'png', 'webp'
    jpeg_quality: int = 95
    webp_quality: int = 80
    min_face_width: int = 30
    min_face_height: int = 30
    padding_percent: float = 5.0
    rule: str = 'both' # ['thirds', 'golden', 'both']
    verbose: bool = False
    dry_run: bool = False
    strip_exif: bool = False

    # Derived or Fixed Values (Not directly from args/config file)
    target_ratio_float: Optional[float] = None
    yunet_model_url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    yunet_model_path: str = field(default_factory=lambda: os.path.join("models", "face_detection_yunet_2023mar.onnx"))
    output_suffix_thirds: str = '_thirds'
    output_suffix_golden: str = '_golden'
    supported_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

    def __post_init__(self):
        """
        Calculate derived values and validate settings after initialization.
        This method is automatically called by dataclasses after __init__.
        """
        self.target_ratio_float = parse_aspect_ratio(self.ratio)
        if self.min_face_width < 0:
            logging.warning(f"Minimum face width must be >= 0 ({self.min_face_width}). Setting to 0.")
            self.min_face_width = 0
        if self.min_face_height < 0:
            logging.warning(f"Minimum face height must be >= 0 ({self.min_face_height}). Setting to 0.")
            self.min_face_height = 0
        if self.padding_percent < 0:
            logging.warning(f"Padding percent must be >= 0 ({self.padding_percent}). Setting to 0.")
            self.padding_percent = 0.0
        if not (1 <= self.jpeg_quality <= 100):
             logging.warning(f"JPEG quality must be between 1 and 100 ({self.jpeg_quality}). Setting to 95.")
             self.jpeg_quality = 95
        if not (1 <= self.webp_quality <= 100):
             logging.warning(f"WebP quality must be between 1 and 100 ({self.webp_quality}). Setting to 80.")
             self.webp_quality = 80

main_process_pid = os.getpid() # Store the main process PID for differentiating logs in multiprocessing

# --- Utility Functions ---

def setup_logging(level: int):
    """Sets the global logging level."""
    logger.setLevel(level)
    logging.debug(f"  -> Debug: Logging level set to {logging.getLevelName(level)}")

def download_model(url: str, file_path: str) -> bool:
    """
    Downloads the model file from the specified URL if it doesn't exist.
    Shows a progress bar during download in the main process.
    """
    model_dir = os.path.dirname(file_path)
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
            logging.info(f"  -> Info: Created model directory: {os.path.abspath(model_dir)}")
        except OSError as e:
            logging.error(f"  -> Error: Failed to create model directory '{os.path.abspath(model_dir)}': {e}")
            return False

    if not os.path.exists(file_path):
        if os.getpid() == main_process_pid: # Log download attempt only in the main process
            logging.info(f"  -> Info: Downloading model file... ({os.path.basename(file_path)}) from {url}")
        try:
            if os.getpid() == main_process_pid: # Show progress bar only in the main process
                with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=f"  Downloading {os.path.basename(file_path)}") as t:
                    def reporthook(blocknum, blocksize, totalsize):
                        if totalsize > 0: 
                            t.total = totalsize
                        t.update(blocksize)
                    urllib.request.urlretrieve(url, file_path, reporthook=reporthook)
            else: # Child processes download without tqdm to avoid multiple bars
                 urllib.request.urlretrieve(url, file_path)

            if os.getpid() == main_process_pid:
                logging.info(f"  -> Info: Download complete. Model saved to {os.path.abspath(file_path)}")
            return True
        except Exception as e:
            logging.error(f"  -> Error: Model file download failed: {e}")
            logging.error(f"       Please manually download from the following URL and save as '{os.path.abspath(file_path)}': {url}")
            return False
    else:
        if os.getpid() == main_process_pid: # Log existence check only in the main process
             logging.info(f"  -> Info: Model file '{os.path.basename(file_path)}' already exists at {os.path.abspath(file_path)}")
        return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    """
    Converts an aspect ratio string (e.g., '16:9', '1.0', 'None') to a float.
    Returns None if the ratio string is invalid or 'None'.
    """
    if ratio_str is None or str(ratio_str).strip().lower() == 'none':
        return None
    try:
        ratio_str = str(ratio_str).strip()
        if ':' in ratio_str:
            w_str, h_str = ratio_str.split(':')
            w, h = float(w_str), float(h_str)
            if h <= 0 or w <= 0: # Width and height must be positive
                logging.warning(f"  -> Warning: Ratio width or height cannot be zero or less: '{ratio_str}'. Using original ratio.")
                return None
            return w / h
        else:
            ratio = float(ratio_str)
            if ratio <= 0: # Ratio as a single number must be positive
                logging.warning(f"  -> Warning: Ratio must be greater than zero: '{ratio_str}'. Using original ratio.")
                return None
            return ratio
    except ValueError:
        logging.warning(f"  -> Warning: Invalid ratio string format: '{ratio_str}'. Using original ratio.")
        return None
    except Exception as e: # Catch any other unexpected errors
        logging.error(f"  -> Error: Unexpected error parsing ratio ('{ratio_str}'): {e}")
        return None

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Loads a JSON configuration file.
    Returns a dictionary with configuration data or an empty dictionary on failure.
    """
    abs_config_path = os.path.abspath(config_path)
    try:
        with open(abs_config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            logging.info(f"  -> Info: Configuration file loaded successfully: {abs_config_path}")
            return config_data
    except FileNotFoundError:
        logging.error(f"  -> Error: Configuration file not found: {abs_config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"  -> Error: Configuration file parsing error ({abs_config_path}): {e}")
        return {}
    except Exception as e:
        logging.error(f"  -> Error: Error loading configuration file ({abs_config_path}): {e}")
        return {}

def create_output_directory(output_dir: str, dry_run: bool) -> bool:
    """
    Creates the output directory if it doesn't exist.
    Returns True if the directory exists or was created, False otherwise.
    """
    abs_output_dir = os.path.abspath(output_dir)
    if dry_run:
        logging.info(f"  -> Info: [DRY RUN] Skipping output directory check/creation: {abs_output_dir}")
        return True
    if not os.path.exists(abs_output_dir):
        try:
            os.makedirs(abs_output_dir)
            logging.info(f"  -> Info: Created output directory: {abs_output_dir}")
            return True
        except OSError as e:
            logging.critical(f"  -> Critical: Failed to create output directory '{abs_output_dir}': {e}")
            return False
    elif not os.path.isdir(abs_output_dir): # Path exists but is not a directory
         logging.critical(f"  -> Critical: The specified output path '{abs_output_dir}' is not a directory.")
         return False
    logging.info(f"  -> Info: Output directory already exists: {abs_output_dir}")
    return True

# --- Core Logic Functions ---

def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray, min_w: int, min_h: int) -> List[Dict[str, Any]]:
    """
    Detects faces using the preloaded DNN model (YuNet).
    Filters faces smaller than the specified minimum size (min_w, min_h).
    Returns a list of dictionaries, each containing face info:
    'bbox': (x, y, w, h), 'bbox_center': (cx, cy), 'eye_center': (ex, ey), 'confidence': score.
    """
    detected_subjects = []
    if image is None or image.size == 0:
        logging.warning("  -> Warning: Input image for face detection is empty.")
        return []
    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0: # Validate image dimensions
        logging.warning(f"  -> Warning: Invalid image size ({img_w}x{img_h}), skipping face detection.")
        return []
    try:
        detector.setInputSize((img_w, img_h)) # Set input size for the detector
        faces = detector.detect(image) # Perform face detection

        if faces is not None and faces[1] is not None: # faces[1] contains the detected face info array
            for idx, face_info in enumerate(faces[1]):
                x, y, w, h = map(int, face_info[:4]) # Bounding box
                if w < min_w or h < min_h: # Filter by minimum face size
                    logging.debug(f"  -> Debug: Face ID {idx} ({w}x{h}) ignored as smaller than min size ({min_w}x{min_h}).")
                    continue
                
                # Landmarks (eyes) and confidence
                r_eye_x, r_eye_y = face_info[4:6]; l_eye_x, l_eye_y = face_info[6:8]
                confidence = face_info[14] # Confidence score

                # Ensure bounding box is within image boundaries
                x = max(0, x); y = max(0, y)
                w = min(img_w - x, w); h = min(img_h - y, h)

                if w > 0 and h > 0: # Proceed only if the adjusted bounding box is valid
                    bbox_center = (x + w // 2, y + h // 2)
                    eye_center = bbox_center # Default to bbox center

                    if r_eye_x > 0 and r_eye_y > 0 and l_eye_x > 0 and l_eye_y > 0: # Calculate eye center if landmarks are valid
                        ecx = int(round((r_eye_x + l_eye_x) / 2)); ecy = int(round((r_eye_y + l_eye_y) / 2))
                        ecx = max(0, min(img_w - 1, ecx)); ecy = max(0, min(img_h - 1, ecy)) # Clamp to image bounds
                        eye_center = (ecx, ecy)
                    else:
                        logging.debug(f"  -> Debug: Eye landmarks for face ID {idx} invalid, using BBox center.")
                    
                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e: # OpenCV specific errors
        logging.error(f"  -> Error: OpenCV error during face detection (image size: {img_w}x{img_h}): {e}")
    except Exception as e: # Other unexpected errors
        logging.error(f"  -> Error: Unexpected problem during DNN face detection: {e}", exc_info=logger.level == logging.DEBUG)
    return detected_subjects

def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'box') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """
    Selects the main subject from the list of detected subjects.
    Returns the selected subject's bounding box and reference point (eye center or bbox center),
    or None if no subject can be selected.
    """
    if not subjects: logging.debug("  -> Debug: Select main subject: No subjects detected."); return None
    img_h, img_w = img_shape; best_subject = None
    try:
        if len(subjects) == 1: best_subject = subjects[0]
        elif method == 'largest': best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3]) # Largest area
        elif method == 'center': # Closest to image center
            img_center = (img_w / 2, img_h / 2)
            best_subject = min(subjects, key=lambda s: math.dist(s['bbox_center'], img_center))
        else: # Default to 'largest' if method is unknown
            logging.warning(f"  -> Warning: Unknown selection method '{method}'. Defaulting to 'largest'.")
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
        
        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        logging.debug(f"  -> Debug: Main subject selected (Method: {method}, Reference: {reference_point_type}). BBox: {best_subject['bbox']}, Ref Point: {ref_center}")
        return best_subject['bbox'], ref_center
    except Exception as e:
        logging.error(f"  -> Error: Error selecting main subject: {e}", exc_info=logger.level == logging.DEBUG)
        return None

def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """
    Calculates the intersection points based on the specified composition rule ('thirds' or 'golden').
    Returns a list of (x, y) tuples representing the intersection points.
    """
    points = []
    if width <= 0 or height <= 0: logging.warning(f"  -> Warning: Cannot calculate rule points for invalid size ({width}x{height})."); return []
    try:
        if rule_type == 'thirds': # Rule of Thirds: 4 inner intersection points
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden': # Golden Ratio points
            phi_inv = (math.sqrt(5) - 1) / 2 # Approx 0.618
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            points = [(w, h) for w in lines_w for h in lines_h]
        else: # Default or unknown rule: use image center
            logging.warning(f"  -> Warning: Unknown composition rule '{rule_type}'. Using image center.")
            points = [(width / 2, height / 2)]
        return [(int(round(px)), int(round(py))) for px, py in points] # Round to integer coordinates
    except Exception as e:
        logging.error(f"  -> Error: Error calculating rule points (Rule: {rule_type}, Size: {width}x{height}): {e}", exc_info=logger.level == logging.DEBUG)
        return []

def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculates the optimal crop area (x1, y1, x2, y2) to place the subject_center
    closest to one of the rule_points, while maintaining the target_aspect_ratio.
    Returns crop coordinates (top-left x, top-left y, bottom-right x, bottom-right y) or None.
    """
    height, width = img_shape
    if height <= 0 or width <= 0: logging.warning(f"  -> Warning: Cannot crop image with zero/negative dimensions ({width}x{height})."); return None
    if not rule_points: logging.warning("  -> Warning: No rule points provided, skipping crop calculation."); return None
    
    cx, cy = subject_center # Subject's reference point
    try:
        # Determine aspect ratio: target if provided, else original image's
        aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        if aspect_ratio <= 0: logging.warning(f"  -> Warning: Invalid target aspect ratio ({aspect_ratio}). Cannot calculate crop."); return None

        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p)) # Find rule point closest to subject
        target_x, target_y = closest_point # This rule point will be the center of the crop

        # Calculate max possible crop dimensions centered at target_x, target_y within image bounds
        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)
        if max_w <= 0 or max_h <= 0: logging.debug(f"  -> Debug: Target rule point ({target_x},{target_y}) too close to boundary."); return None

        # Determine final crop dimensions based on aspect ratio and max dimensions
        crop_h_from_w = max_w / aspect_ratio
        crop_w_from_h = max_h * aspect_ratio

        if crop_h_from_w <= max_h + 1e-6: # Add epsilon for float comparison
            final_w, final_h = max_w, crop_h_from_w # Width is limiting
        else:
            final_w, final_h = crop_w_from_h, max_h # Height is limiting

        # Calculate crop coordinates (top-left x1,y1 and bottom-right x2,y2)
        x1 = target_x - final_w / 2; y1 = target_y - final_h / 2
        x2 = x1 + final_w; y2 = y1 + final_h

        # Ensure coordinates are within image boundaries and convert to integers
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        if x1 >= x2 or y1 >= y2: # Validate final crop size
            logging.warning(f"  -> Warning: Calculated crop area has zero/negative size ({x1},{y1} to {x2},{y2})."); return None
        
        logging.debug(f"  -> Debug: Optimal crop: ({x1}, {y1}) - ({x2}, {y2}) for target ({target_x},{target_y})")
        return x1, y1, x2, y2
    except Exception as e:
        logging.error(f"  -> Error: Error calculating optimal crop (Subject: {subject_center}, RulePt: {closest_point if 'closest_point' in locals() else 'N/A'}): {e}", exc_info=logger.level == logging.DEBUG)
        return None

def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    """
    Applies padding around the calculated crop area, expanding it outwards.
    Adjusts coordinates to stay within original image boundaries.
    """
    x1, y1, x2, y2 = crop_coords; img_h, img_w = img_shape
    crop_w = x2 - x1; crop_h = y2 - y1
    if padding_percent <= 0: return crop_coords # No padding if percent is zero or negative

    # Calculate padding amount for width and height (applied equally to both sides)
    pad_x = int(round(crop_w * padding_percent / 100 / 2))
    pad_y = int(round(crop_h * padding_percent / 100 / 2))

    # Apply padding
    new_x1 = max(0, x1 - pad_x); new_y1 = max(0, y1 - pad_y) # Clamp to image boundaries
    new_x2 = min(img_w, x2 + pad_x); new_y2 = min(img_h, y2 + pad_y)

    if new_x1 >= new_x2 or new_y1 >= new_y2: # If padding results in invalid area
        logging.warning(f"  -> Warning: Crop area invalid after {padding_percent}% padding. No padding applied.")
        return crop_coords # Return original coordinates
    
    logging.debug(f"  -> Debug: Applied {padding_percent}% padding. New crop: ({new_x1}, {new_y1}) - ({new_x2}, {new_y2})")
    return new_x1, new_y1, new_x2, new_y2

# --- Image Processing Sub-Functions ---

def _load_and_prepare_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[bytes], Optional[str]]:
    """Loads image using Pillow, extracts EXIF, converts to OpenCV BGR format."""
    filename = os.path.basename(image_path); exif_data = None; original_ext = os.path.splitext(image_path)[1].lower(); img_bgr = None
    try:
        with Image.open(image_path) as pil_img:
            try: # Preserve orientation using EXIF data
                pil_img = ImageOps.exif_transpose(pil_img)
                exif_data = pil_img.info.get('exif') # Extract raw EXIF
                if exif_data: logging.debug(f"  -> Debug: {filename}: EXIF data found and preserved.")
            except Exception as exif_err:
                logging.warning(f"  -> Warning: {filename}: Error processing EXIF data: {exif_err}. Proceeding without EXIF.")
                exif_data = None
            
            pil_img_rgb = pil_img.convert('RGB') # Convert to RGB for consistency
            img_bgr = np.array(pil_img_rgb)[:, :, ::-1].copy() # Pillow RGB to OpenCV BGR
    except FileNotFoundError: logging.error(f"  -> Error: {filename}: Image file not found."); return None, None, None
    except UnidentifiedImageError: logging.error(f"  -> Error: {filename}: Cannot open or unsupported image format."); return None, None, None
    except Exception as e: logging.error(f"  -> Error: {filename}: Error loading image: {e}", exc_info=logger.level == logging.DEBUG); return None, None, None
    return img_bgr, exif_data, original_ext

def _determine_output_filename(base_filename: str, suffix: str, config: Config, original_ext: str) -> Tuple[str, str]:
    """Determines the output filename and extension based on configuration."""
    target_ratio_str = str(config.ratio) if config.ratio is not None else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"; ref_str = f"_ref{config.reference}" # Add ratio and reference info
    
    output_format = config.output_format.lower() if config.output_format else None
    if output_format: # User specified an output format
        output_ext = f".{output_format.lstrip('.')}"
        if output_ext.lower() not in Image.registered_extensions(): # Validate if Pillow supports it
             logging.warning(f"  -> Warning: {base_filename}: Unsupported output format '{config.output_format}'. Using original '{original_ext}'.")
             output_ext = original_ext
    else: # Keep original format
        output_ext = original_ext
    
    out_filename = f"{base_filename}{suffix}{ratio_str}{ref_str}{output_ext}"
    return out_filename, output_ext

def _save_cropped_image(cropped_bgr: np.ndarray, out_path: str, output_ext: str, config: Config, exif_data: Optional[bytes]) -> bool:
    """Saves the cropped image (BGR format) to the specified path using Pillow."""
    filename = os.path.basename(out_path)
    try:
        if cropped_bgr.size == 0: raise ValueError("Cropped image data is empty.")
        
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for Pillow
        pil_cropped_img = Image.fromarray(cropped_rgb)
        
        save_options = {}
        # Include EXIF if not stripping, and EXIF data exists and is valid
        if not config.strip_exif and exif_data and isinstance(exif_data, bytes):
            save_options['exif'] = exif_data
            logging.debug(f"  -> Debug: {filename}: Saving with EXIF data.")
        elif config.strip_exif: logging.debug(f"  -> Debug: {filename}: Stripping EXIF data.")
        elif exif_data: logging.debug(f"  -> Debug: {filename}: EXIF found but not saving (strip:{config.strip_exif}, type:{type(exif_data)}).")

        # Format-specific save options
        if output_ext.lower() in ['.jpg', '.jpeg']:
            save_options['quality'] = config.jpeg_quality
            save_options['optimize'] = True
            save_options['progressive'] = True
        elif output_ext.lower() == '.webp':
            save_options['quality'] = config.webp_quality
            logging.debug(f"  -> Debug: {filename}: Saving as WebP with quality {config.webp_quality}.")

        pil_cropped_img.save(out_path, **save_options)
        logging.debug(f"  -> Debug: {filename}: Saved successfully to {out_path}")
        return True
    except IOError as e: logging.error(f"  -> Error: {filename}: File write error: {e}"); return False
    except ValueError as e: logging.error(f"  -> Error: {filename}: Cropped image processing error: {e}"); return False
    except Exception as e: logging.error(f"  -> Error: {filename}: Error saving cropped image: {e}", exc_info=config.verbose); return False

def _process_composition_rule(rule_name: str, suffix: str, img_bgr: np.ndarray, ref_center: Tuple[int, int], config: Config, exif_data: Optional[bytes], original_ext: str) -> Tuple[bool, bool, str]:
    """
    Processes a single composition rule (e.g., thirds, golden) for an image.
    Returns:
        saved_actual_file (bool): True if a file was saved or simulated.
        was_skipped_overwrite (bool): True if saving was skipped due to overwrite policy.
        error_message (str): Error message if an error occurred, otherwise empty.
    """
    filename = os.path.basename(config.input_path); base_filename = os.path.splitext(filename)[0]
    img_h, img_w = img_bgr.shape[:2]
    saved_actual_file = False; was_skipped_overwrite = False; error_message = ""

    rule_points = get_rule_points(img_w, img_h, rule_name)
    if not rule_points: error_message = f"Rule '{rule_name}': Failed to get rule points."; logging.warning(f"  -> Warning: {filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, config.target_ratio_float)
    if not crop_coords: error_message = f"Rule '{rule_name}': Failed to calculate optimal crop."; logging.warning(f"  -> Warning: {filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    padded_coords = apply_padding(crop_coords, (img_h, img_w), config.padding_percent)
    x1, y1, x2, y2 = padded_coords
    if x1 >= x2 or y1 >= y2: error_message = f"Rule '{rule_name}': Final crop area invalid after padding."; logging.warning(f"  -> Warning: {filename}: {error_message}"); return saved_actual_file, was_skipped_overwrite, error_message

    out_filename, output_ext = _determine_output_filename(base_filename, suffix, config, original_ext)
    out_path = os.path.join(config.output_dir, out_filename)

    # Check overwrite condition
    if not config.overwrite and os.path.exists(out_path) and not config.dry_run:
        logging.debug(f"  -> Debug: {filename}: File exists, overwrite disabled - skipping: {out_filename}") 
        error_message = f"Rule '{rule_name}': Skipped (file exists, overwrite disabled)."
        was_skipped_overwrite = True
        return saved_actual_file, was_skipped_overwrite, error_message # Not saved, but skipped

    # Perform Dry Run or Actual Save
    if config.dry_run:
        logging.debug(f"  -> Debug: [DRY RUN] {filename}: Would save '{out_filename}' (Rule: {rule_name}, Area: {x1},{y1}-{x2},{y2})")
        saved_actual_file = True # Mark as "saved" for dry run purposes
    else:
        cropped_img_bgr = img_bgr[y1:y2, x1:x2] # Crop the image
        save_success = _save_cropped_image(cropped_img_bgr, out_path, output_ext, config, exif_data)
        if save_success: saved_actual_file = True
        else: error_message = f"Rule '{rule_name}': Failed to save file '{out_filename}'." # Specific error logged in _save_cropped_image
    return saved_actual_file, was_skipped_overwrite, error_message

# --- Main Processing Function ---

def process_image(image_path: str, config: Config, detector: cv2.FaceDetectorYN) -> Dict[str, Any]:
    """
    Processes a single image: loads, detects faces, selects subject,
    calculates crops based on rules, and saves results or logs actions in Dry Run mode.
    Returns a dictionary with processing status.
    """
    filename = os.path.basename(image_path)
    logging.debug(f"  -> Debug: Processing started: {filename}")
    start_time = time.time()
    status = {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules': 0, 'message': '', 'dry_run': config.dry_run}

    img_bgr, exif_data, original_ext = _load_and_prepare_image(image_path)
    if img_bgr is None: status['message'] = "Failed to load or prepare image."; return status # Error logged in helper

    img_h, img_w = img_bgr.shape[:2]
    if img_h <= 0 or img_w <= 0: status['message'] = f"Invalid image dimensions ({img_w}x{img_h})."; logging.warning(f"  -> Warning: {filename}: {status['message']}"); return status

    # Create a config instance specific to this image path for logging within _process_composition_rule
    current_config = Config(**{**config.__dict__, 'input_path': image_path}) 
    
    detected_faces = detect_faces_dnn(detector, img_bgr, current_config.min_face_width, current_config.min_face_height)
    if not detected_faces: status['message'] = "No valid faces detected (considering minimum size)."; logging.debug(f"  -> Debug: {filename}: {status['message']}"); return status

    selection_result = select_main_subject(detected_faces, (img_h, img_w), current_config.method, current_config.reference)
    if not selection_result: status['message'] = "Failed to select main subject."; logging.debug(f"  -> Debug: {filename}: {status['message']}"); return status
    subj_bbox, ref_center = selection_result

    # Determine which composition rules to apply
    rules_to_process = []
    if current_config.rule in ['thirds', 'both']: rules_to_process.append(('thirds', current_config.output_suffix_thirds))
    if current_config.rule in ['golden', 'both']: rules_to_process.append(('golden', current_config.output_suffix_golden))
    if not rules_to_process: status['message'] = f"No composition rule selected ('{current_config.rule}')."; logging.debug(f"  -> Debug: {filename}: {status['message']}"); return status

    saved_rules_count = 0; skipped_overwrite_count_for_image = 0; rule_errors = []
    for rule_name, suffix in rules_to_process:
        saved, skipped_overwrite, error_msg = _process_composition_rule(rule_name, suffix, img_bgr, ref_center, current_config, exif_data, original_ext)
        if saved: saved_rules_count += 1
        if skipped_overwrite: skipped_overwrite_count_for_image +=1
        # Only count as an error if it wasn't saved AND wasn't skipped due to overwrite
        if error_msg and not saved and not skipped_overwrite : rule_errors.append(error_msg) 

    end_time = time.time(); processing_time = end_time - start_time
    status['skipped_overwrite_rules'] = skipped_overwrite_count_for_image

    if saved_rules_count > 0: # At least one rule resulted in a saved/simulated file
        status['success'] = True; status['saved_files'] = saved_rules_count
        action = "simulated" if current_config.dry_run else "saved"
        status['message'] = f"Processed ({saved_rules_count} file(s) {action}, {skipped_overwrite_count_for_image} rule(s) skipped by overwrite, {processing_time:.2f}s)."
        if rule_errors: status['message'] += f" Some rules failed: {'; '.join(rule_errors)}"
        logging.debug(f"  -> Debug: {filename}: {status['message']}") # Log individual file success as debug
    elif skipped_overwrite_count_for_image > 0 and not rule_errors : # All rules skipped, no other errors
        status['success'] = False # Not a full success as no files were generated/simulated
        status['message'] = f"All applicable rules skipped due to overwrite policy ({processing_time:.2f}s)."
        logging.debug(f"  -> Debug: {filename}: {status['message']}")
    else: # No files saved/simulated, and it wasn't (only) due to overwrite skips
        status['success'] = False
        err_summary = '; '.join(rule_errors) if rule_errors else "Crop/Save failed or no faces/rules applicable."
        status['message'] = f"Failed to {'simulate' if current_config.dry_run else 'crop/save'} for all rules ({processing_time:.2f}s). Errors: {err_summary}"
        logging.debug(f"  -> Debug: {filename}: {status['message']}") # Log individual file failure as debug
    return status

# --- Parallel Processing Wrapper ---
def process_image_wrapper(args_tuple: Tuple[str, Config]) -> Dict[str, Any]:
    """
    Wrapper function for concurrent.futures.ProcessPoolExecutor.
    Loads the model and calls process_image for a single image path.
    """
    image_path, config = args_tuple; filename = os.path.basename(image_path); detector = None
    try:
        # Logging within worker processes should be minimal or debug level to avoid clutter
        logging.debug(f"  -> Debug: Worker for {filename}: Loading model ({os.path.basename(config.yunet_model_path)})...")
        if not os.path.exists(config.yunet_model_path): # Ensure model exists
             if not download_model(config.yunet_model_url, config.yunet_model_path): # Child process download
                  raise RuntimeError(f"Model file {config.yunet_model_path} not found and download failed in worker.")
        
        detector = cv2.FaceDetectorYN.create(config.yunet_model_path, "", (0, 0)) # "" for backend, (0,0) for auto input size
        detector.setScoreThreshold(config.confidence); detector.setNMSThreshold(config.nms)
        logging.debug(f"  -> Debug: Worker for {filename}: Model loading complete.")
        return process_image(image_path, config, detector) # Call the main processing function

    except cv2.error as e: # Catch OpenCV errors during model load or detection
        logging.error(f"  -> Error: {filename}: OpenCV error in worker: {e}", exc_info=config.verbose)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules':0, 'message': f"OpenCV Error: {e}", 'dry_run': config.dry_run}
    except Exception as e: # Catch any other critical errors in the worker
        logging.error(f"  -> Error: {filename}: Critical error in worker: {e}", exc_info=True) # Always log traceback for critical worker errors
        return {'filename': filename, 'success': False, 'saved_files': 0, 'skipped_overwrite_rules':0, 'message': f"Critical error in worker: {e}", 'dry_run': config.dry_run}

# --- Command-Line Interface Setup ---

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=f"{os.path.basename(__file__)} - Automatically crop images based on face detection and composition rules.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", help="Path to the image file or directory to process.")
    parser.add_argument("-o", "--output_dir", help=f"Directory to save results (Default: '{Config.output_dir}').")
    parser.add_argument("--config", help="Path to a JSON configuration file to load options from.")

    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=Config.overwrite, help="Allow overwriting output files if they exist.")
    parser.add_argument("--dry-run", action="store_true", default=Config.dry_run, help="Simulate processing without saving files.")

    parser.add_argument("-m", "--method", choices=['largest', 'center'], help=f"Method to select main subject (Default: {Config.method}).")
    parser.add_argument("--ref", "--reference", dest="reference", choices=['eye', 'box'], help=f"Reference point for composition (Default: {Config.reference}).")
    parser.add_argument("-c", "--confidence", type=float, help=f"Min face detection confidence (Default: {Config.confidence}).")
    parser.add_argument("-n", "--nms", type=float, help=f"Face detection NMS threshold (Default: {Config.nms}).")
    parser.add_argument("--min-face-width", type=int, help=f"Min face width in pixels (Default: {Config.min_face_width}).")
    parser.add_argument("--min-face-height", type=int, help=f"Min face height in pixels (Default: {Config.min_face_height}).")

    parser.add_argument("-r", "--ratio", type=str, help="Target crop aspect ratio (e.g., '16:9', '1.0', 'None') (Default: None).")
    parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], help=f"Composition rule(s) (Default: {Config.rule}).")
    parser.add_argument("-p", "--padding-percent", type=float, help=f"Padding percentage around crop (%) (Default: {Config.padding_percent}).")

    parser.add_argument("--output-format", type=str, help="Output image format (e.g., 'jpg', 'png', 'webp'). Default: original.")
    parser.add_argument("-q", "--jpeg-quality", type=int, choices=range(1, 101), metavar="[1-100]", help=f"JPEG quality (1-100) (Default: {Config.jpeg_quality}).")
    parser.add_argument("--webp-quality", type=int, choices=range(1, 101), metavar="[1-100]", help=f"WebP quality (1-100) (Default: {Config.webp_quality}).")
    parser.add_argument("--strip-exif", action="store_true", default=Config.strip_exif, help="Remove EXIF data from output images.")

    parser.add_argument("-v", "--verbose", action="store_true", default=Config.verbose, help="Enable detailed (DEBUG level) logging.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    return parser

def load_and_merge_config(args: argparse.Namespace) -> Config:
    """Loads default config, then merges JSON config file, and finally merges command-line args."""
    default_config_obj = Config()
    config_values = default_config_obj.__dict__.copy() # Start with dataclass defaults
    config_values['input_path'] = args.input_path # Positional argument always from args

    # Load from JSON config file if specified
    if args.config:
        json_config_data = load_config_from_file(args.config)
        for key, value in json_config_data.items():
            if key == 'input_path': logging.warning("  -> Warning: 'input_path' in JSON config ignored; use command line argument instead."); continue
            if key in config_values: config_values[key] = value
            else: logging.warning(f"  -> Warning: Unknown key '{key}' in configuration file '{args.config}' ignored.")

    # Override with command-line arguments (only if they were actually provided by the user)
    parser_instance = setup_arg_parser() # Need a temporary parser instance to check defaults
    for key, value in vars(args).items():
        if key == 'input_path' or key == 'config': continue # Already handled
        
        parser_default = parser_instance.get_default(key)
        # Override if arg was explicitly provided (not None) AND is different from parser's default
        # This ensures CLI args take precedence over JSON/Config defaults if specified.
        if value is not None and value != parser_default:
             if key in config_values: config_values[key] = value
    
    # For BooleanOptionalAction and store_true, check if they differ from Config defaults
    # as their value in 'args' will always be True/False, not None if not specified.
    if args.overwrite != default_config_obj.overwrite: config_values['overwrite'] = args.overwrite
    if args.dry_run != default_config_obj.dry_run: config_values['dry_run'] = args.dry_run
    if args.verbose != default_config_obj.verbose: config_values['verbose'] = args.verbose
    if args.strip_exif != default_config_obj.strip_exif: config_values['strip_exif'] = args.strip_exif
    
    try: final_config = Config(**config_values) # Create final Config object
    except TypeError as e: logging.critical(f"  -> Critical: Configuration error when creating Config object: {e}. Check config file/arguments."); exit(1)
    return final_config

def log_script_settings(config: Config):
    """Logs the script settings in a formatted block."""
    logging.info("=" * 60)
    logging.info(f"{'Script Settings':^60}")
    logging.info("=" * 60)
    logging.info(f"  Input Path: {os.path.abspath(config.input_path)}")
    logging.info(f"  Output Directory: {os.path.abspath(config.output_dir)}")
    logging.info(f"  Face Selection Method: {config.method.capitalize()}")
    logging.info(f"  Reference Point: {config.reference.capitalize()}")
    logging.info(f"  Target Aspect Ratio: {config.ratio if config.ratio else 'Original'}")
    logging.info(f"  Composition Rule: {config.rule.capitalize()}")
    logging.info(f"  Padding: {config.padding_percent}%")
    logging.info(f"  Output Format: {config.output_format.upper() if config.output_format else 'Keep Original'}")
    if config.output_format and config.output_format.lower() in ['jpg', 'jpeg']:
        logging.info(f"  JPEG Quality: {config.jpeg_quality}")
    if config.output_format and config.output_format.lower() == 'webp':
        logging.info(f"  WebP Quality: {config.webp_quality}")
    logging.info(f"  Strip EXIF Data: {'Yes' if config.strip_exif else 'No'}")
    logging.info(f"  Overwrite Existing Files: {'Yes' if config.overwrite else 'No (Rename/Skip)'}") 
    detected_cpus = os.cpu_count()
    logging.info(f"  Parallel Workers: Using all available CPU cores (Detected: {detected_cpus if detected_cpus is not None else 'N/A, defaulting to 1'})")
    logging.info(f"  Dry Run (Simulation): {'Yes' if config.dry_run else 'No'}")
    logging.info(f"  Log Level: {logging.getLevelName(logger.getEffectiveLevel())}")
    logging.info(f"  Processed Extensions: {', '.join(config.supported_extensions)}")
    logging.info("=" * 60)

def log_processing_summary(total_images_to_process: int, successful_images: int, total_files_generated: int,
                           images_with_errors: int, total_skipped_overwrite: int,
                           skipped_scan_items: List[str], output_dir: str,
                           total_processing_time: float, dry_run: bool):
    """Logs the processing summary."""
    action_verb = "simulated" if dry_run else "generated"
    logging.info("-" * 40)
    logging.info(f"{'Processing Summary':^40}")
    logging.info("-" * 40)
    logging.info(f"  Total images scanned for processing: {total_images_to_process}")
    logging.info(f"  Images processed successfully: {successful_images}") # Images for which at least one crop was saved/simulated
    logging.info(f"  Total cropped files {action_verb}: {total_files_generated}") # Total number of output files
    logging.info(f"  Images with errors (no files {action_verb}): {images_with_errors}") # Images that failed entirely
    logging.info(f"  Output files skipped (overwrite policy): {total_skipped_overwrite}") # Individual rule outputs skipped

    if skipped_scan_items:
        logging.info(f"\n  Items skipped during initial scan ({len(skipped_scan_items)} total):")
        for i, item_info in enumerate(skipped_scan_items):
            if i < 10: logging.info(f"    - {item_info}")
            elif i == 10: logging.info("    - ... (and more)"); break
    else:
        logging.info("  No items skipped during initial scan.")
    
    logging.info(f"\n--- All tasks completed ---")
    if not dry_run:
        logging.info(f"Results saved in: '{os.path.abspath(output_dir)}'")
    else:
        logging.info(f"[DRY RUN] No files were saved. Results would be in: '{os.path.abspath(output_dir)}'")
    logging.info(f"Total processing time: {total_processing_time:.2f} seconds")


# --- Main Execution ---

def main():
    """Main execution function of the script."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    config = load_and_merge_config(args)

    setup_logging(logging.DEBUG if config.verbose else logging.INFO) # Set logging level based on config

    logging.info(f"===== Image Cropping Script Started =====")
    if config.dry_run:
        logging.info("***** Running in Dry Run mode. No files will be saved. *****")

    log_script_settings(config) # Log the effective settings

    # Preparation: Download model and create output directory
    if not download_model(config.yunet_model_url, config.yunet_model_path):
        logging.critical(f"  -> Critical: DNN model file is not available or download failed. Aborting.")
        logging.info(f"===== Image Cropping Script Finished Abnormally =====")
        return

    if not create_output_directory(config.output_dir, config.dry_run):
        logging.critical(f"  -> Critical: Could not prepare output directory. Aborting.")
        logging.info(f"===== Image Cropping Script Finished Abnormally =====")
        return

    input_path = config.input_path
    skipped_scan_items_details = [] # Store details of items skipped during initial scan

    if os.path.isfile(input_path):
        # Process a single file
        if not input_path.lower().endswith(config.supported_extensions):
            logging.warning(f"  -> Warning: Single file '{input_path}' is not a supported image type. Skipping.")
            logging.info(f"===== Image Cropping Script Finished =====")
            return

        logging.info(f"  -> Info: Starting single file processing: {os.path.abspath(input_path)}")
        try:
            detector = cv2.FaceDetectorYN.create(config.yunet_model_path, "", (0, 0))
            detector.setScoreThreshold(config.confidence); detector.setNMSThreshold(config.nms)
            result = process_image(input_path, config, detector)
            
            # Log single file result more explicitly
            if result.get('success'):
                logging.info(f"  -> Success: {os.path.basename(input_path)}: {result.get('message', 'Processing finished.')}")
            else:
                # Check if it was only due to overwrite skips or actual failure
                if result.get('skipped_overwrite_rules', 0) > 0 and not result.get('saved_files',0) and "failed" not in result.get('message','').lower() :
                     logging.info(f"  -> Info: {os.path.basename(input_path)}: {result.get('message', 'All rules skipped by overwrite policy.')}")
                else: # Actual failure or no faces/rules
                     logging.warning(f"  -> Warning: {os.path.basename(input_path)}: {result.get('message', 'Processing failed or no faces/rules applicable.')}")

        except cv2.error as e: logging.critical(f"  -> Critical: Failed to load face detection model (single file): {e}")
        except Exception as e: logging.critical(f"  -> Critical: Error during single file processing: {e}", exc_info=True)

    elif os.path.isdir(input_path):
        # Process a directory
        logging.info(f"  -> Info: Starting directory processing: {os.path.abspath(input_path)}")
        try: # Scan input directory
            all_items = os.listdir(input_path)
            image_files = []
            for item_name in all_items:
                item_path = os.path.join(input_path, item_name)
                if os.path.isfile(item_path):
                    if item_name.lower().endswith(config.supported_extensions):
                        image_files.append(item_path)
                    else:
                        skipped_scan_items_details.append(f"{item_name} (Unsupported extension)")
                elif os.path.isdir(item_path): # Directories are skipped in non-recursive scan
                    skipped_scan_items_details.append(f"{item_name} (Directory)")
                else: # Other non-file, non-directory items
                    skipped_scan_items_details.append(f"{item_name} (Not a file or directory)")

        except OSError as e:
            logging.critical(f"  -> Critical: Cannot access input directory '{os.path.abspath(input_path)}': {e}")
            logging.info(f"===== Image Cropping Script Finished Abnormally =====")
            return

        if not image_files: # No processable images found
            logging.info(f"  -> Info: No supported image files ({', '.join(config.supported_extensions)}) found in '{os.path.abspath(input_path)}'.")
            if skipped_scan_items_details:
                 logging.info(f"  -> Info: {len(skipped_scan_items_details)} other item(s) were found and skipped during scan.")
            log_processing_summary(0, 0, 0, 0, 0, skipped_scan_items_details, config.output_dir, 0, config.dry_run) 
            logging.info(f"===== Image Cropping Script Finished =====")
            return
        
        logging.info(f"  -> Info: Scan complete: Found {len(image_files)} image files to process.")
        if skipped_scan_items_details:
            logging.info(f"  -> Info: Skipped {len(skipped_scan_items_details)} other item(s) found during scan (see summary).")

        # Determine number of workers for parallel processing
        available_cpus = os.cpu_count()
        if available_cpus is None: # Fallback if cpu_count is None
            logging.warning("  -> Warning: Could not determine number of CPU cores. Defaulting to 1 worker.")
            available_cpus = 1
        elif available_cpus == 0: # Should not happen, but defensive
            logging.warning("  -> Warning: os.cpu_count() returned 0. Defaulting to 1 worker.")
            available_cpus = 1
        
        # Use all available CPUs, but not more than the number of files. Ensure at least 1.
        actual_workers = min(available_cpus, len(image_files))
        actual_workers = max(1, actual_workers) 
        is_parallel = actual_workers > 1


        logging.info(f"  -> Info: Using {actual_workers} worker process(es) for image processing (Mode: {'Parallel' if is_parallel else 'Sequential'}).")
        total_start_time = time.time()
        results_list = [] 
        
        tasks = [(img_path, config) for img_path in image_files] # Prepare tasks for workers

        if is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                try:
                    future_results = executor.map(process_image_wrapper, tasks)
                    # Configure tqdm for cleaner output when not verbose
                    tqdm_extra_kwargs = {}
                    if not config.verbose: # Less cluttered progress bar if not verbose
                        tqdm_extra_kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                        tqdm_extra_kwargs['ncols'] = 80 # Adjust width as needed

                    for result in tqdm.tqdm(future_results, total=len(tasks), desc="Processing images", unit="file", **tqdm_extra_kwargs):
                        results_list.append(result)
                except Exception as e: # Catch errors during parallel execution
                     logging.critical(f"  -> Critical: Error during parallel processing execution: {e}", exc_info=True)
        else: # Sequential processing
            try:
                detector = cv2.FaceDetectorYN.create(config.yunet_model_path, "", (0, 0)) # Load detector once
                detector.setScoreThreshold(config.confidence); detector.setNMSThreshold(config.nms)
                tqdm_extra_kwargs = {}
                if not config.verbose:
                    tqdm_extra_kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                    tqdm_extra_kwargs['ncols'] = 80

                for task_args in tqdm.tqdm(tasks, desc="Processing images sequentially", unit="file", **tqdm_extra_kwargs):
                     results_list.append(process_image(task_args[0], task_args[1], detector))
            except cv2.error as e: logging.critical(f"  -> Critical: Failed to load face detection model (sequential): {e}"); return
            except Exception as e: logging.critical(f"  -> Critical: Error during sequential processing loop: {e}", exc_info=True)

        # --- Aggregate and Summarize Results from directory processing ---
        successful_image_count = 0; total_generated_count = 0; images_with_errors_count = 0; total_skipped_overwrite_count = 0
        
        for res in results_list: # Iterate through results from workers/sequential processing
            if res and isinstance(res, dict): # Check if result is valid
                if res.get('success', False): # Image was successfully processed (at least one rule)
                    successful_image_count += 1
                # An image is an "error" if it didn't succeed AND it wasn't *only* skipped due to overwrite policy
                elif not res.get('success', False) and \
                     not (res.get('skipped_overwrite_rules', 0) > 0 and \
                          not res.get('saved_files', 0) and \
                          "failed" not in res.get('message','').lower()):
                    images_with_errors_count +=1

                total_generated_count += res.get('saved_files', 0) # Sum of all files saved/simulated
                total_skipped_overwrite_count += res.get('skipped_overwrite_rules', 0) # Sum of all rules skipped by overwrite
            else: # Should not happen if workers return correctly
                 images_with_errors_count +=1 # Count as an error if result is malformed
                 logging.error(f"  -> Error: Received invalid result from worker: {res}")


        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        log_processing_summary(len(image_files), successful_image_count, total_generated_count,
                               images_with_errors_count, total_skipped_overwrite_count,
                               skipped_scan_items_details, config.output_dir,
                               total_processing_time, config.dry_run)

    else: # Input path is not a file or directory
        if not os.path.exists(input_path):
             logging.critical(f"  -> Critical: Input path not found: {os.path.abspath(input_path)}")
        else:
             logging.critical(f"  -> Critical: Input path is neither a file nor a directory: {os.path.abspath(input_path)}")

    logging.info(f"===== Image Cropping Script Finished =====")
    logging.info("Script exited normally.")


if __name__ == "__main__":
    main_process_pid = os.getpid() # Store PID for main process logging differentiation
    main()
