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

# --- Logging Setup ---
# Define detailed log format (timestamp, log level, process name, message)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create console handler (using default stderr)
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)

# Get the root logger
logger = logging.getLogger()

# Remove existing handlers (prevent duplicate logging)
if logger.hasHandlers():
    logger.handlers.clear()

# Add the new handler
logger.addHandler(log_handler)
logger.setLevel(logging.INFO) # Default level INFO

# --- Constants ---
__version__ = "1.7.1" # Version update reflecting default path changes

# --- Configuration Dataclass ---
@dataclass
class Config:
    """Holds all configuration settings for the script."""
    # --- User Configurable Settings ---
    input_path: str = "input" # Changed default input path
    output_dir: str = "output" # Changed default output directory
    config_path: Optional[str] = None # Path to JSON config file
    method: str = 'largest' # ['largest', 'center']
    reference: str = 'eye' # ['eye', 'box']
    ratio: Optional[str] = None # e.g., '16:9', '1.0', 'None'
    confidence: float = 0.6
    nms: float = 0.3
    overwrite: bool = True
    output_format: Optional[str] = None # e.g., 'jpg', 'png'
    jpeg_quality: int = 95
    min_face_width: int = 30
    min_face_height: int = 30
    padding_percent: float = 5.0
    rule: str = 'both' # ['thirds', 'golden', 'both']
    workers: int = field(default_factory=os.cpu_count) # Default to CPU count
    verbose: bool = False
    dry_run: bool = False

    # --- Derived or Fixed Values (Not directly from args/config file) ---
    target_ratio_float: Optional[float] = None # Calculated from ratio string
    yunet_model_url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    yunet_model_path: str = field(default_factory=lambda: os.path.join("models", "face_detection_yunet_2023mar.onnx"))
    output_suffix_thirds: str = '_thirds'
    output_suffix_golden: str = '_golden'
    supported_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

    def __post_init__(self):
        """Calculate derived values and validate settings after initialization."""
        self.target_ratio_float = parse_aspect_ratio(self.ratio)
        # Validate some numeric values
        if self.min_face_width < 0:
            logging.warning(f"Minimum face width must be >= 0 ({self.min_face_width}). Setting to 0.")
            self.min_face_width = 0
        if self.min_face_height < 0:
            logging.warning(f"Minimum face height must be >= 0 ({self.min_face_height}). Setting to 0.")
            self.min_face_height = 0
        if self.padding_percent < 0:
            logging.warning(f"Padding percent must be >= 0 ({self.padding_percent}). Setting to 0.")
            self.padding_percent = 0.0
        if self.workers < 0:
            logging.warning(f"Number of workers must be >= 0 ({self.workers}). Setting to 1.")
            self.workers = 1
        if not (1 <= self.jpeg_quality <= 100):
             logging.warning(f"JPEG quality must be between 1 and 100 ({self.jpeg_quality}). Setting to 95.")
             self.jpeg_quality = 95

# Store the main process PID (for preventing duplicate logs in multiprocessing)
main_process_pid = os.getpid()

# --- Utility Functions ---

def setup_logging(level: int):
    """Sets the global logging level."""
    logger.setLevel(level)
    logging.debug(f"Logging level set to {logging.getLevelName(level)}")

def download_model(url: str, file_path: str) -> bool:
    """Downloads the model file from the specified URL if it doesn't exist."""
    model_dir = os.path.dirname(file_path)
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
            logging.info(f"Created model directory: {model_dir}")
        except OSError as e:
            logging.error(f"Failed to create model directory '{model_dir}': {e}")
            return False

    if not os.path.exists(file_path):
        # Log download attempt only in the main process to avoid clutter
        if os.getpid() == main_process_pid:
            logging.info(f"Downloading model file... ({os.path.basename(file_path)})")
        try:
            urllib.request.urlretrieve(url, file_path)
            if os.getpid() == main_process_pid:
                logging.info("Download complete.")
            return True
        except Exception as e:
            logging.error(f"Model file download failed: {e}")
            logging.error(f"       Please manually download from the following URL and save as '{file_path}': {url}")
            return False
    else:
        # Log existence check only in the main process
        if os.getpid() == main_process_pid:
             logging.info(f"Model file '{os.path.basename(file_path)}' already exists.")
        return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    """Converts a ratio string (e.g., '16:9', '1.0', 'None') to a float."""
    if ratio_str is None or str(ratio_str).strip().lower() == 'none':
        return None
    try:
        ratio_str = str(ratio_str).strip()
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

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            logging.info(f"Configuration file loaded successfully: {config_path}")
            return config_data
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Configuration file parsing error ({config_path}): {e}")
        return {}
    except Exception as e:
        logging.error(f"Error loading configuration file ({config_path}): {e}")
        return {}

def create_output_directory(output_dir: str, dry_run: bool) -> bool:
    """Creates the output directory if it doesn't exist."""
    if dry_run:
        logging.info(f"[DRY RUN] Skipping output directory check/creation: {output_dir}")
        return True
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Output directory created: {output_dir}")
            return True
        except OSError as e:
            logging.critical(f"Failed to create output directory: {e}")
            return False
    elif not os.path.isdir(output_dir):
         logging.critical(f"The specified output path '{output_dir}' is not a directory.")
         return False
    return True

# --- Core Logic Functions ---

def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray, min_w: int, min_h: int) -> List[Dict[str, Any]]:
    """
    Detects faces using the preloaded DNN model (YuNet).
    Excludes faces smaller than the specified minimum size (min_w, min_h).
    Returns a list of dictionaries, each containing face info ('bbox', 'bbox_center', 'eye_center', 'confidence').
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
        # Set input size for the detector
        detector.setInputSize((img_w, img_h))
        # Perform face detection
        faces = detector.detect(image) # Returns tuple: (status, faces_info_array)

        # faces[1] is None if no faces are detected, or a NumPy array otherwise
        if faces is not None and faces[1] is not None:
            for idx, face_info in enumerate(faces[1]):
                # Extract bounding box coordinates and dimensions
                x, y, w, h = map(int, face_info[:4])

                # Filter out faces smaller than the minimum size
                if w < min_w or h < min_h:
                    logging.debug(f"Face ID {idx} ({w}x{h}) ignored as it's smaller than the minimum size ({min_w}x{min_h}).")
                    continue

                # Extract landmarks (eyes) and confidence score
                r_eye_x, r_eye_y = face_info[4:6]
                l_eye_x, l_eye_y = face_info[6:8]
                confidence = face_info[14] # Confidence score is at index 14

                # Ensure bounding box coordinates are within image boundaries
                x = max(0, x); y = max(0, y)
                # Ensure width and height don't extend beyond image boundaries from the starting point (x, y)
                w = min(img_w - x, w); h = min(img_h - y, h)

                # Proceed only if the adjusted bounding box has a valid size
                if w > 0 and h > 0:
                    bbox_center = (x + w // 2, y + h // 2)
                    eye_center = bbox_center # Default to bbox center

                    # Calculate eye center if landmarks are valid (coordinates > 0)
                    if r_eye_x > 0 and r_eye_y > 0 and l_eye_x > 0 and l_eye_y > 0:
                        ecx = int(round((r_eye_x + l_eye_x) / 2))
                        ecy = int(round((r_eye_y + l_eye_y) / 2))
                        # Clamp eye center coordinates to be within image bounds
                        ecx = max(0, min(img_w - 1, ecx))
                        ecy = max(0, min(img_h - 1, ecy))
                        eye_center = (ecx, ecy)
                    else:
                        logging.debug(f"Eye landmarks for face ID {idx} are invalid or out of bounds, using BBox center.")

                    # Append detected subject information
                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e:
        # Log OpenCV specific errors
        logging.error(f"OpenCV error during face detection (image size: {img_w}x{img_h}): {e}")
        return [] # Return empty list on OpenCV error
    except Exception as e:
        # Log any other unexpected errors during detection
        logging.error(f"Unexpected problem during DNN face detection: {e}", exc_info=logging.getLogger().level == logging.DEBUG)
        return []
    return detected_subjects


def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'eye') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """
    Selects the main subject from the list of detected subjects based on the specified method.
    Returns the selected subject's bounding box and reference point (eye center or bbox center).
    """
    if not subjects:
        logging.debug("Select main subject: No subjects detected.")
        return None

    img_h, img_w = img_shape
    best_subject = None

    try:
        if len(subjects) == 1:
            # If only one subject is detected, it's the main subject
            best_subject = subjects[0]
        elif method == 'largest':
            # Select the subject with the largest bounding box area (w * h)
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
        elif method == 'center':
            # Select the subject whose bounding box center is closest to the image center
            img_center = (img_w / 2, img_h / 2)
            best_subject = min(subjects, key=lambda s: math.dist(s['bbox_center'], img_center))
        else:
            # Default to 'largest' if the method is unknown or invalid
            logging.warning(f"Unknown selection method '{method}'. Defaulting to 'largest'.")
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        # Determine the reference point based on the configuration ('eye' or 'box')
        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        logging.debug(f"Main subject selected (Method: {method}, Reference: {reference_point_type}). BBox: {best_subject['bbox']}, Ref Point: {ref_center}")
        return best_subject['bbox'], ref_center

    except Exception as e:
        # Log errors during the selection process
        logging.error(f"Error selecting main subject: {e}", exc_info=logging.getLogger().level == logging.DEBUG)
        return None


def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """
    Calculates the intersection points based on the specified composition rule ('thirds' or 'golden').
    Returns a list of (x, y) tuples representing the intersection points.
    """
    points = []
    if width <= 0 or height <= 0:
        logging.warning(f"Cannot calculate rule points for invalid size ({width}x{height}).")
        return []

    try:
        if rule_type == 'thirds':
            # Rule of Thirds: Divide image into 3x3 grid, points are the 4 inner intersections
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden':
            # Golden Ratio: Use the inverse golden ratio (phi_inv) to find lines
            phi_inv = (math.sqrt(5) - 1) / 2 # Approx 0.618
            lines_w = (width * (1 - phi_inv), width * phi_inv) # Vertical lines
            lines_h = (height * (1 - phi_inv), height * phi_inv) # Horizontal lines
            points = [(w, h) for w in lines_w for h in lines_h] # 4 intersection points
        else:
            # Default or unknown rule: Use the center of the image as the single point
            logging.warning(f"Unknown composition rule '{rule_type}'. Using image center.")
            points = [(width / 2, height / 2)]

        # Round the calculated points to the nearest integer coordinates
        return [(int(round(px)), int(round(py))) for px, py in points]
    except Exception as e:
        # Log errors during rule point calculation
        logging.error(f"Error calculating rule points (Rule: {rule_type}, Size: {width}x{height}): {e}", exc_info=logging.getLogger().level == logging.DEBUG)
        return []


def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculates the optimal crop area (x1, y1, x2, y2) to place the subject_center
    closest to one of the rule_points, while maintaining the target_aspect_ratio.
    Returns the crop coordinates (top-left x, top-left y, bottom-right x, bottom-right y) or None if calculation fails.
    """
    height, width = img_shape
    # Basic validation for image dimensions and rule points
    if height <= 0 or width <= 0:
        logging.warning(f"Cannot crop image with zero or negative dimensions ({width}x{height}).")
        return None
    if not rule_points:
        logging.warning("No rule points provided, skipping crop calculation.")
        return None

    cx, cy = subject_center # Coordinates of the subject's reference point (e.g., eye center)

    try:
        # Determine the aspect ratio to use for cropping
        # Use the target ratio if provided, otherwise use the original image's aspect ratio
        aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)

        if aspect_ratio <= 0:
            logging.warning(f"Invalid target aspect ratio ({aspect_ratio}). Cannot calculate crop.")
            return None

        # Find the rule point that is closest to the subject's reference point
        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point # This rule point will ideally be the center of the final crop

        # Calculate the maximum possible crop dimensions centered around the target rule point,
        # ensuring the crop stays within the image boundaries.
        # max_w is twice the distance from target_x to the nearest vertical edge.
        # max_h is twice the distance from target_y to the nearest horizontal edge.
        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)

        # If the target point is exactly on an edge, a centered crop is not possible.
        if max_w <= 0 or max_h <= 0:
             logging.debug(f"Target rule point ({target_x},{target_y}) is on or too close to the image boundary. Cannot create a valid centered crop.")
             return None # Cannot create a valid crop centered here

        # Determine the final crop dimensions based on the target aspect ratio and the limiting dimension (max_w or max_h).
        # Calculate the height required if we use the maximum possible width (max_w).
        crop_h_from_w = max_w / aspect_ratio
        # Calculate the width required if we use the maximum possible height (max_h).
        crop_w_from_h = max_h * aspect_ratio

        # Choose the smaller crop that fits within both max_w and max_h constraints.
        # We check if the height calculated from max_w fits within max_h.
        # Add a small epsilon (1e-6) to handle potential floating-point inaccuracies.
        if crop_h_from_w <= max_h + 1e-6:
            # If the height fits, use max_w and the calculated height. Width is the limiting factor.
            final_w, final_h = max_w, crop_h_from_w
        else:
            # Otherwise, the height is the limiting factor. Use max_h and the calculated width.
            final_w, final_h = crop_w_from_h, max_h

        # Calculate the crop coordinates (top-left x1, y1 and bottom-right x2, y2)
        # centered around the target rule point (target_x, target_y).
        x1 = target_x - final_w / 2
        y1 = target_y - final_h / 2
        x2 = x1 + final_w
        y2 = y1 + final_h

        # Ensure coordinates are within the image boundaries (0 to width, 0 to height)
        # and convert them to integers.
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        # Final check: Ensure the resulting crop area has a positive width and height.
        if x1 >= x2 or y1 >= y2:
            logging.warning(f"Calculated crop area has zero or negative size ({x1},{y1} to {x2},{y2}).")
            return None

        logging.debug(f"Calculated optimal crop area: ({x1}, {y1}) - ({x2}, {y2}) for target point ({target_x},{target_y})")
        return x1, y1, x2, y2

    except Exception as e:
        # Log any unexpected errors during crop calculation
        logging.error(f"Error calculating optimal crop (Subject: {subject_center}, Rule Pt: {closest_point if 'closest_point' in locals() else 'N/A'}): {e}", exc_info=logging.getLogger().level == logging.DEBUG)
        return None


def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    """
    Applies padding around the calculated crop area, expanding it outwards.
    Adjusts the coordinates to ensure they stay within the original image boundaries.
    """
    x1, y1, x2, y2 = crop_coords
    img_h, img_w = img_shape
    crop_w = x2 - x1
    crop_h = y2 - y1

    # If padding is zero or negative, return the original coordinates
    if padding_percent <= 0:
        return crop_coords

    # Calculate the padding amount for width and height based on the percentage
    # Padding is applied equally to both sides, so divide by 2.
    pad_x = int(round(crop_w * padding_percent / 100 / 2))
    pad_y = int(round(crop_h * padding_percent / 100 / 2))

    # Apply padding by subtracting from top-left (x1, y1) and adding to bottom-right (x2, y2)
    new_x1 = x1 - pad_x
    new_y1 = y1 - pad_y
    new_x2 = x2 + pad_x
    new_y2 = y2 + pad_y

    # Clamp the new coordinates to the image boundaries (0 to img_w, 0 to img_h)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)

    # Check if padding resulted in an invalid (zero or negative size) area
    if new_x1 >= new_x2 or new_y1 >= new_y2:
        logging.warning(f"Crop area size became zero or negative after applying {padding_percent}% padding. Padding will not be applied.")
        return crop_coords # Return original coordinates if padding makes it invalid

    logging.debug(f"Applied {padding_percent}% padding. New crop area: ({new_x1}, {new_y1}) - ({new_x2}, {new_y2})")
    return new_x1, new_y1, new_x2, new_y2


# --- Image Processing Sub-Functions (Refactored from process_image) ---

def _load_and_prepare_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[bytes], Optional[str]]:
    """Loads image using Pillow, extracts EXIF, converts to OpenCV format (BGR)."""
    filename = os.path.basename(image_path)
    exif_data = None
    original_ext = os.path.splitext(image_path)[1].lower()
    img_bgr = None

    try:
        with Image.open(image_path) as pil_img:
            try:
                # Preserve orientation using EXIF data before potential conversion
                pil_img = ImageOps.exif_transpose(pil_img)
                # Extract raw EXIF data
                exif_data = pil_img.info.get('exif')
                if exif_data:
                    logging.debug(f"{filename}: EXIF data found and preserved.")
            except Exception as exif_err:
                logging.warning(f"{filename}: Error processing EXIF data: {exif_err}. Proceeding without EXIF.")
                exif_data = None

            # Convert to RGB (standard color format) before converting to NumPy array
            pil_img_rgb = pil_img.convert('RGB')
            # Convert Pillow image (RGB) to OpenCV format (BGR)
            # NumPy array conversion and color channel swapping
            img_bgr = np.array(pil_img_rgb)[:, :, ::-1].copy()

    except FileNotFoundError:
        logging.error(f"{filename}: Image file not found.")
        return None, None, None
    except UnidentifiedImageError:
        logging.error(f"{filename}: Cannot open image file or unsupported format.")
        return None, None, None
    except Exception as e:
        logging.error(f"{filename}: Error loading image: {e}", exc_info=logging.getLogger().level == logging.DEBUG)
        return None, None, None

    return img_bgr, exif_data, original_ext

def _determine_output_filename(base_filename: str, suffix: str, config: Config, original_ext: str) -> Tuple[str, str]:
    """Determines the output filename and extension based on configuration."""
    target_ratio_str = str(config.ratio) if config.ratio is not None else "Orig"
    # Create a filename-safe string for the ratio (replace ':' with '-')
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"
    ref_str = f"_ref{config.reference}" # Add reference point type ('eye' or 'box')

    # Determine the output file extension
    output_format = config.output_format.lower() if config.output_format else None
    if output_format:
        output_ext = f".{output_format.lstrip('.')}"
        # Validate if Pillow supports the requested format extension
        # Image.registered_extensions() provides a map of supported extensions
        if output_ext.lower() not in Image.registered_extensions():
             logging.warning(f"{base_filename}: Unsupported output format '{config.output_format}'. Using original format '{original_ext}'.")
             output_ext = original_ext
    else:
        # If no output format is specified, keep the original file extension
        output_ext = original_ext

    # Construct the final filename
    out_filename = f"{base_filename}{suffix}{ratio_str}{ref_str}{output_ext}"
    return out_filename, output_ext

def _save_cropped_image(cropped_bgr: np.ndarray, out_path: str, output_ext: str, config: Config, exif_data: Optional[bytes]) -> bool:
    """Saves the cropped image (BGR format) to the specified path using Pillow."""
    filename = os.path.basename(out_path)
    try:
        if cropped_bgr.size == 0:
             raise ValueError("Cropped image data is empty.")

        # Convert cropped image from OpenCV BGR format back to RGB for Pillow saving
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_cropped_img = Image.fromarray(cropped_rgb)

        # Prepare save options (EXIF, quality for JPEG)
        save_options = {}
        # Include EXIF data if it exists and is in bytes format
        if exif_data and isinstance(exif_data, bytes):
            save_options['exif'] = exif_data
        # Apply JPEG specific options
        if output_ext.lower() in ['.jpg', '.jpeg']:
            save_options['quality'] = config.jpeg_quality
            save_options['optimize'] = True    # Try to optimize file size
            save_options['progressive'] = True # Use progressive JPEG format
        # Add other format options here if needed (e.g., PNG compression)
        # Example: if output_ext.lower() == '.png': save_options['compress_level'] = 6

        # Save the image using Pillow
        pil_cropped_img.save(out_path, **save_options)
        logging.debug(f"{filename}: Saved successfully.")
        return True

    except IOError as e:
         logging.error(f"{filename}: File write error: {e}")
         return False
    except ValueError as e: # Catch specific errors like empty image data
         logging.error(f"{filename}: Cropped image processing error: {e}")
         return False
    except Exception as e:
        # Log generic errors during saving, include stack trace if verbose
        logging.error(f"{filename}: Error saving cropped image: {e}", exc_info=config.verbose)
        return False

def _process_composition_rule(rule_name: str, suffix: str, img_bgr: np.ndarray, ref_center: Tuple[int, int], config: Config, exif_data: Optional[bytes], original_ext: str) -> Tuple[bool, str]:
    """Processes a single composition rule (thirds or golden) for an image."""
    # Use the input_path from the config object passed for this specific image
    filename = os.path.basename(config.input_path)
    base_filename = os.path.splitext(filename)[0]
    img_h, img_w = img_bgr.shape[:2]
    saved = False
    error_message = ""

    # 1. Get rule points for the current composition rule
    rule_points = get_rule_points(img_w, img_h, rule_name)
    if not rule_points:
        error_message = f"Rule '{rule_name}': Failed to get rule points."
        logging.warning(f"{filename}: {error_message}")
        return saved, error_message

    # 2. Calculate the optimal crop area based on rule points and subject center
    crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, config.target_ratio_float)
    if not crop_coords:
        error_message = f"Rule '{rule_name}': Failed to calculate optimal crop area."
        logging.warning(f"{filename}: {error_message}")
        return saved, error_message

    # 3. Apply padding to the calculated crop area
    padded_coords = apply_padding(crop_coords, (img_h, img_w), config.padding_percent)
    x1, y1, x2, y2 = padded_coords

    # 4. Validate the final crop area size
    if x1 >= x2 or y1 >= y2:
        error_message = f"Rule '{rule_name}': Final crop area size is zero or negative after padding ({x1},{y1} to {x2},{y2})."
        logging.warning(f"{filename}: {error_message}")
        return saved, error_message

    # 5. Determine the output filename and path
    out_filename, output_ext = _determine_output_filename(base_filename, suffix, config, original_ext)
    out_path = os.path.join(config.output_dir, out_filename)

    # 6. Check overwrite condition
    if not config.overwrite and os.path.exists(out_path) and not config.dry_run:
        logging.info(f"{filename}: File exists and overwrite disabled - skipping: {out_filename}")
        # Technically not an error, but didn't save this specific file
        error_message = f"Rule '{rule_name}': Skipped (file exists, overwrite disabled)."
        return saved, error_message # Return saved=False as this specific file wasn't overwritten

    # 7. Perform Dry Run or Actual Save
    if config.dry_run:
        logging.info(f"[DRY RUN] {filename}: Would save '{out_filename}' (Rule: {rule_name}, Area: {x1},{y1}-{x2},{y2})")
        saved = True # Indicate simulation success for this rule
    else:
        # Crop the original image (BGR)
        cropped_img_bgr = img_bgr[y1:y2, x1:x2]
        # Save the cropped image
        save_success = _save_cropped_image(cropped_img_bgr, out_path, output_ext, config, exif_data)
        if save_success:
            saved = True
        else:
            error_message = f"Rule '{rule_name}': Failed to save file '{out_filename}'."
            # Specific error logged in _save_cropped_image

    return saved, error_message


# --- Main Processing Function ---

def process_image(image_path: str, config: Config, detector: cv2.FaceDetectorYN) -> Dict[str, Any]:
    """
    Processes a single image: loads, detects faces, selects subject, calculates crops
    based on rules, and saves results or logs actions in Dry Run mode.
    Returns a dictionary with processing status and messages.
    """
    filename = os.path.basename(image_path)
    logging.debug(f"Processing started: {filename}")
    start_time = time.time()
    # Initialize status dictionary
    status = {'filename': filename, 'success': False, 'saved_files': 0, 'message': '', 'dry_run': config.dry_run}

    # 1. Load Image and Prepare Data
    img_bgr, exif_data, original_ext = _load_and_prepare_image(image_path)
    if img_bgr is None:
        status['message'] = "Failed to load or prepare image."
        # Specific error logged in _load_and_prepare_image
        return status

    img_h, img_w = img_bgr.shape[:2]
    if img_h <= 0 or img_w <= 0:
        status['message'] = f"Invalid image dimensions ({img_w}x{img_h})."
        logging.warning(f"{filename}: {status['message']}")
        return status

    # Create a config instance specific to this image path for passing down
    # This ensures _process_composition_rule gets the correct input_path for logging
    current_config = Config(**{**config.__dict__, 'input_path': image_path})


    # 2. Detect Faces
    detected_faces = detect_faces_dnn(detector, img_bgr, current_config.min_face_width, current_config.min_face_height)
    if not detected_faces:
        status['message'] = "No valid faces detected (considering minimum size)."
        logging.info(f"{filename}: {status['message']}") # Use info level as it's a common outcome
        return status

    # 3. Select Main Subject
    selection_result = select_main_subject(detected_faces, (img_h, img_w), current_config.method, current_config.reference)
    if not selection_result:
         status['message'] = "Failed to select main subject from detected faces."
         logging.warning(f"{filename}: {status['message']}")
         return status
    subj_bbox, ref_center = selection_result

    # 4. Determine which composition rules to apply
    rules_to_process = []
    if current_config.rule == 'thirds' or current_config.rule == 'both':
        rules_to_process.append(('thirds', current_config.output_suffix_thirds))
    if current_config.rule == 'golden' or current_config.rule == 'both':
        rules_to_process.append(('golden', current_config.output_suffix_golden))

    if not rules_to_process:
        status['message'] = f"No composition rule selected to apply ('{current_config.rule}')."
        logging.warning(f"{filename}: {status['message']}")
        return status

    # 5. Process each applicable composition rule
    saved_count = 0
    rule_errors = []
    for rule_name, suffix in rules_to_process:
        # Pass current_config which has the correct input_path
        saved, error_msg = _process_composition_rule(
            rule_name, suffix, img_bgr, ref_center, current_config, exif_data, original_ext
        )
        if saved:
            saved_count += 1
        elif error_msg: # Only add error if saving/simulation didn't happen for this rule
            rule_errors.append(error_msg)

    # 6. Finalize Status and Log Summary
    end_time = time.time()
    processing_time = end_time - start_time

    if saved_count > 0:
        status['success'] = True
        status['saved_files'] = saved_count
        action_verb = "Simulation complete" if current_config.dry_run else "Processing complete"
        status['message'] = f"{action_verb} ({saved_count} file(s) {'simulated' if current_config.dry_run else 'saved'}, {processing_time:.2f}s)."
        if rule_errors:
            # Add info about partial failures if some rules succeeded but others failed
            status['message'] += f" Some rules failed: {'; '.join(rule_errors)}"
        logging.info(f"{filename}: {status['message']}")
    elif detected_faces: # Faces were detected, but no files were saved/simulated for any rule
        status['message'] = f"Faces detected, but failed to {'simulate' if current_config.dry_run else 'crop/save'} for all rules ({processing_time:.2f}s). Errors: {'; '.join(rule_errors) if rule_errors else 'Crop/Save failed'}"
        logging.warning(f"{filename}: {status['message']}") # Warning as faces were found but output failed
    # If no faces were detected, the message is already set earlier

    return status


# --- Parallel Processing Wrapper ---
def process_image_wrapper(args_tuple: Tuple[str, Config]) -> Dict[str, Any]:
    """
    Wrapper function for concurrent.futures.ProcessPoolExecutor.
    Loads the model and calls process_image for a single image path.
    """
    image_path, config = args_tuple
    filename = os.path.basename(image_path) # For logging errors in this wrapper

    # --- Configure Logging for Child Process ---
    # Inheriting logger setup from main process is usually sufficient.
    # If specific per-process logging is needed, configure it here.

    detector = None # Initialize detector
    try:
        # --- Load detector within each process ---
        logging.debug(f"Process for {filename}: Loading model ({os.path.basename(config.yunet_model_path)})...")
        # Ensure model file exists before creating detector
        if not os.path.exists(config.yunet_model_path):
             # Attempt download again just in case it failed initially
             if not download_model(config.yunet_model_url, config.yunet_model_path):
                  raise RuntimeError(f"Model file {config.yunet_model_path} not found and download failed.")

        detector = cv2.FaceDetectorYN.create(config.yunet_model_path, "", (0, 0)) # "" for backend, (0,0) for auto input size
        detector.setScoreThreshold(config.confidence)
        detector.setNMSThreshold(config.nms)
        logging.debug(f"Process for {filename}: Model loading complete.")
        # --- Detector loading finished ---

        # Call the actual image processing function with the loaded detector
        # Pass the original config, process_image will create a specific one if needed
        return process_image(image_path, config, detector)

    except cv2.error as e:
        logging.error(f"{filename}: OpenCV error during model loading or detection in wrapper: {e}", exc_info=config.verbose)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'message': f"OpenCV Error: {e}", 'dry_run': config.dry_run}
    except Exception as e:
        # Catch exceptions not handled within process_image or during model loading
        logging.error(f"{filename}: Critical error during processing (wrapper): {e}", exc_info=True) # Always log traceback for critical errors
        return {'filename': filename, 'success': False, 'saved_files': 0, 'message': f"Critical error in wrapper: {e}", 'dry_run': config.dry_run}


# --- Command-Line Interface Setup ---

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="""Automatically crop images based on face detection and composition rules (Rule of Thirds, Golden Ratio).""",
        formatter_class=argparse.RawTextHelpFormatter # Preserve help message formatting
    )
    # --- Input/Output Arguments ---
    # Note: Default for input_path is handled by Config, but it's a required positional argument.
    parser.add_argument("input_path", help="Path to the image file or directory to process.")
    # Updated help text for output_dir default
    parser.add_argument("-o", "--output_dir", help=f"Directory to save results (Default: {Config.output_dir}).")
    parser.add_argument("--config", help="Path to a JSON configuration file to load options from.")

    # --- Behavior Control Arguments ---
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=Config.overwrite, help="Allow overwriting output files.")
    parser.add_argument("--dry-run", action="store_true", default=Config.dry_run, help="Simulate processing without saving files.")
    parser.add_argument("-w", "--workers", type=int, help=f"Number of parallel workers (Default: System CPU count, 0 or 1 for sequential).")

    # --- Face Detection and Selection Arguments ---
    parser.add_argument("-m", "--method", choices=['largest', 'center'], help=f"Method to select the main subject (Default: {Config.method}).")
    parser.add_argument("--ref", "--reference", dest="reference", choices=['eye', 'box'], help=f"Reference point type for composition (Default: {Config.reference}).")
    parser.add_argument("-c", "--confidence", type=float, help=f"Minimum face detection confidence (Default: {Config.confidence}).")
    parser.add_argument("-n", "--nms", type=float, help=f"Face detection NMS threshold (Default: {Config.nms}).")
    parser.add_argument("--min-face-width", type=int, help=f"Minimum face width to process in pixels (Default: {Config.min_face_width}).")
    parser.add_argument("--min-face-height", type=int, help=f"Minimum face height to process in pixels (Default: {Config.min_face_height}).")

    # --- Cropping and Composition Arguments ---
    parser.add_argument("-r", "--ratio", type=str, help="Target crop aspect ratio (e.g., '16:9', '1.0', 'None' for original) (Default: None).")
    parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], help=f"Composition rule(s) to apply (Default: {Config.rule}).")
    parser.add_argument("-p", "--padding-percent", type=float, help=f"Padding percentage around the crop area (%) (Default: {Config.padding_percent}).")

    # --- Output Format Arguments ---
    parser.add_argument("--output-format", type=str, help="Output image format (e.g., 'jpg', 'png'). Default is to keep original format.")
    parser.add_argument("-q", "--jpeg-quality", type=int, choices=range(1, 101), metavar="[1-100]", help=f"JPEG save quality (1-100) (Default: {Config.jpeg_quality}).")

    # --- Miscellaneous Arguments ---
    parser.add_argument("-v", "--verbose", action="store_true", default=Config.verbose, help="Enable detailed logging (DEBUG level).")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    return parser

def load_and_merge_config(args: argparse.Namespace) -> Config:
    """Loads default config, merges JSON config file, and merges command-line args."""
    # 1. Start with default config values from the dataclass definition
    # Create an instance to get defaults, including factory defaults
    default_config_obj = Config()
    config_values = default_config_obj.__dict__.copy() # Use copy to avoid modifying the instance dict directly

    # Override input_path from args immediately as it's positional and required
    config_values['input_path'] = args.input_path

    # 2. Load config from JSON file if specified
    json_config_data = {}
    if args.config:
        json_config_data = load_config_from_file(args.config)
        # Merge JSON config into defaults, ensuring keys are valid
        for key, value in json_config_data.items():
            if key == 'input_path':
                 logging.warning("Ignoring 'input_path' from JSON config file; use command line argument instead.")
                 continue # Skip input_path from JSON
            if key in config_values:
                config_values[key] = value
            else:
                logging.warning(f"Unknown key '{key}' in configuration file '{args.config}' ignored.")

    # 3. Override with command-line arguments (only if they were actually provided by the user)
    parser = setup_arg_parser() # Need a temporary parser instance to check defaults
    for key, value in vars(args).items():
        # Skip input_path as it's already handled
        if key == 'input_path':
             continue
        # Check if the argument was provided (not None) AND is different from the *parser's* default
        # This prevents overwriting JSON/Config defaults with CLI defaults if the arg wasn't specified
        parser_default = parser.get_default(key)
        if value is not None and value != parser_default:
             if key in config_values:
                 config_values[key] = value
             # else: # This case should ideally not happen if parser and Config are synced
             #    logging.warning(f"Command line argument '{key}' not found in Config class.")

    # Handle BooleanOptionalAction specifically (overwrite)
    # Check if the value from args is different from the default Config value
    if args.overwrite != default_config_obj.overwrite:
         config_values['overwrite'] = args.overwrite
    # Handle store_true actions (dry_run, verbose)
    if args.dry_run != default_config_obj.dry_run:
         config_values['dry_run'] = args.dry_run
    if args.verbose != default_config_obj.verbose:
         config_values['verbose'] = args.verbose


    # 4. Create the final Config object using the merged values
    try:
        # The __post_init__ will run here, calculating derived values like target_ratio_float
        final_config = Config(**config_values)
    except TypeError as e:
        logging.critical(f"Configuration error when creating Config object: {e}. Check config file/arguments.")
        exit(1) # Exit if config is fundamentally broken

    return final_config

# --- Main Execution ---

def main():
    """
    Main execution function: Parses arguments, loads configuration,
    prepares resources, and processes images sequentially or in parallel.
    """
    # 1. Parse Command Line Arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # 2. Load and Merge Configuration
    # Input path from args is now handled inside load_and_merge_config
    config = load_and_merge_config(args)

    # 3. Setup Logging Level
    setup_logging(logging.DEBUG if config.verbose else logging.INFO)
    if config.verbose:
        # Log the final effective configuration in verbose mode
        logging.debug("--- Effective Configuration ---")
        for key, value in sorted(config.__dict__.items()):
             # Don't log potentially sensitive info if added later
             if key not in ['yunet_model_url']: # Example filter
                 logging.debug(f"{key}: {value}")
        logging.debug("-----------------------------")


    if config.dry_run:
        logging.info("***** Running in Dry Run mode. No files will be saved. *****")

    # 4. Preparation Steps
    #   a. Download DNN model file (only in the main process)
    if not download_model(config.yunet_model_url, config.yunet_model_path):
        logging.critical("DNN model file is not available or download failed. Aborting.")
        return # Critical failure

    #   b. Create output directory
    if not create_output_directory(config.output_dir, config.dry_run):
        return # Critical failure if directory cannot be created

    # 5. Image Processing
    input_path = config.input_path # Get the finalized input path from config
    if os.path.isfile(input_path):
        # --- Process a Single File ---
        logging.info(f"Starting single file processing: {input_path}")
        try:
            # Load detector once for the single file
            detector = cv2.FaceDetectorYN.create(config.yunet_model_path, "", (0, 0))
            detector.setScoreThreshold(config.confidence)
            detector.setNMSThreshold(config.nms)
            # Call process_image directly
            result = process_image(input_path, config, detector)
            logging.info(f"Single file processing finished. Result: {result.get('message', 'No message')}")
        except cv2.error as e:
             logging.critical(f"Failed to load face detection model (single file): {e}")
        except Exception as e:
             logging.critical(f"Error during single file processing: {e}", exc_info=True)

    elif os.path.isdir(input_path):
        # --- Process a Directory ---
        logging.info(f"Starting directory processing: {input_path}")
        try:
            all_files = os.listdir(input_path)
            # Filter for supported image files using the list from Config
            image_files = [os.path.join(input_path, f) for f in all_files
                           if f.lower().endswith(config.supported_extensions) and os.path.isfile(os.path.join(input_path, f))]
        except OSError as e:
            logging.critical(f"Cannot access input directory '{input_path}': {e}")
            return

        if not image_files:
            logging.info(f"No supported image files ({', '.join(config.supported_extensions)}) found in the directory: {input_path}")
            return

        # Determine number of workers for parallel processing
        num_workers = config.workers
        # Use actual CPU count if default is used, ensure at least 1 worker
        # Clamp workers to number of files if fewer files than workers
        actual_workers = min(num_workers, os.cpu_count(), len(image_files)) if num_workers > 0 else 1
        is_parallel = actual_workers > 1

        logging.info(f"Found {len(image_files)} image file(s). Starting processing (Mode: {'Parallel' if is_parallel else 'Sequential'}, Workers: {actual_workers})...")
        total_start_time = time.time()
        results = []
        processed_count = 0
        success_count = 0
        total_saved_files = 0
        failed_files = []

        # Create task list (tuples of image path and config object)
        # Pass the same config object to all workers; it's read-only after creation
        tasks = [(img_path, config) for img_path in image_files]

        if is_parallel:
             # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                try:
                    # Map tasks to the wrapper function
                    results_iterator = executor.map(process_image_wrapper, tasks)
                    # Use tqdm for progress bar if available
                    if TQDM_AVAILABLE:
                         results = list(tqdm(results_iterator, total=len(tasks), desc="Processing images", unit="file"))
                    else:
                         # Log progress periodically if tqdm is not available
                         temp_results = []
                         for i, result in enumerate(results_iterator):
                              temp_results.append(result)
                              processed_count = i + 1 # Count as processed when result arrives
                              if processed_count % 50 == 0 or processed_count == len(tasks): # Log every 50 or at the end
                                   logging.info(f"Progress: {processed_count}/{len(tasks)} files processed...")
                         results = temp_results
                except Exception as e:
                     logging.critical(f"Critical error during parallel processing execution: {e}", exc_info=True)
                     # Results list might be incomplete here
        else:
            # Sequential processing (load detector once in main process)
            logging.info("Running in sequential processing mode (1 worker).")
            try:
                detector = cv2.FaceDetectorYN.create(config.yunet_model_path, "", (0, 0))
                detector.setScoreThreshold(config.confidence)
                detector.setNMSThreshold(config.nms)
                # Iterate through files, using tqdm if available
                iterator = tqdm(tasks, desc="Processing images", unit="file") if TQDM_AVAILABLE else tasks
                for task_args in iterator:
                     # Call process_image directly since detector is loaded
                     results.append(process_image(task_args[0], task_args[1], detector))
            except cv2.error as e:
                 logging.critical(f"Failed to load face detection model (sequential): {e}")
                 return # Stop sequential processing if model fails to load
            except Exception as e:
                 logging.critical(f"Critical error during sequential processing loop: {e}", exc_info=True)
                 # Results list might be incomplete


        # --- Aggregate and Summarize Results ---
        processed_count = 0 # Reset counter, count based on results received
        for result in results:
            processed_count += 1
            if result and isinstance(result, dict): # Check if result is valid
                if result.get('success', False):
                    success_count += 1
                    total_saved_files += result.get('saved_files', 0)
                if not result.get('success', False):
                    # Store filename and message for failed files
                    failed_files.append(f"{result.get('filename', 'UnknownFile')} ({result.get('message', 'Unknown error')})")
            else:
                 # Handle cases where a worker might have returned None or unexpected data
                 logging.error(f"Received invalid result from worker: {result}")
                 failed_files.append("UnknownFile (Invalid result from worker)")


        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        action_verb = "Simulation" if config.dry_run else "Processing"

        # Print summary
        logging.info("-" * 40)
        logging.info(f"          Directory {action_verb} Summary          ")
        logging.info("-" * 40)
        logging.info(f"Total files attempted: {len(image_files)}")
        logging.info(f"Files processed (results received): {processed_count}")
        logging.info(f"Successfully processed files: {success_count}")
        logging.info(f"Total cropped images {'simulated' if config.dry_run else 'saved'}: {total_saved_files}")
        logging.info(f"Files with errors or failures: {len(failed_files)}")
        if failed_files:
            logging.info("Failure details (Top 10 - check logs for full list):")
            for i, fail_info in enumerate(failed_files):
                 if i < 10:
                    # Show summary of failure
                    logging.info(f"  - {fail_info}")
                 elif i == 10:
                    logging.info("  - ... (Check logs for more)")
                    break
        logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
        logging.info("-" * 40)

    else:
        # Check if input path exists at all
        if not os.path.exists(input_path):
             logging.critical(f"Input path not found: {input_path}")
        else:
             logging.critical(f"Input path is neither a file nor a directory: {input_path}")

if __name__ == "__main__":
    # Store main process PID (used to differentiate logging in child processes)
    main_process_pid = os.getpid()
    main()
