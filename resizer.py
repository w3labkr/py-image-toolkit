# -*- coding: utf-8 -*-
import os
import sys
import importlib.util # Kept for potential future use
import argparse # For command line argument processing
import logging # For logging functionality
from dataclasses import dataclass, field # For configuration class
from typing import Dict, Any, Optional, Tuple # For type hints

# --- Third-party Library Imports ---
# If required libraries are not installed, an ImportError will occur here and the script will exit.
from PIL import Image, UnidentifiedImageError
import piexif
from tqdm import tqdm

# --- Logging Setup (Console Output Only) ---
log_console_handler = logging.StreamHandler(sys.stdout) # Console output handler
log_console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s')) # Simple format for console

logger = logging.getLogger()
logger.setLevel(logging.INFO) # Default log level
logger.addHandler(log_console_handler) # Add only the console handler


# --- Constants ---
SCRIPT_VERSION = "2.9" # Version update (Removed unnecessary comments, English comments/output)

# Define resampling filters for Pillow version compatibility
# Needs definition before Config class instantiation
try:
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST
    }
except AttributeError: # Compatibility for older Pillow versions
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }

# User-friendly names for filters
FILTER_NAMES = {
    "lanczos": "LANCZOS (High quality)",
    "bicubic": "BICUBIC (Medium quality)",
    "bilinear": "BILINEAR (Low quality)",
    "nearest": "NEAREST (Lowest quality)"
}

# Supported output formats and their display names
SUPPORTED_OUTPUT_FORMATS = {
    "original": "Keep Original", "png": "PNG", "jpg": "JPG", "webp": "WEBP",
}
# Supported input file extensions
SUPPORTED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')


# --- Configuration Dataclass ---
@dataclass
class Config:
    """Dataclass to store script configuration."""
    # Values directly from command-line arguments
    input_dir: str = 'input' # Input directory path, default 'input'
    output_dir_arg: Optional[str] = None # Temporary storage for --output-dir argument
    resize_mode: str = field(default_factory=str) # Required argument, no default
    output_format: str = field(default_factory=str) # Required argument, no default
    width: int = 0
    height: int = 0
    filter: Optional[str] = None
    recursive: bool = False
    quality: Optional[int] = None

    # Calculated or prepared values
    absolute_input_dir: str = field(init=False, default='')
    absolute_output_dir: str = field(init=False, default='')
    resize_options: Dict[str, Any] = field(init=False, default_factory=dict)
    output_format_options: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Perform initial validation and calculations after Config object creation."""
        self._validate_paths()
        self._prepare_options()

    def _validate_paths(self):
        """Validate input/output paths and calculate absolute paths."""
        if not os.path.isdir(self.input_dir):
            logger.critical(f"(!) Error: Invalid input folder path: {self.input_dir}")
            if self.input_dir == 'input':
                 logger.info("    Default input folder 'input' not found. Please create it or specify a path.")
            sys.exit(1)

        self.absolute_input_dir = os.path.abspath(self.input_dir)

        if self.output_dir_arg:
            self.absolute_output_dir = os.path.abspath(self.output_dir_arg)
        else:
            # Default output dir is 'resized_images' under the input directory
            self.absolute_output_dir = os.path.join(self.absolute_input_dir, "resized_images")
            logger.info(f"   -> Info: Output folder not specified. Using default: '{self.absolute_output_dir}'")

        if self.absolute_input_dir == self.absolute_output_dir:
            logger.critical("(!) Error: Input folder and output folder cannot be the same.")
            sys.exit(1)

        # Prevent infinite loops if output is inside input with recursion enabled
        try:
            rel_path = os.path.relpath(self.absolute_output_dir, start=self.absolute_input_dir)
            if self.recursive and not rel_path.startswith(os.pardir) and rel_path != '.':
                 if os.path.commonpath([self.absolute_input_dir]) == os.path.commonpath([self.absolute_input_dir, self.absolute_output_dir]):
                    logger.critical("(!) Error: When --recursive is used, the output folder cannot be inside the input folder to prevent loops.")
                    sys.exit(1)
        except ValueError: # Paths might be on different drives (Windows)
            logger.debug("Input/output paths on different drives, skipping containment check.")
            pass

        # Create output directory if it doesn't exist
        try:
            if not os.path.exists(self.absolute_output_dir):
                os.makedirs(self.absolute_output_dir)
                logger.info(f"   -> Info: Created output folder: '{self.absolute_output_dir}'")
            else:
                logger.debug(f"Output folder already exists: '{self.absolute_output_dir}'")
        except OSError as e:
            logger.critical(f"(!) Error: Cannot create output folder: {self.absolute_output_dir} ({e})")
            sys.exit(1)

        logger.debug(f"Path setup complete: Input='{self.absolute_input_dir}', Output='{self.absolute_output_dir}'")


    def _prepare_options(self):
        """Prepare resize and output format option dictionaries."""
        # Resize options
        self.resize_options = {'mode': self.resize_mode}
        if self.resize_mode != 'none':
            if not self.filter: # Filter validation already done in parse_arguments
                 logger.critical("(!) Internal Error: Filter required for resize mode but not set.")
                 sys.exit(1)
            self.resize_options.update({
                'width': self.width,
                'height': self.height,
                'filter_str': self.filter,
                'filter_obj': _PIL_RESAMPLE_FILTERS[self.filter] # Get actual Pillow filter object
            })
        logger.debug(f"Resize options set: {self.resize_options}")

        # Output format options
        self.output_format_options = {'format_str': self.output_format}
        if self.output_format in ('jpg', 'webp'):
            default_quality = 95 if self.output_format == 'jpg' else 80
            # Quality validation already done in parse_arguments
            self.output_format_options['quality'] = self.quality if self.quality is not None else default_quality
        logger.debug(f"Output format options set: {self.output_format_options}")

        logger.info("Environment and options prepared.")


# --- Utility Functions ---

def get_unique_filepath(filepath):
    """
    Checks if a file path exists. If it does, generates a unique path by appending '_<number>'
    before the extension.
    """
    if not os.path.exists(filepath): return filepath
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            logger.debug(f"Filename conflict detected: '{filename}' -> '{new_filename}'")
            return new_filepath
        counter += 1

# --- Core Image Processing Functions ---

def resize_image_maintain_aspect_ratio(img, target_width, target_height, resample_filter):
    """Resizes image maintaining aspect ratio."""
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0:
        logger.warning("Original image dimensions are zero or negative. Skipping resize.")
        return img

    new_width, new_height = 0, 0
    if target_width > 0 and target_height > 0:
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(original_width * ratio))
        new_height = max(1, int(original_height * ratio))
    elif target_width > 0:
        ratio = target_width / original_width
        new_width = target_width
        new_height = max(1, int(original_height * ratio))
    elif target_height > 0:
        ratio = target_height / original_height
        new_height = target_height
        new_width = max(1, int(original_width * ratio))
    else:
        logger.warning("Target dimensions for aspect ratio resize not specified. Returning original.")
        return img

    if (new_width, new_height) == (original_width, original_height):
        logger.debug("Calculated resize dimensions are same as original. Skipping resize.")
        return img
    try:
        logger.debug(f"Resizing (aspect ratio): ({original_width},{original_height}) -> ({new_width},{new_height})")
        return img.resize((new_width, new_height), resample_filter)
    except ValueError as e:
        # Use tqdm.write for immediate feedback during the loop
        tqdm.write(f"   -> Warning: Error during aspect ratio resize (({original_width},{original_height}) -> ({new_width},{new_height})): {e}. Using original.")
        return img

def resize_image_fixed_size(img, target_width, target_height, resample_filter):
    """Forces image resize to specified dimensions."""
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height):
        logger.debug("Target dimensions are same as original. Skipping fixed resize.")
        return img
    if target_width <= 0 or target_height <= 0:
        logger.warning("Target dimensions for fixed resize are zero or negative. Skipping resize.")
        return img
    try:
        logger.debug(f"Resizing (fixed): ({original_width},{original_height}) -> ({target_width},{target_height})")
        return img.resize((target_width, target_height), resample_filter)
    except ValueError as e:
        tqdm.write(f"   -> Warning: Error during fixed size resize (({original_width},{original_height}) -> ({target_width},{target_height})): {e}. Using original.")
        return img

def prepare_image_for_save(img, output_format_str):
    """Converts image mode if necessary for the target output format."""
    save_img = img
    original_mode = img.mode
    # output_format_str is expected to be uppercase ('JPG', 'PNG', 'WEBP') or None/Original
    if output_format_str == 'JPG':
        # Convert modes with alpha or palette to RGB for JPG saving
        if img.mode in ('RGBA', 'LA', 'P'):
            if img.mode == 'P' and 'transparency' in img.info:
                save_img = img.convert('RGBA') # Convert palette with transparency to RGBA first
                logger.debug("Converted 'P' (with transparency) -> 'RGBA' for JPG saving.")
            elif img.mode == 'P':
                 save_img = img.convert('RGB')
                 logger.debug("Converted 'P' -> 'RGB' for JPG saving.")

            # Handle RGBA/LA (potentially after P conversion)
            if save_img.mode in ('RGBA', 'LA'):
                logger.debug(f"Processing '{save_img.mode}' mode for JPG saving...")
                # Create a white background image
                background = Image.new("RGB", save_img.size, (255, 255, 255))
                try:
                    # Paste the image onto the background using the alpha channel as mask
                    mask = save_img.split()[-1]
                    background.paste(save_img, mask=mask)
                    save_img = background
                    logger.debug(f"Converted '{original_mode}' -> 'RGB' by merging alpha channel.")
                except (IndexError, ValueError):
                    # Fallback if alpha channel processing fails
                    save_img = save_img.convert('RGB')
                    logger.warning(f"Error processing alpha channel. Forcing conversion '{original_mode}' -> 'RGB'.")
    elif output_format_str == 'WEBP':
         # Convert palette mode for WEBP saving
         if img.mode == 'P':
              save_img = img.convert("RGBA") if 'transparency' in img.info else img.convert("RGB")
              logger.debug(f"Converted 'P' -> '{save_img.mode}' for WEBP saving.")

    if save_img.mode != original_mode:
        logger.info(f"Image mode converted: '{original_mode}' -> '{save_img.mode}' (Output format: {output_format_str or 'Original'})")
    return save_img

def process_single_image_file(input_path: str, relative_path: str, config: Config) -> Tuple[bool, Optional[str], str]:
    """
    Processes a single image file: loads, resizes, converts format, handles EXIF, and saves.
    Gets settings from the Config object.
    Returns a tuple: (success_bool, error_message_or_none, affected_path).
    """
    base_name, original_ext = os.path.splitext(os.path.basename(input_path))
    output_format_str = config.output_format_options['format_str'] # 'jpg', 'png', 'webp', 'original'
    logger.info(f"Processing: '{relative_path}'")

    output_ext_map = {'jpg': '.jpg', 'webp': '.webp', 'png': '.png'}
    output_ext = output_ext_map.get(output_format_str.lower(), original_ext)

    # Construct output path maintaining subdirectory structure
    output_relative_dir = os.path.dirname(relative_path)
    output_dir_for_file = os.path.join(config.absolute_output_dir, output_relative_dir)

    # Ensure output subdirectory exists (should be pre-created by batch_process_images)
    if not os.path.isdir(output_dir_for_file):
         logger.error(f"Internal Error: Output subdirectory does not exist: '{output_dir_for_file}'")
         try:
             os.makedirs(output_dir_for_file)
             logger.warning(f"Warning: Created missing output subdirectory: '{output_dir_for_file}'")
         except OSError as e:
              error_msg = f"Failed to create output subdirectory: {output_dir_for_file} ({e})"
              logger.error(error_msg)
              return False, error_msg, input_path # Failure related to input path

    output_filename = base_name + output_ext
    output_path_base = os.path.join(output_dir_for_file, output_filename)
    output_path = get_unique_filepath(output_path_base) # Ensure unique output filename

    original_exif_bytes = None
    exif_data = None

    try:
        with Image.open(input_path) as img:
            logger.debug(f"Image loaded: '{relative_path}' (Size: {img.size}, Mode: {img.mode})")
            # --- EXIF Handling (Load) ---
            if 'exif' in img.info and img.info['exif']:
                original_exif_bytes = img.info['exif']
                try:
                    exif_data = piexif.load(original_exif_bytes)
                    logger.debug(f"EXIF data loaded successfully: '{relative_path}'")
                except Exception as e: # Catch various piexif errors
                    tqdm.write(f"   -> Warning: Failed to parse EXIF for '{relative_path}' ({type(e).__name__}). Attempting to keep original bytes.")
                    exif_data = None # Keep original_exif_bytes to try saving raw if dump fails

            # --- Prepare Image Mode for Saving ---
            save_format_upper = output_format_str.upper() if output_format_str != 'original' else None
            img_prepared = prepare_image_for_save(img, save_format_upper)

            # --- Resize ---
            img_resized = img_prepared
            resize_opts = config.resize_options
            if resize_opts['mode'] == 'aspect_ratio':
                img_resized = resize_image_maintain_aspect_ratio(
                    img_prepared, resize_opts['width'], resize_opts['height'], resize_opts['filter_obj']
                )
            elif resize_opts['mode'] == 'fixed':
                img_resized = resize_image_fixed_size(
                    img_prepared, resize_opts['width'], resize_opts['height'], resize_opts['filter_obj']
                )
            else: # 'none'
                 logger.debug("Resize mode is 'none', skipping resize.")

            # --- Prepare Save Options ---
            save_kwargs = {}
            save_format_arg = save_format_upper if output_format_str != 'original' else None
            if save_format_arg == 'JPG': save_format_arg = 'JPEG' # Pillow uses 'JPEG'

            # --- EXIF Handling (Prepare for Save) ---
            final_exif_bytes = None
            if output_format_str != 'original': # Don't modify EXIF if keeping original format
                if exif_data:
                    try:
                        # Reset orientation tag if present
                        if piexif.ImageIFD.Orientation in exif_data.get('0th', {}):
                            exif_data['0th'][piexif.ImageIFD.Orientation] = 1 # Normal orientation
                            logger.debug("Reset EXIF Orientation tag to 1.")
                        # Remove thumbnail to prevent issues and save space
                        if 'thumbnail' in exif_data and exif_data['thumbnail']:
                             exif_data['thumbnail'] = None
                             logger.debug("Removed EXIF thumbnail data.")
                        final_exif_bytes = piexif.dump(exif_data)
                        logger.debug("Successfully dumped modified EXIF data.")
                    except Exception as e:
                        tqdm.write(f"   -> Warning: Failed to dump modified EXIF for '{relative_path}' ({type(e).__name__}). Attempting to use original EXIF.")
                        final_exif_bytes = original_exif_bytes # Fallback to raw original
                elif original_exif_bytes: # Parsing failed, but we have raw bytes
                     final_exif_bytes = original_exif_bytes
                     logger.debug("Using original EXIF bytes as parsing failed.")
            else: # Keep original format, just pass original EXIF if available
                 final_exif_bytes = original_exif_bytes
                 if final_exif_bytes: logger.debug("Passing original EXIF data for original format.")

            # --- Apply Format-Specific Options & EXIF ---
            output_opts = config.output_format_options
            if output_format_str == 'jpg':
                quality = output_opts.get('quality', 95)
                save_kwargs.update({'quality': quality, 'optimize': True, 'progressive': True})
                logger.debug(f"JPG save options: quality={quality}, optimize=True, progressive=True")
                if final_exif_bytes:
                    save_kwargs['exif'] = final_exif_bytes
                    logger.debug("Including EXIF data for JPG save.")
            elif output_format_str == 'png':
                save_kwargs['optimize'] = True
                logger.debug("PNG save options: optimize=True")
                if final_exif_bytes:
                     try:
                         save_kwargs['exif'] = final_exif_bytes
                         logger.debug("Including EXIF data for PNG save.")
                     except TypeError: # Older Pillow might not support EXIF for PNG
                         logger.warning(f"'{relative_path}': This Pillow version might not support EXIF saving for PNG.")
                         pass
            elif output_format_str == 'webp':
                quality = output_opts.get('quality', 80)
                save_kwargs.update({'quality': quality, 'lossless': False}) # Assuming lossy WEBP is default
                logger.debug(f"WEBP save options: quality={quality}, lossless=False")
                if final_exif_bytes:
                    save_kwargs['exif'] = final_exif_bytes
                    logger.debug("Including EXIF data for WEBP save.")
            elif output_format_str == 'original' and final_exif_bytes:
                # Only add EXIF if the original format supports it
                if original_ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.webp']:
                     save_kwargs['exif'] = final_exif_bytes
                     logger.debug(f"Including EXIF data for original format ({original_ext}) save.")

            # --- Save Result ---
            logger.info(f"Attempting to save image: '{output_path}' (Format: {save_format_arg or 'Original'})")
            img_resized.save(output_path, format=save_format_arg, **save_kwargs)
            logger.info(f"Image saved successfully: '{output_path}'")
            return True, None, output_path # Success, return output path

    except UnidentifiedImageError:
        logger.error(f"Processing failed: '{relative_path}' - Invalid or corrupted image file.")
        return False, "Invalid or corrupted image file", input_path
    except PermissionError:
        logger.error(f"Processing failed: '{relative_path}' - File read/write permission denied.")
        return False, "File read/write permission denied", input_path
    except OSError as e:
        logger.error(f"Processing failed: '{relative_path}' - File system error ({e})")
        # Attempt to clean up partially created file
        if os.path.exists(output_path):
            try: os.remove(output_path); logger.warning(f"Removed partially created file after error: '{output_path}'")
            except OSError: pass
        return False, f"File system error ({e})", input_path
    except ValueError as e: # Pillow internal processing error
         logger.error(f"Processing failed: '{relative_path}' - Image processing value error ({e})")
         return False, f"Image processing value error ({e})", input_path
    except Exception as e:
        logger.critical(f"Processing failed: '{relative_path}' - Unexpected error ({type(e).__name__}: {e})", exc_info=True) # Log stack trace
        return False, f"Unexpected error ({type(e).__name__}: {e})", input_path


def scan_for_image_files(config: Config) -> Tuple[list[Tuple[str, str]], list[str]]:
    """Scans the input folder for image files using settings from Config object."""
    files_to_process = []
    skipped_scan_files = []
    logger.info(f"Scanning input folder: '{config.absolute_input_dir}' (Recursive: {config.recursive})")

    if config.recursive:
        for root, dirs, files in os.walk(config.absolute_input_dir):
            # Skip the output directory and its subdirectories
            if os.path.abspath(root).startswith(config.absolute_output_dir):
                skipped_msg = os.path.relpath(root, config.absolute_input_dir) + " (Output folder subdirectory - skipped)"
                logger.debug(f"Skipping scan: '{skipped_msg}'")
                skipped_scan_files.append(skipped_msg)
                dirs[:] = [] # Don't traverse deeper into this directory
                continue

            logger.debug(f"Scanning directory: '{root}'")
            for filename in files:
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, config.absolute_input_dir)
                if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                    logger.debug(f"Found image file to process: '{relative_path}'")
                    files_to_process.append((input_path, relative_path))
                else:
                    skipped_msg = relative_path + " (Unsupported format)"
                    logger.debug(f"Skipping scan: '{skipped_msg}'")
                    skipped_scan_files.append(skipped_msg)
    else: # Non-recursive scan
        logger.debug(f"Listing files/folders in '{config.absolute_input_dir}' (non-recursive)")
        for filename in os.listdir(config.absolute_input_dir):
            input_path = os.path.join(config.absolute_input_dir, filename)
            if os.path.isfile(input_path):
                if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                    logger.debug(f"Found image file to process: '{filename}'")
                    files_to_process.append((input_path, filename)) # Relative path is just the filename
                else:
                    skipped_msg = filename + " (Unsupported format)"
                    logger.debug(f"Skipping scan: '{skipped_msg}'")
                    skipped_scan_files.append(skipped_msg)
            elif os.path.isdir(input_path):
                 # Check if it's the output directory itself
                if os.path.abspath(input_path) == config.absolute_output_dir:
                    skipped_msg = filename + " (Output folder)"
                    logger.debug(f"Skipping scan: '{skipped_msg}'")
                    skipped_scan_files.append(skipped_msg)
                else: # It's another directory, skip in non-recursive mode
                    skipped_msg = filename + " (Directory - non-recursive)"
                    logger.debug(f"Skipping scan: '{skipped_msg}'")
                    skipped_scan_files.append(skipped_msg)

    logger.info(f"Scan complete: Found {len(files_to_process)} files to process, skipped {len(skipped_scan_files)} items.")
    return files_to_process, skipped_scan_files


def batch_process_images(config: Config) -> Tuple[int, int, list[Tuple[str, str]], list[str]]:
    """Processes all supported image files based on the provided Config."""
    processed_count = 0
    error_count = 0
    error_files = [] # Stores tuples of (relative_path, error_message)
    all_skipped_files = []

    logger.info(f"Starting batch processing. Output folder: '{config.absolute_output_dir}'")

    try:
        files_to_process, skipped_scan_files = scan_for_image_files(config)
        all_skipped_files.extend(skipped_scan_files) # Add files skipped during scan

        total_files = len(files_to_process)
        if total_files == 0:
             logger.warning("(!) No image files found to process in the specified path.")
             return 0, 0, [], all_skipped_files
        else:
            logger.info(f"-> Found {total_files} image files. Starting processing.")

    except Exception as e:
        logger.critical(f"(!) Critical Error: Failed to access input path '{config.absolute_input_dir}'. ({e})", exc_info=True)
        return 0, 0, [], all_skipped_files # Cannot proceed if scan fails

    # --- Pre-create Output Subdirectories ---
    # Avoids checking/creating in the loop for each file
    required_subdirs = set()
    for _, relative_path in files_to_process:
        output_relative_dir = os.path.dirname(relative_path)
        if output_relative_dir: # Only add if not in the root
             required_subdirs.add(output_relative_dir)

    for subdir in required_subdirs:
        output_dir_to_create = os.path.join(config.absolute_output_dir, subdir)
        if not os.path.exists(output_dir_to_create):
            try:
                os.makedirs(output_dir_to_create)
                logger.info(f"Created output subdirectory: '{output_dir_to_create}'")
            except OSError as e:
                # This could be a significant issue affecting multiple files
                logger.error(f"Error: Failed to create output subdirectory '{output_dir_to_create}' ({e}). Files in this directory may fail.")
                # Consider adding logic here to skip files destined for this failed directory

    # --- Image Processing Loop ---
    for input_path, relative_path in tqdm(files_to_process, desc="Processing images", unit="file", ncols=100, leave=True):
        logger.debug(f"Processing loop start for: '{relative_path}'")

        success, error_message, affected_path = process_single_image_file(
            input_path, relative_path, config
        )

        if success:
            processed_count += 1
            logger.debug(f"File processed successfully: '{relative_path}' -> '{affected_path}'")
        else:
            # Error already logged within process_single_image_file
            # tqdm.write provides immediate feedback in the console near the progress bar
            tqdm.write(f" ✗ Error: '{os.path.basename(affected_path)}' (in '{os.path.dirname(relative_path)}' folder) - {error_message}")
            error_files.append((relative_path, error_message)) # Use relative path for consistent reporting
            error_count += 1

    logger.info(f"Batch processing complete. Success: {processed_count}, Errors: {error_count}")
    return processed_count, error_count, error_files, all_skipped_files

# --- Argument Parsing and Setup ---

def parse_arguments() -> Config:
    """Parses command line arguments and creates the Config object."""
    parser = argparse.ArgumentParser(
        description=f"Batch Image Resizer Script (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawTextHelpFormatter, # Preserve formatting in help text
        usage="%(prog)s [input_dir] -m <mode> -O <format> [options]" # Example usage
    )
    # Positional argument for input directory
    parser.add_argument("input_dir", nargs='?', default='input',
                        help="Path to the source folder (default: 'input')")

    # Required options group
    required_group = parser.add_argument_group('Required Options')
    required_group.add_argument("-m", "--resize-mode", required=True, choices=['aspect_ratio', 'fixed', 'none'],
                                help="Resize mode:\n"
                                     "  aspect_ratio: Maintain aspect ratio (specify width and/or height)\n"
                                     "  fixed: Force resize to exact dimensions (may distort)\n"
                                     "  none: No resizing")
    required_group.add_argument("-O", "--output-format", required=True, choices=SUPPORTED_OUTPUT_FORMATS.keys(),
                                help="Output file format:\n" +
                                     "\n".join([f"  {k}: {v}" for k, v in SUPPORTED_OUTPUT_FORMATS.items()]))

    # Optional arguments
    parser.add_argument("-o", "--output-dir", dest='output_dir_arg', # Match Config field name
                        help="Path to save results (default: 'resized_images' under input folder)")

    # Resize options group (ignored if mode is 'none')
    resize_group = parser.add_argument_group('Resize Options (ignored if mode is "none")')
    resize_group.add_argument("-w", "--width", type=int, default=0, help="Target width (pixels)")
    resize_group.add_argument("-H", "--height", type=int, default=0, help="Target height (pixels)")
    resize_group.add_argument("-f", "--filter", choices=FILTER_NAMES.keys(), default=None,
                              help="Resize filter:\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]) +
                                   "\n(Required if mode is 'aspect_ratio' or 'fixed')")

    # Other optional arguments group
    optional_group = parser.add_argument_group('Other Optional Options')
    optional_group.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    optional_group.add_argument("-q", "--quality", type=int, default=None, help="JPG/WEBP quality (1-100)")
    optional_group.add_argument('--version', action='version', version=f'%(prog)s {SCRIPT_VERSION}')

    args = parser.parse_args()
    logger.info(f"Command line arguments parsed: {vars(args)}")

    # --- Argument Validation (before creating Config) ---
    if args.resize_mode == 'aspect_ratio':
        if args.width <= 0 and args.height <= 0:
            parser.error("Mode 'aspect_ratio' requires --width (-w) or --height (-H) > 0.")
        if args.width < 0 or args.height < 0:
             parser.error("--width (-w) and --height (-H) cannot be negative.")
        if not args.filter:
            parser.error("--filter (-f) is required for mode 'aspect_ratio'.")
    elif args.resize_mode == 'fixed':
        if args.width <= 0 or args.height <= 0:
            parser.error("Mode 'fixed' requires both --width (-w) and --height (-H) > 0.")
        if not args.filter:
            parser.error("--filter (-f) is required for mode 'fixed'.")
    elif args.resize_mode == 'none' and (args.width > 0 or args.height > 0 or args.filter):
        # Not an error, just inform the user
        logger.warning("   -> Info: --width, --height, and --filter are ignored when --resize-mode is 'none'.")

    # Quality option validation
    if args.output_format in ('jpg', 'webp') and args.quality is not None:
        if not (1 <= args.quality <= 100):
            parser.error(f"--quality must be between 1 and 100 (got {args.quality}).")
    elif args.quality is not None: # Quality specified for non-applicable format
        logger.warning(f"   -> Warning: --quality (-q) is ignored for output format '{args.output_format}'.")

    logger.debug("Argument validation passed.")

    # Create Config object (this runs __post_init__ for path/option setup)
    try:
        config = Config(
            input_dir=args.input_dir,
            output_dir_arg=args.output_dir_arg,
            resize_mode=args.resize_mode,
            output_format=args.output_format,
            width=args.width,
            height=args.height,
            filter=args.filter,
            recursive=args.recursive,
            quality=args.quality
        )
        return config
    except SystemExit: # Handle sys.exit calls within Config.__post_init__
        raise # Re-raise to exit script
    except Exception as e:
        logger.critical(f"(!) Error during configuration object creation: {e}", exc_info=True)
        sys.exit(1)


def display_settings(config: Config):
    """Displays the final settings before processing starts (uses Config object)."""
    # Use print for settings display as it's direct user feedback
    print("\n" + "="*30 + " Script Settings " + "="*30)
    print(f"Input folder: {config.absolute_input_dir}")
    print(f"Output folder: {config.absolute_output_dir}")
    print(f"Include subfolders: {'Yes' if config.recursive else 'No'}")
    print(f"Resize mode: {config.resize_mode}")
    if config.resize_mode != 'none':
        resize_opts = config.resize_options
        size_info = []
        if resize_opts.get('width', 0) > 0: size_info.append(f"Width={resize_opts['width']}px")
        if resize_opts.get('height', 0) > 0: size_info.append(f"Height={resize_opts['height']}px")
        # Display size description differently based on mode
        size_desc = ' and/or '.join(size_info) if config.resize_mode == 'aspect_ratio' else f"{resize_opts['width']}x{resize_opts['height']}px"
        print(f"  Target size: {size_desc}{' (maintaining aspect ratio)' if config.resize_mode == 'aspect_ratio' else ''}")
        print(f"  Resize filter: {FILTER_NAMES[resize_opts['filter_str']]}")
    print(f"Output format: {SUPPORTED_OUTPUT_FORMATS[config.output_format]}")
    if 'quality' in config.output_format_options:
        print(f"  Quality: {config.output_format_options['quality']}")
    print(f"Preserve EXIF metadata: Yes (default, where supported by format)")
    print("="*72)
    logger.info("Script settings displayed.")

def print_summary(processed: int, errors: int, error_list: list, skipped_list: list, output_dir: str):
    """Prints the final processing summary."""
    # Use print for summary as it's direct user feedback
    print(f"\n\n--- Processing Summary ---")
    print(f"Successfully processed images: {processed}")
    print(f"Errors encountered: {errors}")

    logger.info(f"Processing summary: Success={processed}, Errors={errors}, Skipped={len(skipped_list)}")

    if error_list:
        print("\n[Files with Errors]")
        # Show max 20 errors in console
        for i, (filepath, errmsg) in enumerate(error_list[:20]):
            print(f"  - {filepath}: {errmsg}")
            # Log all errors regardless
            logger.error(f"File processing error: '{filepath}' - {errmsg}")
        if len(error_list) > 20:
            print(f"  ... and {len(error_list) - 20} more error(s).") # No log file reference needed
            # Log remaining errors
            for i, (filepath, errmsg) in enumerate(error_list[20:]):
                 logger.error(f"File processing error (additional): '{filepath}' - {errmsg}")

    if skipped_list:
        print(f"\nSkipped files/folders during scan ({len(skipped_list)} total):")
        # Show max 10 skipped items in console
        print(f"  {', '.join(skipped_list[:10])}{'...' if len(skipped_list) > 10 else ''}")
        # Log all skipped items
        for skipped_item in skipped_list:
            logger.warning(f"Skipped during scan: {skipped_item}")
    else:
        print("\nNo files or folders were skipped during scan.")
        logger.info("No files/folders skipped during scan.")

    print("\n--- All tasks completed ---")
    print(f"Results saved in: '{output_dir}'")
    logger.info(f"All tasks completed. Results saved in: '{output_dir}'")

# --- Main Execution Logic ---
def main():
    """Main function to orchestrate script execution."""
    try:
        logger.info(f"===== Image Resizer Script v{SCRIPT_VERSION} Started =====")
        # Parse arguments and create Config object (includes validation and option prep)
        config = parse_arguments()

        # Display settings to the user
        display_settings(config)

        # Optional: Add a user confirmation step here if desired
        # proceed = input("Confirm settings and proceed? (y/n): ")
        # if proceed.lower() != 'y':
        #     logger.info("User cancelled operation.")
        #     print("Operation cancelled.")
        #     sys.exit(0)

        # Perform batch image processing
        processed_count, error_count, error_files, skipped_files = batch_process_images(config)

        # Print the final summary
        print_summary(processed_count, error_count, error_files, skipped_files, config.absolute_output_dir)

        logger.info(f"===== Image Resizer Script Finished =====")

        # Set exit code based on errors
        if error_count > 0:
            sys.exit(1) # Exit with error code 1 if errors occurred
        else:
            sys.exit(0) # Exit with code 0 for success

    except SystemExit as e: # Handle explicit sys.exit calls (e.g., from argparse or Config)
        # Log based on exit code (0 is usually success)
        if e.code is None or e.code == 0: # None can happen from parser.error()
             logger.info("Script exited normally (or via parser).")
        else:
             logger.error(f"Script exited with error code {e.code}.")
        raise # Re-raise SystemExit to actually exit
    except Exception as e:
        # Catch any other unexpected exceptions
        logger.critical(f"Unhandled exception occurred: {e}", exc_info=True) # Log stack trace
        print(f"\n(!) Critical Error: {e}")
        # No log file to refer to anymore
        sys.exit(2) # Exit with code 2 for unexpected errors

if __name__ == "__main__":
    main()
