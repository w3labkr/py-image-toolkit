# -*- coding: utf-8 -*-
import os
import sys
import argparse # For command line argument processing
import logging # For logging functionality
from dataclasses import dataclass, field # For configuration class
from typing import Dict, Any, Optional, Tuple # For type hints
import multiprocessing # For parallel processing

# --- Third-party Library Imports ---
from PIL import Image, UnidentifiedImageError
import piexif
from tqdm import tqdm

# --- Logging Setup (Console Output Only) ---
log_console_handler = logging.StreamHandler(sys.stdout)
log_console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logger = logging.getLogger()
logger.setLevel(logging.INFO) # Default log level is INFO. DEBUG messages won't show.
logger.addHandler(log_console_handler)


# --- Constants ---
SCRIPT_VERSION = "3.3" # Version update (Cleaned up unused code/comments)

try:
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST
    }
except AttributeError: # For older Pillow versions
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }

FILTER_NAMES = {
    "lanczos": "LANCZOS (High quality)",
    "bicubic": "BICUBIC (Medium quality)",
    "bilinear": "BILINEAR (Low quality)",
    "nearest": "NEAREST (Lowest quality)"
}

SUPPORTED_OUTPUT_FORMATS = {
    "original": "Keep Original", "png": "PNG", "jpg": "JPG", "webp": "WEBP",
}
SUPPORTED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')


# --- Global Helper for Multiprocessing ---
# This function must be defined at the top level of a module to be picklable by multiprocessing.
def static_process_image_worker(paths_tuple_and_config_tuple: Tuple[Tuple[str, str], 'Config']) -> Tuple[bool, Optional[str], str, str]:
    """
    Worker function for multiprocessing pool.
    Unpacks arguments and calls the main image processing function.
    Args:
        paths_tuple_and_config_tuple: A tuple containing ((input_path, relative_path), config_object)
    Returns:
        A tuple: (success_bool, error_message_or_none, affected_path, original_relative_path)
    """
    paths_tuple, config_obj = paths_tuple_and_config_tuple
    input_path, relative_path = paths_tuple
    return process_single_image_file(input_path, relative_path, config_obj)

# --- Configuration Dataclass ---
@dataclass
class Config:
    """Dataclass to store script configuration."""
    input_dir: str = 'input'
    output_dir_arg: Optional[str] = None
    resize_mode: str = field(default_factory=str)
    output_format: str = field(default_factory=str)
    width: int = 0
    height: int = 0
    filter: Optional[str] = None
    recursive: bool = False
    quality: Optional[int] = None
    verbose: bool = False

    absolute_input_dir: str = field(init=False, default='')
    absolute_output_dir: str = field(init=False, default='')
    resize_options: Dict[str, Any] = field(init=False, default_factory=dict)
    output_format_options: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._validate_paths()
        self._prepare_options()
        # Set logger level based on verbose flag from config
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled via Config.")


    def _validate_paths(self):
        if not os.path.isdir(self.input_dir):
            logger.critical(f"(!) Error: Invalid input folder path: {self.input_dir}")
            if self.input_dir == 'input':
                 logger.info("    Default input folder 'input' not found. Please create it or specify a path.")
            sys.exit(1)

        self.absolute_input_dir = os.path.abspath(self.input_dir)

        if self.output_dir_arg:
            self.absolute_output_dir = os.path.abspath(self.output_dir_arg)
        else:
            self.absolute_output_dir = os.path.join(self.absolute_input_dir, "resized_images")
            logger.info(f"   -> Info: Output folder not specified. Using default: '{self.absolute_output_dir}'")

        if self.absolute_input_dir == self.absolute_output_dir:
            logger.critical("(!) Error: Input folder and output folder cannot be the same.")
            sys.exit(1)

        try:
            # Check if output directory is inside input directory when recursive is on
            rel_path = os.path.relpath(self.absolute_output_dir, start=self.absolute_input_dir)
            if self.recursive and not rel_path.startswith(os.pardir) and rel_path != '.':
                 if os.path.commonpath([self.absolute_input_dir]) == os.path.commonpath([self.absolute_input_dir, self.absolute_output_dir]):
                    logger.critical("(!) Error: When --recursive is used, the output folder cannot be inside the input folder to prevent loops.")
                    sys.exit(1)
        except ValueError: # Paths are on different drives, relpath raises ValueError
            logger.debug("Input/output paths on different drives, skipping containment check.")
            pass

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
        self.resize_options = {'mode': self.resize_mode}
        if self.resize_mode != 'none':
            if not self.filter:
                 logger.critical("(!) Internal Error: Filter required for resize mode but not set.")
                 sys.exit(1)
            self.resize_options.update({
                'width': self.width,
                'height': self.height,
                'filter_str': self.filter,
                'filter_obj': _PIL_RESAMPLE_FILTERS[self.filter]
            })
        logger.debug(f"Resize options set: {self.resize_options}")

        self.output_format_options = {'format_str': self.output_format}
        if self.output_format in ('jpg', 'webp'):
            default_quality = 95 if self.output_format == 'jpg' else 80
            self.output_format_options['quality'] = self.quality if self.quality is not None else default_quality
        logger.debug(f"Output format options set: {self.output_format_options}")
        logger.debug("Environment and options prepared.")

# --- Utility Functions ---
def get_unique_filepath(filepath: str) -> str:
    """Generates a unique filepath by appending a counter if the file already exists."""
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
def resize_image_maintain_aspect_ratio(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    """Resizes an image while maintaining its aspect ratio to fit within target dimensions."""
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0:
        logger.warning(f"Original image dimensions ({original_width}x{original_height}) are invalid. Skipping resize.")
        return img

    new_width, new_height = 0, 0
    if target_width > 0 and target_height > 0: # Fit within both width and height
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(original_width * ratio))
        new_height = max(1, int(original_height * ratio))
    elif target_width > 0: # Fit to width
        ratio = target_width / original_width
        new_width = target_width
        new_height = max(1, int(original_height * ratio))
    elif target_height > 0: # Fit to height
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
        logger.warning(f"Error during aspect ratio resize (({original_width},{original_height}) -> ({new_width},{new_height})): {e}. Using original.")
        return img

def resize_image_fixed_size(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    """Resizes an image to a fixed target width and height, potentially changing aspect ratio."""
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height):
        logger.debug("Target dimensions are same as original. Skipping fixed resize.")
        return img
    if target_width <= 0 or target_height <= 0:
        logger.warning(f"Target dimensions for fixed resize ({target_width}x{target_height}) are invalid. Skipping resize.")
        return img
    try:
        logger.debug(f"Resizing (fixed): ({original_width},{original_height}) -> ({target_width},{target_height})")
        return img.resize((target_width, target_height), resample_filter)
    except ValueError as e:
        logger.warning(f"Error during fixed size resize (({original_width},{original_height}) -> ({target_width},{target_height})): {e}. Using original.")
        return img

def prepare_image_for_save(img: Image.Image, output_format_str: Optional[str]) -> Image.Image:
    """Prepares image mode for saving, e.g., handling transparency for JPG."""
    save_img = img
    original_mode = img.mode
    if output_format_str == 'JPG': # JPG does not support alpha
        if img.mode in ('RGBA', 'LA', 'P'): # Modes that might have alpha or need conversion
            if img.mode == 'P' and 'transparency' in img.info: # Palette mode with transparency
                save_img = img.convert('RGBA') # Convert to RGBA first to handle transparency correctly
                logger.debug("Converted 'P' (with transparency) -> 'RGBA' for JPG saving.")
            elif img.mode == 'P': # Palette mode without transparency info (or not RGBA convertible directly)
                 save_img = img.convert('RGB')
                 logger.debug("Converted 'P' -> 'RGB' for JPG saving.")

            # If now RGBA or LA, composite onto a white background
            if save_img.mode in ('RGBA', 'LA'):
                logger.debug(f"Processing '{save_img.mode}' mode for JPG saving...")
                background = Image.new("RGB", save_img.size, (255, 255, 255)) # White background
                try:
                    mask = save_img.split()[-1] # Get alpha channel as mask
                    background.paste(save_img, mask=mask)
                    save_img = background
                    logger.debug(f"Converted '{original_mode}' -> 'RGB' by merging alpha channel onto white background.")
                except (IndexError, ValueError): # Fallback if alpha channel processing fails
                    save_img = save_img.convert('RGB')
                    logger.warning(f"Error processing alpha channel for JPG. Forcing conversion '{original_mode}' -> 'RGB'.")
    elif output_format_str == 'WEBP': # WEBP supports alpha, but P mode might need conversion
         if img.mode == 'P':
              save_img = img.convert("RGBA") if 'transparency' in img.info else img.convert("RGB")
              logger.debug(f"Converted 'P' -> '{save_img.mode}' for WEBP saving.")

    if save_img.mode != original_mode:
        logger.debug(f"Image mode converted: '{original_mode}' -> '{save_img.mode}' (Output format: {output_format_str or 'Original'})")
    return save_img

def process_single_image_file(input_path: str, relative_path: str, config: Config) -> Tuple[bool, Optional[str], str, str]:
    """
    Processes a single image file: loads, resizes, prepares, and saves it.
    Returns: (success_bool, error_message_or_none, affected_path, original_relative_path).
    'affected_path' is output_path on success, input_path on error.
    'original_relative_path' is the relative_path argument.
    """
    base_name, original_ext = os.path.splitext(os.path.basename(input_path))
    output_format_str = config.output_format_options['format_str']
    logger.debug(f"Processing: '{relative_path}'") # This log will appear from worker.

    output_ext_map = {'jpg': '.jpg', 'webp': '.webp', 'png': '.png'}
    output_ext = output_ext_map.get(output_format_str.lower(), original_ext) # Keep original if not specified

    output_relative_dir = os.path.dirname(relative_path)
    output_dir_for_file = os.path.join(config.absolute_output_dir, output_relative_dir)

    if not os.path.isdir(output_dir_for_file):
         logger.error(f"Internal Error: Output subdirectory does not exist: '{output_dir_for_file}'")
         try:
             os.makedirs(output_dir_for_file)
             logger.warning(f"Warning: Created missing output subdirectory: '{output_dir_for_file}'")
         except OSError as e:
              error_msg = f"Failed to create output subdirectory: {output_dir_for_file} ({e})"
              logger.error(error_msg) # Logged by worker
              return False, error_msg, input_path, relative_path

    output_filename = base_name + output_ext
    output_path_base = os.path.join(output_dir_for_file, output_filename)
    output_path = get_unique_filepath(output_path_base)

    original_exif_bytes = None
    exif_data = None

    try:
        with Image.open(input_path) as img:
            logger.debug(f"Image loaded: '{relative_path}' (Size: {img.size}, Mode: {img.mode})")
            if 'exif' in img.info and img.info['exif']:
                original_exif_bytes = img.info['exif']
                try:
                    exif_data = piexif.load(original_exif_bytes)
                    logger.debug(f"EXIF data loaded successfully: '{relative_path}'")
                except Exception as e: # piexif can raise various errors
                    logger.warning(f"Failed to parse EXIF for '{relative_path}' ({type(e).__name__}). Attempting to keep original bytes.")
                    exif_data = None # Ensure exif_data is None if parsing fails

            save_format_upper = output_format_str.upper() if output_format_str != 'original' else None
            img_prepared = prepare_image_for_save(img, save_format_upper)

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
            # No 'else' needed as img_resized is already img_prepared if mode is 'none'

            save_kwargs = {}
            save_format_arg = save_format_upper if output_format_str != 'original' else None
            if save_format_arg == 'JPG': save_format_arg = 'JPEG' # Pillow uses 'JPEG'

            final_exif_bytes = None
            if output_format_str != 'original': # Only modify/process EXIF if format is changing
                if exif_data:
                    try:
                        # Reset Orientation to 1 (Normal) to avoid auto-rotation issues by viewers
                        if piexif.ImageIFD.Orientation in exif_data.get('0th', {}):
                            exif_data['0th'][piexif.ImageIFD.Orientation] = 1
                            logger.debug("Reset EXIF Orientation tag to 1.")
                        # Remove thumbnail to save space and prevent outdated thumbnails
                        if 'thumbnail' in exif_data and exif_data['thumbnail']:
                             exif_data['thumbnail'] = None
                             logger.debug("Removed EXIF thumbnail data.")
                        final_exif_bytes = piexif.dump(exif_data)
                        logger.debug("Successfully dumped modified EXIF data.")
                    except Exception as e:
                        logger.warning(f"Failed to dump modified EXIF for '{relative_path}' ({type(e).__name__}). Attempting to use original EXIF.")
                        final_exif_bytes = original_exif_bytes # Fallback to original if modification fails
                elif original_exif_bytes: # If parsing failed but we have original bytes
                     final_exif_bytes = original_exif_bytes
                     logger.debug("Using original EXIF bytes as parsing failed but bytes are available.")
            else: # Keep original EXIF if format is 'original'
                 final_exif_bytes = original_exif_bytes
                 if final_exif_bytes: logger.debug("Passing original EXIF data for original format.")

            output_opts = config.output_format_options
            if output_format_str == 'jpg':
                quality = output_opts.get('quality', 95)
                save_kwargs.update({'quality': quality, 'optimize': True, 'progressive': True})
                if final_exif_bytes: save_kwargs['exif'] = final_exif_bytes
            elif output_format_str == 'png':
                save_kwargs['optimize'] = True
                if final_exif_bytes:
                     try: save_kwargs['exif'] = final_exif_bytes
                     except TypeError: logger.warning(f"'{relative_path}': Pillow version might not support EXIF for PNG.")
            elif output_format_str == 'webp':
                quality = output_opts.get('quality', 80)
                save_kwargs.update({'quality': quality, 'lossless': False}) # Default to lossy WEBP
                if final_exif_bytes: save_kwargs['exif'] = final_exif_bytes
            elif output_format_str == 'original' and final_exif_bytes:
                # Only add EXIF if the original format supports it (common ones)
                if original_ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.webp']:
                     save_kwargs['exif'] = final_exif_bytes

            logger.debug(f"Attempting to save image: '{output_path}' (Format: {save_format_arg or 'Original'})")
            img_resized.save(output_path, format=save_format_arg, **save_kwargs)
            logger.debug(f"Image saved successfully: '{output_path}'")
            return True, None, output_path, relative_path

    except UnidentifiedImageError:
        msg = "Invalid or corrupted image file"
        logger.error(f"Processing failed: '{relative_path}' - {msg}") # Logged by worker
        return False, msg, input_path, relative_path
    except PermissionError:
        msg = "File read/write permission denied"
        logger.error(f"Processing failed: '{relative_path}' - {msg}") # Logged by worker
        return False, msg, input_path, relative_path
    except OSError as e: # Covers various file system errors
        msg = f"File system error ({e})"
        logger.error(f"Processing failed: '{relative_path}' - {msg}") # Logged by worker
        if os.path.exists(output_path): # Attempt to clean up partially created file
            try: os.remove(output_path); logger.warning(f"Removed partially created file after error: '{output_path}'")
            except OSError: pass
        return False, msg, input_path, relative_path
    except ValueError as e: # e.g. from Pillow operations
         msg = f"Image processing value error ({e})"
         logger.error(f"Processing failed: '{relative_path}' - {msg}") # Logged by worker
         return False, msg, input_path, relative_path
    except Exception as e: # Catch-all for unexpected errors
        msg = f"Unexpected error ({type(e).__name__}: {e})"
        logger.critical(f"Processing failed: '{relative_path}' - {msg}", exc_info=True) # Log with stack trace
        return False, msg, input_path, relative_path

def scan_for_image_files(config: Config) -> Tuple[list[Tuple[str, str]], list[str]]:
    """Scans the input directory for image files based on configuration."""
    files_to_process = []
    skipped_scan_files = []
    logger.info(f"Scanning input folder: '{config.absolute_input_dir}' (Recursive: {config.recursive})")

    if config.recursive:
        for root, dirs, files in os.walk(config.absolute_input_dir):
            # Prevent recursion into the output directory if it's inside the input directory
            if os.path.abspath(root).startswith(config.absolute_output_dir):
                skipped_msg = os.path.relpath(root, config.absolute_input_dir) + " (Output folder subdirectory - skipped)"
                logger.debug(f"Skipping scan: '{skipped_msg}'")
                skipped_scan_files.append(skipped_msg)
                dirs[:] = [] # Don't traverse into this directory further
                continue
            for filename in files:
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, config.absolute_input_dir)
                if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                    files_to_process.append((input_path, relative_path))
                else:
                    skipped_scan_files.append(relative_path + " (Unsupported format)")
    else: # Non-recursive scan
        for filename in os.listdir(config.absolute_input_dir):
            input_path = os.path.join(config.absolute_input_dir, filename)
            if os.path.isfile(input_path):
                if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                    files_to_process.append((input_path, filename)) # relative_path is just filename
                else:
                    skipped_scan_files.append(filename + " (Unsupported format)")
            elif os.path.isdir(input_path) and os.path.abspath(input_path) == config.absolute_output_dir:
                skipped_scan_files.append(filename + " (Output folder)")
            elif os.path.isdir(input_path):
                skipped_scan_files.append(filename + " (Directory - non-recursive)")


    logger.info(f"Scan complete: Found {len(files_to_process)} files to process, skipped {len(skipped_scan_files)} items.")
    return files_to_process, skipped_scan_files

def batch_process_images(config: Config) -> Tuple[int, int, list[Tuple[str, str]], list[str]]:
    """Manages the batch processing of images using a multiprocessing pool."""
    processed_count = 0
    error_count = 0
    error_files = []
    all_skipped_files = []

    logger.info(f"Starting batch processing. Output folder: '{config.absolute_output_dir}'")

    try:
        files_to_process, skipped_scan_files = scan_for_image_files(config)
        all_skipped_files.extend(skipped_scan_files)
        total_files = len(files_to_process)
        if total_files == 0:
             logger.warning("(!) No image files found to process in the specified path.")
             return 0, 0, [], all_skipped_files
        else:
            logger.info(f"-> Found {total_files} image files. Starting processing.")
    except Exception as e:
        logger.critical(f"(!) Critical Error: Failed to access input path '{config.absolute_input_dir}'. ({e})", exc_info=True)
        return 0, 0, [], all_skipped_files

    # Pre-create necessary output subdirectories
    required_subdirs = set()
    for _, relative_path in files_to_process:
        output_relative_dir = os.path.dirname(relative_path)
        if output_relative_dir: required_subdirs.add(output_relative_dir)

    for subdir in required_subdirs:
        output_dir_to_create = os.path.join(config.absolute_output_dir, subdir)
        if not os.path.exists(output_dir_to_create):
            try:
                os.makedirs(output_dir_to_create)
                logger.info(f"Created output subdirectory: '{output_dir_to_create}'")
            except OSError as e:
                logger.error(f"Error: Failed to create output subdirectory '{output_dir_to_create}' ({e}). Files in this directory may fail.")

    # --- Image Processing Loop with Multiprocessing ---
    num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} worker processes for image processing.")

    tasks_with_config = [((input_p, rel_p), config) for input_p, rel_p in files_to_process]

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm for overall progress bar
        with tqdm(total=total_files, desc="Processing images", unit="file", ncols=100, leave=True) as pbar:
            # imap_unordered gets results as they complete, good for progress bar updates
            for res_success, res_error_msg, res_affected_path, res_relative_path in pool.imap_unordered(static_process_image_worker, tasks_with_config):
                if res_success:
                    processed_count += 1
                else:
                    error_count += 1
                    # Use tqdm.write for immediate feedback for errors in the main process console
                    tqdm.write(f" ✗ Error: '{os.path.basename(res_affected_path)}' (in '{os.path.dirname(res_relative_path)}' folder) - {res_error_msg}")
                    error_files.append((res_relative_path, res_error_msg or "Unknown error"))
                pbar.update(1) # Update progress bar for each completed task

    logger.info(f"Batch processing complete. Success: {processed_count}, Errors: {error_count}")
    return processed_count, error_count, error_files, all_skipped_files

# --- Argument Parsing and Setup ---
def parse_arguments() -> Config:
    """Parses command line arguments and returns a Config object."""
    parser = argparse.ArgumentParser(
        description=f"Batch Image Resizer Script (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawTextHelpFormatter, # Allows for newlines in help text
        usage="%(prog)s [input_dir] -m <mode> -O <format> [options]"
    )
    parser.add_argument("input_dir", nargs='?', default='input',
                        help="Path to the source folder (default: 'input')")

    required_group = parser.add_argument_group('Required Options')
    required_group.add_argument("-m", "--resize-mode", required=True, choices=['aspect_ratio', 'fixed', 'none'],
                                help="Resize mode:\n"
                                     "  aspect_ratio: Maintain aspect ratio to fit target W/H\n"
                                     "  fixed: Force resize to target WxH (may distort)\n"
                                     "  none: No resizing (only format conversion/EXIF handling)")
    required_group.add_argument("-O", "--output-format", required=True, choices=SUPPORTED_OUTPUT_FORMATS.keys(),
                                help="Output file format:\n" +
                                     "\n".join([f"  {k}: {v}" for k, v in SUPPORTED_OUTPUT_FORMATS.items()]))

    parser.add_argument("-o", "--output-dir", dest='output_dir_arg',
                        help="Path to save results (default: 'resized_images' under input folder)")

    resize_group = parser.add_argument_group('Resize Options (ignored if mode is "none")')
    resize_group.add_argument("-w", "--width", type=int, default=0, help="Target width (pixels)")
    resize_group.add_argument("-H", "--height", type=int, default=0, help="Target height (pixels)")
    resize_group.add_argument("-f", "--filter", choices=FILTER_NAMES.keys(), default=None, # Default set to None, validated later
                              help="Resize filter (Required if resizing):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]))

    optional_group = parser.add_argument_group('Other Optional Options')
    optional_group.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    optional_group.add_argument("-q", "--quality", type=int, default=None, # Default None, applied if relevant
                                help="JPG/WEBP quality (1-100, higher is better). Default: JPG=95, WEBP=80")
    optional_group.add_argument('--version', action='version', version=f'%(prog)s {SCRIPT_VERSION}')
    optional_group.add_argument("-v", "--verbose", action="store_true",
                                help="Enable verbose (DEBUG level) logging")

    args = parser.parse_args()

    # Validate resize options based on mode
    if args.resize_mode == 'aspect_ratio':
        if args.width <= 0 and args.height <= 0:
            parser.error("Mode 'aspect_ratio' requires --width or --height (or both) to be > 0.")
        if args.width < 0 or args.height < 0:
            parser.error("--width and --height cannot be negative.")
        if not args.filter:
            parser.error("--filter is required for resize mode 'aspect_ratio'.")
    elif args.resize_mode == 'fixed':
        if args.width <= 0 or args.height <= 0:
            parser.error("Mode 'fixed' requires both --width and --height to be > 0.")
        if not args.filter:
            parser.error("--filter is required for resize mode 'fixed'.")
    elif args.resize_mode == 'none' and (args.width > 0 or args.height > 0 or args.filter):
        logger.warning("   -> Info: --width, --height, and --filter are ignored when --resize-mode is 'none'.")

    # Validate quality
    if args.output_format in ('jpg', 'webp') and args.quality is not None:
        if not (1 <= args.quality <= 100):
            parser.error(f"--quality must be between 1 and 100 (got {args.quality}).")
    elif args.quality is not None and args.output_format not in ('jpg', 'webp'):
        logger.warning(f"   -> Warning: --quality is ignored for output format '{args.output_format}'.")

    try:
        return Config(**vars(args))
    except SystemExit: # Raised by parser.error()
        raise
    except Exception as e: # Catch any other error during Config creation
        logger.critical(f"(!) Error during configuration object creation: {e}", exc_info=True)
        sys.exit(1)

def display_settings(config: Config):
    """Prints the effective script settings to the console."""
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
        size_desc = ' and/or '.join(size_info) if config.resize_mode == 'aspect_ratio' else f"{resize_opts['width']}x{resize_opts['height']}px"
        print(f"  Target size: {size_desc}{' (maintaining aspect ratio)' if config.resize_mode == 'aspect_ratio' else ''}")
        print(f"  Resize filter: {FILTER_NAMES[resize_opts['filter_str']]}")
    print(f"Output format: {SUPPORTED_OUTPUT_FORMATS[config.output_format]}")
    if 'quality' in config.output_format_options:
        print(f"  Quality: {config.output_format_options['quality']}")
    print(f"Preserve EXIF metadata: Yes (default, where supported by format)")
    print(f"Log Level: {logging.getLevelName(logger.getEffectiveLevel())}")
    print("="*72)

def print_summary(processed: int, errors: int, error_list: list, skipped_list: list, output_dir: str):
    """Prints a summary of the processing results."""
    print(f"\n\n--- Processing Summary ---")
    print(f"Successfully processed images: {processed}")
    print(f"Errors encountered: {errors}")
    if error_list:
        print("\n[Files with Errors]")
        for i, (filepath, errmsg) in enumerate(error_list[:20]): # Show first 20 errors
            print(f"  - {filepath}: {errmsg}")
        if len(error_list) > 20:
            print(f"  ... and {len(error_list) - 20} more error(s).")
    if skipped_list:
        print(f"\nSkipped files/folders during scan ({len(skipped_list)} total):")
        display_skipped = skipped_list[:5] # Show first 5 skipped
        print(f"  {', '.join(display_skipped)}{'...' if len(skipped_list) > 5 else ''}")
        if logger.getEffectiveLevel() <= logging.DEBUG and skipped_list: # Show all if verbose
            print("  Full list of skipped items (debug mode):")
            for item in skipped_list: print(f"    - {item}")
    else:
        print("\nNo files or folders were skipped during scan.")
    print("\n--- All tasks completed ---")
    print(f"Results saved in: '{output_dir}'")

# --- Main Execution Logic ---
def main():
    """Main function to orchestrate script execution."""
    try:
        config = parse_arguments()

        logger.info(f"===== Image Resizer Script v{SCRIPT_VERSION} Started =====")
        display_settings(config)

        processed_count, error_count, error_files, skipped_files = batch_process_images(config)
        print_summary(processed_count, error_count, error_files, skipped_files, config.absolute_output_dir)

        logger.info(f"===== Image Resizer Script Finished =====")
        sys.exit(1 if error_count > 0 else 0) # Exit with 1 if errors occurred
    except SystemExit as e: # Handle sys.exit() calls, e.g., from argparse or explicit exits
        if e.code is None or e.code == 0:
            logger.info("Script exited normally.")
        else:
            logger.error(f"Script exited with error code {e.code}.")
        # No re-raise needed as sys.exit() already handles program termination.
    except Exception as e: # Catch any other unhandled exceptions
        logger.critical(f"Unhandled exception occurred: {e}", exc_info=True)
        print(f"\n(!) Critical Error: {e}") # Also print to console for visibility
        sys.exit(2) # General error exit code

if __name__ == "__main__":
    # This check is crucial for multiprocessing on Windows and for PyInstaller.
    # It prevents child processes from re-executing the main script logic.
    multiprocessing.freeze_support()
    main()
