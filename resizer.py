# -*- coding: utf-8 -*-
import os
import sys
import argparse # For command line argument processing
import logging # For logging functionality
from dataclasses import dataclass, field # For configuration class
from typing import Dict, Any, Optional, Tuple, List, ClassVar # For type hints
import multiprocessing # For parallel processing
from PIL import Image, UnidentifiedImageError
import piexif
from tqdm import tqdm

# --- Logging Setup ---
# Define detailed log format (timestamp, log level, process name, message)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create console handler (using default stderr)
log_handler = logging.StreamHandler() # Changed from sys.stdout
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
__version__ = "3.13" # Script version (Use stderr for Logging)

# These constants remain global as they are widely used and define capabilities
try:
    # Pillow >= 9.1.0 uses Resampling enums
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
    # --- User configurable via args ---
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
    strip_exif: bool = False
    overwrite_policy: str = "rename"
    include_extensions: Optional[List[str]] = None
    exclude_extensions: Optional[List[str]] = None
    webp_lossless: bool = False

    # --- Internal or derived ---
    absolute_input_dir: str = field(init=False, default='')
    absolute_output_dir: str = field(init=False, default='')
    resize_options: Dict[str, Any] = field(init=False, default_factory=dict)
    output_format_options: Dict[str, Any] = field(init=False, default_factory=dict)

    # --- Defaults moved into Config ---
    default_jpg_quality: int = 95
    default_webp_quality: int = 80 # Default for lossy WEBP
    default_supported_input_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    
    # --- Constants moved into Config ---
    filter_names: Dict[str, str] = field(default_factory=lambda: {
        "lanczos": "LANCZOS (High quality)",
        "bicubic": "BICUBIC (Medium quality)",
        "bilinear": "BILINEAR (Low quality)",
        "nearest": "NEAREST (Lowest quality)"
    })
    supported_output_formats: Dict[str, str] = field(default_factory=lambda: {
        "original": "Keep Original", "png": "PNG", "jpg": "JPG", "webp": "WEBP",
    })
    skipped_existing_msg: str = "SKIPPED_EXISTING_FILE_POLICY"


    def __post_init__(self):
        self._validate_paths()
        self._prepare_options()
        self._normalize_extension_filters()
        if self.verbose:
            logger.setLevel(logging.DEBUG) # Change logger level to DEBUG
            logger.debug("Verbose logging enabled via Config.")

    def _normalize_extension_filters(self):
        """Normalizes include/exclude extensions to lowercase and ensures they start with a dot."""
        if self.include_extensions:
            self.include_extensions = [f".{ext.lower().lstrip('.')}" for ext in self.include_extensions]
            logger.debug(f"Normalized include_extensions: {self.include_extensions}")
        if self.exclude_extensions:
            self.exclude_extensions = [f".{ext.lower().lstrip('.')}" for ext in self.exclude_extensions]
            logger.debug(f"Normalized exclude_extensions: {self.exclude_extensions}")


    def _validate_paths(self):
        """Validates input and output paths."""
        if not os.path.isdir(self.input_dir):
            logger.critical(f"(!) Error: Invalid input folder path: {self.input_dir}")
            if self.input_dir == 'input': # Default input folder name check
                 logger.info(f"    Default input folder '{self.input_dir}' not found. Please create it or specify a path.")
            sys.exit(1)

        self.absolute_input_dir = os.path.abspath(self.input_dir)

        if self.output_dir_arg:
            self.absolute_output_dir = os.path.abspath(self.output_dir_arg)
        else:
            # Use direct string for default output subdir name
            self.absolute_output_dir = os.path.join(self.absolute_input_dir, "resized_images")
            logger.info(f"   -> Info: Output folder not specified. Using default: '{self.absolute_output_dir}'")

        if self.absolute_input_dir == self.absolute_output_dir:
            logger.critical("(!) Error: Input folder and output folder cannot be the same.")
            sys.exit(1)

        # Prevent processing loops when recursive is enabled and output is inside input
        try:
            rel_path = os.path.relpath(self.absolute_output_dir, start=self.absolute_input_dir)
            # Check if output is a direct or indirect subdirectory of input
            if self.recursive and not rel_path.startswith(os.pardir) and rel_path != '.':
                 # A more robust check using commonpath
                 if os.path.commonpath([self.absolute_input_dir]) == os.path.commonpath([self.absolute_input_dir, self.absolute_output_dir]):
                    logger.critical(f"(!) Error: When --recursive is used, the output folder ('{self.absolute_output_dir}') cannot be inside the input folder ('{self.absolute_input_dir}') to prevent processing loops.")
                    sys.exit(1)
        except ValueError: # Paths are on different drives, relpath raises ValueError
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
        """Prepares resize and output format options based on configuration."""
        # Prepare resize options
        self.resize_options = {'mode': self.resize_mode}
        if self.resize_mode != 'none':
            if not self.filter: # Should be caught by argparse, but as a safeguard
                 logger.critical("(!) Internal Error: Filter required for resize mode but not set.")
                 sys.exit(1)
            self.resize_options.update({
                'width': self.width,
                'height': self.height,
                'filter_str': self.filter,
                'filter_obj': _PIL_RESAMPLE_FILTERS[self.filter] # Global filter map still used here
            })
        logger.debug(f"Resize options prepared: {self.resize_options}")

        # Prepare output format options
        self.output_format_options = {'format_str': self.output_format}
        if self.output_format == 'jpg':
            # Use quality from args if provided, otherwise use default from Config
            self.output_format_options['quality'] = self.quality if self.quality is not None else self.default_jpg_quality
        elif self.output_format == 'webp':
            # Use quality from args if provided, otherwise use default from Config
            self.output_format_options['quality'] = self.quality if self.quality is not None else self.default_webp_quality
            self.output_format_options['lossless'] = self.webp_lossless

        logger.debug(f"Output format options prepared: {self.output_format_options}")
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
    else: # Should not happen if argparse validation is correct
        logger.warning("Target dimensions for aspect ratio resize not specified. Returning original.")
        return img

    if (new_width, new_height) == (original_width, original_height):
        logger.debug("Calculated resize dimensions are same as original. Skipping resize.")
        return img
    try:
        logger.debug(f"Resizing (aspect ratio): ({original_width},{original_height}) -> ({new_width},{new_height})")
        return img.resize((new_width, new_height), resample_filter)
    except ValueError as e: # e.g., if new_width or new_height is zero after calculation (though max(1,...) should prevent)
        logger.warning(f"Error during aspect ratio resize (({original_width},{original_height}) -> ({new_width},{new_height})): {e}. Using original.")
        return img

def resize_image_fixed_size(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    """Resizes an image to a fixed target width and height, potentially changing aspect ratio."""
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height):
        logger.debug("Target dimensions are same as original. Skipping fixed resize.")
        return img
    if target_width <= 0 or target_height <= 0: # Should be caught by argparse
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
    # Handle JPG conversion: ensure RGB mode, composite alpha if present
    if output_format_str == 'JPG':
        if img.mode in ('RGBA', 'LA', 'P'): # Modes that might have alpha or need conversion
            # Convert Palette to RGBA/RGB first if necessary
            if img.mode == 'P':
                save_img = img.convert('RGBA') if 'transparency' in img.info else img.convert('RGB')
                logger.debug(f"Converted 'P' (Palette) mode to '{save_img.mode}' for JPG preparation.")
            
            # If image has alpha channel (RGBA or LA after potential P conversion)
            if save_img.mode in ('RGBA', 'LA'):
                logger.debug(f"Processing '{save_img.mode}' mode for JPG saving (requires alpha compositing)...")
                # Create a new RGB image with a white background
                background = Image.new("RGB", save_img.size, (255, 255, 255)) # White background
                try:
                    # Paste the image onto the background using its alpha channel as a mask
                    mask = save_img.split()[-1] # Get the alpha channel
                    background.paste(save_img, (0,0), mask=mask)
                    save_img = background
                    logger.debug(f"Converted '{original_mode}' to 'RGB' by merging alpha channel onto white background.")
                except (IndexError, ValueError) as e: # Fallback if alpha processing fails
                    logger.warning(f"Error processing alpha channel for JPG conversion ('{original_mode}' -> 'RGB'): {e}. Forcing direct conversion.")
                    save_img = save_img.convert('RGB') # Force conversion to RGB
            elif save_img.mode != 'RGB': # If it was 'P' without transparency and converted to something other than RGB
                save_img = save_img.convert('RGB')
                logger.debug(f"Converted '{original_mode}' to 'RGB' for JPG saving.")

    # Handle WEBP conversion: P mode might need conversion for alpha
    elif output_format_str == 'WEBP':
         if img.mode == 'P':
              save_img = img.convert("RGBA") if 'transparency' in img.info else img.convert("RGB")
              logger.debug(f"Converted 'P' (Palette) mode to '{save_img.mode}' for WEBP saving.")

    if save_img.mode != original_mode:
        logger.debug(f"Image mode converted: '{original_mode}' -> '{save_img.mode}' (Output format: {output_format_str or 'Original'})")
    return save_img

def process_single_image_file(input_path: str, relative_path: str, config: Config) -> Tuple[bool, Optional[str], str, str]:
    """
    Processes a single image file: loads, resizes, prepares, and saves it.
    Returns: (success_bool, error_message_or_none, affected_path, original_relative_path).
    'affected_path' is output_path on success, input_path on error or skip.
    'original_relative_path' is the relative_path argument.
    """
    base_name, original_ext = os.path.splitext(os.path.basename(input_path))
    output_format_str = config.output_format_options['format_str']
    logger.debug(f"Processing: '{relative_path}'")

    output_ext_map = {'jpg': '.jpg', 'webp': '.webp', 'png': '.png'}
    output_ext = output_ext_map.get(output_format_str.lower(), original_ext)

    output_relative_dir = os.path.dirname(relative_path)
    output_dir_for_file = os.path.join(config.absolute_output_dir, output_relative_dir)

    # Ensure output subdirectory exists
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
    
    output_path = output_path_base # Default output path
    # Handle overwrite policy
    if os.path.exists(output_path_base):
        if config.overwrite_policy == "skip":
            logger.info(f"Skipping existing file due to policy: '{relative_path}' (target: '{output_path_base}')")
            # Use the skipped message from the config object
            return True, config.skipped_existing_msg, input_path, relative_path 
        elif config.overwrite_policy == "rename":
            output_path = get_unique_filepath(output_path_base)
            if output_path != output_path_base: # If a new name was generated
                 logger.info(f"Output file '{os.path.basename(output_path_base)}' exists. Renaming to '{os.path.basename(output_path)}' due to policy.")
        elif config.overwrite_policy == "overwrite":
            logger.info(f"Overwriting existing file due to policy: '{output_path_base}'")
        # If 'overwrite', output_path remains output_path_base, no specific log here, done before save attempt.

    original_exif_bytes = None
    exif_data = None # Parsed EXIF data dictionary

    try:
        with Image.open(input_path) as img:
            logger.debug(f"Image loaded: '{relative_path}' (Size: {img.size}, Mode: {img.mode})")
            
            # Load original EXIF bytes if not stripping and EXIF exists
            if not config.strip_exif and 'exif' in img.info and img.info['exif']:
                original_exif_bytes = img.info['exif']
                try:
                    exif_data = piexif.load(original_exif_bytes) # Try to parse it
                    logger.debug(f"EXIF data loaded and parsed successfully for '{relative_path}'")
                except Exception as e: 
                    logger.warning(f"Failed to parse EXIF for '{relative_path}' ({type(e).__name__}). Will attempt to keep original EXIF bytes if possible.")
                    exif_data = None # Ensure exif_data is None if parsing fails
            elif config.strip_exif:
                logger.debug(f"EXIF stripping enabled for '{relative_path}'. All EXIF data will be removed.")
            # If strip_exif is False but no EXIF in img.info, original_exif_bytes and exif_data remain None

            # Prepare image (mode conversion, alpha handling)
            save_format_upper = output_format_str.upper() if output_format_str != 'original' else None
            img_prepared = prepare_image_for_save(img, save_format_upper)

            # Resize image if needed
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

            # Prepare save arguments (format, quality, EXIF)
            save_kwargs = {}
            save_format_arg = save_format_upper if output_format_str != 'original' else None
            if save_format_arg == 'JPG': save_format_arg = 'JPEG' # Pillow uses 'JPEG'

            # Determine final EXIF bytes to save
            final_exif_bytes = None
            if not config.strip_exif:
                if output_format_str == 'original':
                    final_exif_bytes = original_exif_bytes # Keep original EXIF as is
                    if final_exif_bytes: logger.debug(f"Passing original EXIF data for '{relative_path}' (original format).")
                else: # Output format is changing, so EXIF might be modified
                    if exif_data: # EXIF was successfully parsed
                        try:
                            # Reset Orientation to 1 (Normal)
                            if piexif.ImageIFD.Orientation in exif_data.get('0th', {}):
                                exif_data['0th'][piexif.ImageIFD.Orientation] = 1
                                logger.debug(f"Reset EXIF Orientation tag to 1 for '{relative_path}'.")
                            # Remove thumbnail
                            if 'thumbnail' in exif_data and exif_data['thumbnail'] is not None:
                                 exif_data['thumbnail'] = None
                                 logger.debug(f"Removed EXIF thumbnail data for '{relative_path}'.")
                            final_exif_bytes = piexif.dump(exif_data)
                            logger.debug(f"Successfully prepared modified EXIF data for '{relative_path}'.")
                        except Exception as e:
                            logger.warning(f"Failed to dump modified EXIF for '{relative_path}' ({type(e).__name__}). Attempting to use original EXIF bytes.")
                            final_exif_bytes = original_exif_bytes # Fallback to original bytes
                    elif original_exif_bytes: # EXIF parsing failed, but we have the original bytes
                         final_exif_bytes = original_exif_bytes
                         logger.debug(f"Using original (unparsed) EXIF bytes for '{relative_path}' as parsing failed.")
            # If config.strip_exif is True, final_exif_bytes remains None

            # Add EXIF to save_kwargs if it's determined and applicable
            if final_exif_bytes:
                # Check format compatibility for EXIF (Pillow specific)
                if output_format_str == 'jpg' or \
                   (output_format_str == 'original' and original_ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.webp']) or \
                   output_format_str == 'webp':
                    save_kwargs['exif'] = final_exif_bytes
                elif output_format_str == 'png':
                    try:
                        save_kwargs['exif'] = final_exif_bytes
                    except TypeError: # Some Pillow versions might not support EXIF for PNG with piexif bytes
                        logger.warning(f"'{relative_path}': Pillow version might not support EXIF for PNG or EXIF data is incompatible.")
            
            # Set quality and other format-specific options
            output_opts = config.output_format_options # Contains quality and lossless settings
            if output_format_str == 'jpg':
                save_kwargs.update({
                    'quality': output_opts.get('quality', config.default_jpg_quality), # Use default from config
                    'optimize': True,
                    'progressive': True
                })
            elif output_format_str == 'png':
                save_kwargs['optimize'] = True # PNG optimization
            elif output_format_str == 'webp':
                save_kwargs.update({
                    'quality': output_opts.get('quality', config.default_webp_quality), # Use default from config
                    'lossless': output_opts.get('lossless', False) # Get from config
                })

            logger.debug(f"Attempting to save image: '{output_path}' (Format: {save_format_arg or 'Original'}) with kwargs: {save_kwargs.keys()}")
            img_resized.save(output_path, format=save_format_arg, **save_kwargs)
            logger.debug(f"Image saved successfully: '{output_path}'")
            return True, None, output_path, relative_path

    # Exception handling remains the same
    except UnidentifiedImageError:
        msg = "Invalid or corrupted image file"
        logger.error(f"Processing failed: '{relative_path}' - {msg}")
        return False, msg, input_path, relative_path
    except PermissionError:
        msg = "File read/write permission denied"
        logger.error(f"Processing failed: '{relative_path}' - {msg}")
        return False, msg, input_path, relative_path
    except OSError as e: 
        msg = f"File system error ({e})"
        logger.error(f"Processing failed: '{relative_path}' - {msg}")
        if os.path.exists(output_path) and output_path != input_path : # Avoid deleting source on error
            try: os.remove(output_path); logger.warning(f"Removed partially created file after error: '{output_path}'")
            except OSError: pass # Ignore if removal fails
        return False, msg, input_path, relative_path
    except ValueError as e: # e.g. from Pillow operations
         msg = f"Image processing value error ({e})"
         logger.error(f"Processing failed: '{relative_path}' - {msg}")
         return False, msg, input_path, relative_path
    except Exception as e: # Catch-all for unexpected errors
        msg = f"Unexpected error ({type(e).__name__}: {e})"
        logger.critical(f"Processing failed: '{relative_path}' - {msg}", exc_info=True) # Log with stack trace
        return False, msg, input_path, relative_path

def scan_for_image_files(config: Config) -> Tuple[list[Tuple[str, str]], list[str]]:
    """Scans the input directory for image files based on configuration."""
    files_to_process = []
    skipped_scan_files = [] # Stores reasons for skipping
    logger.info(f"Scanning input folder: '{config.absolute_input_dir}' (Recursive: {config.recursive})")

    items_to_scan = [] # List of (absolute_path, relative_path)
    if config.recursive:
        for root, dirs, files in os.walk(config.absolute_input_dir):
            # Prevent recursion into the output directory if it's inside the input directory
            if os.path.abspath(root).startswith(config.absolute_output_dir):
                skipped_msg = os.path.relpath(root, config.absolute_input_dir) + " (Output folder subdirectory - skipped)"
                logger.debug(f"Skipping scan of directory: '{skipped_msg}'")
                skipped_scan_files.append(skipped_msg)
                dirs[:] = [] # Don't traverse into this directory further
                continue
            for filename in files:
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, config.absolute_input_dir)
                items_to_scan.append((abs_path, rel_path))
    else: # Non-recursive scan
        for filename in os.listdir(config.absolute_input_dir):
            abs_path = os.path.join(config.absolute_input_dir, filename)
            if os.path.isfile(abs_path):
                 items_to_scan.append((abs_path, filename)) # relative_path is just filename
            elif os.path.isdir(abs_path):
                # Skip if it's the output directory itself or any other directory in non-recursive mode
                if os.path.abspath(abs_path) == config.absolute_output_dir:
                    skipped_scan_files.append(filename + " (Is the output folder)")
                else:
                    skipped_scan_files.append(filename + " (Directory - non-recursive mode)")
    
    for input_path, relative_path in items_to_scan:
        filename = os.path.basename(input_path)
        file_ext = os.path.splitext(filename)[1].lower()

        if not file_ext: # Skip files with no extension
            skipped_scan_files.append(relative_path + " (No extension)")
            continue

        # Determine if the file should be processed based on extension filters
        should_process_based_on_extension = False
        # Use default extensions from Config object
        supported_extensions = config.default_supported_input_extensions 
        
        if config.include_extensions: # --include-extensions takes precedence
            if file_ext in config.include_extensions:
                should_process_based_on_extension = True
            else:
                skipped_scan_files.append(relative_path + f" (Extension '{file_ext}' not in --include-extensions list)")
        elif file_ext in supported_extensions: # No --include, use default from Config
            should_process_based_on_extension = True
        else: # Not in default list and no --include
            skipped_scan_files.append(relative_path + f" (Extension '{file_ext}' not in default supported list)")

        # Apply --exclude-extensions if the file was otherwise going to be processed
        if should_process_based_on_extension and config.exclude_extensions and file_ext in config.exclude_extensions:
            should_process_based_on_extension = False
            skipped_scan_files.append(relative_path + f" (Extension '{file_ext}' in --exclude-extensions list)")
        
        if should_process_based_on_extension:
            files_to_process.append((input_path, relative_path))

    logger.info(f"Scan complete: Found {len(files_to_process)} files to process, skipped {len(skipped_scan_files)} items.")
    if logger.getEffectiveLevel() <= logging.DEBUG and skipped_scan_files:
        logger.debug("Skipped items during scan:")
        for item_reason in skipped_scan_files:
            logger.debug(f"  - {item_reason}")
            
    return files_to_process, skipped_scan_files


def batch_process_images(config: Config) -> Tuple[int, int, int, list[Tuple[str, str]], list[str]]:
    """Manages the batch processing of images using a multiprocessing pool."""
    processed_count = 0
    error_count = 0
    skipped_overwrite_count = 0 
    error_files = [] # Tuples of (relative_path, error_message)
    all_skipped_scan_items = [] # Items skipped during the initial scan phase

    logger.info(f"Starting batch processing. Output folder: '{config.absolute_output_dir}'")

    try:
        files_to_process, skipped_scan_items_list = scan_for_image_files(config)
        all_skipped_scan_items.extend(skipped_scan_items_list)
        total_files_to_attempt = len(files_to_process)

        if total_files_to_attempt == 0:
             logger.warning("(!) No image files found to process with current filters and paths.")
             return 0, 0, 0, [], all_skipped_scan_items # No files to process
        else:
            logger.info(f"-> Found {total_files_to_attempt} image files for processing.")
    except Exception as e:
        logger.critical(f"(!) Critical Error: Failed during file scanning or input path access for '{config.absolute_input_dir}'. ({e})", exc_info=True)
        return 0, 0, 0, [], all_skipped_scan_items # Return empty/zero counts

    # Pre-create necessary output subdirectories based on files_to_process
    required_subdirs = set()
    for _, relative_path in files_to_process:
        output_relative_dir = os.path.dirname(relative_path)
        if output_relative_dir: # Ensure it's not an empty string (for files in root of input_dir)
            required_subdirs.add(output_relative_dir)

    for subdir in required_subdirs:
        output_dir_to_create = os.path.join(config.absolute_output_dir, subdir)
        if not os.path.exists(output_dir_to_create):
            try:
                os.makedirs(output_dir_to_create)
                logger.info(f"Created output subdirectory: '{output_dir_to_create}'")
            except OSError as e:
                # Log error but continue, individual file processing might still succeed if dir exists by then or for other files
                logger.error(f"Error: Failed to create output subdirectory '{output_dir_to_create}' ({e}). Files destined for this directory may fail.")

    num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} worker processes for image processing.")

    # Prepare arguments for the worker: each item is ((input_path, relative_path), config_object)
    tasks_with_config = [((input_p, rel_p), config) for input_p, rel_p in files_to_process]

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=total_files_to_attempt, desc="Processing images", unit="file", ncols=100, leave=True) as pbar:
            # imap_unordered gets results as they complete, good for progress bar updates
            for res_success, res_error_msg, res_affected_path, res_relative_path in pool.imap_unordered(static_process_image_worker, tasks_with_config):
                # Use the skipped message from the config object
                if res_success and res_error_msg == config.skipped_existing_msg: 
                    skipped_overwrite_count +=1
                elif res_success:
                    processed_count += 1
                else: # An actual error occurred
                    error_count += 1
                    # Use tqdm.write for immediate feedback for the error in the main process console
                    # Note: tqdm.write might interfere slightly with the progress bar appearance
                    tqdm.write(f" ✗ Error: '{os.path.basename(res_affected_path)}' (in '{os.path.dirname(res_relative_path)}' folder) - {res_error_msg}")
                    error_files.append((res_relative_path, res_error_msg or "Unknown error"))
                pbar.update(1) # Update progress bar for each completed task (success, error, or policy skip)

    logger.info(f"Batch processing complete. Success: {processed_count}, Errors: {error_count}, Skipped (overwrite policy): {skipped_overwrite_count}")
    return processed_count, error_count, skipped_overwrite_count, error_files, all_skipped_scan_items

# --- Argument Parsing and Setup ---
def parse_arguments() -> Config:
    """Parses command line arguments and returns a Config object."""
    # Create a temporary Config instance to get default values for help text
    temp_config = Config()

    parser = argparse.ArgumentParser(
        description=f"Batch Image Resizer Script (v{__version__})", # Use __version__
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
    # Use config default for help text generation
    required_group.add_argument("-O", "--output-format", required=True, choices=temp_config.supported_output_formats.keys(),
                                help="Output file format:\n" +
                                     "\n".join([f"  {k}: {v}" for k, v in temp_config.supported_output_formats.items()]))

    parser.add_argument("-o", "--output-dir", dest='output_dir_arg',
                        help="Path to save results (default: 'resized_images' under input folder)") 

    resize_group = parser.add_argument_group('Resize Options (ignored if mode is "none")')
    resize_group.add_argument("-w", "--width", type=int, default=0, help="Target width (pixels)")
    resize_group.add_argument("-H", "--height", type=int, default=0, help="Target height (pixels)")
    # Use config default for help text generation
    resize_group.add_argument("-f", "--filter", choices=temp_config.filter_names.keys(), default=None, 
                              help="Resize filter (Required if resizing):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in temp_config.filter_names.items()]))

    optional_group = parser.add_argument_group('Other Optional Options')
    optional_group.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    optional_group.add_argument("-q", "--quality", type=int, default=None, # Default None, applied if relevant
                                help=f"JPG/WEBP quality (1-100, higher is better).\n"
                                     f"Default: JPG={temp_config.default_jpg_quality}, WEBP={temp_config.default_webp_quality} (lossy)")
    optional_group.add_argument("--strip-exif", action="store_true", help="Remove all EXIF data from images.")
    optional_group.add_argument("--overwrite-policy", choices=['rename', 'overwrite', 'skip'], default='rename',
                                help="Policy for existing output files:\n"
                                     "  rename: Add suffix (e.g., _1) (default)\n"
                                     "  overwrite: Replace existing file\n"
                                     "  skip: Do not process if output file exists")
    optional_group.add_argument("--include-extensions", nargs='+', metavar='EXT',
                                help="Process only these extensions (e.g., jpg png).\n"
                                     "This REPLACES the default list of supported extensions.\n"
                                     "Provide extensions without dots (e.g., 'jpg' not '.jpg').\n"
                                     "Cannot be used with --exclude-extensions.")
    optional_group.add_argument("--exclude-extensions", nargs='+', metavar='EXT',
                                help="Exclude these extensions from processing (e.g., gif tiff).\n"
                                     "Applied AFTER the default list (or --include-extensions list if specified).\n"
                                     "Provide extensions without dots (e.g., 'gif' not '.gif').\n"
                                     "Cannot be used with --include-extensions.")
    optional_group.add_argument("--webp-lossless", action="store_true",
                                help="Use lossless compression for WEBP output. Ignored if output format is not WEBP.\n"
                                     "If set, --quality for WEBP is still used by Pillow but might have less effect.")
    
    optional_group.add_argument('--version', action='version', version=f'%(prog)s {__version__}') # Use __version__
    optional_group.add_argument("-v", "--verbose", action="store_true",
                                help="Enable verbose (DEBUG level) logging")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.include_extensions and args.exclude_extensions:
        parser.error("argument --include-extensions: not allowed with argument --exclude-extensions.")

    if args.resize_mode == 'aspect_ratio':
        if args.width <= 0 and args.height <= 0:
            parser.error("Mode 'aspect_ratio' requires --width or --height (or both) to be > 0.")
        if args.width < 0 or args.height < 0: # Width/Height cannot be negative
            parser.error("--width and --height cannot be negative.")
        if not args.filter:
            parser.error("--filter is required for resize mode 'aspect_ratio'.")
    elif args.resize_mode == 'fixed':
        if args.width <= 0 or args.height <= 0:
            parser.error("Mode 'fixed' requires both --width and --height to be > 0.")
        if not args.filter:
            parser.error("--filter is required for resize mode 'fixed'.")
    elif args.resize_mode == 'none' and (args.width > 0 or args.height > 0 or args.filter is not None):
        logger.warning("   -> Info: --width, --height, and --filter are ignored when --resize-mode is 'none'.")

    if args.output_format in ('jpg', 'webp') and args.quality is not None:
        if not (1 <= args.quality <= 100):
            parser.error(f"--quality must be between 1 and 100 (got {args.quality}).")
    elif args.quality is not None and args.output_format not in ('jpg', 'webp'): # Quality given for non-JPG/WEBP
        logger.warning(f"   -> Warning: --quality is ignored for output format '{args.output_format}'.")
    
    if args.webp_lossless and args.output_format != 'webp':
        logger.warning("   -> Warning: --webp-lossless is ignored when output format is not WEBP.")


    try:
        # Create the final Config object using parsed arguments
        # Default values defined in Config will be used if args are None
        return Config(**vars(args))
    except SystemExit: # Raised by parser.error()
        raise # Re-raise to exit script
    except Exception as e: # Catch any other error during Config object creation
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
        # Use filter names from config object
        print(f"  Resize filter: {config.filter_names[resize_opts['filter_str']]}")
    
    # Use supported formats from config object
    print(f"Output format: {config.supported_output_formats[config.output_format]}")
    if config.output_format == 'webp':
        # Use the actual values from output_format_options which considered defaults
        webp_mode = "Lossless" if config.output_format_options.get('lossless') else f"Lossy (Quality: {config.output_format_options.get('quality')})"
        print(f"  WEBP Mode: {webp_mode}")
    elif config.output_format == 'jpg': # Only show quality for JPG if not WEBP
        print(f"  JPG Quality: {config.output_format_options.get('quality')}")
    # No quality display for PNG or 'original'
    
    exif_handling = "Strip all EXIF" if config.strip_exif else "Preserve/Modify EXIF (default)"
    print(f"EXIF Handling: {exif_handling}")
    print(f"Overwrite Policy: {config.overwrite_policy.capitalize()}")
    
    # Display extension filter settings
    if config.include_extensions:
        print(f"Include Extensions: {', '.join(config.include_extensions)}")
    elif config.exclude_extensions: # Only shown if include_extensions is not set
        print(f"Exclude Extensions (from default): {', '.join(config.exclude_extensions)}")
    else: # Neither include nor exclude is set
        # Use default extensions from Config object
        print(f"Processed Extensions: Default ({', '.join(config.default_supported_input_extensions)})")

    print(f"Log Level: {logging.getLevelName(logger.getEffectiveLevel())}")
    print("="*72)

def print_summary(processed: int, errors: int, skipped_overwrite: int, error_list: list, skipped_scan_list: list, output_dir: str):
    """Prints a summary of the processing results."""
    print(f"\n\n--- Processing Summary ---")
    print(f"Successfully processed images: {processed}")
    print(f"Errors encountered: {errors}")
    print(f"Skipped (due to overwrite policy): {skipped_overwrite}")
    if error_list:
        print("\n[Files with Errors]")
        for i, (filepath, errmsg) in enumerate(error_list[:20]): # Show first 20 errors
            print(f"  - {filepath}: {errmsg}")
        if len(error_list) > 20:
            print(f"  ... and {len(error_list) - 20} more error(s).")
    
    if skipped_scan_list: # Now this list contains reasons
        print(f"\nSkipped items during scan ({len(skipped_scan_list)} total):")
        # Display only a few skipped files if the list is long, for brevity
        display_skipped_reasons = skipped_scan_list[:5]
        for reason in display_skipped_reasons:
            print(f"  - {reason}")
        if len(skipped_scan_list) > 5:
            print(f"  ... and {len(skipped_scan_list) - 5} more skipped items.")

        if logger.getEffectiveLevel() <= logging.DEBUG and skipped_scan_list: # Show all if verbose
            print("  Full list of skipped items with reasons (debug mode):")
            for item_reason in skipped_scan_list: print(f"    - {item_reason}")
    else:
        print("\nNo items were skipped during scan phase.") # Or all items found were processable
    print("\n--- All tasks completed ---")
    print(f"Results saved in: '{output_dir}'")

# --- Main Execution Logic ---
def main():
    """Main function to orchestrate script execution."""
    try:
        config = parse_arguments() # Parse arguments and create Config object

        logger.info(f"===== Image Resizer Script v{__version__} Started =====") # Use __version__
        display_settings(config) # Display the settings that will be used

        # Start processing
        processed_count, error_count, skipped_overwrite_count, error_files, skipped_scan_files = batch_process_images(config)
        
        # Print summary of results
        print_summary(processed_count, error_count, skipped_overwrite_count, error_files, skipped_scan_files, config.absolute_output_dir)

        logger.info(f"===== Image Resizer Script v{__version__} Finished =====") # Use __version__
        sys.exit(1 if error_count > 0 else 0) # Exit with 1 if errors occurred, 0 otherwise
    except SystemExit as e: # Handle sys.exit() calls, e.g., from argparse or explicit exits
        if e.code is None or e.code == 0: # Normal exit
            logger.info("Script exited normally.")
        else: # Exit due to error (e.g., argparse error)
            logger.error(f"Script exited with error code {e.code}.")
        # No re-raise needed as sys.exit() already handles program termination.
    except Exception as e: # Catch any other unhandled exceptions
        logger.critical(f"Unhandled critical exception occurred: {e}", exc_info=True)
        print(f"\n(!) Critical Error: {e}") # Also print to console for visibility
        sys.exit(2) # General error exit code

if __name__ == "__main__":
    # This check is crucial for multiprocessing on Windows and for PyInstaller.
    # It prevents child processes from re-executing the main script logic.
    multiprocessing.freeze_support()
    main()
