# -*- coding: utf-8 -*-
import os
import sys
import argparse # For command-line argument processing
import logging # For logging functionality
from dataclasses import dataclass, field # For the configuration dataclass
from typing import Dict, Any, Optional, Tuple, List # For type hinting
import multiprocessing # For parallel processing

# --- Third-party Library Imports ---
from PIL import Image, UnidentifiedImageError # For image manipulation
import piexif # For EXIF data handling
from tqdm import tqdm # For progress bar display

# --- Logging Setup ---
# Define a detailed log format: timestamp, log level, process name, and message.
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create a console handler that writes log messages to stderr.
log_handler = logging.StreamHandler() 
log_handler.setFormatter(log_formatter)

# Get the root logger instance.
logger = logging.getLogger()

# Remove any existing handlers to prevent duplicate log output.
if logger.hasHandlers():
    logger.handlers.clear()

# Add the new console handler to the root logger.
logger.addHandler(log_handler) 

# Set the default logging level to INFO. Messages with INFO level or higher will be displayed.
logger.setLevel(logging.INFO)


# --- Constants ---
__version__ = "3.37" # Script version (Renamed Config.output_dir_arg to Config.output_dir)

# Define PIL/Pillow resampling filters, handling potential API changes across versions.
try:
    # Pillow version 9.1.0 and later use Resampling enums.
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS, # Highest quality
        "bicubic": Image.Resampling.BICUBIC,  # Medium quality
        "bilinear": Image.Resampling.BILINEAR, # Low quality
        "nearest": Image.Resampling.NEAREST   # Lowest quality, fastest
    }
except AttributeError: # Fallback for older Pillow versions.
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
    Worker function for the multiprocessing pool.
    It unpacks arguments and calls the main image processing function for a single image.

    Args:
        paths_tuple_and_config_tuple: A tuple containing:
            - ((input_path, relative_path), config_object)
              - input_path: Absolute path to the input image file.
              - relative_path: Path of the image relative to the input directory.
              - config_object: The script's configuration object.

    Returns:
        A tuple: (success_bool, error_message_or_none, affected_path, original_relative_path)
            - success_bool: True if processing was successful or skipped by policy, False on error.
            - error_message_or_none: An error message string if an error occurred, 
                                     a special skip message, or None on success.
            - affected_path: The output path on success, or the input path on error/skip.
            - original_relative_path: The original relative path of the processed image.
    """
    paths_tuple, config_obj = paths_tuple_and_config_tuple
    input_path, relative_path = paths_tuple
    return process_single_image_file(input_path, relative_path, config_obj)

# --- Configuration Dataclass ---
@dataclass
class Config:
    """
    Dataclass to store and manage all script configurations.
    It centralizes settings derived from command-line arguments and internal defaults.
    Fields without default values must come before fields with default values.
    """
    # --- User-configurable via command-line arguments ---
    # Fields without dataclass-defined defaults (values are expected from argparse)
    output_format: str 

    # Fields with dataclass-defined defaults
    overwrite: bool = True # Policy for handling existing output files. True to overwrite, False to skip.
    input_dir: str = 'input' # Default input directory.
    output_dir: Optional[str] = None # User-specified output directory (if any). Renamed from output_dir_arg.
    ratio: str = 'aspect_ratio' # Default resize ratio mode.
    width: int = 0 # Target width for resizing.
    height: int = 0 # Target height for resizing.
    filter: Optional[str] = None # Resampling filter to use. Argparse default is 'lanczos'.
    verbose: bool = False # Enable verbose (DEBUG level) logging.
    strip_exif: bool = False # Whether to remove EXIF data.
    include_extensions: Optional[List[str]] = None # List of extensions to explicitly include.
    exclude_extensions: Optional[List[str]] = None # List of extensions to explicitly exclude.
    webp_lossless: bool = False # Use lossless compression for WEBP output.

    # --- Internal or derived attributes (not directly set by user args) ---
    absolute_input_dir: str = field(init=False, default='') # Absolute path to the input directory.
    absolute_output_dir: str = field(init=False, default='') # Absolute path to the output directory.
    resize_options: Dict[str, Any] = field(init=False, default_factory=dict) # Parsed resize options.
    output_format_options: Dict[str, Any] = field(init=False, default_factory=dict) # Parsed output format options.

    # --- Default values for specific settings, can be overridden by args ---
    # These are also fields with dataclass defaults. Argparse values will override these.
    jpg_quality: int = 95 # Quality for JPG output. Can be overridden by --jpeg-quality.
    webp_quality: int = 80 # Quality for lossy WEBP output. Can be overridden by --webp-quality.
    
    supported_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp') # Default supported image extensions.
    
    # --- Constant mappings used within the configuration ---
    filter_names: Dict[str, str] = field(default_factory=lambda: { # User-friendly names for resize filters.
        "lanczos": "LANCZOS (High quality)",
        "bicubic": "BICUBIC (Medium quality)",
        "bilinear": "BILINEAR (Low quality)",
        "nearest": "NEAREST (Lowest quality)"
    })
    supported_output_formats: Dict[str, str] = field(default_factory=lambda: { # User-friendly names for output formats.
        "original": "Keep Original", "png": "PNG", "jpg": "JPG", "webp": "WEBP",
    })
    skipped_existing_msg: str = "SKIPPED_EXISTING_FILE_POLICY" # Internal message for files skipped due to overwrite policy.
    
    _skip_post_init_actions: bool = field(default=False, kw_only=True, repr=False) # Internal flag to skip post_init for temp instances.


    def __post_init__(self):
        """
        Performs initialization and validation after the dataclass has been populated.
        This includes path validation, preparing processing options, and setting up logging.
        """
        if self._skip_post_init_actions:
            return 
        
        self._validate_paths()
        self._prepare_options()
        self._normalize_extension_filters()
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled via Config.")

    def _normalize_extension_filters(self):
        """
        Normalizes include/exclude extension lists.
        Converts extensions to lowercase and ensures they start with a dot (e.g., "jpg" -> ".jpg").
        """
        if self.include_extensions:
            self.include_extensions = [f".{ext.lower().lstrip('.')}" for ext in self.include_extensions]
            logger.debug(f"Normalized include_extensions: {self.include_extensions}")
        if self.exclude_extensions:
            self.exclude_extensions = [f".{ext.lower().lstrip('.')}" for ext in self.exclude_extensions]
            logger.debug(f"Normalized exclude_extensions: {self.exclude_extensions}")


    def _validate_paths(self):
        """
        Validates the input and output directory paths.
        Ensures input directory exists and output directory can be created.
        Prevents input and output directories from being the same.
        Exits script on critical path errors.
        """
        if not os.path.isdir(self.input_dir):
            logger.critical(f"(!) Error: Invalid input folder path: {self.input_dir}")
            if self.input_dir == 'input': 
                 logger.info(f"    Default input folder '{self.input_dir}' not found. Please create it or specify a path.")
            sys.exit(1)

        self.absolute_input_dir = os.path.abspath(self.input_dir)

        # Use the renamed 'output_dir' field
        if self.output_dir: 
            self.absolute_output_dir = os.path.abspath(self.output_dir)
        else:
            self.absolute_output_dir = os.path.abspath("output") 
            logger.info(f"   -> Info: Output folder not specified. Using default: '{self.absolute_output_dir}'")

        if self.absolute_input_dir == self.absolute_output_dir:
            logger.critical("(!) Error: Input folder and output folder cannot be the same.")
            sys.exit(1)

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
        """
        Prepares dictionaries for resize and output format options based on the configuration.
        This centralizes how these options are accessed during image processing.
        """
        self.resize_options = {'mode': self.ratio} 
        if self.ratio != 'none': 
            self.resize_options.update({
                'width': self.width,
                'height': self.height,
                'filter_str': self.filter, 
                'filter_obj': _PIL_RESAMPLE_FILTERS[self.filter] 
            })
        logger.debug(f"Resize options prepared: {self.resize_options}")

        self.output_format_options = {'format_str': self.output_format}
        if self.output_format == 'jpg':
            self.output_format_options['quality'] = self.jpg_quality
        elif self.output_format == 'webp':
            self.output_format_options['quality'] = self.webp_quality
            self.output_format_options['lossless'] = self.webp_lossless

        logger.debug(f"Output format options prepared: {self.output_format_options}")
        logger.debug("Environment and options prepared.")

# --- Core Image Processing Functions ---
def resize_image_maintain_aspect_ratio(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    """
    Resizes an image while maintaining its aspect ratio to fit within the target dimensions.
    The image will be scaled down to fit either the target_width or target_height,
    whichever results in a smaller image that still fits both dimensions.

    Args:
        img: The PIL Image object to resize.
        target_width: The maximum width for the resized image.
        target_height: The maximum height for the resized image.
        resample_filter: The PIL resampling filter to use.

    Returns:
        The resized PIL Image object, or the original if no resize was needed or possible.
    """
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0: 
        logger.warning(f"Original image dimensions ({original_width}x{original_height}) are invalid. Skipping resize.")
        return img

    new_width, new_height = 0, 0
    if target_width > 0 and target_height > 0: 
        ratio_calc = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(original_width * ratio_calc))
        new_height = max(1, int(original_height * ratio_calc))
    elif target_width > 0: 
        ratio_calc = target_width / original_width
        new_width = target_width
        new_height = max(1, int(original_height * ratio_calc))
    elif target_height > 0: 
        ratio_calc = target_height / original_height
        new_height = target_height
        new_width = max(1, int(original_width * ratio_calc))
    else: 
        logger.warning("Target dimensions for aspect ratio resize not properly specified. Returning original image.")
        return img

    if (new_width, new_height) == (original_width, original_height):
        logger.debug("Calculated resize dimensions are the same as original. Skipping resize.")
        return img
    try:
        logger.debug(f"Resizing (aspect ratio): ({original_width},{original_height}) -> ({new_width},{new_height})")
        return img.resize((new_width, new_height), resample_filter)
    except ValueError as e: 
        logger.warning(f"Error during aspect ratio resize (({original_width},{original_height}) -> ({new_width},{new_height})): {e}. Using original image.")
        return img

def resize_image_fixed_size(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    """
    Resizes an image to a fixed target width and height.
    This may change the aspect ratio of the image.

    Args:
        img: The PIL Image object to resize.
        target_width: The exact target width for the resized image.
        target_height: The exact target height for the resized image.
        resample_filter: The PIL resampling filter to use.

    Returns:
        The resized PIL Image object, or the original if no resize was needed or possible.
    """
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height):
        logger.debug("Target dimensions are the same as original. Skipping fixed resize.")
        return img
    if target_width <= 0 or target_height <= 0: 
        logger.warning(f"Target dimensions for fixed resize ({target_width}x{target_height}) are invalid. Skipping resize.")
        return img
    try:
        logger.debug(f"Resizing (fixed): ({original_width},{original_height}) -> ({target_width},{target_height})")
        return img.resize((target_width, target_height), resample_filter)
    except ValueError as e:
        logger.warning(f"Error during fixed size resize (({original_width},{original_height}) -> ({target_width},{target_height})): {e}. Using original image.")
        return img

def prepare_image_for_save(img: Image.Image, output_format_str: Optional[str]) -> Image.Image:
    """
    Prepares an image for saving by handling mode conversions, especially for formats like JPG.
    For JPG, this involves converting images with alpha channels (e.g., RGBA, LA, P with transparency)
    to RGB by compositing them onto a white background.
    For WEBP, Palette ('P') mode images might be converted to RGBA or RGB.

    Args:
        img: The PIL Image object to prepare.
        output_format_str: The target output format string (e.g., "JPG", "WEBP"). Case-insensitive.

    Returns:
        The prepared PIL Image object, possibly with a changed mode.
    """
    save_img = img
    original_mode = img.mode
    
    if output_format_str: 
        output_format_upper = output_format_str.upper()
    else: 
        return img # No format conversion needed if output_format_str is None (e.g. 'original' format)

    if output_format_upper == 'JPG':
        if img.mode in ('RGBA', 'LA', 'P'): 
            if img.mode == 'P': # Palette mode
                # Convert to RGBA if transparency exists, otherwise RGB
                save_img = img.convert('RGBA') if 'transparency' in img.info else img.convert('RGB')
                logger.debug(f"Converted 'P' (Palette) mode to '{save_img.mode}' for JPG preparation.")
            
            if save_img.mode in ('RGBA', 'LA'): # Modes with Alpha channel
                logger.debug(f"Processing '{save_img.mode}' mode for JPG saving (requires alpha compositing)...")
                # Create a white background image
                background = Image.new("RGB", save_img.size, (255, 255, 255)) 
                try:
                    # Paste the image onto the background using its alpha channel as a mask
                    mask = save_img.split()[-1] # Get alpha channel
                    background.paste(save_img, (0,0), mask=mask)
                    save_img = background
                    logger.debug(f"Converted '{original_mode}' to 'RGB' by merging alpha channel onto white background.")
                except (IndexError, ValueError) as e: # Fallback if alpha processing fails
                    logger.warning(f"Error processing alpha channel for JPG conversion ('{original_mode}' -> 'RGB'): {e}. Forcing direct conversion to RGB.")
                    save_img = save_img.convert('RGB') 
            elif save_img.mode != 'RGB': # Other modes that need conversion to RGB (e.g. 'L')
                save_img = save_img.convert('RGB')
                logger.debug(f"Converted '{original_mode}' to 'RGB' for JPG saving.")

    elif output_format_upper == 'WEBP':
         if img.mode == 'P': # Palette mode for WEBP
              # Convert to RGBA if transparency exists, otherwise RGB
              save_img = img.convert("RGBA") if 'transparency' in img.info else img.convert("RGB")
              logger.debug(f"Converted 'P' (Palette) mode to '{save_img.mode}' for WEBP saving.")

    if save_img.mode != original_mode:
        logger.debug(f"Image mode converted: '{original_mode}' -> '{save_img.mode}' (Target output format: {output_format_str or 'Original'})")
    return save_img

def process_single_image_file(input_path: str, relative_path: str, config: Config) -> Tuple[bool, Optional[str], str, str]:
    """
    Processes a single image file: loads, resizes (if applicable), handles EXIF,
    converts format (if applicable), and saves it to the output directory.

    Args:
        input_path: Absolute path to the input image file.
        relative_path: Path of the image relative to the input directory.
                       This is used to reconstruct the path in the output directory.
        config: The script's configuration object.

    Returns:
        A tuple: (success_bool, error_message_or_none, affected_path, original_relative_path).
            - success_bool: True if processing was successful or skipped by policy, False on error.
            - error_message_or_none: An error message string if an error occurred, 
                                     a special skip message from config.skipped_existing_msg, or None on success.
            - affected_path: The output path on success, or the input path on error/skip.
            - original_relative_path: The original relative_path argument.
    """
    base_name, original_ext = os.path.splitext(os.path.basename(input_path))
    output_format_str = config.output_format_options['format_str'] 
    logger.debug(f"Processing: '{relative_path}'")

    # Determine output extension based on target format
    output_ext_map = {'jpg': '.jpg', 'webp': '.webp', 'png': '.png'}
    output_ext = output_ext_map.get(output_format_str.lower(), original_ext)

    # Construct output path
    output_relative_dir = os.path.dirname(relative_path) 
    output_dir_for_file = os.path.join(config.absolute_output_dir, output_relative_dir)

    # This check is mostly redundant if batch_process_images pre-creates dirs, but good for safety
    if not os.path.isdir(output_dir_for_file):
         logger.error(f"Internal Error: Output subdirectory does not exist: '{output_dir_for_file}'")
         try:
             os.makedirs(output_dir_for_file) 
             logger.warning(f"Warning: Created missing output subdirectory during single file processing: '{output_dir_for_file}'")
         except OSError as e:
              error_msg = f"Failed to create output subdirectory '{output_dir_for_file}' during single file processing ({e})"
              logger.error(error_msg)
              return False, error_msg, input_path, relative_path 

    output_filename = base_name + output_ext
    output_path = os.path.join(output_dir_for_file, output_filename)
    
    # Handle existing output files based on overwrite policy
    if os.path.exists(output_path):
        if config.overwrite: 
            logger.debug(f"Output file '{output_filename}' exists. Overwriting due to --overwrite policy.")
        else: # --no-overwrite was specified
            logger.debug(f"Skipping existing file due to --no-overwrite policy: '{relative_path}' (target: '{output_path}')")
            return True, config.skipped_existing_msg, input_path, relative_path # Success, but skipped

    original_exif_bytes = None 
    exif_data = None # Parsed EXIF data

    try:
        with Image.open(input_path) as img:
            logger.debug(f"Image loaded: '{relative_path}' (Size: {img.size}, Mode: {img.mode})")
            
            # Load EXIF data if not stripping
            if not config.strip_exif and 'exif' in img.info and img.info['exif']:
                original_exif_bytes = img.info['exif']
                try:
                    exif_data = piexif.load(original_exif_bytes) 
                    logger.debug(f"EXIF data loaded and parsed successfully for '{relative_path}'")
                except Exception as e: 
                    logger.warning(f"Failed to parse EXIF for '{relative_path}' ({type(e).__name__}). Will attempt to keep original EXIF bytes if possible for compatible formats.")
                    exif_data = None # Mark as unparsed
            elif config.strip_exif:
                logger.debug(f"EXIF stripping enabled for '{relative_path}'. All EXIF data will be removed.")

            # Prepare image for saving (mode conversion, e.g., for JPG alpha)
            save_format_for_prepare = output_format_str.upper() if output_format_str != 'original' else None
            img_prepared = prepare_image_for_save(img, save_format_for_prepare)

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
            # If resize_opts['mode'] is 'none', img_resized remains img_prepared

            # Prepare save arguments (format, quality, EXIF)
            save_kwargs = {}
            save_format_arg = output_format_str.upper() if output_format_str != 'original' else None # Pillow format string
            if save_format_arg == 'JPG': save_format_arg = 'JPEG' # Pillow uses 'JPEG'

            # Handle EXIF data for saving
            final_exif_bytes = None
            if not config.strip_exif:
                if output_format_str == 'original': # Keep original EXIF if format is original
                    final_exif_bytes = original_exif_bytes 
                    if final_exif_bytes: logger.debug(f"Passing original EXIF data for '{relative_path}' (output format: original).")
                else: # For converted formats, try to use parsed and modified EXIF
                    if exif_data: # If EXIF was successfully parsed
                        try:
                            # Reset Orientation tag to 1 (normal) to avoid rotation issues after resize
                            if piexif.ImageIFD.Orientation in exif_data.get('0th', {}):
                                exif_data['0th'][piexif.ImageIFD.Orientation] = 1
                                logger.debug(f"Reset EXIF Orientation tag to 1 for '{relative_path}'.")
                            # Remove thumbnail to save space and prevent issues with outdated thumbnails
                            if 'thumbnail' in exif_data and exif_data['thumbnail'] is not None:
                                 exif_data['thumbnail'] = None
                                 logger.debug(f"Removed EXIF thumbnail data for '{relative_path}'.")
                            final_exif_bytes = piexif.dump(exif_data) 
                            logger.debug(f"Successfully prepared modified EXIF data for '{relative_path}'.")
                        except Exception as e: # Fallback if dumping modified EXIF fails
                            logger.warning(f"Failed to dump modified EXIF for '{relative_path}' ({type(e).__name__}). Attempting to use original EXIF bytes as fallback.")
                            final_exif_bytes = original_exif_bytes # Use original raw bytes as a last resort
                    elif original_exif_bytes: # If parsing failed but we have original bytes
                         final_exif_bytes = original_exif_bytes 
                         logger.debug(f"Using original (unparsed) EXIF bytes for '{relative_path}' as parsing failed earlier.")
            
            if final_exif_bytes:
                # Add EXIF to save_kwargs if format supports it
                if output_format_str == 'jpg' or \
                   (output_format_str == 'original' and original_ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.webp']) or \
                   output_format_str == 'webp':
                    save_kwargs['exif'] = final_exif_bytes
                elif output_format_str == 'png':
                    # Pillow's PNG EXIF support can be tricky with piexif bytes directly
                    try:
                        save_kwargs['exif'] = final_exif_bytes
                    except TypeError: # Pillow might raise TypeError if it can't handle the exif format for PNG
                        logger.warning(f"'{relative_path}': Pillow version might not support EXIF for PNG with piexif bytes, or EXIF data is incompatible. EXIF will not be saved for this PNG.")
            
            # Set format-specific save options
            output_opts = config.output_format_options 
            if output_format_str == 'jpg':
                save_kwargs.update({
                    'quality': output_opts.get('quality', config.jpg_quality), 
                    'optimize': True,
                    'progressive': True
                })
            elif output_format_str == 'png':
                save_kwargs['optimize'] = True 
            elif output_format_str == 'webp':
                save_kwargs.update({
                    'quality': output_opts.get('quality', config.webp_quality), 
                    'lossless': output_opts.get('lossless', False) 
                })

            logger.debug(f"Attempting to save image: '{output_path}' (Format arg for Pillow: {save_format_arg or 'Inferred'}) with kwargs keys: {list(save_kwargs.keys())}")
            img_resized.save(output_path, format=save_format_arg, **save_kwargs)
            logger.debug(f"Image saved successfully: '{output_path}'")
            return True, None, output_path, relative_path # Success

    except UnidentifiedImageError:
        msg = "Invalid or corrupted image file. Pillow could not identify the image format."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        return False, msg, input_path, relative_path
    except PermissionError:
        msg = "File read/write permission denied."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        return False, msg, input_path, relative_path
    except OSError as e: # Catch broader OS errors (disk full, etc.)
        msg = f"File system or OS-level error occurred ({e})."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        if os.path.exists(output_path) and output_path != input_path : # Clean up partially created file
            try: 
                os.remove(output_path)
                logger.warning(f"Removed partially created/failed output file: '{output_path}'")
            except OSError: pass # Ignore if removal fails
        return False, msg, input_path, relative_path
    except ValueError as e: # Catch errors from Pillow operations (e.g. invalid parameters)
         msg = f"Image processing value error, likely from Pillow ({e})."
         logger.error(f"Processing failed for '{relative_path}': {msg}")
         return False, msg, input_path, relative_path
    except Exception as e: # Catch-all for any other unexpected errors
        msg = f"An unexpected error occurred ({type(e).__name__}: {e})."
        logger.critical(f"Processing failed for '{relative_path}': {msg}", exc_info=config.verbose) # Log traceback if verbose
        return False, msg, input_path, relative_path

def scan_for_image_files(config: Config) -> Tuple[list[Tuple[str, str]], list[str]]:
    """
    Scans the input directory for image files based on the provided configuration.
    This version is always non-recursive (scans only the top level of input_dir).

    Args:
        config: The script's configuration object.

    Returns:
        A tuple containing:
            - files_to_process: A list of (absolute_path, relative_path) tuples for images to be processed.
            - skipped_scan_files: A list of reasons (strings) why certain items were skipped during the scan.
    """
    files_to_process = [] 
    skipped_scan_items_reasons = [] 
    logger.info(f"Scanning input folder: '{config.absolute_input_dir}'")

    items_found_to_evaluate = [] # (absolute_path, relative_path_to_input_dir)
    try:
        for filename in os.listdir(config.absolute_input_dir):
            abs_path = os.path.join(config.absolute_input_dir, filename)
            if os.path.isfile(abs_path):
                 items_found_to_evaluate.append((abs_path, filename)) # relative_path is just filename for non-recursive
            elif os.path.isdir(abs_path):
                # Skip if it's the output directory itself or if recursion is not enabled (always the case here)
                if os.path.abspath(abs_path) == config.absolute_output_dir:
                    skipped_scan_items_reasons.append(filename + " (Is the output folder - skipped scan)")
                else:
                    skipped_scan_items_reasons.append(filename + " (Is a directory - skipped scan as recursive is not enabled)")
    except OSError as e:
        logger.critical(f"Error scanning input directory '{config.absolute_input_dir}': {e}")
        return [], [f"Error scanning input directory: {e}"]


    for input_path, relative_path in items_found_to_evaluate:
        filename = os.path.basename(input_path) # Should be same as relative_path here
        file_ext = os.path.splitext(filename)[1].lower() 

        if not file_ext: # Skip files without extensions
            skipped_scan_items_reasons.append(relative_path + " (File has no extension)")
            continue

        should_process_based_on_extension = False
        
        # Apply include/exclude extension filters
        if config.include_extensions: # If --include-extensions is used
            if file_ext in config.include_extensions:
                should_process_based_on_extension = True
            else:
                skipped_scan_items_reasons.append(relative_path + f" (Extension '{file_ext}' not in --include-extensions list: {config.include_extensions})")
        elif file_ext in config.supported_extensions: # Default: check against supported_extensions
            should_process_based_on_extension = True
        else: # Not in default supported list
            skipped_scan_items_reasons.append(relative_path + f" (Extension '{file_ext}' not in default supported list: {config.supported_extensions})")

        # Apply exclude filter if the file was initially marked for processing
        if should_process_based_on_extension and config.exclude_extensions and file_ext in config.exclude_extensions:
            should_process_based_on_extension = False 
            skipped_scan_items_reasons.append(relative_path + f" (Extension '{file_ext}' is in --exclude-extensions list: {config.exclude_extensions})")
        
        if should_process_based_on_extension:
            files_to_process.append((input_path, relative_path))

    logger.info(f"Scan complete: Found {len(files_to_process)} files to process. Skipped {len(skipped_scan_items_reasons)} other items found during scan.")
    if logger.getEffectiveLevel() <= logging.DEBUG and skipped_scan_items_reasons: 
        logger.debug("Items skipped during scan phase (with reasons):")
        for item_reason in skipped_scan_items_reasons:
            logger.debug(f"  - {item_reason}")
            
    return files_to_process, skipped_scan_items_reasons


def batch_process_images(config: Config) -> Tuple[int, int, int, list[Tuple[str, str]], list[str]]:
    """
    Manages the batch processing of images using a multiprocessing pool.
    It scans for files, (pre-creates output subdirectories for non-recursive scan), 
    and distributes the processing tasks to worker processes.

    Args:
        config: The script's configuration object.

    Returns:
        A tuple containing:
            - processed_count: Number of successfully processed images.
            - error_count: Number of images that failed to process.
            - skipped_overwrite_count: Number of images skipped due to overwrite policy.
            - error_files: A list of (relative_path, error_message) tuples for failed images.
            - all_skipped_scan_items_reasons: A list of reasons for items skipped during the initial scan phase.
    """
    processed_count = 0
    error_count = 0
    skipped_overwrite_count = 0 
    error_files_details = [] # List of (relative_path, error_message)
    all_skipped_scan_items_reasons = [] 

    logger.info(f"Starting batch processing. Output will be saved to: '{config.absolute_output_dir}'")

    try:
        files_to_process, skipped_scan_reasons_list = scan_for_image_files(config)
        all_skipped_scan_items_reasons.extend(skipped_scan_reasons_list)
        total_files_to_attempt_processing = len(files_to_process)

        if total_files_to_attempt_processing == 0:
             logger.warning("(!) No image files found to process with the current filters and input path.")
             return 0, 0, 0, [], all_skipped_scan_items_reasons
        else:
            logger.info(f"-> Found {total_files_to_attempt_processing} image files for processing.")
    except Exception as e: # Should be caught by scan_for_image_files, but as a safeguard
        logger.critical(f"(!) Critical Error: Failed during file scanning for '{config.absolute_input_dir}'. Error: {e}", exc_info=config.verbose)
        return 0, 0, 0, [], all_skipped_scan_items_reasons

    # Pre-create output subdirectories (only one level deep for non-recursive)
    # For non-recursive, relative_path is just the filename, so os.path.dirname(relative_path) will be empty.
    # This means only the main output_dir needs to exist, which Config handles.
    # If the script were recursive, this loop would be more relevant for deeper structures.
    # However, keeping it for consistency if scan_for_image_files ever changes.
    required_output_subdirs = set()
    for _, relative_path in files_to_process:
        output_relative_dir = os.path.dirname(relative_path) # This will be '' for top-level files
        if output_relative_dir: # Only add if it's a non-empty path (i.e., a subdirectory)
            required_output_subdirs.add(output_relative_dir)

    for subdir_rel_path in required_output_subdirs: 
        output_dir_to_create = os.path.join(config.absolute_output_dir, subdir_rel_path)
        if not os.path.exists(output_dir_to_create):
            try:
                os.makedirs(output_dir_to_create)
                logger.info(f"Created output subdirectory: '{output_dir_to_create}'")
            except OSError as e:
                logger.error(f"Error: Failed to pre-create output subdirectory '{output_dir_to_create}' ({e}). Files destined for this directory may fail.")


    num_processes = multiprocessing.cpu_count() 
    logger.info(f"Using {num_processes} worker processes for image processing.")

    # Prepare tasks for the pool: each task is a tuple ((input_p, rel_p), config_obj)
    tasks_with_config = [((input_p, rel_p), config) for input_p, rel_p in files_to_process]

    # Use multiprocessing pool for parallel processing
    with multiprocessing.Pool(processes=num_processes) as pool:
        # tqdm provides a progress bar
        with tqdm(total=total_files_to_attempt_processing, desc="Processing images", unit="file", ncols=100, leave=True) as pbar:
            # imap_unordered processes tasks as they complete, good for progress updates
            for res_success, res_error_msg, res_affected_path, res_relative_path in pool.imap_unordered(static_process_image_worker, tasks_with_config):
                if res_success and res_error_msg == config.skipped_existing_msg: 
                    skipped_overwrite_count +=1
                elif res_success:
                    processed_count += 1
                else: # Error occurred
                    error_count += 1
                    # Write error to tqdm to not mess up the progress bar
                    tqdm.write(f" ✗ Error processing '{os.path.basename(res_affected_path)}' (original: '{res_relative_path}'): {res_error_msg}")
                    error_files_details.append((res_relative_path, res_error_msg or "Unknown error"))
                pbar.update(1) # Update progress bar

    logger.info(f"Batch processing complete. Success: {processed_count}, Errors: {error_count}, Skipped (due to --no-overwrite): {skipped_overwrite_count}")
    return processed_count, error_count, skipped_overwrite_count, error_files_details, all_skipped_scan_items_reasons

# --- Argument Parsing and Setup ---
def parse_arguments() -> Config:
    """
    Parses command-line arguments using argparse and returns a Config object.
    Includes validation for argument combinations and values.
    Exits script on argument parsing errors.
    """
    # Create a temporary Config instance to access default values for help strings and validation
    # _skip_post_init_actions=True prevents it from running full validation logic prematurely.
    temp_config_for_defaults = Config(output_format='original', overwrite=True, filter='lanczos', ratio='aspect_ratio', _skip_post_init_actions=True) 
    filter_choices_map = temp_config_for_defaults.filter_names
    output_format_choices_map = temp_config_for_defaults.supported_output_formats
    
    parser = argparse.ArgumentParser(
        description=f"Batch Image Resizer Script (v{__version__})",
        formatter_class=argparse.RawTextHelpFormatter, # Allows for more control over help text formatting
        usage="%(prog)s [input_dir] [options]" # Custom usage message
    )
    parser.add_argument("input_dir", nargs='?', default='input', # Positional argument, optional, default 'input'
                        help="Path to the source image folder (default: 'input' in current directory).")

    parser.add_argument("-f", "--output-format", 
                        default='original', 
                        choices=output_format_choices_map.keys(),
                        help="Target output file format (default: original):\n" +
                             "\n".join([f"  {k}: {v}" for k, v in output_format_choices_map.items()]))

    # Group for resize ratio options
    ratio_group = parser.add_argument_group('Resize Ratio Option') 
    ratio_group.add_argument("-r", "--ratio",  # Changed short command from -m to -r
                                choices=['aspect_ratio', 'fixed', 'none'], 
                                default='aspect_ratio', 
                                dest='ratio', # Maps to Config.ratio
                                help="Resize ratio behavior (default: aspect_ratio):\n"
                                     "  aspect_ratio: Maintain aspect ratio to fit target Width/Height.\n"
                                     "  fixed: Force resize to target Width x Height (may distort).\n"
                                     "  none: No resizing (only format conversion/EXIF handling).")

    # Changed dest to 'output_dir' to match the Config field name
    parser.add_argument("-o", "--output-dir", dest='output_dir', 
                        help="Path to save processed images (default: 'output' in current directory).") 

    # Group for resize dimension options (relevant if ratio is not 'none')
    resize_group = parser.add_argument_group('Resize Dimensions (if --ratio is not "none")')
    resize_group.add_argument("-w", "--width", type=int, default=0, help="Target width in pixels for resizing.")
    resize_group.add_argument("-H", "--height", type=int, default=0, help="Target height in pixels for resizing.") # Uppercase H for height
    resize_group.add_argument("--filter", 
                              default='lanczos', 
                              choices=filter_choices_map.keys(), 
                              help="Resampling filter to use for resizing (default: lanczos):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in filter_choices_map.items()]))

    # Group for other optional settings
    optional_group = parser.add_argument_group('Other Optional Options')
    
    optional_group.add_argument("-q", "--jpeg-quality", type=int, dest='jpg_quality', # -q alias
                                default=temp_config_for_defaults.jpg_quality,
                                help=f"Quality for JPG output (1-100, higher is better).\n"
                                     f"Default: {temp_config_for_defaults.jpg_quality}")
    optional_group.add_argument("--webp-quality", type=int, dest='webp_quality',
                                default=temp_config_for_defaults.webp_quality,
                                help=f"Quality for WEBP output (1-100, higher is better).\n"
                                     f"Default: {temp_config_for_defaults.webp_quality} (for lossy WEBP).")

    optional_group.add_argument("--strip-exif", action="store_true", help="Remove all EXIF metadata from images.")
    
    # Mutually exclusive group for overwrite policy
    overwrite_group = optional_group.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", dest='overwrite', action='store_true',
                                 help="Overwrite existing output files.") # Help text simplified as default is now clear
    overwrite_group.add_argument("--no-overwrite", dest='overwrite', action='store_false',
                                 help="Do not overwrite existing output files; skip them instead.")
    parser.set_defaults(overwrite=True) # Ensures 'overwrite' is True if neither flag is given

    # Extension filtering options
    optional_group.add_argument("--include-extensions", nargs='+', metavar='EXT',
                                help="Process only files with these extensions (e.g., jpg png).\n"
                                     "This REPLACES the default list of supported extensions.\n"
                                     "Provide extensions without dots (e.g., 'jpg' not '.jpg').\n"
                                     "Cannot be used with --exclude-extensions.")
    optional_group.add_argument("--exclude-extensions", nargs='+', metavar='EXT',
                                help="Exclude files with these extensions from processing (e.g., gif tiff).\n"
                                     "Applied AFTER the default list (or --include-extensions list if specified).\n"
                                     "Provide extensions without dots (e.g., 'gif' not '.gif').\n"
                                     "Cannot be used with --include-extensions.")
    optional_group.add_argument("--webp-lossless", action="store_true",
                                help="Use lossless compression for WEBP output. Ignored if output format is not WEBP.\n"
                                     "If set, --webp-quality is still used by Pillow but might have less visual impact.")
    
    optional_group.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    optional_group.add_argument("-v", "--verbose", action="store_true",
                                help="Enable verbose (DEBUG level) logging for detailed output.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.include_extensions and args.exclude_extensions:
        parser.error("The arguments --include-extensions and --exclude-extensions cannot be used together.")
        
    if args.ratio == 'aspect_ratio':
        if args.width <= 0 and args.height <= 0:
            parser.error("For '--ratio aspect_ratio', either --width or --height (or both) must be greater than 0.")
        if args.width < 0 or args.height < 0: # Ensure non-negative
            parser.error("--width and --height cannot be negative values.")
    elif args.ratio == 'fixed':
        if args.width <= 0 or args.height <= 0:
            parser.error("For '--ratio fixed', both --width and --height must be greater than 0.")
    elif args.ratio == 'none': # No resize
        if args.width > 0 or args.height > 0: # Dimensions specified but not used
             logger.warning(f"   -> Info: --width and --height arguments are ignored when --ratio is 'none'.")

    # Validate quality ranges
    if not (1 <= args.jpg_quality <= 100):
        parser.error(f"--jpeg-quality must be an integer between 1 and 100 (inclusive), got {args.jpg_quality}.")
    if not (1 <= args.webp_quality <= 100):
        parser.error(f"--webp-quality must be an integer between 1 and 100 (inclusive), got {args.webp_quality}.")

    # Warnings for irrelevant quality settings
    if args.output_format != 'jpg' and args.jpg_quality != temp_config_for_defaults.jpg_quality:
        logger.warning(f"   -> Warning: --jpeg-quality argument is ignored when output format is not JPG.")
    if args.output_format != 'webp' and args.webp_quality != temp_config_for_defaults.webp_quality:
        logger.warning(f"   -> Warning: --webp-quality argument is ignored when output format is not WEBP.")
    
    if args.webp_lossless and args.output_format != 'webp':
        logger.warning("   -> Warning: --webp-lossless argument is ignored when output format is not WEBP.")

    try:
        # Create Config instance from parsed arguments.
        # args.overwrite will be True by default (from set_defaults) if no flag is given,
        # or True/False if a flag is explicitly used. This value is then passed to Config.
        return Config(**vars(args))
    except SystemExit: # Raised by parser.error()
        raise # Re-raise to exit script
    except Exception as e: # Catch any other errors during Config instantiation
        logger.critical(f"(!) Error: Error during configuration object creation: {e}", exc_info=True)
        sys.exit(1) # Exit with error

def display_settings(config: Config):
    """Prints the effective script settings to the console for user review."""
    print("\n" + "="*30 + " Script Settings " + "="*30)
    print(f"Input folder: {config.absolute_input_dir}")
    print(f"Output folder: {config.absolute_output_dir}")
    print(f"Resize ratio policy (-r, --ratio): {config.ratio}") 
    
    if config.ratio != 'none':
        resize_opts = config.resize_options
        size_info_parts = []
        if resize_opts.get('width', 0) > 0: size_info_parts.append(f"Width={resize_opts['width']}px")
        if resize_opts.get('height', 0) > 0: size_info_parts.append(f"Height={resize_opts['height']}px")
        
        size_desc = ""
        if config.ratio == 'aspect_ratio':
            size_desc = ' or '.join(size_info_parts) + " (maintaining aspect ratio)" if size_info_parts else "(maintaining aspect ratio, target dimensions not fully specified)"
        elif config.ratio == 'fixed':
             size_desc = f"{resize_opts.get('width','N/A')}x{resize_opts.get('height','N/A')}px"
        print(f"  Target size: {size_desc}")
        print(f"  Resize filter (--filter): {config.filter_names.get(resize_opts.get('filter_str'), 'N/A')} (Default: lanczos)")
    
    print(f"Output format (-f, --output-format): {config.supported_output_formats.get(config.output_format, 'N/A')} (Default: original)") 
    if config.output_format == 'webp':
        webp_mode_desc = "Lossless" if config.output_format_options.get('lossless') else f"Lossy (Quality: {config.output_format_options.get('quality')})"
        print(f"  WEBP Mode: {webp_mode_desc}")
    elif config.output_format == 'jpg':
        print(f"  JPG Quality (-q, --jpeg-quality): {config.output_format_options.get('quality')}")
    
    exif_handling_desc = "Strip all EXIF data" if config.strip_exif else "Preserve/Modify EXIF (default: reset orientation, remove thumbnail)"
    print(f"EXIF Handling: {exif_handling_desc}")
    
    overwrite_action = "Overwrite existing files" if config.overwrite else "Skip existing files"
    print(f"File Overwrite Policy (--overwrite/--no-overwrite): {overwrite_action}")
    
    if config.include_extensions:
        print(f"Include Extensions: {', '.join(config.include_extensions)}")
    elif config.exclude_extensions:
        print(f"Exclude Extensions (from default list): {', '.join(config.exclude_extensions)}")
    else:
        print(f"Processed Extensions: Default ({', '.join(config.supported_extensions)})")

    print(f"Log Level: {logging.getLevelName(logger.getEffectiveLevel())}")
    print("="*72)

def print_summary(processed: int, errors: int, skipped_overwrite: int, error_list: list, skipped_scan_list_reasons: list, output_dir: str):
    """Prints a summary of the batch processing results to the console."""
    print(f"\n\n--- Processing Summary ---")
    print(f"Successfully processed images: {processed}")
    print(f"Errors encountered during processing: {errors}")
    print(f"Skipped (due to --no-overwrite policy): {skipped_overwrite}")
    
    if error_list:
        print("\n[Files with Processing Errors]")
        for i, (filepath, errmsg) in enumerate(error_list[:20]): # Show first 20 errors
            print(f"  - '{filepath}': {errmsg}")
        if len(error_list) > 20:
            print(f"  ... and {len(error_list) - 20} more error(s). Check logs for full details if verbose.")
    
    if skipped_scan_list_reasons:
        print(f"\nItems skipped during initial scan ({len(skipped_scan_list_reasons)} total):")
        display_skipped_reasons = skipped_scan_list_reasons[:5] # Show first 5 reasons
        for reason in display_skipped_reasons:
            print(f"  - {reason}")
        if len(skipped_scan_list_reasons) > 5:
            print(f"  ... and {len(skipped_scan_list_reasons) - 5} more items skipped during scan. Enable verbose logging (-v) for full list.")
    else:
        print("\nNo items were skipped during the initial scan phase (or all found items were processable based on filters).")
        
    print("\n--- All tasks completed ---")
    print(f"Results saved in: '{output_dir}'")

# --- Main Execution Logic ---
def main():
    """
    Main function to orchestrate the script's execution.
    It handles argument parsing, configuration setup, batch processing, and summary display.
    """
    try:
        config = parse_arguments() # Parse arguments and create Config object

        logger.info(f"===== Image Resizer Script Started (v{__version__}) =====")
        display_settings(config) # Display the effective settings

        # Perform the core image processing
        processed_count, error_count, skipped_overwrite_count, error_files_details, skipped_scan_reasons = batch_process_images(config)
        
        # Display the summary of results
        print_summary(processed_count, error_count, skipped_overwrite_count, error_files_details, skipped_scan_reasons, config.absolute_output_dir)

        logger.info(f"===== Image Resizer Script Finished =====")
        # Exit with error code 1 if there were errors, 0 otherwise
        sys.exit(1 if error_count > 0 else 0) 
    except SystemExit as e: # Catch exits from argparse or sys.exit()
        if e.code is None or e.code == 0: 
            logger.info("Script exited normally.")
        else: 
            # Argparse errors usually print their own messages.
            # This will catch other sys.exit(N) calls.
            logger.error(f"Script exited with error code {e.code}.") 
            # No need to print to console here as argparse/logger already did.
    except Exception as e: # Catch any other unhandled exceptions
        logger.critical(f"An unhandled critical exception occurred: {e}", exc_info=True) # Log with traceback
        print(f"\n(!) Critical Error: {e}. Check logs for details.") # User-friendly message
        sys.exit(2) # Exit with a different error code for critical failures

if __name__ == "__main__":
    multiprocessing.freeze_support() # Important for PyInstaller compatibility on Windows
    main()
