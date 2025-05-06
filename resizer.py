# -*- coding: utf-8 -*-
import os
import sys
import importlib.util # For library check
import argparse # For command line argument processing

# --- Library Check and Load ---
REQUIRED_LIBS = {
    "Pillow": "PIL",
    "piexif": "piexif", # For EXIF processing
    "tqdm": "tqdm"      # For progress display
}

missing_libs = []
for package_name, import_name in REQUIRED_LIBS.items():
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        missing_libs.append(package_name)

if missing_libs:
    print("(!) Error: Required libraries for script execution are not installed.")
    print("    Please install them using the following commands:")
    for lib in missing_libs:
        print(f"    pip install {lib}")
    sys.exit(1)

# Load libraries
from PIL import Image, UnidentifiedImageError
import piexif
from tqdm import tqdm

# --- Constants ---
SCRIPT_VERSION = "2.1" # Version update (tqdm progress bar added)

# Define resampling filters for Pillow version compatibility
try:
    # Pillow 9.1.0 or later
    RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST
    }
    FILTER_NAMES = {
        "lanczos": "LANCZOS (High quality)",
        "bicubic": "BICUBIC (Medium quality)",
        "bilinear": "BILINEAR (Low quality)",
        "nearest": "NEAREST (Lowest quality)"
    }
except AttributeError:
    # Older Pillow version compatibility
    RESAMPLE_FILTERS = {
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

# Define supported output formats (names to be used in argparse)
SUPPORTED_OUTPUT_FORMATS = {
    "original": "Keep Original",
    "png": "PNG",
    "jpg": "JPG",
    "webp": "WEBP",
}

# Supported input image extensions
SUPPORTED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# --- Core Image Processing Functions ---

def resize_image_maintain_aspect_ratio(img, target_width, target_height, resample_filter):
    """
    Resizes the image while maintaining the aspect ratio.
    If only target_width or target_height is provided, the other is calculated.
    If both are provided, the image is resized to fit within the bounds, maintaining ratio.

    Args:
        img (PIL.Image.Image): The image object to resize.
        target_width (int): The target width in pixels. Use 0 to calculate based on height.
        target_height (int): The target height in pixels. Use 0 to calculate based on width.
        resample_filter (int): The resampling filter to use (e.g., Image.Resampling.LANCZOS).

    Returns:
        PIL.Image.Image: The resized image object, or the original if no resize is needed or possible.
    """
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0: return img # Invalid original size

    new_width = 0
    new_height = 0

    if target_width > 0 and target_height > 0:
        # If both width and height are specified: maintain ratio within the specified maximum size
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(original_width * ratio))
        new_height = max(1, int(original_height * ratio))
    elif target_width > 0:
        # If only width is specified: calculate height based on width
        ratio = target_width / original_width
        new_width = target_width
        new_height = max(1, int(original_height * ratio))
    elif target_height > 0:
        # If only height is specified: calculate width based on height
        ratio = target_height / original_height
        new_height = target_height
        new_width = max(1, int(original_width * ratio))
    else:
        # If neither width nor height is specified (should not happen with arg validation, but defensive)
        return img

    # If no size change, return original
    if (new_width, new_height) == (original_width, original_height): return img

    try:
        return img.resize((new_width, new_height), resample_filter)
    except ValueError as e:
        tqdm.write(f"   -> Warning: Error during aspect ratio resize (({original_width},{original_height}) -> ({new_width},{new_height})): {e}. Using original image.")
        return img

def resize_image_fixed_size(img, target_width, target_height, resample_filter):
    """
    Forces image resize to the specified dimensions, potentially distorting aspect ratio.

    Args:
        img (PIL.Image.Image): The image object to resize.
        target_width (int): The exact target width in pixels. Must be > 0.
        target_height (int): The exact target height in pixels. Must be > 0.
        resample_filter (int): The resampling filter to use.

    Returns:
        PIL.Image.Image: The resized image object, or the original if dimensions match or are invalid.
    """
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height): return img
    if target_width <= 0 or target_height <= 0:
        return img
    try:
        return img.resize((target_width, target_height), resample_filter)
    except ValueError as e:
        tqdm.write(f"   -> Warning: Error during fixed size resize (({original_width},{original_height}) -> ({target_width},{target_height})): {e}. Using original image.")
        return img


def get_unique_filepath(filepath):
    """
    Checks if a file path exists. If it does, generates a unique path by appending '_<number>'
    before the extension.

    Args:
        filepath (str): The desired file path.

    Returns:
        str: The original filepath if it doesn't exist, or a unique version.
    """
    if not os.path.exists(filepath): return filepath
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def prepare_image_for_save(img, output_format_str):
    """
    Converts the image mode if necessary for the target output format (e.g., RGBA to RGB for JPG).

    Args:
        img (PIL.Image.Image): The input image object.
        output_format_str (str | None): The target format ('JPG', 'PNG', 'WEBP', etc.) in uppercase,
                                        or None if format is 'original'.

    Returns:
        PIL.Image.Image: The image object, potentially converted to a suitable mode.
    """
    save_img = img
    if output_format_str == 'JPG':
        # JPG does not support transparency. Convert modes with alpha or palette to RGB.
        if img.mode in ('RGBA', 'LA', 'P'):
            if img.mode == 'P' and 'transparency' in img.info:
                # Convert P with transparency to RGBA first, then handle RGBA
                save_img = img.convert('RGBA')
            elif img.mode == 'P': # P without transparency
                save_img = img.convert('RGB')

            # Handle RGBA or LA (which might have come from P with transparency)
            if save_img.mode in ('RGBA', 'LA'):
                # Create a white background and paste the image onto it
                background = Image.new("RGB", save_img.size, (255, 255, 255))
                try:
                    # Use alpha channel as mask if available
                    mask = save_img.split()[-1]
                    background.paste(save_img, mask=mask)
                    save_img = background
                except (IndexError, ValueError): # If no alpha channel or cannot split
                    save_img = save_img.convert('RGB') # Fallback to simple conversion

    elif output_format_str == 'WEBP':
         # WebP supports RGBA, RGB, L. Convert P mode.
         if img.mode == 'P':
              if 'transparency' in img.info:
                   save_img = img.convert("RGBA") # Preserve transparency
              else:
                   save_img = img.convert("RGB") # No transparency

    return save_img


def process_images(input_folder, output_folder, resize_options, output_format_options, process_recursive):
    """
    Processes all supported image files in the input folder (and optionally subfolders).
    Applies resizing, format conversion, and preserves EXIF metadata.

    Args:
        input_folder (str): Path to the source image folder.
        output_folder (str): Path to the destination folder for processed images.
        resize_options (dict): Dictionary containing resize parameters:
                                'mode' (str: 'aspect_ratio', 'fixed', 'none'),
                                'width' (int), 'height' (int),
                                'filter_obj' (PIL.Image.Resampling filter).
        output_format_options (dict): Dictionary containing output format parameters:
                                     'format_str' (str: 'jpg', 'png', 'webp', 'original'),
                                     'quality' (int, optional for jpg/webp).
        process_recursive (bool): Whether to process images in subfolders.

    Returns:
        tuple: Contains:
               - processed_count (int): Number of successfully processed images.
               - error_count (int): Number of images that failed processing.
               - error_files (list): List of tuples (relative_filepath, error_message).
               - skipped_files (list): List of skipped file/folder names with reasons.
    """
    processed_count = 0
    error_count = 0
    skipped_files = []
    error_files = [] # List of files that caused errors (filename, error message)
    absolute_output_folder = os.path.abspath(output_folder) # Use absolute path

    # --- Scan for image files ---
    files_to_process = [] # Store (input_path, relative_path) tuples
    try:
        print(f"\nScanning for image files in input folder '{input_folder}'...")
        if process_recursive:
            for root, dirs, files in os.walk(input_folder):
                # Prevent scanning the output folder itself if it's inside the input folder
                if os.path.abspath(root).startswith(absolute_output_folder):
                    dirs[:] = [] # Don't traverse into subdirectories of the output folder
                    continue
                for filename in files:
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        input_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(input_path, input_folder)
                        files_to_process.append((input_path, relative_path))
                    else:
                         relative_path = os.path.relpath(os.path.join(root, filename), input_folder)
                         skipped_files.append(relative_path + " (Unsupported format)")
        else:
            # Scan only the top-level directory
            for filename in os.listdir(input_folder):
                input_path = os.path.join(input_folder, filename)
                if os.path.isfile(input_path):
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        files_to_process.append((input_path, filename)) # Relative path is just filename
                    else:
                        skipped_files.append(filename + " (Unsupported format)")
                elif os.path.isdir(input_path):
                    # Add output folder to skip list if it's directly inside input folder
                    if os.path.abspath(input_path) == absolute_output_folder:
                         skipped_files.append(filename + " (Output folder)")
                    else:
                         skipped_files.append(filename + " (Folder)")

        total_files = len(files_to_process)
        if total_files == 0:
             print("(!) No image files to process found in the specified path.")
             return 0, 0, [], []
        else:
            print(f"-> Found a total of {total_files} image files. Starting processing.")

    except Exception as e:
        print(f"(!) Critical Error: Failed to access input path '{input_folder}'. ({e})")
        return 0, 0, [], []

    # --- Process each file with tqdm progress bar ---
    for input_path, relative_path in tqdm(files_to_process, desc="Processing images", unit="file", ncols=100, leave=True):
        filename = os.path.basename(input_path)

        # Determine output path, maintaining subfolder structure if recursive
        output_relative_dir = os.path.dirname(relative_path)
        output_dir_for_file = os.path.join(absolute_output_folder, output_relative_dir)
        if not os.path.exists(output_dir_for_file):
            try:
                os.makedirs(output_dir_for_file)
            except OSError as e:
                error_msg = f"Failed to create output subfolder: {output_dir_for_file} ({e})"
                tqdm.write(f" ✗ Error: {error_msg} ({relative_path})")
                error_files.append((relative_path, error_msg))
                error_count += 1
                continue # Skip to next file

        # Determine output filename and extension
        base_name, original_ext = os.path.splitext(filename)
        output_format_str = output_format_options['format_str']
        output_ext = ""
        if output_format_str == 'original': output_ext = original_ext
        elif output_format_str == 'jpg': output_ext = '.jpg'
        elif output_format_str == 'webp': output_ext = '.webp'
        elif output_format_str == 'png': output_ext = '.png'
        else: output_ext = original_ext # Fallback

        output_filename = base_name + output_ext
        output_path_base = os.path.join(output_dir_for_file, output_filename)
        output_path = get_unique_filepath(output_path_base) # Ensure unique filename

        exif_data = None # Parsed EXIF dictionary
        original_exif_bytes = None # Raw EXIF bytes

        try:
            with Image.open(input_path) as img:
                # --- EXIF Handling (Load) ---
                if 'exif' in img.info and img.info['exif']:
                    original_exif_bytes = img.info['exif']
                    try:
                        exif_data = piexif.load(original_exif_bytes)
                    except Exception as exif_err:
                        # Warning if EXIF parsing fails, but continue processing
                        # tqdm.write(f"   -> Warning: Failed to load/parse EXIF data for '{filename}'. ({type(exif_err).__name__})")
                        exif_data = None
                        original_exif_bytes = None # Don't try to use corrupted raw bytes

                # --- Prepare Image Mode for Saving ---
                save_format_upper = output_format_str.upper() if output_format_str != 'original' else None
                img_prepared = prepare_image_for_save(img, save_format_upper)

                # --- Resize ---
                resample_filter = resize_options.get('filter_obj')
                img_resized = img_prepared
                if resize_options['mode'] == 'aspect_ratio':
                    img_resized = resize_image_maintain_aspect_ratio(
                        img_prepared, resize_options['width'], resize_options['height'], resample_filter
                    )
                elif resize_options['mode'] == 'fixed':
                    img_resized = resize_image_fixed_size(
                        img_prepared, resize_options['width'], resize_options['height'], resample_filter
                    )
                # No action needed for 'none' mode

                # --- Prepare Save Options ---
                save_kwargs = {}
                save_format_arg = None # Format string for save() method
                if output_format_str != 'original':
                    save_format_arg = save_format_upper
                    if save_format_arg == 'JPG':
                         save_format_arg = 'JPEG' # Pillow uses 'JPEG'

                # --- EXIF Handling (Prepare for Save) ---
                final_exif_bytes = None
                if exif_data: # If EXIF was successfully parsed
                    try:
                        # Reset orientation tag (as image is now potentially resized/reoriented)
                        if piexif.ImageIFD.Orientation in exif_data.get('0th', {}):
                            exif_data['0th'][piexif.ImageIFD.Orientation] = 1 # 1 = Normal
                        # Remove thumbnail data (often becomes invalid after resize)
                        if 'thumbnail' in exif_data and exif_data['thumbnail']:
                             exif_data['thumbnail'] = None
                        final_exif_bytes = piexif.dump(exif_data)
                    except Exception as dump_err:
                        # Warning if dumping modified EXIF fails
                        # tqdm.write(f"   -> Warning: Failed to dump modified EXIF data for '{filename}'. ({dump_err})")
                        final_exif_bytes = original_exif_bytes # Try saving original bytes if dump failed
                elif original_exif_bytes: # If parsing failed but we have original bytes
                     final_exif_bytes = original_exif_bytes

                # --- Apply Format-Specific Options & EXIF ---
                if output_format_str == 'jpg':
                    save_kwargs['quality'] = output_format_options.get('quality', 95)
                    save_kwargs['optimize'] = True
                    save_kwargs['progressive'] = True
                    if final_exif_bytes: save_kwargs['exif'] = final_exif_bytes
                elif output_format_str == 'png':
                    save_kwargs['optimize'] = True
                    # PNG EXIF support can be tricky/version-dependent; attempt it
                    if final_exif_bytes:
                         try: save_kwargs['exif'] = final_exif_bytes
                         except TypeError: pass # Silently ignore if Pillow version doesn't support exif for PNG
                elif output_format_str == 'webp':
                    save_kwargs['quality'] = output_format_options.get('quality', 80)
                    save_kwargs['lossless'] = False # Assuming lossy WEBP by default
                    if final_exif_bytes: save_kwargs['exif'] = final_exif_bytes
                elif output_format_str == 'original' and final_exif_bytes:
                    # Only attempt to save EXIF if the original format supports it
                    if original_ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.png', '.webp']:
                         save_kwargs['exif'] = final_exif_bytes

                # --- Save Result ---
                img_resized.save(output_path, format=save_format_arg, **save_kwargs)
                processed_count += 1

        # --- Individual File Error Handling ---
        except UnidentifiedImageError:
            error_msg = "Invalid or corrupted image file"
            tqdm.write(f" ✗ Error: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
        except PermissionError:
            error_msg = "File read/write permission denied"
            tqdm.write(f" ✗ Error: '{relative_path}' or output path ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
        except OSError as e:
            error_msg = f"File system error ({e})"
            tqdm.write(f" ✗ Error: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
            # Attempt to clean up partially created file on OS error
            if os.path.exists(output_path):
                  try: os.remove(output_path)
                  except OSError: pass
        except ValueError as e: # e.g., Pillow internal processing error
             error_msg = f"Image processing value error ({e})"
             tqdm.write(f" ✗ Error: '{relative_path}' ({error_msg})")
             error_files.append((relative_path, error_msg))
             error_count += 1
        except Exception as e:
            # Catch-all for unexpected errors during processing of a single file
            import traceback
            error_msg = f"Unexpected error ({type(e).__name__}: {e})"
            tqdm.write(f" ✗ Error: '{relative_path}' ({error_msg})")
            # traceback.print_exc() # Uncomment for detailed stack trace during debugging
            error_files.append((relative_path, error_msg))
            error_count += 1

    # --- Return Summary ---
    return processed_count, error_count, error_files, skipped_files


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Batch Image Resizer Script (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawTextHelpFormatter, # Preserve help message line breaks
        usage="%(prog)s [input_directory] -m <mode> -O <format> [options]"
    )

    # --- Argument Definitions ---
    parser.add_argument("input_directory", nargs='?', default='.',
                        help="Path to the source folder containing images (default: current directory)")

    required_group = parser.add_argument_group('Required Options')
    required_group.add_argument("-m", "--resize-mode", required=True, choices=['aspect_ratio', 'fixed', 'none'],
                                help="Resize mode:\n"
                                     "  aspect_ratio: Maintain aspect ratio (specify width and/or height)\n"
                                     "  fixed: Force resize to specified dimensions (may distort ratio)\n"
                                     "  none: No resizing (only format conversion, EXIF processing)")
    required_group.add_argument("-O", "--output-format", required=True, choices=SUPPORTED_OUTPUT_FORMATS.keys(),
                                help="Output file format:\n" +
                                     "\n".join([f"  {k}: {v}" for k, v in SUPPORTED_OUTPUT_FORMATS.items()]))

    parser.add_argument("-o", "--output-dir",
                        help="Path to the folder where results will be saved (default: 'resized_images' under input folder)")

    resize_group = parser.add_argument_group('Resize Options (ignored if mode is "none")')
    resize_group.add_argument("-w", "--width", type=int, default=0,
                              help="Target width (px). Required for 'fixed'. Used for 'aspect_ratio' if height is 0.")
    resize_group.add_argument("-H", "--height", type=int, default=0,
                              help="Target height (px). Required for 'fixed'. Used for 'aspect_ratio' if width is 0.")
    resize_group.add_argument("-f", "--filter", choices=FILTER_NAMES.keys(), default=None,
                              help="Resize filter (quality/speed):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]) +
                                   "\n(Required if mode is 'aspect_ratio' or 'fixed')")

    optional_group = parser.add_argument_group('Other Optional Options')
    optional_group.add_argument("-r", "--recursive", action="store_true",
                                help="Include images in subfolders for processing")
    optional_group.add_argument("-q", "--quality", type=int, default=None,
                                help="JPG/WEBP save quality (1-100). Default: JPG=95, WEBP=80")
    optional_group.add_argument('--version', action='version', version=f'%(prog)s {SCRIPT_VERSION}')

    args = parser.parse_args()

    # --- Argument Validation and Setup ---

    if not os.path.isdir(args.input_directory):
        parser.error(f"Invalid input folder path: {args.input_directory}")
    absolute_input_dir = os.path.abspath(args.input_directory)

    resize_opts = {'mode': args.resize_mode}
    if args.resize_mode == 'aspect_ratio':
        if args.width <= 0 and args.height <= 0:
            parser.error("Mode 'aspect_ratio' requires --width (-w) or --height (-H) > 0.")
        if args.width < 0 or args.height < 0:
             parser.error("--width (-w) and --height (-H) cannot be negative.")
        if not args.filter:
            parser.error("--filter (-f) is required for mode 'aspect_ratio'.")
        resize_opts.update({
            'width': args.width,
            'height': args.height,
            'filter_str': args.filter,
            'filter_obj': RESAMPLE_FILTERS[args.filter]
        })
    elif args.resize_mode == 'fixed':
        if args.width <= 0 or args.height <= 0:
            parser.error("Mode 'fixed' requires both --width (-w) and --height (-H) > 0.")
        if not args.filter:
            parser.error("--filter (-f) is required for mode 'fixed'.")
        resize_opts.update({
            'width': args.width,
            'height': args.height,
            'filter_str': args.filter,
            'filter_obj': RESAMPLE_FILTERS[args.filter]
        })
    else: # args.resize_mode == 'none'
        if args.width > 0 or args.height > 0 or args.filter:
             print("   -> Info: --width, --height, and --filter are ignored when --resize-mode is 'none'.")

    if args.output_dir:
        absolute_output_dir = os.path.abspath(args.output_dir)
    else:
        absolute_output_dir = os.path.join(absolute_input_dir, "resized_images")
        print(f"   -> Info: Output folder not specified. Using default: '{absolute_output_dir}'")

    if absolute_input_dir == absolute_output_dir:
        parser.error("Input folder and output folder cannot be the same.")

    # Prevent output folder inside input folder when using recursive mode
    try:
        rel_path = os.path.relpath(absolute_output_dir, start=absolute_input_dir)
        # Check if rel_path doesn't start with '..' (meaning it's not outside) and isn't '.' (same folder)
        if args.recursive and not rel_path.startswith(os.pardir) and rel_path != '.':
             # More specific check: ensure it's not a direct subfolder or deeper
             # This check works correctly even if paths are like /a/b and /a/b/c
             if os.path.commonpath([absolute_input_dir]) == os.path.commonpath([absolute_input_dir, absolute_output_dir]):
                parser.error("When using --recursive, the output folder cannot be inside the input folder to prevent infinite loops.")
    except ValueError:
        # Paths are on different drives (Windows), which is not a conflict.
        pass


    try:
        if not os.path.exists(absolute_output_dir):
            os.makedirs(absolute_output_dir)
            print(f"   -> Info: Created output folder: '{absolute_output_dir}'")
    except OSError as e:
        parser.error(f"Cannot create output folder: {absolute_output_dir} ({e})")

    output_format_opts = {'format_str': args.output_format}
    if args.output_format in ('jpg', 'webp'):
        default_quality = 95 if args.output_format == 'jpg' else 80
        quality = args.quality if args.quality is not None else default_quality
        if not (1 <= quality <= 100):
            parser.error(f"--quality must be between 1 and 100 (got {args.quality}).")
        output_format_opts['quality'] = quality
    elif args.quality is not None:
        print(f"   -> Warning: --quality (-q) is ignored for output format '{args.output_format}'.")


    # --- Display Settings and Start ---
    print("\n" + "="*30 + " Script Settings " + "="*30)
    print(f"Input folder: {absolute_input_dir}")
    print(f"Output folder: {absolute_output_dir}")
    print(f"Include subfolders: {'Yes' if args.recursive else 'No'}")
    print(f"Resize mode: {args.resize_mode}")
    if args.resize_mode != 'none':
        size_info = []
        if resize_opts.get('width', 0) > 0: size_info.append(f"Width={resize_opts['width']}px")
        if resize_opts.get('height', 0) > 0: size_info.append(f"Height={resize_opts['height']}px")
        if args.resize_mode == 'aspect_ratio': print(f"  Target size: {' and/or '.join(size_info)} (maintaining aspect ratio)")
        else: print(f"  Fixed size: {resize_opts['width']}x{resize_opts['height']}px")
        print(f"  Resize filter: {FILTER_NAMES[resize_opts['filter_str']]}")
    print(f"Output format: {SUPPORTED_OUTPUT_FORMATS[output_format_opts['format_str']]}")
    if 'quality' in output_format_opts: print(f"  Quality: {output_format_opts['quality']}")
    print(f"Preserve EXIF metadata: Yes (default, where supported by format)")
    print("="*72)

    # --- Run Processing ---
    processed_count, error_count, error_files, skipped_files = process_images(
        absolute_input_dir,
        absolute_output_dir,
        resize_opts,
        output_format_opts,
        args.recursive,
    )

    # --- Final Summary ---
    print(f"\n\n--- Processing Summary ---")
    print(f"Successfully processed images: {processed_count}")
    print(f"Errors encountered: {error_count}")
    if error_files:
        print("\n[Files with Errors]")
        max_errors_to_show = 20
        for i, (filepath, errmsg) in enumerate(error_files):
            if i >= max_errors_to_show:
                print(f"  ... and {len(error_files) - max_errors_to_show} more error(s) not listed.")
                break
            print(f"  - {filepath}: {errmsg}")

    if skipped_files:
        print(f"\nSkipped files/folders ({len(skipped_files)} total):")
        limit = 10
        display_list = skipped_files[:limit]
        print(f"  {', '.join(display_list)}{'...' if len(skipped_files) > limit else ''}")
    else:
        print("\nNo files or folders were skipped.")

    print("\n--- All tasks completed ---")
    print(f"Results saved in: '{absolute_output_dir}'")

    # Exit with non-zero status if errors occurred
    if error_count > 0:
        sys.exit(1)
