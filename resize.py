# -*- coding: utf-8 -*-
import os
import logging
from typing import Dict, Any, Optional, Tuple, List, NoReturn
import multiprocessing
import argparse
import sys

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
    logger.setLevel(logging.WARNING)

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

FILTER_NAMES = {
    "lanczos": "LANCZOS (High quality)",
    "bicubic": "BICUBIC (Medium quality)",
    "bilinear": "BILINEAR (Low quality)",
    "nearest": "NEAREST (Lowest quality)"
}

try:
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST
    }
except AttributeError:
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }

def static_process_image_worker(paths_tuple_and_config_tuple: Tuple[Tuple[str, str], dict]) -> Tuple[bool, Optional[str], str, str]:
    paths_tuple, config_obj = paths_tuple_and_config_tuple
    input_path, relative_path = paths_tuple
    return process_single_image_file(input_path, relative_path, config_obj)

def resize_image_maintain_aspect_ratio(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
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

def process_single_image_file(input_path: str, relative_path: str, config: dict) -> Tuple[bool, Optional[str], str, str]:
    base_name, original_ext = os.path.splitext(os.path.basename(input_path))
    resize_opts = config['resize_options']
    logger.debug(f"Processing: '{relative_path}'")

    output_relative_dir = os.path.dirname(relative_path)
    output_dir_for_file = os.path.join(config['absolute_output_dir'], output_relative_dir)

    if not os.path.isdir(output_dir_for_file):
         try:
             os.makedirs(output_dir_for_file, exist_ok=True)
             logger.info(f"Created output subdirectory: '{output_dir_for_file}'")
         except OSError as e:
              error_msg = f"Failed to create output subdirectory '{output_dir_for_file}' ({e})"
              logger.error(error_msg)
              return False, error_msg, input_path, relative_path

    output_path = os.path.join(output_dir_for_file, f"{base_name}{original_ext}")

    if not config['overwrite'] and os.path.exists(output_path):
        logger.info(f"Skipping '{relative_path}' as output file '{output_path}' already exists and overwrite is disabled.")
        return True, "skipped_overwrite", input_path, relative_path

    try:
        with Image.open(input_path) as img:
            logger.debug(f"Opened '{relative_path}'. Original size: {img.size}, mode: {img.mode}")

            processed_img = img
            if resize_opts['mode'] != 'none':
                if resize_opts['mode'] == 'aspect_ratio':
                    processed_img = resize_image_maintain_aspect_ratio(
                        img, resize_opts['width'], resize_opts['height'], resize_opts['filter_obj']
                    )
                elif resize_opts['mode'] == 'fixed':
                    processed_img = resize_image_fixed_size(
                        img, resize_opts['width'], resize_opts['height'], resize_opts['filter_obj']
                    )

            processed_img.save(output_path)

            logger.debug(f"Successfully processed and saved '{relative_path}' to '{output_path}'")
            return True, None, input_path, relative_path

    except UnidentifiedImageError:
        msg = "Invalid or corrupted image file. Pillow could not identify the image format."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        return False, msg, input_path, relative_path
    except PermissionError:
        msg = "File read/write permission denied."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        return False, msg, input_path, relative_path
    except FileNotFoundError:
        msg = "Input file not found during processing (should have been caught earlier)."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        return False, msg, input_path, relative_path
    except OSError as e:
        msg = f"File system or OS-level error occurred ({e})."
        logger.error(f"Processing failed for '{relative_path}': {msg}")
        if os.path.exists(output_path) and output_path != input_path:
            try:
                os.remove(output_path)
                logger.warning(f"Removed partially written/failed output file: '{output_path}'")
            except OSError as rm_e:
                logger.error(f"Could not remove partially written/failed output file '{output_path}': {rm_e}")
        return False, msg, input_path, relative_path
    except ValueError as e:
         msg = f"Image processing value error, likely from Pillow ({e})."
         logger.error(f"Processing failed for '{relative_path}': {msg}")
         return False, msg, input_path, relative_path
    except Exception as e:
        verbose_logging = config.get('verbose_global', False)
        msg = f"An unexpected error occurred ({type(e).__name__}: {e})."
        logger.critical(f"Processing failed for '{relative_path}': {msg}", exc_info=verbose_logging)
        return False, msg, input_path, relative_path

def scan_for_image_files(config: dict) -> Tuple[list[Tuple[str, str]], list[str]]:
    files_to_process = []
    skipped_scan_items_reasons = []
    input_dir = config['absolute_input_dir']

    items_found_to_evaluate = []
    try:
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
            return [], [f"Input directory '{input_dir}' not found or invalid."]

        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir)
                items_found_to_evaluate.append((full_path, relative_path))
        
        if not items_found_to_evaluate and os.path.isfile(input_dir):
             logger.warning(f"Input path '{input_dir}' is a file. Treating it as a single image to process.")
             items_found_to_evaluate.append((input_dir, os.path.basename(input_dir)))

    except OSError as e:
        logger.error(f"Error scanning input directory '{input_dir}': {e}")
        return [], [f"OS error while scanning '{input_dir}': {e}"]
    
    if not items_found_to_evaluate:
        logger.warning(f"No files found in input directory '{input_dir}'.")

    for input_path, relative_path in items_found_to_evaluate:
        filename = os.path.basename(input_path)
        file_ext = os.path.splitext(filename)[1].lower()

        if not file_ext:
            skipped_scan_items_reasons.append(f"Skipped '{relative_path}': No file extension.")
            logger.debug(f"Skipping '{relative_path}': No file extension.")
            continue

        if file_ext in SUPPORTED_EXTENSIONS:
            files_to_process.append((input_path, relative_path))
            logger.debug(f"Queued '{relative_path}' for processing.")
        else:
            reason = f"Skipped '{relative_path}': Extension '{file_ext}' is not in the supported list."
            skipped_scan_items_reasons.append(reason)
            logger.debug(reason)

    return files_to_process, skipped_scan_items_reasons

def batch_process_images(config: dict) -> Tuple[int, int, int, list[Tuple[str, str]], list[str]]:
    processed_count = 0
    error_count = 0
    skipped_overwrite_count = 0
    error_files_details = []
    all_skipped_scan_items_reasons = []

    try:
        os.makedirs(config['absolute_output_dir'], exist_ok=True)
    except OSError as e:
        logger.error(f"Fatal: Could not create base output directory '{config['absolute_output_dir']}': {e}")
        return 0, 0, 0, [], [f"Failed to create base output directory: {e}"]

    files_to_process, skipped_scan_reasons_list = scan_for_image_files(config)
    all_skipped_scan_items_reasons.extend(skipped_scan_reasons_list)
    total_files_to_attempt_processing = len(files_to_process)

    if total_files_to_attempt_processing == 0:
        logger.warning("No image files found to process after filtering.")
        return 0, 0, 0, [], all_skipped_scan_items_reasons

    num_processes = multiprocessing.cpu_count()
    tasks_with_config = [((input_p, rel_p), config) for input_p, rel_p in files_to_process]

    logger.info(f"Starting batch processing of {total_files_to_attempt_processing} images using up to {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=total_files_to_attempt_processing, desc="Processing images", unit="file", ncols=100, leave=True) as pbar:
            for result in pool.imap_unordered(static_process_image_worker, tasks_with_config):
                success, message, processed_input_path, processed_relative_path = result
                if success:
                    if message == "skipped_overwrite":
                        skipped_overwrite_count += 1
                    else:
                        processed_count += 1
                else:
                    error_count += 1
                    error_files_details.append((processed_relative_path or processed_input_path, message or "Unknown error"))
                pbar.update(1)

    return processed_count, error_count, skipped_overwrite_count, error_files_details, all_skipped_scan_items_reasons

def execute_resize_operation(config: dict) -> NoReturn:
    logger.info("===== Image Resizing Script Logic Started =====")

    try:
        processed_count, error_count, skipped_overwrite_count, error_files_details, all_skipped_scan_items_reasons = batch_process_images(config)
        
        summary_messages = []
        summary_messages.append(f"Total files processed: {processed_count}")
        if skipped_overwrite_count > 0:
            summary_messages.append(f"Files skipped (overwrite disabled): {skipped_overwrite_count}")
        if all_skipped_scan_items_reasons:
            summary_messages.append(f"Files/items skipped during scan: {len(all_skipped_scan_items_reasons)}")
        if error_count > 0:
            summary_messages.append(f"Errors encountered: {error_count}")
        
        final_summary = " | ".join(summary_messages)
        logger.info(f"Resize operation summary: {final_summary}")

        if error_count > 0:
            logger.warning(f"Image resizing finished with {error_count} errors.")
        else:
            logger.info("Image resizing finished successfully.")
        logger.info("===== Image Resizing Script Logic Finished =====")
        sys.exit(1 if error_count > 0 else 0)

    except Exception as e:
        verbose_logging = config.get('verbose_global', False)
        logger.critical(f"CRITICAL: An unhandled critical exception occurred in resize operation: {e}", exc_info=verbose_logging)
        print(f"\n(!) Critical Error in resize operation: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser(
        description="Py Image Toolkit CLI - Image Resizer",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_dir", nargs='?', default='input',
                               help="Path to the source image folder or a single image file (default: 'input' in current directory).")
    parser.add_argument("-o", "--output-dir", dest='output_dir', default='output',
                               help="Path to save processed images (default: 'output' in current directory).")
    parser.add_argument("--overwrite", dest='overwrite', action='store_true', default=False,
                                 help="Overwrite existing output files. If not specified, existing files will be skipped.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose_global", 
                        help="Enable verbose (DEBUG level) logging for detailed output across all commands.")

    parser.add_argument("-r", "--ratio",
                             choices=['aspect_ratio', 'fixed', 'none'],
                             default='aspect_ratio',
                             help="Resize ratio behavior (default: aspect_ratio):\n"
                                  "  aspect_ratio: Maintain aspect ratio to fit target Width/Height.\n"
                                  "  fixed: Force resize to target Width x Height (may distort).\n"
                                  "  none: No resizing.")

    parser.add_argument("-w", "--width", type=int, default=0,
                              help="Target width in pixels for resizing (used if --ratio is not 'none').")
    parser.add_argument("-H", "--height", type=int, default=0,
                              help="Target height in pixels for resizing (used if --ratio is not 'none').")
    parser.add_argument("--filter",
                              default='lanczos',
                              choices=FILTER_NAMES.keys(),
                              help="Resampling filter for resizing (default: lanczos):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]))

    try:
        args = parser.parse_args()

        if args.verbose_global:
            logger.setLevel(logging.DEBUG)
            log_handler.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled.")

        config = {
            'verbose_global': args.verbose_global,
            'overwrite': args.overwrite,
            'absolute_input_dir': os.path.abspath(args.input_dir),
            'absolute_output_dir': os.path.abspath(args.output_dir),
            'resize_options': {
                'mode': args.ratio,
                'width': args.width,
                'height': args.height,
                'filter_str': args.filter,
                'filter_obj': _PIL_RESAMPLE_FILTERS[args.filter]
            }
        }

        execute_resize_operation(config)
            
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)
    except Exception as e:
        print(f"(!) An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
