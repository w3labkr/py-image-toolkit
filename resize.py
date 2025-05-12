# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Optional, Tuple, List, NoReturn
import argparse
import sys
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

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

def resize_image_maintain_aspect_ratio(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0:
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
        return img

    if (new_width, new_height) == (original_width, original_height):
        return img
    try:
        return img.resize((new_width, new_height), resample_filter)
    except ValueError:
        return img

def resize_image_fixed_size(img: Image.Image, target_width: int, target_height: int, resample_filter: Any) -> Image.Image:
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height):
        return img
    if target_width <= 0 or target_height <= 0:
        return img
    try:
        return img.resize((target_width, target_height), resample_filter)
    except ValueError:
        return img

def process_single_image_file(input_path: str, relative_path: str, config: dict) -> Tuple[bool, Optional[str], str, str]:
    base_name, original_ext = os.path.splitext(os.path.basename(input_path))
    resize_opts = config['resize_options']

    output_relative_dir = os.path.dirname(relative_path)
    output_dir_for_file = os.path.join(config['output_dir'], output_relative_dir)

    if not os.path.isdir(output_dir_for_file):
         try:
             os.makedirs(output_dir_for_file, exist_ok=True)
         except OSError as e:
              error_msg = f"Failed to create output subdirectory '{output_dir_for_file}' ({e})"
              return False, error_msg, input_path, relative_path

    output_path = os.path.join(output_dir_for_file, f"{base_name}{original_ext}")

    if not config['overwrite'] and os.path.exists(output_path):
        return True, "skipped_overwrite", input_path, relative_path

    try:
        with Image.open(input_path) as img:
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
            return True, None, input_path, relative_path

    except UnidentifiedImageError:
        msg = "Invalid or corrupted image file. Pillow could not identify the image format."
        return False, msg, input_path, relative_path
    except PermissionError:
        msg = "File read/write permission denied."
        return False, msg, input_path, relative_path
    except FileNotFoundError:
        msg = "Input file not found during processing (should have been caught earlier)."
        return False, msg, input_path, relative_path
    except OSError as e:
        msg = f"File system or OS-level error occurred ({e})."
        if os.path.exists(output_path) and output_path != input_path:
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False, msg, input_path, relative_path
    except ValueError as e:
         msg = f"Image processing value error, likely from Pillow ({e})."
         return False, msg, input_path, relative_path
    except Exception as e:
        msg = f"An unexpected error occurred ({type(e).__name__}: {e})."
        return False, msg, input_path, relative_path

def scan_for_image_files(config: dict) -> Tuple[list[Tuple[str, str]], list[str]]:
    files_to_process = []
    skipped_scan_items_reasons = []
    input_dir = config['input_dir']

    items_found_to_evaluate = []
    try:
        if not os.path.isdir(input_dir):
            return [], [f"Input directory '{input_dir}' not found or invalid."]

        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir)
                items_found_to_evaluate.append((full_path, relative_path))
        
        if not items_found_to_evaluate and os.path.isfile(input_dir):
             items_found_to_evaluate.append((input_dir, os.path.basename(input_dir)))

    except OSError as e:
        return [], [f"OS error while scanning '{input_dir}': {e}"]
    
    if not items_found_to_evaluate:
        pass

    for input_path, relative_path in items_found_to_evaluate:
        filename = os.path.basename(input_path)
        file_ext = os.path.splitext(filename)[1].lower()

        if not file_ext:
            skipped_scan_items_reasons.append(f"Skipped '{relative_path}': No file extension.")
            continue

        if file_ext in SUPPORTED_EXTENSIONS:
            files_to_process.append((input_path, relative_path))
        else:
            reason = f"Skipped '{relative_path}': Extension '{file_ext}' is not in the supported list."
            skipped_scan_items_reasons.append(reason)

    return files_to_process, skipped_scan_items_reasons

def batch_process_images(config: dict) -> Tuple[int, int, int, list[Tuple[str, str]], list[str]]:
    processed_count = 0
    error_count = 0
    skipped_overwrite_count = 0
    error_files_details = []
    all_skipped_scan_items_reasons = []

    try:
        os.makedirs(config['output_dir'], exist_ok=True)
    except OSError as e:
        return 0, 0, 0, [], [f"Failed to create base output directory: {e}"]

    files_to_process, skipped_scan_reasons_list = scan_for_image_files(config)
    all_skipped_scan_items_reasons.extend(skipped_scan_reasons_list)
    total_files_to_attempt_processing = len(files_to_process)

    if total_files_to_attempt_processing == 0:
        return 0, 0, 0, [], all_skipped_scan_items_reasons

    with tqdm(total=total_files_to_attempt_processing, desc="Processing images", unit="file", ncols=100, leave=True) as pbar:
        for input_p, rel_p in files_to_process:
            result = process_single_image_file(input_p, rel_p, config)
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

def execute_resize_operation(input_dir: str, output_dir: str, overwrite: bool, resize_mode: str,
                       width: int, height: int, filter_str: str, filter_obj: Any) -> NoReturn:
    try:
        options = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'overwrite': overwrite,
            'resize_options': {
                'mode': resize_mode,
                'width': width,
                'height': height,
                'filter_str': filter_str,
                'filter_obj': filter_obj
            }
        }
        processed_count, error_count, skipped_overwrite_count, error_files_details, all_skipped_scan_items_reasons = batch_process_images(options)
        
        summary_messages = []
        summary_messages.append(f"Total files processed: {processed_count}")
        if skipped_overwrite_count > 0:
            summary_messages.append(f"Files skipped (overwrite disabled): {skipped_overwrite_count}")
        if all_skipped_scan_items_reasons:
            summary_messages.append(f"Files/items skipped during scan: {len(all_skipped_scan_items_reasons)}")
        if error_count > 0:
            summary_messages.append(f"Errors encountered: {error_count}")
        
        final_summary = " | ".join(summary_messages)
        
        print(final_summary)
        
        if error_count > 0:
            print("\nError Details:")
            for file_path, error_msg in error_files_details:
                print(f"- {file_path}: {error_msg}")
                
        if all_skipped_scan_items_reasons and len(all_skipped_scan_items_reasons) < 10:
            print("\nSkipped Files List:")
            for reason in all_skipped_scan_items_reasons:
                print(f"- {reason}")
        
        sys.exit(1 if error_count > 0 else 0)

    except Exception as e:
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser(
        description="Py Image Toolkit CLI - Image Resizer",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_dir", nargs='?', default='input', help="Path to the source image folder or a single image file (default: 'input' in current directory).")
    parser.add_argument("-o", "--output-dir", dest='output_dir', default='output', help="Path to save processed images (default: 'output' in current directory).")
    parser.add_argument("--overwrite", dest='overwrite', action='store_true', default=False, help="Overwrite existing output files. If not specified, existing files will be skipped.")

    parser.add_argument("--ratio", choices=['aspect_ratio', 'fixed', 'none'], default='aspect_ratio',
                       help="Resize ratio behavior (default: aspect_ratio):\n"
                            "  aspect_ratio: Maintain aspect ratio to fit target Width/Height.\n"
                            "  fixed: Force resize to target Width x Height (may distort).\n"
                            "  none: No resizing.")

    parser.add_argument("-w", "--width", type=int, default=0, help="Target width in pixels for resizing (used if --ratio is not 'none').")
    parser.add_argument("-H", "--height", type=int, default=0, help="Target height in pixels for resizing (used if --ratio is not 'none').")
    parser.add_argument("--filter", default='lanczos', choices=FILTER_NAMES.keys(),
                       help="Resampling filter for resizing (default: lanczos):\n" +
                            "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]))

    try:
        args = parser.parse_args()

        execute_resize_operation(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            overwrite=args.overwrite, 
            resize_mode=args.ratio,
            width=args.width, 
            height=args.height, 
            filter_str=args.filter,
            filter_obj=_PIL_RESAMPLE_FILTERS[args.filter]
        )
            
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)
    except Exception as e:
        sys.exit(2)

if __name__ == "__main__":
    main()
