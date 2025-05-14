# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Optional, Tuple, NoReturn
import argparse
import sys
from PIL import Image, UnidentifiedImageError

RESIZE_SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
)

RESIZE_FILTER_NAMES = {
    "lanczos": "LANCZOS (High quality)",
    "bicubic": "BICUBIC (Medium quality)",
    "bilinear": "BILINEAR (Low quality)",
    "nearest": "NEAREST (Lowest quality)",
}

try:
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST,
    }
except AttributeError:
    _PIL_RESAMPLE_FILTERS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }


def resize_image_maintain_aspect_ratio(
    img: Image.Image, target_width: int, target_height: int, resample_filter: Any
) -> Image.Image:
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


def resize_image_fixed_size(
    img: Image.Image, target_width: int, target_height: int, resample_filter: Any
) -> Image.Image:
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height):
        return img
    if target_width <= 0 or target_height <= 0:
        return img
    try:
        return img.resize((target_width, target_height), resample_filter)
    except ValueError:
        return img


def process_image_file(
    input_path: str, output_path: str, resize_opts: dict
) -> Tuple[bool, Optional[str]]:
    if not os.path.isfile(input_path):
        return False, f"Input file '{input_path}' not found."

    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext not in RESIZE_SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file extension: '{file_ext}'"

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            return False, f"Failed to create output directory '{output_dir}': {e}"

    try:
        with Image.open(input_path) as img:
            processed_img = img
            if resize_opts["mode"] != "none":
                if resize_opts["mode"] == "aspect_ratio":
                    processed_img = resize_image_maintain_aspect_ratio(
                        img,
                        resize_opts["width"],
                        resize_opts["height"],
                        resize_opts["filter_obj"],
                    )
                elif resize_opts["mode"] == "fixed":
                    processed_img = resize_image_fixed_size(
                        img,
                        resize_opts["width"],
                        resize_opts["height"],
                        resize_opts["filter_obj"],
                    )

            processed_img.save(output_path)
            return True, None

    except UnidentifiedImageError:
        return (
            False,
            "Invalid or corrupt image file. Pillow cannot recognize the image format.",
        )
    except PermissionError:
        return False, "Permission denied for file read/write operations."
    except FileNotFoundError:
        return False, "Input file not found."
    except OSError as e:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False, f"File system or OS level error occurred: {e}"
    except ValueError as e:
        return False, f"Image processing value error from Pillow: {e}"
    except Exception as e:
        return False, f"Unexpected error occurred: {type(e).__name__}: {e}"


def run(
    input_file: str,
    output_file: str,
    overwrite: bool,
    resize_mode: str,
    width: int,
    height: int,
    filter_str: str,
    filter_obj: Any,
) -> NoReturn:
    try:
        if not overwrite and os.path.exists(output_file):
            print(
                f"'{os.path.basename(output_file)}' file already exists and overwrite option is disabled, skipping."
            )
            sys.exit(0)

        resize_options = {
            "mode": resize_mode,
            "width": width,
            "height": height,
            "filter_str": filter_str,
            "filter_obj": filter_obj,
        }

        success, error_msg = process_image_file(input_file, output_file, resize_options)

        if success:
            relative_output_file = os.path.relpath(output_file)
            print(
                f"Image processing complete: {os.path.relpath(input_file)} -> {relative_output_file}"
            )
            sys.exit(0)
        else:
            print(f"Image processing failed: {error_msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(2)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Py Image Toolkit CLI - Image Resizer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", help="Input image file with path")
    parser.add_argument(
        "-o", "--output-dir", help="Path to save resized images", default="output"
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not specified, existing files will be skipped.",
    )

    parser.add_argument(
        "--ratio",
        choices=["aspect_ratio", "fixed", "none"],
        default="aspect_ratio",
        help="Resizing ratio behavior (default: aspect_ratio):\n"
        "  aspect_ratio: Maintains aspect ratio to fit target dimensions.\n"
        "  fixed: Forces resize to exact target dimensions (may distort image).\n"
        "  none: No resizing.",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=0,
        help="Target width in pixels for resizing (used when --ratio is not 'none').",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=0,
        help="Target height in pixels for resizing (used when --ratio is not 'none').",
    )
    parser.add_argument(
        "--filter",
        default="lanczos",
        choices=RESIZE_FILTER_NAMES.keys(),
        help="Resampling filter for resizing (default: lanczos):\n"
        + "\n".join([f"  {k}: {v}" for k, v in RESIZE_FILTER_NAMES.items()]),
    )
    return parser


def main():
    parser = get_parser()
    try:
        args = parser.parse_args()

        input_file = os.path.abspath(args.input_file)
        output_dir = os.path.abspath(args.output_dir)

        os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, output_filename)

        run(
            input_file=input_file,
            output_file=output_file,
            overwrite=args.overwrite,
            resize_mode=args.ratio,
            width=args.width,
            height=args.height,
            filter_str=args.filter,
            filter_obj=_PIL_RESAMPLE_FILTERS[args.filter],
        )

    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
