# -*- coding: utf-8 -*-
from PIL import Image, UnidentifiedImageError
import os
import argparse
import sys

OPTIMIZE_JPG_QUALITY = 85
OPTIMIZE_WEBP_QUALITY = 85
OPTIMIZE_LOSSLESS = False


def run(
    input_file,
    output_path=None,
    jpg_quality=OPTIMIZE_JPG_QUALITY,
    webp_quality=OPTIMIZE_WEBP_QUALITY,
    optimize=True,
    overwrite=False,
    lossless=OPTIMIZE_LOSSLESS,
):
    display_input_file = os.path.relpath(input_file)

    try:
        if output_path is None:
            output_path = os.path.abspath("output")

        os.makedirs(output_path, exist_ok=True)

        img = Image.open(input_file)

        file_extension = os.path.splitext(input_file)[1].lower()
        output_file_name_abs = os.path.join(output_path, os.path.basename(input_file))

        if os.path.exists(output_file_name_abs) and not overwrite:
            display_output_file = os.path.relpath(output_file_name_abs)
            print(f"Skipping '{display_output_file}' - File already exists")
            return

        save_options = {}

        if file_extension in [".jpg", ".jpeg"]:
            save_options.update({"quality": jpg_quality, "optimize": optimize})
        elif file_extension == ".png":
            save_options.update(
                {"optimize": optimize, "compress_level": 9 if optimize else 6}
            )
        elif file_extension == ".webp":
            save_options.update({"lossless": lossless})
            if not lossless:
                save_options.update({"quality": webp_quality, "method": 6})
        elif file_extension == ".tiff":
            save_options.update({"compression": "tiff_lzw"})

        img.save(output_file_name_abs, **save_options)

    except UnidentifiedImageError:
        print(f"Cannot identify image file '{display_input_file}'")
    except FileNotFoundError:
        print(f"Input file '{display_input_file}' not found")
    except PermissionError:
        print(f"Permission denied accessing '{display_input_file}'")
    except Exception as e:
        print(f"Error occurred during image optimization: {e}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Image Compression and Optimization Script"
    )
    parser.add_argument("input_file", help="Input file")
    parser.add_argument(
        "-o", "--output-dir", help="Path to save optimized images", default="output"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=OPTIMIZE_JPG_QUALITY,
        help=f"JPEG image quality (1-100, default: {OPTIMIZE_JPG_QUALITY})",
    )
    parser.add_argument(
        "--webp-quality",
        type=int,
        default=OPTIMIZE_WEBP_QUALITY,
        help=f"WebP image quality (1-100, default: {OPTIMIZE_WEBP_QUALITY}, ignored when --lossless option is used)",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        default=OPTIMIZE_LOSSLESS,
        help="Use lossless compression for WebP and PNG (ignores WebP quality setting)",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file_abs = os.path.abspath(args.input_file)
    output_path_abs = os.path.abspath(args.output_dir)

    display_input_file = os.path.relpath(input_file_abs)

    try:
        if os.path.isfile(input_file_abs):
            print(f"Processing file '{display_input_file}'...")
            run(
                input_file_abs,
                output_path_abs,
                jpg_quality=args.jpg_quality,
                webp_quality=args.webp_quality,
                optimize=True,
                overwrite=args.overwrite,
                lossless=args.lossless,
            )
            print(f"Processing complete")
        else:
            print(f"'{display_input_file}' is not a valid file")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
