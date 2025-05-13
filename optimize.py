# -*- coding: utf-8 -*-
from PIL import Image, UnidentifiedImageError
import os
import argparse
import sys

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

def optimize_image(input_path, output_path=None, jpg_quality=85, webp_quality=85, optimize=True, overwrite=False, lossless=False):
    try:
        if output_path is None:
            output_path = 'output'
        os.makedirs(output_path, exist_ok=True)

        img = Image.open(input_path)
        
        file_extension = os.path.splitext(input_path)[1].lower()
        output_file = os.path.join(output_path, os.path.basename(input_path))

        if os.path.exists(output_file) and not overwrite:
            print(f"Skipping '{os.path.basename(output_file)}' - File already exists")
            return

        save_options = {}
        
        if file_extension in ['.jpg', '.jpeg']:
            save_options.update({'quality': jpg_quality, 'optimize': optimize})
        elif file_extension == '.png':
            save_options.update({'optimize': optimize, 'compress_level': 9 if optimize else 6})
        elif file_extension == '.webp':
            save_options.update({'lossless': lossless})
            if not lossless:
                save_options.update({'quality': webp_quality, 'method': 6})
        elif file_extension == '.tiff':
            save_options.update({'compression': 'tiff_lzw'})
        
        img.save(output_file, **save_options)

    except UnidentifiedImageError:
        print(f"Cannot identify image file '{input_path}'")
    except FileNotFoundError:
        print(f"Input file '{input_path}' not found")
    except PermissionError:
        print(f"Permission denied accessing '{input_path}'")
    except Exception as e:
        print(f"Error occurred during image optimization: {e}")

def main():
    parser = argparse.ArgumentParser(description="Image Compression and Optimization Script")
    parser.add_argument("input_path", help="Input file path")
    parser.add_argument("-o", "--output-dir", help="Path to save optimized images", default="output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    parser.add_argument("--jpg-quality", type=int, default=85, help="JPEG image quality (1-100, default: 85)")
    parser.add_argument("--webp-quality", type=int, default=85, help="WebP image quality (1-100, default: 85, ignored when --lossless option is used)")
    parser.add_argument("--lossless", action="store_true", help="Use lossless compression for WebP and PNG (ignores WebP quality setting)")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_dir)

    try:
        if os.path.isfile(input_path):
            print(f"Processing file '{input_path}'...")
            optimize_image(input_path, output_path, jpg_quality=args.jpg_quality, webp_quality=args.webp_quality, optimize=True, overwrite=args.overwrite, lossless=args.lossless)
            print(f"Processing complete")
        else:
            print(f"'{input_path}' is not a valid file")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
