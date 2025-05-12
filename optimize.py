# -*- coding: utf-8 -*-
from PIL import Image, UnidentifiedImageError
import os
import argparse
import sys
from tqdm import tqdm

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

def optimize_image(input_path, output_path=None, quality=85, optimize=True, overwrite=False):
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
            save_options.update({'quality': quality, 'optimize': optimize})
        elif file_extension == '.png':
            save_options.update({'optimize': optimize})
        elif file_extension == '.webp':
            save_options.update({'quality': quality, 'method': 6})
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

def process_directory(input_dir, output_dir, quality, optimize=True, overwrite=False):
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    files_processed = 0
    errors = 0
    
    files = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file)) and os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS]
    
    for file in tqdm(files, desc="Optimizing images", unit="file", ncols=100, leave=True):
        file_path = os.path.join(input_dir, file)
        try:
            optimize_image(file_path, output_dir, quality, optimize, overwrite)
            files_processed += 1
        except Exception as e:
            print(f"Error processing file '{file}': {e}")
            errors += 1
    
    return files_processed, errors

def main():
    parser = argparse.ArgumentParser(description="Image Compression and Optimization Script")
    parser.add_argument("input_path", help="Input file or directory path", nargs='?', default="input")
    parser.add_argument("-o", "--output-dir", help="Path to save optimized images", default="output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--jpg-quality", type=int, default=85, help="JPEG image quality (1-100, default: 85)")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_dir)

    try:
        if os.path.isdir(input_path):
            print(f"Processing directory '{input_path}'...")
            files_processed, errors = process_directory(
                input_path, 
                output_path, 
                args.jpg_quality, 
                optimize=True,
                overwrite=args.overwrite
            )
            print(f"Processing complete: {files_processed} files processed, {errors} errors occurred")
        elif os.path.isfile(input_path):
            print(f"Processing file '{input_path}'...")
            optimize_image(input_path, output_path, args.jpg_quality, optimize=True, overwrite=args.overwrite)
        else:
            print(f"'{input_path}' is not a valid file or directory")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
