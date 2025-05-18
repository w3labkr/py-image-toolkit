# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import argparse
import multiprocessing
from tqdm import tqdm
from resize import (
    RESIZE_SUPPORTED_EXTENSIONS,
    RESIZE_FILTER_NAMES,
    _PIL_RESAMPLE_FILTERS,
    get_parser,
)


def process_image(args_tuple):
    input_item_path, output_dir, overwrite, resize_mode, width, height, filter_str, resize_script_path, suppress_output = args_tuple
    item = os.path.basename(input_item_path)
    try:
        if not suppress_output:
            print(f"Processing '{item}'...")

        output_filename = os.path.basename(input_item_path)
        output_item_path = os.path.join(output_dir, output_filename)

        try:
            relative_output_item_path = os.path.relpath(output_item_path)
        except ValueError:
            relative_output_item_path = output_item_path

        command = [
            sys.executable,
            resize_script_path,
            input_item_path,
            "--output-dir",
            output_dir,
            "--ratio",
            resize_mode,
            "--width",
            str(width),
            "--height",
            str(height),
            "--filter",
            filter_str,
        ]
        if overwrite:
            command.append("--overwrite")

        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        success_message = f"Successfully processed '{item}' to '{relative_output_item_path}'."
        output_details = ""
        if process.stdout:
            output_details += f"\\nOutput:\\n{process.stdout.strip()}"
        if process.stderr:
            output_details += f"\\nError output:\\n{process.stderr.strip()}"
        
        if not suppress_output:
            print(success_message + output_details)
            print("-" * 30)
        
        return success_message + (output_details if suppress_output and output_details else ""), "" # Return details if suppressed and present
    except subprocess.CalledProcessError as e:
        error_message = f"Error processing '{item}':\\n  Standard output:\\n{e.stdout.strip()}\\n  Standard error:\\n{e.stderr.strip()}"
        if not suppress_output:
            print(error_message)
            print("-" * 30)
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error processing '{item}': {e}"
        if not suppress_output:
            print(error_message)
            print("-" * 30)
        return None, error_message


def run(
    input_dir,
    output_dir="output",
    overwrite=False,
    resize_mode="aspect_ratio",
    width=0,
    height=0,
    filter_str="lanczos",
):
    if not os.path.isdir(input_dir):
        print(f"Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    resize_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resize.py"
    )
    if not os.path.isfile(resize_script_path):
        print(
            f"'resize.py' script not found. It should be in the same directory as 'batch_resize.py'."
        )
        sys.exit(1)

    try:
        input_dir_display = os.path.relpath(input_dir)
        output_dir_display = os.path.relpath(output_dir)
    except ValueError:
        input_dir_display = input_dir
        output_dir_display = output_dir

    print(f"Input directory: {input_dir_display}")
    print(f"Output directory: {output_dir_display}")
    print(
        f"Overwrite: {overwrite}, Resize mode: {resize_mode}, Width: {width}, Height: {height}, Filter: {filter_str}"
    )
    print("-" * 30)

    tasks = []
    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            file_name, file_extension = os.path.splitext(item)
            if file_extension.lower() in RESIZE_SUPPORTED_EXTENSIONS:
                tasks.append((input_item_path, output_dir, overwrite, resize_mode, width, height, filter_str, resize_script_path, True)) # Add suppress_output=True
            else:
                # Only print if not using progress bar for other files, or if it's a standalone skip
                if not tasks: # Or some other logic if you want to always print skips
                    print(f"Skipping '{item}' (unsupported extension).")
        else:
            # Only print if not using progress bar for other files
            if not tasks:
                print(f"Skipping '{item}' (directory).")

    if not tasks:
        print("No supported image files found to process.")
        sys.exit(0)

    num_processes = min(multiprocessing.cpu_count(), len(tasks))
    print(f"Using {num_processes} processes for parallel execution.")
    print("-" * 30)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Resizing images"))

    print("-" * 30) # Print separator once before all results
    for success_msg, error_msg in results:
        if success_msg: # This will now include details if suppress_output was True
            print(success_msg)
        if error_msg:
            print(error_msg, file=sys.stderr)
        if success_msg or error_msg: # Print separator only if there was some message
            print("-" * 30)

    print("All tasks completed.")


def main():
    parser = get_parser()

    parser.description = "Batch resize images in a directory."

    for i, action in enumerate(parser._actions):
        if action.dest == "input_file":
            parser._actions[i] = argparse._StoreAction(
                option_strings=[],
                dest="input_dir",
                nargs=None,
                const=None,
                default=argparse.SUPPRESS,
                type=str,
                choices=None,
                help="Input directory containing images to resize",
            )
            break

    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = (
                "Output directory to save resized images (default: 'output' folder)"
            )
            action.default = "output"
            break

    args = parser.parse_args()

    run(
        input_dir=os.path.abspath(args.input_dir),
        output_dir=os.path.abspath(args.output_dir),
        overwrite=args.overwrite,
        resize_mode=args.ratio,
        width=args.width,
        height=args.height,
        filter_str=args.filter,
    )


if __name__ == "__main__":
    main()
