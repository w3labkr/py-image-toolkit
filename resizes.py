# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import argparse
import multiprocessing
from tqdm import tqdm
from resize import (
    RESIZE_SUPPORTED_EXTENSIONS,
    get_parser,
)


def process_image(args_tuple):
    input_item_path, output_dir, overwrite, resize_mode, width, height, filter_str, resize_script_path, _suppress_output_always_true = args_tuple
    item = os.path.basename(input_item_path)
    try:
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
        return item, None
    except subprocess.CalledProcessError as e:
        error_message = f"Error processing '{item}':\n  Standard output:\n{e.stdout.strip()}\n  Standard error:\n{e.stderr.strip()}"
        return item, error_message
    except Exception as e:
        error_message = f"Unexpected error processing '{item}': {e}"
        return item, error_message


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
    
    print(
        f"Overwrite: {overwrite}, Resize mode: {resize_mode}, Width: {width}, Height: {height}, Filter: {filter_str}"
    )
    print("-" * 30)

    tasks = []
    skipped_items_count = 0
    
    all_items_in_dir = os.listdir(input_dir)
    for item_name_in_dir in all_items_in_dir:
        input_item_path = os.path.join(input_dir, item_name_in_dir)
        if os.path.isfile(input_item_path):
            _file_name, file_extension = os.path.splitext(item_name_in_dir)
            if file_extension.lower() in RESIZE_SUPPORTED_EXTENSIONS:
                tasks.append((input_item_path, output_dir, overwrite, resize_mode, width, height, filter_str, resize_script_path, True))
            else:
                skipped_items_count += 1
        else:
            skipped_items_count += 1

    if not tasks:
        print("No supported image files found to process.")
        if skipped_items_count > 0:
             print(f"0 file(s) processed successfully.")
             print(f"{skipped_items_count} additional item(s) skipped (unsupported or directory).")
        sys.exit(0)

    num_processes = min(multiprocessing.cpu_count(), len(tasks))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc=""))

    print("-" * 30)

    successful_count = 0
    failed_count = 0
    error_reports = []

    for item_name, error_msg_content in results:
        if error_msg_content is None:
            successful_count += 1
        else:
            failed_count += 1
            error_reports.append({'item': item_name, 'error': error_msg_content})

    print(f"{successful_count} file(s) processed successfully.")
    print(f"{skipped_items_count} additional item(s) skipped (unsupported or directory).")

    print("All tasks completed.")

    if failed_count > 0:
        print("\nDetails for failed images:")
        for report in error_reports:
            print(f"  - File: {report['item']}")
            indented_error = "\n".join(["      " + line for line in report['error'].splitlines()])
            print(f"    Error:\n{indented_error}")
    print("-" * 30)


def main():
    parser = get_parser()
    parser.description = "Batch resize images in a directory."

    input_dir_arg_found = False
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
                metavar="INPUT_DIR" 
            )
            input_dir_arg_found = True
            break
    
    if not input_dir_arg_found:
        pass


    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = (
                "Output directory to save resized images (default: 'output' folder)"
            )
            action.default = "output"
            break
    
    args = parser.parse_args()

    if not hasattr(args, 'input_dir'):
        parser.error("the following arguments are required: INPUT_DIR (input directory)")


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