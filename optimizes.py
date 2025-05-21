# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import concurrent.futures
from tqdm import tqdm
from optimize import (
    OPTIMIZE_JPG_QUALITY,
    OPTIMIZE_WEBP_QUALITY,
    OPTIMIZE_LOSSLESS,
    get_parser,
)

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")


def process_image(input_item_path, output_dir, jpg_quality, webp_quality, lossless, overwrite, optimize_script_path, suppress_output=False):
    item = os.path.basename(input_item_path)

    command = [
        sys.executable,
        optimize_script_path,
        input_item_path,
        "--output-dir",
        output_dir,
        "--jpg-quality",
        str(jpg_quality),
        "--webp-quality",
        str(webp_quality),
    ]
    if lossless:
        command.append("--lossless")
    if overwrite:
        command.append("--overwrite")

    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if suppress_output:
            return None 
        else:
            output_message = ""
            if process.stdout:
                output_message += f"  Output for {item}:\n{process.stdout.strip()}"
            if process.stderr: 
                output_message += f"  Error output for {item}:\n{process.stderr.strip()}"
            if output_message:
                print(output_message)
            print(f"'{item}' processed successfully.")
            return f"'{item}' processed successfully."

    except subprocess.CalledProcessError as e:
        error_message = f"Error processing '{item}':"
        if e.stdout:
            error_message += f"\n  optimize.py stdout:\n{e.stdout.strip()}"
        if e.stderr:
            error_message += f"\n  optimize.py stderr:\n{e.stderr.strip()}"
        return error_message

    except Exception as e:
        error_message = f"Unexpected error processing '{item}': {e}"
        return error_message


def run(
    input_dir,
    output_dir="output",
    jpg_quality=OPTIMIZE_JPG_QUALITY,
    webp_quality=OPTIMIZE_WEBP_QUALITY,
    lossless=OPTIMIZE_LOSSLESS,
    overwrite=False,
    max_workers=None,
):
    if not os.path.isdir(input_dir):
        print(f"Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    optimize_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "optimize.py"
    )
    if not os.path.isfile(optimize_script_path):
        print(
            f"'optimize.py' script not found. It should be in the same directory as 'optimizes.py'."
        )
        sys.exit(1)

    print(
        f"JPEG quality: {jpg_quality}, WebP quality: {webp_quality}, Lossless: {lossless}, Overwrite: {overwrite}"
    )
    print("-" * 30)

    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    futures_map = {} 
    tasks_to_submit = []

    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            _file_name, file_extension = os.path.splitext(item)
            if file_extension.lower() in SUPPORTED_EXTENSIONS:
                tasks_to_submit.append((item, input_item_path))
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    total_processable_files = len(tasks_to_submit)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for item, input_item_path in tasks_to_submit:
            future = executor.submit(
                process_image,
                input_item_path,
                output_dir,
                jpg_quality,
                webp_quality,
                lossless,
                overwrite,
                optimize_script_path,
                True,
            )
            futures_map[future] = item

        error_results = []
        if futures_map:
            for future in tqdm(concurrent.futures.as_completed(futures_map), total=total_processable_files, desc=None, unit="file"):
                item_name = futures_map[future] 
                try:
                    result = future.result()
                    if result: 
                        failure_count += 1
                        error_results.append(result)
                    else: 
                        success_count += 1
                except Exception as exc: 
                    failure_count += 1
                    error_msg = f"Task for '{item_name}' failed with an unexpected exception: {exc}"
                    error_results.append(error_msg)
        
    print("-" * 30)

    if total_processable_files == 0 and skipped_count == 0:
        print("No files or items found in the input directory.")
    else:
        if total_processable_files > 0:
            if success_count > 0:
                print(f"{success_count} file(s) processed successfully.")
            if failure_count > 0:
                print(f"{failure_count} file(s) failed.")
        
        if skipped_count > 0:
            if total_processable_files == 0:
                print(f"{skipped_count} item(s) found and skipped (unsupported or directory).")
                print("No processable image files found to optimize.")
            else:
                print(f"{skipped_count} additional item(s) skipped (unsupported or directory).")

    if error_results:
        print("\n--- Error Details ---")
        for error_message in error_results:
            print(error_message)

    print("All tasks completed.")


def main():
    parser = get_parser()

    parser.description = "Batch optimize images in a directory."

    input_path_action = None
    for act in parser._actions:
        if act.dest == "input_file":
            input_path_action = act
            break

    if input_path_action:
        parser._actions.remove(input_path_action)
        for group in parser._action_groups:
            if input_path_action in group._group_actions:
                group._group_actions.remove(input_path_action)
                break

    parser.add_argument(
        "input_dir", help="Input directory path containing images to optimize"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of processes to use for parallel processing (default: number of CPUs)",
    )

    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = "Output directory path to save optimized images (default: 'output' folder)"
            action.default = "output"
            break
    
    args = parser.parse_args()

    run(
        input_dir=os.path.abspath(args.input_dir),
        output_dir=os.path.abspath(args.output_dir),
        jpg_quality=args.jpg_quality,
        webp_quality=args.webp_quality,
        lossless=args.lossless,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()