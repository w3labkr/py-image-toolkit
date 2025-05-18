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
    if not suppress_output:
        print(f"Processing '{item}'...")

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
        output_message = ""
        if process.stdout:
            output_message += f"  Output for {item}:\\n{process.stdout.strip()}"
        if process.stderr:
            output_message += f"  Error output for {item}:\\n{process.stderr.strip()}"
        
        if not suppress_output:
            if output_message:
                print(output_message)
            print(f"'{item}' processed successfully.")
        
        # Return a success message, potentially with output/error details if not suppressed
        return f"'{item}' processed successfully." + (f"\\n{output_message}" if output_message and suppress_output else "")
    except subprocess.CalledProcessError as e:
        error_message = f"Error processing '{item}':"
        if e.stdout:
            error_message += f"\\n  Standard output:\\n{e.stdout.strip()}"
        if e.stderr:
            error_message += f"\\n  Standard error:\\n{e.stderr.strip()}"
        if not suppress_output:
            print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Unexpected error processing '{item}': {e}"
        if not suppress_output:
            print(error_message)
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

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(
        f"JPEG quality: {jpg_quality}, WebP quality: {webp_quality}, Lossless: {lossless}, Overwrite: {overwrite}"
    )
    if max_workers:
        print(f"Max workers: {max_workers}")
    print("-" * 30)

    tasks = []
    # Initialize futures_list here to ensure it's always a list
    futures_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for item in os.listdir(input_dir):
            input_item_path = os.path.join(input_dir, item)
            if os.path.isfile(input_item_path):
                file_name, file_extension = os.path.splitext(item)
                if file_extension.lower() in SUPPORTED_EXTENSIONS:
                    futures_list.append(
                        executor.submit(
                            process_image,
                            input_item_path,
                            output_dir,
                            jpg_quality,
                            webp_quality,
                            lossless,
                            overwrite,
                            optimize_script_path,
                            True,  # suppress_output = True
                        )
                    )
                else:
                    if not futures_list: # Only print if not using progress bar for other files
                        print(f"Skipping '{item}' (unsupported extension).")
            else:
                if not futures_list: # Only print if not using progress bar for other files
                    print(f"Skipping '{item}' (directory).")

        results_summary = []
        if futures_list:
            for future in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list), desc="Optimizing images"):
                try:
                    result = future.result()
                    if result: # Collect results that might contain error messages or important info
                        results_summary.append(result)
                except Exception as exc:
                    # Construct a message similar to what process_image would return for an exception
                    # This part might need adjustment based on how you want to identify the failing task
                    item_name = "Unknown item" # Placeholder, ideally get from future if possible
                    error_msg = f"A task for {item_name} generated an exception: {exc}"
                    print(error_msg)
                    results_summary.append(error_msg)
        
        for summary in results_summary:
            print(summary)
            print("-" * 30)

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
