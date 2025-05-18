# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import concurrent.futures
from optimize import (
    OPTIMIZE_JPG_QUALITY,
    OPTIMIZE_WEBP_QUALITY,
    OPTIMIZE_LOSSLESS,
    get_parser,
)

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")


def process_image(input_item_path, output_dir, jpg_quality, webp_quality, lossless, overwrite, optimize_script_path):
    item = os.path.basename(input_item_path)
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
        if process.stdout:
            print(f"  Output for {item}:\n{process.stdout.strip()}")
        if process.stderr:
            print(
                f"  Error output for {item}:\n{process.stderr.strip()}",
                file=sys.stderr,
            )
        print(f"'{item}' processed successfully.")
        return f"'{item}' processed successfully."
    except subprocess.CalledProcessError as e:
        error_message = f"Error processing '{item}':"
        if e.stdout:
            error_message += f"\\n  Standard output:\\n{e.stdout.strip()}"
        if e.stderr:
            error_message += f"\\n  Standard error:\\n{e.stderr.strip()}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Unexpected error processing '{item}': {e}"
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for item in os.listdir(input_dir):
            input_item_path = os.path.join(input_dir, item)
            if os.path.isfile(input_item_path):
                file_name, file_extension = os.path.splitext(item)
                if file_extension.lower() in SUPPORTED_EXTENSIONS:
                    tasks.append(
                        executor.submit(
                            process_image,
                            input_item_path,
                            output_dir,
                            jpg_quality,
                            webp_quality,
                            lossless,
                            overwrite,
                            optimize_script_path,
                        )
                    )
                else:
                    print(f"Skipping '{item}' (unsupported extension).")
            else:
                print(f"Skipping '{item}' (directory).")

        for future in concurrent.futures.as_completed(tasks):
            try:
                result = future.result()
                # print(result) # 이미 process_image 함수 내부에서 출력하므로 주석 처리
            except Exception as exc:
                print(f"A task generated an exception: {exc}")
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
