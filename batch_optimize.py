# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from optimize import (
    OPTIMIZE_JPG_QUALITY,
    OPTIMIZE_WEBP_QUALITY,
    OPTIMIZE_LOSSLESS,
    get_parser,
)

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")


def run(
    input_dir,
    output_dir="output",
    jpg_quality=OPTIMIZE_JPG_QUALITY,
    webp_quality=OPTIMIZE_WEBP_QUALITY,
    lossless=OPTIMIZE_LOSSLESS,
    overwrite=False,
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
            f"'optimize.py' script not found. It should be in the same directory as 'batch_optimize.py'."
        )
        sys.exit(1)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(
        f"JPEG quality: {jpg_quality}, WebP quality: {webp_quality}, Lossless: {lossless}, Overwrite: {overwrite}"
    )
    print("-" * 30)

    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            file_name, file_extension = os.path.splitext(item)
            if file_extension.lower() in SUPPORTED_EXTENSIONS:
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
                        print(f"  Output:\n{process.stdout.strip()}")
                    if process.stderr:
                        print(
                            f"  Error output:\n{process.stderr.strip()}",
                            file=sys.stderr,
                        )
                    print(f"'{item}' processed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing '{item}':")
                    if e.stdout:
                        print(f"  Standard output:\n{e.stdout.strip()}")
                    if e.stderr:
                        print(f"  Standard error:\n{e.stderr.strip()}")
                except Exception as e:
                    print(f"Unexpected error processing '{item}': {e}")
                print("-" * 30)
            else:
                print(f"Skipping '{item}' (unsupported extension).")
        else:
            print(f"Skipping '{item}' (directory).")

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
    )


if __name__ == "__main__":
    main()
