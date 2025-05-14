# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from crop import get_parser, CROP_SUPPORTED_EXTENSIONS


def run(
    input_dir,
    output_dir="output",
    overwrite=False,
    rule=None,
    ratio=None,
    padding_percent=None,
    reference=None,
    method=None,
    min_face_width=None,
    min_face_height=None,
    model_path=None,
):
    if not os.path.isdir(input_dir):
        print(f"Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    crop_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "crop.py"
    )
    if not os.path.isfile(crop_script_path):
        print(
            f"'crop.py' script not found. It should be in the same directory as 'batch_crop.py'."
        )
        sys.exit(1)

    display_model_path = model_path
    if model_path and os.path.isabs(model_path):
        display_model_path = os.path.relpath(model_path)

    print(f"Input directory: {os.path.relpath(input_dir)}")
    print(f"Output directory: {os.path.relpath(output_dir)}")
    print(
        f"Overwrite: {overwrite}, Rule: {rule}, Ratio: {ratio}, Padding: {padding_percent}%, Reference: {reference}, Subject selection method: {method}"
    )
    print(
        f"Min face width: {min_face_width}, Min face height: {min_face_height}, Model path: {display_model_path}"
    )
    print("-" * 30)

    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            file_name, file_extension = os.path.splitext(item)
            if file_extension.lower() in CROP_SUPPORTED_EXTENSIONS:
                print(f"Processing '{item}'...")

                command = [
                    sys.executable,
                    crop_script_path,
                    input_item_path,
                    "--output-dir",
                    output_dir,
                ]
                if overwrite:
                    command.append("--overwrite")

                if rule is not None:
                    command.extend(["--rule", str(rule)])
                if ratio is not None:
                    command.extend(["--ratio", str(ratio)])
                if padding_percent is not None:
                    command.extend(["--padding-percent", str(padding_percent)])
                if reference is not None:
                    command.extend(["--reference", str(reference)])
                if method is not None:
                    command.extend(["--method", str(method)])
                if min_face_width is not None:
                    command.extend(["--min-face-width", str(min_face_width)])
                if min_face_height is not None:
                    command.extend(["--min-face-height", str(min_face_height)])
                if model_path is not None:
                    command.extend(["--model-path", str(model_path)])

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
                    print(f"Successfully processed '{item}'.")
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
    parser.description = "Batch crop images in a directory."

    input_file_action = None
    for act in parser._actions:
        if act.dest == "input_file":
            input_file_action = act
            break

    if input_file_action:
        parser._actions.remove(input_file_action)
        for group in parser._action_groups:
            if input_file_action in group._group_actions:
                group._group_actions.remove(input_file_action)
                break

    parser.add_argument(
        "input_dir", help="Input directory path containing images to crop"
    )

    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = "Output directory path to save cropped images (default: 'output' folder)"
            action.default = "output"
            break

    args = parser.parse_args()

    run(
        input_dir=os.path.abspath(args.input_dir),
        output_dir=os.path.abspath(args.output_dir),
        overwrite=getattr(args, "overwrite", False),
        rule=getattr(args, "rule", None),
        ratio=getattr(args, "ratio", None),
        padding_percent=getattr(args, "padding_percent", None),
        reference=getattr(args, "reference", None),
        method=getattr(args, "method", None),
        min_face_width=getattr(args, "min_face_width", None),
        min_face_height=getattr(args, "min_face_height", None),
        model_path=getattr(args, "model_path", None),
    )


if __name__ == "__main__":
    main()
