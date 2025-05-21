# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import multiprocessing
from tqdm import tqdm
from crop import get_parser, CROP_SUPPORTED_EXTENSIONS


def _process_image_item(args):
    input_item_path, output_dir, overwrite, rule, ratio, padding_percent, reference, method, min_face_width, min_face_height, model_path, crop_script_path, suppress_output = args
    item = os.path.basename(input_item_path)
    
    try:
        file_name, file_extension = os.path.splitext(item)
        if file_extension.lower() in CROP_SUPPORTED_EXTENSIONS:
            if not suppress_output:
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

            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if not suppress_output:
                output_message_display = f"Successfully processed '{item}'."
                if process.stdout:
                    output_message_display += f"\n  Output:\n{process.stdout.strip()}"
                if process.stderr:
                    output_message_display += f"\n  Process stderr:\n{process.stderr.strip()}"
                print(output_message_display)
                print("-" * 30)
            return None
        else:
            if not suppress_output:
                print(f"Skipping '{item}' (unsupported extension).")
                print("-" * 30)
            return f"Skipped '{item}' (unsupported extension)."
    except subprocess.CalledProcessError as e:
        error_message = f"Error processing '{item}':"
        if e.stdout:
            error_message += f"\n  Standard output:\n{e.stdout.strip()}"
        if e.stderr:
            error_message += f"\n  Standard error:\n{e.stderr.strip()}"
        if not suppress_output:
            print(error_message)
            print("-" * 30)
        return error_message
    except Exception as e:
        error_message = f"Unexpected error processing '{item}': {e}"
        if not suppress_output:
            print(error_message)
            print("-" * 30)
        return error_message


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
        try:
            display_model_path = os.path.relpath(model_path)
        except ValueError:
            pass

    print(
        f"Overwrite: {overwrite}, Rule: {rule}, Ratio: {ratio}, Padding: {padding_percent}%, Reference: {reference}, Subject selection method: {method}"
    )
    print(
        f"Min face width: {min_face_width}, Min face height: {min_face_height}, Model path: {display_model_path}"
    )
    print("-" * 30)

    tasks = []
    skipped_directories_count = 0
    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            tasks.append((
                input_item_path, output_dir, overwrite, rule, ratio, 
                padding_percent, reference, method, min_face_width, 
                min_face_height, model_path, crop_script_path, True
            ))
        else:
            skipped_directories_count += 1

    successful_processed_count = 0
    skipped_unsupported_files_count = 0

    if not tasks:
        pass
    else:
        num_processes = min(multiprocessing.cpu_count(), len(tasks))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(_process_image_item, tasks), total=len(tasks), desc="Processing images"))

        for result in results:
            if result is None:
                successful_processed_count += 1
            elif isinstance(result, str) and result.startswith("Skipped"):
                skipped_unsupported_files_count += 1
            elif result:
                pass

    print(f"{successful_processed_count} file(s) processed successfully.")
    
    total_skipped_items = skipped_directories_count + skipped_unsupported_files_count
    if total_skipped_items > 0:
        print(f"{total_skipped_items} additional item(s) skipped (unsupported or directory).")

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