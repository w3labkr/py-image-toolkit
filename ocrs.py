# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import argparse
import multiprocessing
from tqdm import tqdm
from ocr import OCR_SUPPORTED_EXTENSIONS


def process_image(args_tuple):
    input_item_path, item_name, paddleocr_cli_args, ocr_script_path = args_tuple
    file_name, file_extension = os.path.splitext(item_name)
    if file_extension.lower() in OCR_SUPPORTED_EXTENSIONS:
        print(f"Processing '{item_name}'...")

        command = [sys.executable, ocr_script_path, input_item_path]
        command.extend(paddleocr_cli_args)

        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            output = f"Successfully processed '{item_name}'."
            if process.stdout:
                output += f"\\n  Output:\\n{process.stdout.strip()}"
            if process.stderr:
                output += f"\\n  Error output:\\n{process.stderr.strip()}"
            return output
        except subprocess.CalledProcessError as e:
            error_output = f"Error processing '{item_name}':"
            if e.stdout:
                error_output += f"\\n  Standard output:\\n{e.stdout.strip()}"
            if e.stderr:
                error_output += f"\\n  Standard error:\\n{e.stderr.strip()}"
            return error_output
        except Exception as e:
            return f"Unexpected error processing '{item_name}': {e}"
    else:
        return f"Skipping '{item_name}' (unsupported extension)."


def run(input_dir, paddleocr_cli_args):
    if not os.path.isdir(input_dir):
        print(f"Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    ocr_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr.py")
    if not os.path.isfile(ocr_script_path):
        print(
            f"'ocr.py' script not found. It should be in the same directory as 'ocrs.py'."
        )
        sys.exit(1)

    print(f"Input directory: {input_dir}")
    print(f"PaddleOCR arguments (passed to ocr.py): {' '.join(paddleocr_cli_args)}")
    print("-" * 30)

    tasks = []
    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            tasks.append((input_item_path, item, paddleocr_cli_args, ocr_script_path))
        else:
            print(f"Skipping '{item}' (directory).")
            print("-" * 30)

    # Determine the number of processes to use, e.g., number of CPU cores
    # Or a fixed number, e.g., 4. For CPU-bound tasks, os.cpu_count() is a good default.
    # For I/O bound tasks or tasks that release GIL, more processes might be beneficial.
    # Let's use os.cpu_count() or a sensible default if it's None.
    num_processes = os.cpu_count() or 4 

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Performing OCR"))

    for result in results:
        if result: # process_image can return None for skipped directories
            print(result)
            print("-" * 30)

    print("All tasks completed.")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Batch OCR images in a directory using ocr.py."
    )

    parser.add_argument(
        "input_dir", help="Input directory path containing images for OCR"
    )

    paddleocr_group = parser.add_argument_group(
        "PaddleOCR Arguments (passed to ocr.py)",
        "These arguments are passed directly to the ocr.py script. Do not specify them to use ocr.py defaults.",
    )

    paddleocr_group.add_argument(
        "--lang",
        type=str,
        help="OCR language (e.g., korean, en). ocr.py default: korean",
    )
    paddleocr_group.add_argument(
        "--rec_model_dir", type=str, help="Path to recognition model directory."
    )
    paddleocr_group.add_argument(
        "--det_model_dir", type=str, help="Path to detection model directory."
    )
    paddleocr_group.add_argument(
        "--cls_model_dir", type=str, help="Path to classification model directory."
    )
    paddleocr_group.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for OCR. ocr.py default: False"
    )
    paddleocr_group.add_argument(
        "--rec_char_dict_path",
        type=str,
        help="Path to recognition character dictionary.",
    )
    paddleocr_group.add_argument(
        "--rec_batch_num", type=int, help="Recognition batch size. ocr.py default: 6"
    )
    paddleocr_group.add_argument(
        "--det_db_thresh",
        type=float,
        help="Detection DB threshold. ocr.py default: 0.4",
    )
    paddleocr_group.add_argument(
        "--det_db_box_thresh",
        type=float,
        help="Detection DB box threshold. ocr.py default: 0.6",
    )
    paddleocr_group.add_argument(
        "--det_db_unclip_ratio",
        type=float,
        help="Detection DB unclip ratio. ocr.py default: 1.8",
    )
    paddleocr_group.add_argument(
        "--drop_score",
        type=float,
        help="Text detection drop score. ocr.py default: 0.6",
    )
    paddleocr_group.add_argument(
        "--cls_thresh", type=float, help="Classification threshold. ocr.py default: 0.9"
    )
    paddleocr_group.add_argument(
        "--use_angle_cls",
        action="store_true",
        help="Use angle classification. ocr.py default: False",
    )
    paddleocr_group.add_argument(
        "--use_space_char",
        action=argparse.BooleanOptionalAction,
        help="Use space character. ocr.py default: True",
    )
    paddleocr_group.add_argument(
        "--use_dilation",
        action=argparse.BooleanOptionalAction,
        help="Use dilation on text areas. ocr.py default: True",
    )
    paddleocr_group.add_argument(
        "--show_log",
        action="store_true",
        help="Show PaddleOCR logs. ocr.py default: False",
    )

    return parser


def main():
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    paddleocr_cli_args = []

    for action in parser._actions:
        if action.dest == "input_dir" or action.dest == "help":
            continue

        value = getattr(args, action.dest)

        if isinstance(action, argparse._StoreTrueAction):
            if value:
                paddleocr_cli_args.append(action.option_strings[0])
        elif isinstance(action, argparse.BooleanOptionalAction):
            if value is True:
                paddleocr_cli_args.append(action.option_strings[0])
            elif value is False:
                no_option_flag = next(
                    (opt for opt in action.option_strings if opt.startswith("--no-")),
                    None,
                )
                if no_option_flag:
                    paddleocr_cli_args.append(no_option_flag)

        elif value is not None:
            paddleocr_cli_args.append(action.option_strings[0])
            paddleocr_cli_args.append(str(value))

    paddleocr_cli_args.extend(unknown_args)

    run(
        input_dir=os.path.abspath(args.input_dir), paddleocr_cli_args=paddleocr_cli_args
    )


if __name__ == "__main__":
    main()
