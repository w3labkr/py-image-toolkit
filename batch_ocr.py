# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import argparse
from ocr import OCR_SUPPORTED_EXTENSIONS


def run(input_dir, paddleocr_cli_args):
    if not os.path.isdir(input_dir):
        print(f"Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    ocr_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr.py")
    if not os.path.isfile(ocr_script_path):
        print(
            f"'ocr.py' script not found. It should be in the same directory as 'batch_ocr.py'."
        )
        sys.exit(1)

    print(f"Input directory: {input_dir}")
    print(f"PaddleOCR arguments (passed to ocr.py): {' '.join(paddleocr_cli_args)}")
    print("-" * 30)

    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        if os.path.isfile(input_item_path):
            file_name, file_extension = os.path.splitext(item)
            if file_extension.lower() in OCR_SUPPORTED_EXTENSIONS:
                print(f"Processing '{item}'...")

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
