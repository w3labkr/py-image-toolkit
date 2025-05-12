# py-image-toolkit

`py-image-toolkit` is a Python-based toolkit for image processing. It provides a command-line interface (CLI) through individual scripts (`resize.py`, `crop.py`, and `ocr.py`) for resizing images, performing automatic cropping based on face detection and composition rules, and extracting text using Optical Character Recognition (OCR). It's designed to be fast, efficient, and easy to use.

---

## Folder and File Structure

```plaintext
py-image-toolkit/
├── models/            # Directory for storing models
│   ├── face_detection_yunet_2023mar.onnx # YuNet model for face detection
│   ├── ch_PP-OCRv3_det_infer/             # Example: PaddleOCR detection model
│   ├── ko_PP-OCRv3_rec_infer/             # Example: PaddleOCR Korean recognition model
│   └── ch_ppocr_mobile_v2.0_cls_infer/    # Example: PaddleOCR classification model
├── resize.py          # CLI script for image resizing logic
├── crop.py            # CLI script for face detection and auto-cropping logic
├── ocr.py             # CLI script for Optical Character Recognition (OCR)
├── requirements.txt   # List of required libraries
└── README.md          # Project documentation
```

- `resize.py`: CLI script containing the core logic for image resizing and aspect ratio adjustments.
- `crop.py`: CLI script containing the core logic for detecting faces and cropping images based on composition rules.
- `ocr.py`: CLI script providing OCR functionality to extract text from images.
- `models/`: Directory for storing machine learning models.
    - `face_detection_yunet_2023mar.onnx`: Pretrained YuNet face detection model, automatically downloaded if not present for the `crop.py` script.
    - PaddleOCR models (e.g., `ch_PP-OCRv3_det_infer`, `ko_PP-OCRv3_rec_infer`): Models for text detection, recognition, and classification used by `ocr.py`. These models need to be downloaded manually and placed in the default locations if not specified otherwise via arguments.
- `requirements.txt`: Lists Python packages required to run the project.

---

## Installation

### Requirements

- Python 3.8 or higher
- Required libraries (see `requirements.txt`):
  - `opencv-python`
  - `Pillow`
  - `tqdm`
  - `piexif`
  - `numpy`
  - `paddleocr` (for `ocr.py` script)
  - `paddlepaddle` or `paddlepaddle-gpu` (for `ocr.py` script)

### Installation Guide

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/w3labkr/py-image-toolkit.git
cd py-image-toolkit
# Example using pyenv (adjust to your Python version management)
# pyenv install 3.12.9
# pyenv virtualenv 3.12.9 py-image-toolkit-3.12.9
# pyenv local py-image-toolkit-3.12.9
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

To use the `ocr.py` script with GPU acceleration, ensure you install `paddlepaddle-gpu` instead of `paddlepaddle` and have the necessary CUDA drivers and toolkit installed.

To deactivate and remove the virtual environment created with pyenv, use the following commands:

```bash
# Deactivate the current pyenv virtual environment
pyenv deactivate

# Remove the local virtual environment setting for the project directory
pyenv local --unset

# Delete the virtual environment
# Replace py-image-toolkit-3.12.9 with your virtual environment name if different
pyenv virtualenv-delete py-image-toolkit-3.12.9

# Optional: Uninstall the Python version if it's no longer needed by other projects
# pyenv uninstall 3.12.9 # Be cautious with this command

# Verify the changes
pyenv versions
```

---

## General CLI Options

Each script (`resize.py`, `crop.py`, `ocr.py`) is run directly and has its own set of options. Common options include:

- `-v, --verbose`: `resize.py` and `crop.py` support this flag to enable detailed (DEBUG level) logging.
- `ocr.py` provides CLI options for PaddleOCR configuration, and `--show_log` can be used to display PaddleOCR's internal logs. See the `Optical Character Recognition (OCR)` section below for details.

---

## Image Resizing

The `resize.py` script processes images from an input directory or a single image file by resizing them. It preserves EXIF metadata where possible (this is the default behavior of the underlying Pillow library) and saves images in their original format. It employs multiprocessing to speed up batch processing and offers detailed logging.

- Aspect Ratio Resizing: Maintains aspect ratio while resizing.
- Fixed Size Resizing: Forces images to fit specific dimensions.
- EXIF Metadata Preservation: Preserves original EXIF metadata where possible.
- Recursive Processing: Processes directories and subdirectories.

Syntax:

```bash
python resize.py <input_path> [options]
```

Key Options for `resize`:

- `input_dir`: Path to the source image folder or a single image file (default: `input`).
- `-o, --output-dir`: Directory to save processed images (default: `output`).
- `-r, --ratio`: Resize ratio behavior (`aspect_ratio`, `fixed`, `none`). (default: `aspect_ratio`).
  - `aspect_ratio`: Maintain aspect ratio to fit the target size. Requires at least one of `--width` or `--height` to be positive.
  - `none`: No resizing. If `width` or `height` are specified with `none`, a warning will be shown as they are ignored.
- `-w, --width`: Target width for resizing in pixels (default: `0`).
- `-H, --height`: Target height for resizing in pixels (default: `0`).
- `--filter`: Resampling filter to use when resizing (`lanczos`, `bicubic`, `bilinear`, `nearest`). (default: `lanczos`).
- `--overwrite`: Overwrite existing output files. If not specified, existing files will be skipped.

Examples for `resize`:

- Resize images in `./input` to a width of 1280px, maintaining aspect ratio:

  ```bash
  python resize.py ./input -w 1280
  ```

- Resize images to fixed 720x600 dimensions:

  ```bash
  python resize.py ./input -o ./output_resized -r fixed -w 720 -H 600
  ```

- Process images in `./input` without resizing (e.g., to copy to output directory, useful with `--overwrite`):

  ```bash
  python resize.py ./input -r none --overwrite
  ```

---

## Image Cropping

The `crop.py` script is designed to detect faces in images and automatically crop them based on composition rules like the rule of thirds or the golden ratio. It identifies subjects in the image, determines the main subject, and then calculates the optimal crop area. It automatically downloads the face detection model if needed.

Syntax:

```bash
python crop.py <input_path> [options]
```

Key Options for `crop`:

- `input_path`: (Required) Path to the image file or directory to process.
- `-o, --output_dir`: Directory to save results (Default: `output`).
- `--overwrite`: Overwrite existing output files (Default: False).
- `-v, --verbose`: Enable detailed (DEBUG level) logging for the crop operation (Default: False).
- `-m, --method`: Method to select main subject (`largest`, `center`). (Default: `largest`).
- `--ref, --reference`: Reference point for composition (`eye`, `box`). (Default: `box`).
- `-c, --confidence`: Min face detection confidence (Default: `0.6`).
- `-n, --nms`: Face detection NMS threshold (Default: `0.3`).
- `--min-face-width`: Min face width in pixels (Default: `30`).
- `--min-face-height`: Min face height in pixels (Default: `30`).
- `-r, --ratio`: Target crop aspect ratio (e.g., '16:9', '1.0', 'None') (Default: `None`).
- `--rule`: Composition rule(s) (`thirds`, `golden`, `both`). (Default: `both`).
- `-p, --padding-percent`: Padding percentage around crop (%) (Default: `5.0`).
- `--yunet-model-path`: Path to the YuNet ONNX model file. If not specified, it defaults to `models/face_detection_yunet_2023mar.onnx` and will be downloaded if missing.

Examples for `crop`:

- Crop an image using default settings:

  ```bash
  python crop.py ./input/sample.jpg
  ```

- Crop all images in a directory, saving to `./cropped_images`, using the 'thirds' rule and a 16:9 aspect ratio:

  ```bash
  python crop.py ./input -o ./cropped_images --rule thirds --ratio 16:9
  ```

- Crop an image, focusing on the 'eye' as reference, with 10% padding and overwriting existing files:

  ```bash
  python crop.py ./input/sample.png --reference eye --padding-percent 10 --overwrite
  ```

---

## Optical Character Recognition (OCR)

The `ocr.py` script uses PaddleOCR to extract text from images. It can process single image files or multiple image files within a directory and supports multiprocessing for efficient batch processing of large numbers of images.

Syntax:

```bash
python ocr.py <input_path> [options]
```

Key Options for `ocr`:

- `input_path`: (Required) Path to the image file or directory to process.
- `--lang`: OCR language (default: `korean`). Refer to PaddleOCR documentation for supported languages.
- `--rec_model_dir`: Path to the recognition model directory (default: `./models/ko_PP-OCRv3_rec_infer`).
- `--det_model_dir`: Path to the detection model directory (default: `./models/ch_PP-OCRv3_det_infer`).
- `--cls_model_dir`: Path to the direction classification model directory (default: `./models/ch_ppocr_mobile_v2.0_cls_infer`).
- `--use_gpu`: Whether to use GPU for OCR processing (default: `False`).
- `--rec_char_dict_path`: Path to recognition character dictionary (default: `None`, uses PaddleOCR default).
- `--rec_batch_num`: Recognition batch size (default: `6`).
- `--det_db_thresh`: Detection DB threshold (default: `0.4`).
- `--det_db_box_thresh`: Detection DB box threshold (default: `0.6`).
- `--det_db_unclip_ratio`: Detection DB unclip ratio (default: `1.8`).
- `--drop_score`: Drop score for text detection (default: `0.6`).
- `--cls_thresh`: Classification threshold (default: `0.9`).
- `--use_angle_cls`: Whether to use angle classification (default: `False`).
- `--use_space_char`: Whether to use space character (default: `True`).
- `--use_dilation`: Whether to use dilation on text regions (default: `True`).
- `--show_log`: Whether to display PaddleOCR's internal logs (default: `False`).

Examples for `ocr`:

- Extract text from a single image file:

  ```bash
  python ocr.py ./input/sample.png
  ```

- Extract Korean text from all images in the `input_images` directory (using GPU and specifying a custom Korean recognition model):

  ```bash
  python ocr.py ./input_images --lang korean --use_gpu --rec_model_dir ./models/my_custom_korean_ocr_model
  ```

---

## Troubleshooting

If you encounter issues while using the toolkit, refer to the following common problems and solutions:

### Missing Libraries

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Missing Model File

- **YuNet Model (for `crop.py` script)**: The `crop.py` script will automatically attempt to download the YuNet model if it is not found in the `models/` directory. Ensure you have an active internet connection during the first run or if the model is missing.
- **PaddleOCR Models (for `ocr.py` script)**: The `ocr.py` script requires PaddleOCR models.
  - Ensure you have downloaded the necessary detection, recognition (e.g., for Korean), and classification models.
  - By default, the script looks for models in subdirectories under `./models/` (e.g., `./models/ko_PP-OCRv3_rec_infer`). You may need to create these directories and place the model files there, or specify custom model paths using options like `--rec_model_dir`, `--det_model_dir`, and `--cls_model_dir`.
  - If using GPU (`--use_gpu`), ensure `paddlepaddle-gpu` is installed and your CUDA environment is correctly configured.

### Image Not Processed

- Verify that the input path is correct and contains valid image files with supported extensions.
- Use verbose flags for each script to enable detailed logs for debugging:
  - For `resize.py`: `python resize.py ./input --verbose`
  - For `crop.py`: `python crop.py ./input -v` (or `--verbose`)
  - For `ocr.py`: `python ocr.py ./input_image.png --show_log` (for PaddleOCR logs)
- If `python resize.py` command fails due to invalid arguments or critical processing errors, it will exit with a non-zero status code. Check terminal output for specific error messages.
- If `python crop.py` command encounters critical setup issues (e.g., model download failure, output directory problems, invalid input path) or unhandled processing errors, it will exit with a non-zero status code. `CropSetupError` indicates a problem that prevented the operation from starting. Check terminal output and logs for details.
- If `python ocr.py` command fails, check for PaddleOCR model availability, correct paths, and library installation. Error messages in the terminal or logs should provide more details.

---

## License

This project is licensed under the [MIT License](LICENSE).
