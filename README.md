# py-image-toolkit

`py-image-toolkit` is a Python-based toolkit for image processing. It provides a command-line interface (CLI) through individual scripts (`resize.py`, `crop.py`, `ocr.py`, `optimize.py`) for resizing images, performing automatic cropping based on face detection and composition rules, extracting text using Optical Character Recognition (OCR), and optimizing images. It's designed to be fast, efficient, and easy to use.

---

## Folder and File Structure

```plaintext
py-image-toolkit/
├── models/            # Directory for storing models
│   ├── face_detection_yunet_2023mar.onnx # YuNet model for face detection
│   ├── ch_PP-OCRv3_det_infer/           # Example: PaddleOCR detection model
│   ├── ko_PP-OCRv3_rec_infer/           # Example: PaddleOCR Korean recognition model
│   └── ch_ppocr_mobile_v2.0_cls_infer/  # Example: PaddleOCR classification model
├── resize.py          # CLI script for image resizing logic
├── crop.py            # CLI script for face detection and auto-cropping logic
├── ocr.py             # CLI script for Optical Character Recognition (OCR)
├── optimize.py        # CLI script for image optimization and compression
├── requirements.txt   # List of required libraries
└── README.md          # Project documentation
```

- `resize.py`: CLI script containing the core logic for image resizing and aspect ratio adjustments.
- `crop.py`: CLI script containing the core logic for detecting faces and cropping images based on composition rules.
- `ocr.py`: CLI script providing OCR functionality to extract text from images.
- `optimize.py`: CLI script providing image optimization and compression functionality.
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

Each script (`resize.py`, `crop.py`, `ocr.py`, `optimize.py`) is run directly and has its own set of options.
- `ocr.py` provides CLI options for PaddleOCR configuration, and `--show_log` can be used to display PaddleOCR's internal logs. See the `Optical Character Recognition (OCR)` section below for details.

---

## Image Resizing

The `resize.py` script processes a single image file by resizing it. It preserves EXIF metadata where possible (this is the default behavior of the underlying Pillow library) and saves images in their original format. It offers detailed logging for errors.

- Aspect Ratio Resizing: Maintains aspect ratio while resizing.
- Fixed Size Resizing: Forces images to fit specific dimensions.
- EXIF Metadata Preservation: Preserves original EXIF metadata where possible.

Syntax:

```bash
python resize.py <input_file> [options]
```

Key Options for `resize`:

- `input_file`: Path to the source image file.
- `-o, --output-dir`: Directory to save processed images (default: `output`).
- `-r, --ratio`: Resize ratio behavior (`aspect_ratio`, `fixed`, `none`). (default: `aspect_ratio`).
  - `aspect_ratio`: Maintain aspect ratio to fit the target size. Requires at least one of `--width` or `--height` to be positive.
  - `none`: No resizing. If `width` or `height` are specified with `none`, a warning will be shown as they are ignored.
- `-w, --width`: Target width for resizing in pixels (default: `0`).
- `-H, --height`: Target height for resizing in pixels (default: `0`).
- `--filter`: Resampling filter to use when resizing (`lanczos`, `bicubic`, `bilinear`, `nearest`). (default: `lanczos`).
- `--overwrite`: Overwrite existing output files. If not specified, existing files will be skipped.

Examples for `resize`:

- Resize an image `./input/sample.jpg` to a width of 1280px, maintaining aspect ratio:

  ```bash
  python resize.py ./input/sample.jpg -w 1280
  ```

- Resize an image to fixed 720x600 dimensions:

  ```bash
  python resize.py ./input/sample.jpg -o ./output_resized -r fixed -w 720 -H 600
  ```

- Process an image `./input/sample.jpg` without resizing (e.g., to copy to output directory, useful with `--overwrite`):

  ```bash
  python resize.py ./input/sample.jpg -r none --overwrite
  ```

---

## Image Cropping

The `crop.py` script is designed to detect faces in a single image and automatically crop it based on composition rules like the rule of thirds or the golden ratio. It identifies subjects in the image, determines the main subject, and then calculates the optimal crop area. It automatically downloads the face detection model if needed.

Syntax:

```bash
python crop.py <input_file> [options]
```

Key Options for `crop`:

- `input_file`: (Required) Path to the image file to process.
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

- Crop an image, saving to `./cropped_images`, using the 'thirds' rule and a 16:9 aspect ratio:

  ```bash
  python crop.py ./input/sample.jpg -o ./cropped_images --rule thirds --ratio 16:9
  ```

- Crop an image, focusing on the 'eye' as reference, with 10% padding and overwriting existing files:

  ```bash
  python crop.py ./input/sample.png --reference eye --padding-percent 10 --overwrite
  ```

---

## Image Optimization

The `optimize.py` script allows you to compress and optimize a single image without significant visible quality loss. It supports various formats including JPEG, PNG, WebP, and TIFF, each with format-specific optimizations.

Syntax:

```bash
python optimize.py <input_file> [options]
```

Key Options for `optimize`:

- `input_file`: Path to the image file to process.
- `-o, --output-dir`: Directory to save optimized images (default: `output`).
- `--overwrite`: Overwrite existing files if they already exist in the output directory (Default: False).
- `--jpg-quality`: JPEG image quality setting (1-100, default: `85`). Lower values produce smaller files but may reduce quality.
- `--webp-quality`: WebP image quality (1-100, default: `85`, ignored when `--lossless` option is used).
- `--lossless`: Use lossless compression for WebP (ignores WebP quality setting).

Examples for `optimize`:

- Optimize an image `./input/sample.jpg` with default settings:

  ```bash
  python optimize.py ./input/sample.jpg
  ```

- Optimize a single image with custom JPEG quality (70%) and save to a specific directory:

  ```bash
  python optimize.py ./input/large_image.jpg -o ./optimized_images --jpg-quality 70
  ```

- Optimize a WebP image with lossless compression:

  ```bash
  python optimize.py ./input/sample.webp -o ./optimized_output --lossless
  ```

- Process an image `./photos/another.png` and overwrite any existing file:

  ```bash
  python optimize.py ./photos/another.png -o ./photos_optimized --overwrite
  ```

---

## Optical Character Recognition (OCR)

The `ocr.py` script uses PaddleOCR to extract text from a single image. It then attempts to identify and label key information such as document title, name, address, resident registration number, issue date, and issuer.

Syntax:

```bash
python ocr.py <input_file> [options]
```

Key Options for `ocr`:

- `input_file`: (Required) Path to the image file to process.
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

- Extract Korean text from an image (using GPU and specifying a custom Korean recognition model):

  ```bash
  python ocr.py ./input_images/my_document.jpg --lang korean --use_gpu --rec_model_dir ./models/my_custom_korean_ocr_model
  ```

The script will output extracted fields like "문서 제목", "이름", "주소", "주민등록번호", "발급일", "발급기관".

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

- Verify that the input path is correct and points to a valid image file with supported extensions.
- For `ocr.py`: `python ocr.py ./input_image.png --show_log` (for PaddleOCR logs)
- If `python resize.py` command fails due to invalid arguments or critical processing errors, it will exit with a non-zero status code. Check terminal output for specific error messages.
- If `python crop.py` command encounters critical setup issues (e.g., model download failure, output directory problems, invalid input path) or unhandled processing errors, it will exit with a non-zero status code. `CropSetupError` indicates a problem that prevented the operation from starting. Check terminal output and logs for details.
- If `python ocr.py` command fails, check for PaddleOCR model availability, correct paths, and library installation. Error messages in the terminal or logs should provide more details.
- If `python optimize.py` command fails, ensure that the image files are valid and that you have appropriate file access permissions. Check terminal output for specific error messages.

---

## License

This project is licensed under the [MIT License](LICENSE).
