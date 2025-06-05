# py-image-toolkit

A fast and easy-to-use Python toolkit for image processing with CLI tools for resizing, cropping, OCR, and optimization, including batch processing support.

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
├── resizes.py    # CLI script for batch image resizing
├── crops.py      # CLI script for batch image cropping
├── ocrs.py       # CLI script for batch Optical Character Recognition (OCR)
├── optimizes.py  # CLI script for batch image optimization
├── requirements.txt   # List of required libraries
└── README.md          # Project documentation
```

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

Each script (`resize.py`, `crop.py`, `ocr.py`, `optimize.py`) and their batch counterparts (`resizes.py`, `crops.py`, `ocrs.py`, `optimizes.py`) are run directly and have their own set of options.

- Individual scripts generally take a single `input_file` as an argument.
- Batch scripts generally take an `input_dir` as an argument to process multiple files.
- `ocr.py` and `batch_ocr.py` provide CLI options for PaddleOCR configuration, and `--show_log` can be used to display PaddleOCR's internal logs. See the `Optical Character Recognition (OCR)` section below for details.

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

## Batch Image Resizing

The `resizes.py` script processes all images within a specified input directory, applying the same resizing logic as `resize.py` to each image.

Syntax:

```bash
python resizes.py <input_dir> [options]
```

Key Options for `batch_resize`:

- `input_dir`: Path to the directory containing source image files.
- `-o, --output-dir`: Directory to save processed images (default: `output`).
- `-r, --ratio`: Resize ratio behavior (`aspect_ratio`, `fixed`, `none`). (default: `aspect_ratio`).
  - `aspect_ratio`: Maintain aspect ratio to fit the target size. Requires at least one of `--width` or `--height` to be positive.
  - `none`: No resizing. If `width` or `height` are specified with `none`, a warning will be shown as they are ignored.
- `-w, --width`: Target width for resizing in pixels (default: `0`).
- `-H, --height`: Target height for resizing in pixels (default: `0`).
- `--filter`: Resampling filter to use when resizing (`lanczos`, `bicubic`, `bilinear`, `nearest`). (default: `lanczos`).
- `--overwrite`: Overwrite existing output files. If not specified, existing files will be skipped.

Examples for `batch_resize`:

- Resize all images in `./input_images` to a width of 1280px, maintaining aspect ratio, and save to `./output_resized_batch`:

  ```bash
  python resizes.py ./input_images -o ./output_resized_batch -w 1280
  ```

- Resize all images in `./input_images` to fixed 720x600 dimensions:

  ```bash
  python resizes.py ./input_images -r fixed -w 720 -H 600
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
- `-o, --output-dir`: Directory to save results (Default: `output`).
- `--overwrite`: Overwrite existing output files (Default: False).
- `-v, --verbose`: Enable detailed (DEBUG level) logging for the crop operation (Default: False).
- `-m, --method`: Method to select main subject (`largest`, `center`). (Default: `largest`).
- `--ref, --reference`: Reference point for composition (`eye`, `box`). (Default: `box`).
- `-c, --confidence`: Min face detection confidence (Default: `0.6`).
- `-n, --nms`: Face detection NMS threshold (Default: `0.3`).
- `--min-face-width`: Min face width in pixels (Default: `30`).
- `--min-face-height`: Min face height in pixels (Default: `30`).
- `-r, --ratio`: Target crop aspect ratio (e.g., '16:9', '1.0', 'None') (Default: `None`).
- `--rule`: Composition rule(s) (`thirds`, `golden`, `both`, `none`). (Default: `both`).
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

## Batch Image Cropping

The `crops.py` script processes all images within a specified input directory, applying the same face detection and auto-cropping logic as `crop.py` to each image.

Syntax:

```bash
python crops.py <input_dir> [options]
```

Key Options for `batch_crop`:

- `input_dir`: (Required) Path to the directory containing image files to process.
- `-o, --output-dir`: Directory to save results (Default: `output`).
- `--overwrite`: Overwrite existing output files (Default: False).
- `-v, --verbose`: Enable detailed (DEBUG level) logging for the crop operation (Default: False).
- `-m, --method`: Method to select main subject (`largest`, `center`). (Default: `largest`).
- `--ref, --reference`: Reference point for composition (`eye`, `box`). (Default: `box`).
- `-c, --confidence`: Min face detection confidence (Default: `0.6`).
- `-n, --nms`: Face detection NMS threshold (Default: `0.3`).
- `--min-face-width`: Min face width in pixels (Default: `30`).
- `--min-face-height`: Min face height in pixels (Default: `30`).
- `-r, --ratio`: Target crop aspect ratio (e.g., '16:9', '1.0', 'None') (Default: `None`).
- `--rule`: Composition rule(s) (`thirds`, `golden`, `both`, `none`). (Default: `both`).
- `-p, --padding-percent`: Padding percentage around crop (%) (Default: `5.0`).
- `--yunet-model-path`: Path to the YuNet ONNX model file. If not specified, it defaults to `models/face_detection_yunet_2023mar.onnx` and will be downloaded if missing.

Examples for `batch_crop`:

- Crop all images in `./input_folder` using default settings:

  ```bash
  python crops.py ./input_folder
  ```

- Crop all images in `./input_folder`, saving to `./cropped_batch`, using the 'thirds' rule and a 16:9 aspect ratio:

  ```bash
  python crops.py ./input_folder -o ./cropped_batch --rule thirds --ratio 16:9
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

## Batch Image Optimization

The `optimizes.py` script processes all images within a specified input directory, applying the same optimization logic as `optimize.py` to each image.

Syntax:

```bash
python optimizes.py <input_dir> [options]
```

Key Options for `batch_optimize`:

- `input_dir`: Path to the directory containing image files to process.
- `-o, --output-dir`: Directory to save optimized images (default: `output`).
- `--overwrite`: Overwrite existing files if they already exist in the output directory (Default: False).
- `--jpg-quality`: JPEG image quality setting (1-100, default: `85`).
- `--webp-quality`: WebP image quality (1-100, default: `85`, ignored when `--lossless` option is used).
- `--lossless`: Use lossless compression for WebP (ignores WebP quality setting).
- `--max-workers`: Maximum number of processes to use for parallel processing (default: number of CPUs).

Examples for `batch_optimize`:

- Optimize all images in `./source_images` with default settings:

  ```bash
  python optimizes.py ./source_images
  ```

- Optimize all images in `./source_images` with custom JPEG quality (70%) and save to `./optimized_batch`:

  ```bash
  python optimizes.py ./source_images -o ./optimized_batch --jpg-quality 70
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
- `--use_space_char/--no-use_space_char`: Enable or disable space character support (default: enabled).
- `--use_dilation/--no-use_dilation`: Enable or disable dilation on text regions (default: enabled).
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

## Batch Optical Character Recognition (OCR)

The `ocrs.py` script processes all images within a specified input directory, applying the same OCR logic as `ocr.py` to each image and saving the results.

Syntax:

```bash
python ocrs.py <input_dir> [options]
```

Key Options for `batch_ocr`:

- `input_dir`: (Required) Path to the directory containing image files to process.
- `-o, --output-dir`: Directory to save the output CSV files (passed to `ocr.py`). `ocr.py` default: `output`.
- `--lang`: OCR language (default: `korean`). (`ocr.py` default)
- `--rec_model_dir`: Path to the recognition model directory (default: `./models/ko_PP-OCRv3_rec_infer`). (`ocr.py` default)
- `--det_model_dir`: Path to the detection model directory (default: `./models/ch_PP-OCRv3_det_infer`). (`ocr.py` default)
- `--cls_model_dir`: Path to the direction classification model directory (default: `./models/ch_ppocr_mobile_v2.0_cls_infer`). (`ocr.py` default)
- `--use_gpu`: Whether to use GPU for OCR processing (default: `False`). (`ocr.py` default)
- `--rec_char_dict_path`: Path to recognition character dictionary (default: `None`, uses PaddleOCR default). (`ocr.py` default)
- `--rec_batch_num`: Recognition batch size (default: `6`). (`ocr.py` default)
- `--det_db_thresh`: Detection DB threshold (default: `0.4`). (`ocr.py` default)
- `--det_db_box_thresh`: Detection DB box threshold (default: `0.6`). (`ocr.py` default)
- `--det_db_unclip_ratio`: Detection DB unclip ratio (default: `1.8`). (`ocr.py` default)
- `--drop_score`: Drop score for text detection (default: `0.6`). (`ocr.py` default)
- `--cls_thresh`: Classification threshold (default: `0.9`). (`ocr.py` default)
- `--use_angle_cls`: Whether to use angle classification (default: `False`). (`ocr.py` default)
- `--use_space_char/--no-use_space_char`: Enable or disable space character support (default: enabled). (`ocr.py` default)
- `--use_dilation/--no-use_dilation`: Enable or disable dilation on text regions (default: enabled). (`ocr.py` default)
- `--show_log`: Whether to display PaddleOCR's internal logs (default: `False`). (`ocr.py` default)

Examples for `batch_ocr`:

- Extract text from all images in `./scan_docs`:

  ```bash
  python ocrs.py ./scan_docs
  ```

- Extract Korean text from all images in `./id_cards` (using GPU):

  ```bash
  python ocrs.py ./id_cards --lang korean --use_gpu
  ```

The script will save extracted text or structured data for each image in the specified output directory.

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
