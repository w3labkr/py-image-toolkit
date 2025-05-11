# py-image-toolkit

`py-image-toolkit` is a Python-based toolkit for image processing. It provides a command-line interface (CLI) through `image.py` for resizing images and performing automatic cropping based on face detection and composition rules. It's designed to be fast, efficient, and easy to use.

---

## Folder and File Structure

```plaintext
py-image-toolkit/
├── modules/
│   ├── resize.py      # Module for image resizing logic
│   └── crop.py        # Module for face detection and auto-cropping logic
├── models/
│   └── face_detection_yunet_2023mar.onnx # YuNet model for face detection
├── image.py           # Main CLI script for resize and crop commands
├── requirements.txt   # List of required libraries
└── README.md          # Project documentation
```

- `modules/resize.py`: Contains the core logic for image resizing, aspect ratio adjustments, and format conversion.
- `modules/crop.py`: Contains the core logic for detecting faces and cropping images based on composition rules.
- `models/face_detection_yunet_2023mar.onnx`: Pretrained YuNet face detection model, automatically downloaded if not present.
- `image.py`: The main entry point for accessing `resize` and `crop` functionalities.
- `requirements.txt`: Lists Python packages required to run the project.

---

## Installation

### Requirements

- Python 3.8 or higher
- Required libraries (see `requirements.txt`):
  - `opencv-python`
  - `Pillow`
  - `tqdm`
  - `piexif` (for `resize` command if preserving EXIF in some cases)
  - `numpy`

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

The main `image.py` script provides the following global options:

- `--version`: Show program's version number and exit.
- `-v, --verbose`: Enable verbose (DEBUG level) logging for detailed output across all commands. This is a global flag; individual commands like `crop` might also have their own verbose flags for more specific debug output.

---

## Image Resizing

The `resize` command processes images from an input directory by resizing them while optionally converting formats and handling EXIF metadata. It employs multiprocessing to speed up batch processing and offers detailed logging.

- Aspect Ratio Resizing: Maintains aspect ratio while resizing.
- Fixed Size Resizing: Forces images to fit specific dimensions.
- Format Conversion: Supports PNG, JPG, WEBP, and keeping the original format.
- EXIF Metadata Preservation: Keeps original metadata intact by default.
- Recursive Processing: Processes directories and subdirectories.

Syntax:

```bash
python image.py resize <input_dir> [options]
```

Key Options for `resize`:

- `input_dir`: Path to the source image folder or a single image file (Default: `input`).
- `-o, --output-dir`: Output directory for processed images (Default: `output`).
- `-f, --output-format`: Target output file format (`original`, `png`, `jpg`, `webp`). (Default: `original`).
- `-r, --ratio`: Resize ratio behavior (`aspect_ratio`, `fixed`, `none`). (Default: `aspect_ratio`).
  - `aspect_ratio`: Maintain aspect ratio to fit the target size. Requires at least one of `--width` or `--height` to be positive.
  - `fixed`: Force resize to exact dimensions (may distort). Both `--width` and `--height` must be positive.
  - `none`: No resizing; only format conversion and EXIF handling will be applied. If `width` or `height` are specified with `none`, a warning will be shown as they are ignored.
- `-w, --width`: Target width in pixels for resizing (Default: `0`).
- `-H, --height`: Target height in pixels for resizing (Default: `0`).
- `--filter`: Resampling filter for resizing (`lanczos`, `bicubic`, `bilinear`, `nearest`). (Default: `lanczos`).
- `-q, --jpeg-quality`: Quality for JPG output (1-100). (Default: `95`). A warning is shown if used when output format is not JPG.
- `--webp-quality`: Quality for WEBP output (1-100). (Default: `80` for lossy). A warning is shown if used when output format is not WEBP.
- `--webp-lossless`: Use lossless compression for WEBP output. A warning is shown if used when output format is not WEBP.
- `--strip-exif`: Remove all EXIF metadata from images.
- `--overwrite`: Overwrite existing output files. If not specified, existing files will be skipped.
- `--include-extensions`: Process only files with these extensions (e.g., `jpg png`). Replaces default list. Cannot be used with `--exclude-extensions`.
- `--exclude-extensions`: Exclude files with these extensions from processing (e.g., `gif tiff`). Applied after default/include list. Cannot be used with `--include-extensions`.

Examples for `resize`:

- Resize images in `./input` to a width of 1280px, maintaining aspect ratio:
  ```bash
  python image.py resize ./input -w 1280
  ```
- Resize images to fixed 720x600 dimensions and output as PNG:
  ```bash
  python image.py resize ./input -f png -r fixed -w 720 -H 600
  ```
- Convert images to WEBP format without resizing and strip EXIF data:
  ```bash
  python image.py resize ./input -f webp -r none --strip-exif
  ```

---

## Face Detection & Auto-Cropping

The `crop` command detects faces in images and automatically crops them based on face detection and composition rules.

- DNN-based Face Detection: Uses OpenCV’s YuNet model for face detection.
- Composition-Aware Cropping: Applies Rule of Thirds, Golden Ratio, or both.
- Main Subject Selection: Selects either the largest face or the one closest to the image center.
- Aspect Ratio Cropping: Supports standard ratios like 16:9, 1:1, or custom.
- Parallel Processing: Speeds up batch processing using multiple CPU cores for directory input.

Syntax:

```bash
python image.py crop <input_path> [options]
```

Key Options for `crop`:

- `input_path`: Path to the image file or directory to process (Default: `input`).
- `-o, --output_dir`: Directory to save results (Default: `output`).
- `--overwrite`: Overwrite existing output files (Default: False, files are skipped).
- `--dry-run`: Simulate processing without saving files (Default: False).
- `-m, --method`: Method to select main subject (`largest`, `center`) (Default: `largest`).
- `--ref, --reference`: Reference point for composition (`eye` center, `box` center of the face) (Default: `box`).
- `-c, --confidence`: Minimum face detection confidence (0.0-1.0) (Default: `0.6`).
- `-n, --nms`: Face detection Non-Maximum Suppression (NMS) threshold (0.0-1.0) (Default: `0.3`).
- `--min-face-width`: Minimum detected face width in pixels (Default: `30`).
- `--min-face-height`: Minimum detected face height in pixels (Default: `30`).
- `-r, --ratio`: Target crop aspect ratio (e.g., `16:9`, `1.0`, `None` for original) (Default: `None`). Use 'None' (case-insensitive string) or omit for original aspect ratio.
- `--rule`: Composition rule(s) to apply (`thirds`, `golden`, `both`) (Default: `both`).
- `-p, --padding-percent`: Padding percentage around crop area (%) (Default: `5.0`).
- `--output-format`: Output image format (e.g., `jpg`, `png`, `webp`). (Default: original format is kept).
- `-q, --jpeg-quality`: JPEG quality for JPG output (1-100) (Default: `95`).
- `--webp-quality`: Quality for WEBP output (1-100) (Default: `80`).
- `--strip-exif`: Remove EXIF data from output images (Default: False, EXIF is preserved).
- `--yunet-model-path`: Path to the YuNet ONNX model file. (Default: `models/face_detection_yunet_2023mar.onnx`, downloaded if missing).
- `-v, --verbose`: Enable detailed (DEBUG level) logging *specifically for the crop operation* (Default: False).

Examples for `crop`:

- Crop a single image `./input/image.jpg` to 16:9 ratio using rule of thirds:
  ```bash
  python image.py crop ./input/image.jpg -o ./output -r 16:9 --rule thirds
  ```
- Crop all images in `./input`, selecting main subject by proximity to center, using bounding box reference, 1:1 ratio, and both rules:
  ```bash
  python image.py crop ./input -o ./output -r 1:1 -m center --ref box --rule both
  ```
- Perform a dry run for cropping images in `./input`:
  ```bash
  python image.py crop ./input -o ./output --dry-run
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

The `crop` command (via `image.py crop`) will automatically attempt to download the YuNet model if it is not found in the `models/` directory. Ensure you have an active internet connection during the first run or if the model is missing.

### Image Not Processed

- Verify that the input path is correct and contains valid image files with supported extensions.
- Use the global `--verbose` flag with `image.py` or specific verbose flags for commands (like `-v` for `image.py crop`) to enable detailed logs for debugging:
  ```bash
  python image.py resize ./input --verbose
  python image.py crop ./input -v
  ```
- If `image.py resize` command fails due to invalid arguments or critical processing errors, it will exit with a non-zero status code. Check terminal output for specific error messages.
- If `image.py crop` command encounters critical setup issues (e.g., model download failure, output directory problems, invalid input path) or unhandled processing errors, it will exit with a non-zero status code. `CropSetupError` indicates a problem that prevented the operation from starting. Check terminal output and logs for details.

---

## License

This project is licensed under the [MIT License](LICENSE).
