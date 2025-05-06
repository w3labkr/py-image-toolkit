# py-image-toolkit

`py-image-toolkit` is a Python-based toolkit for image processing. It provides tools for resizing images and performing automatic cropping based on face detection and composition rules. With a simple command-line interface (CLI), it's designed to be fast, efficient, and easy to use.

---

## Folder and File Structure

```plaintext
py-image-toolkit/
├── resizer.py         # Script for image resizing
├── cropper.py         # Script for face detection and auto-cropping
├── README.md          # Project documentation
├── requirements.txt   # List of required libraries
└── models/
    └── yunet.onnx     # YuNet model for face detection (used by OpenCV)
```

- `resizer.py`: Handles resizing, aspect ratio adjustments, and format conversion
- `cropper.py`: Detects faces and crops images based on composition rules (e.g., Rule of Thirds, Golden Ratio)
- `requirements.txt`: Lists Python packages required to run the project
- `models/yunet.onnx`: Pretrained YuNet face detection model

---

## Key Features

### Image Resizing

- Aspect Ratio Resizing: Maintains aspect ratio while resizing
- Fixed Size Resizing: Forces images to fit specific dimensions
- Format Conversion: Supports PNG, JPG, WEBP, and more
- EXIF Metadata Preservation: Keeps original metadata intact
- Recursive Processing: Processes directories and subdirectories

### Face Detection & Auto-Cropping

- DNN-based Face Detection: Uses OpenCV’s YuNet for face detection
- Composition-Aware Cropping: Applies Rule of Thirds, Golden Ratio, or both
- Main Subject Selection: Selects either the largest face or the one closest to the center
- Aspect Ratio Cropping: Supports standard ratios like 16:9, 1:1, etc.
- Parallel Processing: Speeds up processing using multiple cores

---

## Installation

### Requirements

- Python 3.8 or higher
- Required libraries:
  - `opencv-python`
  - `opencv-contrib-python`
  - `numpy`
  - `Pillow`
  - `tqdm`
  - `piexif`

### Setup Instructions

1. Clone the repository and set up a virtual environment:

   ```bash
   git clone https://github.com/w3labkr/py-image-toolkit.git
   cd py-image-toolkit
   pyenv install 3.12.9
   pyenv virtualenv 3.12.9 py-image-toolkit-3.12.9
   pyenv local py-image-toolkit-3.12.9
   ```

2. Install the required dependencies:

   ```bash
   pip install opencv-python opencv-contrib-python numpy Pillow tqdm piexif
   ```

---

## Usage

### Image Resizing

#### Basic Command

```bash
python resizer.py [input_directory] -m <mode> -O <format> [options]
```

#### Key Options

* `-m`, `--resize-mode`: Resize mode (`aspect_ratio`, `fixed`, `none`)
* `-O`, `--output-format`: Output format (`original`, `png`, `jpg`, `webp`)
* `-w`, `--width`: Target width in pixels
* `-H`, `--height`: Target height in pixels
* `-f`, `--filter`: Resize filter (`lanczos`, `bicubic`, `bilinear`, `nearest`)
* `-o`, `--output-dir`: Output directory path
* `-r`, `--recursive`: Enable recursive directory processing

#### Examples

* Aspect Ratio Resize:

  ```bash
  python resizer.py ./input -o ./output -m aspect_ratio -f nearest -w 1280 -O jpg
  ```

* Fixed Size Resize:

  ```bash
  python resizer.py ./input -o ./output -m fixed -w 720 -H 600 -f bicubic -O png
  ```

---

### Face Detection & Auto-Cropping

#### Basic Command

```bash
python cropper.py <input_path> [options]
```

#### Key Options

* `-o`, `--output_dir`: Output directory
* `-m`, `--method`: Main subject selection (`largest`, `center`)
* `--ref`, `--reference`: Reference point (`eye`, `box`)
* `-r`, `--ratio`: Desired crop aspect ratio (e.g., `16:9`, `1.0`)
* `--rule`: Composition rule (`thirds`, `golden`, `both`)
* `-p`, `--padding-percent`: Padding percentage around crop area
* `--dry-run`: Preview without saving output

#### Examples

* Crop a single image:

  ```bash
  python cropper.py ./input -o ./output -r 16:9 --rule thirds
  ```

* Crop all images in a folder:

  ```bash
  python cropper.py ./input -o ./output -r 1:1 -m center --ref box --rule both
  ```

---

## Example Results

### Resized Image

* Input: `image.jpg` (1920x1080)
* Output: `image_resized.jpg` (1280x720)

### Cropped Image

* Input: `portrait.jpg` (3000x2000)
* Output:

  * `portrait_thirds_r16-9_refEye.jpg`
  * `portrait_golden_r16-9_refEye.jpg`

---

## Error Handling & Debugging

* Missing Libraries: Use `pip install` to install dependencies
* Missing Model File: `cropper.py` will auto-download YuNet if missing
* Image Not Processed:

  * Ensure input path and image format are valid
  * Use `--verbose` for detailed logs

---

## License

This project is licensed under the MIT License.
