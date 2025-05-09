# py-image-toolkit

`py-image-toolkit` is a Python-based toolkit for image processing. It provides tools for resizing images and performing automatic cropping based on face detection and composition rules. With a simple command-line interface (CLI), it's designed to be fast, efficient, and easy to use.

---

## Folder and File Structure

```plaintext
py-image-toolkit/
├── resizer.py         # Script for image resizing
├── cropper.py         # Script for face detection and auto-cropping
├── ocr.py             # Script for OCR text extraction
├── README.md          # Project documentation
├── requirements.txt   # List of required libraries
└── models/
    └── yunet.onnx     # YuNet model for face detection (used by OpenCV)
```

- `resizer.py`: Handles resizing, aspect ratio adjustments, and format conversion
- `cropper.py`: Detects faces and crops images based on composition rules (e.g., Rule of Thirds, Golden Ratio)
- `ocr.py`: Extracts text from images using OCR with preprocessing and labeling features
- `requirements.txt`: Lists Python packages required to run the project
- `models/yunet.onnx`: Pretrained YuNet face detection model

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
  - `paddleocr`

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
   pip install -r requirements.txt
   ```

---

## Image Resizing

The `resizer.py` script (v3.38) processes images from an input directory by resizing them while optionally converting formats and handling EXIF metadata. It employs multiprocessing to speed up batch processing and offers detailed logging.

- Aspect Ratio Resizing: Maintains aspect ratio while resizing
- Fixed Size Resizing: Forces images to fit specific dimensions
- Format Conversion: Supports PNG, JPG, WEBP, and more
- EXIF Metadata Preservation: Keeps original metadata intact
- Recursive Processing: Processes directories and subdirectories

Syntax:

```bash
python resizer.py <input_path> [options]
```

Key Options:

- `-f`, `--output-format`: Target output file format (`original`, `png`, `jpg`, `webp`). (Default: `original`)
- `-r`, `--ratio`: Resize ratio behavior. (Default: `aspect_ratio`)
  - `aspect_ratio`: Maintain aspect ratio to fit the target size. Requires at least one of `--width` or `--height`.
  - `fixed`: Force resize to exact dimensions (may distort the image). Both `--width` and `--height` must be specified.
  - `none`: No resizing; only format conversion and EXIF handling will be applied.
- `-w`, `--width`: Target width in pixels.
- `-H`, `--height`: Target height in pixels.
- `--filter`: Resampling filter for resizing (`lanczos`, `bicubic`, `bilinear`, `nearest`). (Default: `lanczos`)
- `-o`, `--output-dir`: Output directory for processed images. (Default: `output`)
- `--jpeg-quality`: Quality for JPG output (1-100, default: `95`).
- `--webp-quality`: Quality for WEBP output (1-100, default: `80` for lossy WEBP).
- `--strip-exif`: Remove all EXIF metadata from images. (Default: EXIF is preserved unless this flag is used).
- `--overwrite`: Overwrite existing output files. By default, existing files are skipped.
- `--include-extensions`: Process only files with these extensions (e.g., `jpg png`).
- `--exclude-extensions`: Exclude files with these extensions from processing (e.g., `gif tiff`).
- `--webp-lossless`: Use lossless compression for WEBP output (only applicable when the output format is WEBP).
- `-v`, `--verbose`: Enable detailed (DEBUG level) logging.
- `--version`: Show program's version number and exit.

Examples

- Resize while maintaining aspect ratio:

  ```bash
  python resizer.py ./input -w 1280
  ```

- Resize to fixed dimensions:

  ```bash
  python resizer.py ./input -f png -r fixed -w 720 -H 600
  ```

- Convert format without resizing and strip EXIF data:

  ```bash
  python resizer.py ./input -f webp -r none --strip-exif
  ```

---

## Face Detection & Auto-Cropping

The `cropper.py` script (v1.8.0) detects faces in images and automatically crops them based on composition rules.

- DNN-based Face Detection: Uses OpenCV’s YuNet for face detection
- Composition-Aware Cropping: Applies Rule of Thirds, Golden Ratio, or both
- Main Subject Selection: Selects either the largest face or the one closest to the center
- Aspect Ratio Cropping: Supports standard ratios like 16:9, 1:1, etc.
- Parallel Processing: Speeds up processing using multiple cores

Syntax:

```bash
python cropper.py <input_path> [options]
```

Key Options:

- `--config`: Path to a JSON configuration file to load options from.
- `-o`, `--output_dir`: Output directory (Default: `output`).
- `--output-format`: Output image format (e.g., `jpg`, `png`, `webp`). (Default: original format is kept).
- `-q, --jpeg-quality`: JPEG quality for JPG output (1-100) (Default: `95`).
- `--webp-quality`: Quality for WEBP output (1-100) (Default: `80`).
- `--overwrite`: Overwrite existing output files. (Default: files are skipped if they exist).
- `--strip-exif`: Remove all EXIF metadata from output images. (Default: EXIF is preserved. Use this flag to strip).
- `-m`, `--method`: Main subject selection method (`largest` face area, or face `center` closest to image center) (Default: `largest`).
- `--ref`, `--reference`: Reference point on the subject for composition (`eye` center, `box` center of the face) (Default: `box`).
- `-r`, `--ratio`: Desired crop aspect ratio (e.g., `16:9`, `1.0`). (Default: original image ratio).
- `--rule`: Composition rule(s) to apply (`thirds`, `golden`, or `both`) (Default: `both`).
- `-p`, `--padding-percent`: Percentage of padding to add around the main subject crop area (Default: `5.0`).
- `-c`, `--confidence`: Minimum confidence score for face detection (0.0-1.0) (Default: `0.6`).
- `-n`, `--nms`: Non-maximum suppression (NMS) threshold for face detection (0.0-1.0) (Default: `0.3`).
- `--min-face-width`: Minimum detected face width in pixels (Default: `30`).
- `--min-face-height`: Minimum detected face height in pixels (Default: `30`).
- `--dry-run`: Perform a trial run, showing intended actions without writing output files.
- `-v, --verbose`: Enable verbose logging for detailed process information.
- `--version`: Show program's version number and exit.

Examples

- Crop a single image:

  ```bash
  python cropper.py ./input/image.jpg -o ./output -r 16:9 --rule thirds
  ```

- Crop all images in a folder, selecting the main subject based on center proximity and using bounding box as a reference:

  ```bash
  python cropper.py ./input -o ./output -r 1:1 -m center --ref box --rule both
  ```

- Perform a dry run without writing output files:

  ```bash
  python cropper.py ./input -o ./output --dry-run
  ```

---

## OCR Text Extraction

The ocr.py script (v1.2.0) uses PaddleOCR to extract and label text from images, especially Korean ID fields. It supports preprocessing and parallel processing, and outputs results in detailed and summary CSVs, plus optional annotated images.

- **Text Recognition**: Leverages PaddleOCR for multilingual text extraction.
- **Image Preprocessing**: Optional steps to enhance image quality for OCR.
- **Automatic Labeling**: Heuristic-based labeling for common ID document fields.
- **Parallel Processing**: Speeds up batch processing using multiple CPU cores.
- **CSV Output**: Generates both detailed and summary reports of extracted text.
- **Visualization**: Overlays extracted text and labels on output images.

Syntax:

```bash
python ocr.py <input_dir> [options]
```

Key Options:

- `input_dir`: Path to the folder containing images for text extraction. (Default: `input`)
- `--output_dir`: Path to the folder where OCR result images and CSV files will be saved. (Default: `output`)
- `--lang`: Language to use for OCR (e.g., `korean`, `en`, `ch_sim`). (Default: `korean`)
- `--show_image`: Display each processed image with OCR results on screen.
- `--no_preprocess`: Skip the image preprocessing step.
- `--use_gpu`: Attempt to use GPU for OCR processing if available (requires NVIDIA GPU and CUDA).
- `--num_workers`: Number of worker processes for parallel processing. (Default: Number of CPU cores)
- `--debug`: Enable debug level logging for more detailed output.
- `--version`: Show script's version number and exit.

Examples

- Extract text from Korean ID cards in `./input` and save results to `./output`:

  ```bash
  python ocr.py ./input --output_dir ./output --lang korean
  ```

- Process images in `./input` using GPU, 4 worker processes, and skip preprocessing:

  ```bash
  python ocr.py ./input --use_gpu --num_workers 4 --no_preprocess
  ```

---

## Troubleshooting

If you encounter issues while using the toolkit, refer to the following common problems and solutions:

### Missing Libraries

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

For PaddleOCR-specific dependencies:

```bash
pip install paddleocr
```

### Missing Model File

The `cropper.py` script will automatically download the YuNet model if it is not found in the `models/` directory. Ensure you have an active internet connection.

### Image Not Processed

- Verify that the input path is correct and contains valid image files.
- Use the `--verbose` flag to enable detailed logs for debugging:
  ```bash
  python resizer.py ./input --verbose
  ```

---

## License

This project is licensed under the [MIT License](LICENSE).
