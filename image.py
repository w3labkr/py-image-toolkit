import argparse
import sys
import os

from modules.resize import (
    execute_resize_operation,
    SUPPORTED_OUTPUT_FORMATS,
    FILTER_NAMES,
    _PIL_RESAMPLE_FILTERS
)
from modules.crop import (
    execute_crop_operation,
    CropSetupError,
    MODEL_DIR_NAME,
    YUNET_MODEL_FILENAME
)

def handle_resize_command(args: argparse.Namespace):
    parser = args.parser_ref

    if args.include_extensions and args.exclude_extensions:
        parser.error("The arguments --include-extensions and --exclude-extensions cannot be used together.")

    if args.ratio == 'aspect_ratio':
        if args.width <= 0 and args.height <= 0:
            parser.error("For --ratio=aspect_ratio, at least --width or --height must be a positive integer.")
        if args.width < 0 or args.height < 0:
            parser.error("--width and --height must be non-negative for --ratio=aspect_ratio.")
    elif args.ratio == 'fixed':
        if args.width <= 0 or args.height <= 0:
            parser.error("For --ratio=fixed, both --width and --height must be positive integers.")
    elif args.ratio == 'none':
        if args.width > 0 or args.height > 0:
             print("   -> Warning: --width and --height arguments are ignored when --ratio=none.")

    if not (1 <= args.jpg_quality <= 100):
        parser.error(f"--jpeg-quality must be an integer between 1 and 100 (inclusive), got {args.jpg_quality}.")
    if not (1 <= args.webp_quality <= 100):
        parser.error(f"--webp-quality must be an integer between 1 and 100 (inclusive), got {args.webp_quality}.")

    if args.output_format != 'jpg' and args.jpg_quality != 95:
        print(f"   -> Warning: --jpeg-quality argument is ignored when output format is not JPG.")
    if args.output_format != 'webp' and args.webp_quality != 80:
        print(f"   -> Warning: --webp-quality argument is ignored when output format is not WEBP.")
    if args.webp_lossless and args.output_format != 'webp':
        print("   -> Warning: --webp-lossless argument is ignored when output format is not WEBP.")
    
    config = {k: v for k, v in vars(args).items() if k not in ['func', 'parser_ref']}

    config['absolute_input_dir'] = os.path.abspath(args.input_dir)
    config['absolute_output_dir'] = os.path.abspath(args.output_dir)

    config['resize_options'] = {
        'mode': args.ratio,
        'width': args.width,
        'height': args.height,
        'filter_str': args.filter,
        'filter_obj': _PIL_RESAMPLE_FILTERS[args.filter]
    }
    
    config['output_format_options'] = {
        'format_str': args.output_format
    }
    if args.output_format == 'jpg':
        config['output_format_options']['quality'] = args.jpg_quality
    elif args.output_format == 'webp':
        config['output_format_options'].update({
            'quality': args.webp_quality,
            'lossless': args.webp_lossless
        })
    
    try:
        error_count = execute_resize_operation(config)
        sys.exit(1 if error_count > 0 else 0)
    except Exception as e:
        print(f"CRITICAL: An unhandled critical exception occurred in resize operation: {e}", file=sys.stderr)
        print(f"\n(!) Critical Error in resize operation: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(2)


def _create_resize_parser(subparsers: argparse._SubParsersAction):
    resize_parser = subparsers.add_parser(
        "resize",
        help="Batch resize images.",
        description="Batch Image Resizing",
        formatter_class=argparse.RawTextHelpFormatter
    )
    resize_parser.set_defaults(func=handle_resize_command, parser_ref=resize_parser)

    resize_parser.add_argument("input_dir", nargs='?', default='input',
                               help="Path to the source image folder or a single image file (default: 'input' in current directory).")
    
    resize_parser.add_argument("-o", "--output-dir", dest='output_dir', default='output',
                               help="Path to save processed images (default: 'output' in current directory).")

    resize_parser.add_argument("-f", "--output-format",
                               default='original',
                               choices=SUPPORTED_OUTPUT_FORMATS.keys(),
                               help="Target output file format (default: original):\n" +
                                    "\n".join([f"  {k}: {v}" for k, v in SUPPORTED_OUTPUT_FORMATS.items()]))

    ratio_group = resize_parser.add_argument_group('Resize Ratio Option')
    ratio_group.add_argument("-r", "--ratio",
                             choices=['aspect_ratio', 'fixed', 'none'],
                             default='aspect_ratio',
                             help="Resize ratio behavior (default: aspect_ratio):\n"
                                  "  aspect_ratio: Maintain aspect ratio to fit target Width/Height.\n"
                                  "  fixed: Force resize to target Width x Height (may distort).\n"
                                  "  none: No resizing (only format conversion/EXIF handling).")

    resize_group = resize_parser.add_argument_group('Resize Dimensions (if --ratio is not "none")')
    resize_group.add_argument("-w", "--width", type=int, default=0, 
                              help="Target width in pixels for resizing.")
    resize_group.add_argument("-H", "--height", type=int, default=0, 
                              help="Target height in pixels for resizing.")
    resize_group.add_argument("--filter",
                              default='lanczos',
                              choices=FILTER_NAMES.keys(),
                              help="Resampling filter for resizing (default: lanczos):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]))

    optional_group = resize_parser.add_argument_group('Other Optional Options for Resize')
    optional_group.add_argument("-q", "--jpeg-quality", type=int, dest='jpg_quality',
                                default=95,
                                help="Quality for JPG output (1-100, higher is better). Default: 95")
    optional_group.add_argument("--webp-quality", type=int, dest='webp_quality',
                                default=80,
                                help="Quality for WEBP output (1-100, higher is better). Default: 80 (for lossy WEBP).")
    optional_group.add_argument("--webp-lossless", action="store_true",
                                help="Use lossless compression for WEBP output. Ignored if output format is not WEBP.")
    optional_group.add_argument("--strip-exif", action="store_true", 
                                help="Remove all EXIF metadata from images.")
    
    optional_group.add_argument("--overwrite", dest='overwrite', action='store_true', default=False,
                                 help="Overwrite existing output files. If not specified, existing files will be skipped.")

    optional_group.add_argument("--include-extensions", nargs='+', metavar='EXT',
                                help="Process only files with these extensions (e.g., jpg png).\n"
                                     "Replaces the default list. Provide without dots (e.g., 'jpg').\n"
                                     "Cannot be used with --exclude-extensions.")
    optional_group.add_argument("--exclude-extensions", nargs='+', metavar='EXT',
                                help="Exclude files with these extensions (e.g., gif tiff).\n"
                                     "Applied AFTER default/include list. Provide without dots.\n"
                                     "Cannot be used with --include-extensions.")
    return resize_parser


def handle_crop_command(args: argparse.Namespace):
    crop_op_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != 'parser_ref'})

    try:
        error_image_count = execute_crop_operation(crop_op_args)
        if error_image_count > 0:
            print(f"Crop operation completed with {error_image_count} image(s) having errors. Check logs.", file=sys.stderr)
            sys.exit(1)
        else:
            sys.exit(0)
    except CropSetupError as e:
        print(f"(!) Critical Crop Setup Error: {e}", file=sys.stderr)
        sys.exit(2) 
    except Exception as e:
        print(f"CRITICAL: An unhandled critical exception occurred in crop operation: {e}", file=sys.stderr)
        print(f"\n(!) Critical Error in crop operation: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(3) 


def _create_crop_parser(subparsers: argparse._SubParsersAction):
    crop_parser = subparsers.add_parser(
        "crop",
        help="Automatically crop images based on face detection and composition rules.",
        description="Batch image cropping using face detection (YuNet) and composition rules (thirds, golden ratio).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    crop_parser.set_defaults(func=handle_crop_command, parser_ref=crop_parser)

    crop_parser.add_argument("input_path", nargs='?', default="input", help="Path to the image file or directory to process (Default: 'input').")
    crop_parser.add_argument("-o", "--output_dir", default="output", help="Directory to save results (Default: 'output').")
    
    crop_parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output files (Default: False).")
    crop_parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate processing without saving files (Default: False).")

    crop_parser.add_argument("-m", "--method", choices=['largest', 'center'], default='largest', help="Method to select main subject (Default: largest).")
    crop_parser.add_argument("--ref", "--reference", dest="reference", choices=['eye', 'box'], default='box', help="Reference point for composition (Default: box).")
    crop_parser.add_argument("-c", "--confidence", type=float, default=0.6, help="Min face detection confidence (Default: 0.6).")
    crop_parser.add_argument("-n", "--nms", type=float, default=0.3, help="Face detection NMS threshold (Default: 0.3).")
    crop_parser.add_argument("--min-face-width", type=int, default=30, help="Min face width in pixels (Default: 30).")
    crop_parser.add_argument("--min-face-height", type=int, default=30, help="Min face height in pixels (Default: 30).")

    crop_parser.add_argument("-r", "--ratio", type=str, default=None, help="Target crop aspect ratio (e.g., '16:9', '1.0', 'None') (Default: None).")
    crop_parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], default='both', help="Composition rule(s) (Default: both).")
    crop_parser.add_argument("-p", "--padding-percent", type=float, default=5.0, help="Padding percentage around crop (%) (Default: 5.0).")

    crop_parser.add_argument("--output-format", type=str, default=None, help="Output image format (e.g., 'jpg', 'png', 'webp'). Default: original.")
    crop_parser.add_argument("-q", "--jpeg-quality", type=int, choices=range(1, 101), metavar="[1-100]", default=95, help="JPEG quality (1-100) (Default: 95).")
    crop_parser.add_argument("--webp-quality", type=int, choices=range(1, 101), metavar="[1-100]", default=80, help="WebP quality (1-100) (Default: 80).")
    crop_parser.add_argument("--strip-exif", action="store_true", default=False, help="Remove EXIF data from output images (Default: False).")
    
    crop_parser.add_argument("--yunet-model-path", type=str, default=None, help=f"Path to the YuNet ONNX model file. If not specified, it defaults to '{MODEL_DIR_NAME}/{YUNET_MODEL_FILENAME}' and will be downloaded if missing.")

    crop_parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable detailed (DEBUG level) logging for the crop operation (Default: False).")
    return crop_parser


def main():
    parser = argparse.ArgumentParser(
        description="Py Image Toolkit CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s (module version not available)'
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose_global",
        help="Enable verbose (DEBUG level) logging for detailed output across all commands."
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True,
                                       help="Available commands")

    _create_resize_parser(subparsers)
    _create_crop_parser(subparsers)

    try:
        args = parser.parse_args()
        args.func(args)
            
    except Exception as e:
        print(f"(!) An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()