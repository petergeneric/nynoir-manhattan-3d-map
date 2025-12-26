#!/usr/bin/env python3
"""
Text Detection and Inpainting Experiment

Uses EasyOCR to detect text regions in an image, then uses OpenCV
inpainting to remove the text and fill in the background.

Usage:
    uv run python inpaint_text.py
    uv run python inpaint_text.py -i /path/to/input.png -o /path/to/output.png
    uv run python inpaint_text.py --method telea  # Use Telea inpainting
    uv run python inpaint_text.py --expand 5      # Expand mask by 5 pixels
"""

import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import easyocr
from PIL import Image

# Default paths
DEFAULT_INPUT = Path(__file__).parent.parent / "media" / "p2-mono.png"
DEFAULT_OUTPUT = Path(__file__).parent / "inpainted.png"

# OCR parameters
TEXT_THRESHOLD = 0.7
LINK_THRESHOLD = 0.1
LOW_TEXT = 0.4


def get_cache_path(image_path: Path) -> Path:
    """Get cache file path for OCR results."""
    return image_path.with_suffix(image_path.suffix + '.ocr_cache.json')


def load_ocr_cache(cache_path: Path, image_path: Path) -> tuple[list, list] | None:
    """Load OCR results from cache if valid."""
    if not cache_path.exists():
        return None

    if image_path.stat().st_mtime > cache_path.stat().st_mtime:
        print("OCR cache is stale - will regenerate")
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded OCR cache from {cache_path}")
        return data['horizontal'], data['free']
    except (json.JSONDecodeError, KeyError) as e:
        print(f"OCR cache invalid ({e}) - will regenerate")
        return None


def save_ocr_cache(cache_path: Path, horizontal: list, free: list):
    """Save OCR results to cache."""
    data = {'horizontal': horizontal, 'free': free}
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    print(f"Saved OCR cache to {cache_path}")


def detect_text_regions(image_path: Path) -> tuple[list, list]:
    """
    Run EasyOCR to detect text regions.

    Returns:
        tuple: (horizontal_boxes, free_boxes)
            - horizontal_boxes: [[x_min, x_max, y_min, y_max], ...]
            - free_boxes: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
    """
    cache_path = get_cache_path(image_path)
    cached = load_ocr_cache(cache_path, image_path)

    if cached is not None:
        return cached

    # Check for GPU
    use_gpu = False
    if torch.backends.mps.is_available():
        print("MPS (Metal) GPU detected - enabling GPU acceleration")
        use_gpu = True
    elif torch.cuda.is_available():
        print("CUDA GPU detected - enabling GPU acceleration")
        use_gpu = True
    else:
        print("No GPU available - using CPU")

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=use_gpu)

    with Image.open(image_path) as img:
        width, height = img.size

    print(f"Running text detection on {image_path.name} ({width}x{height})...")
    horizontal_boxes, free_boxes = reader.detect(
        str(image_path),
        text_threshold=TEXT_THRESHOLD,
        link_threshold=LINK_THRESHOLD,
        low_text=LOW_TEXT,
        canvas_size=max(width, height),
    )

    raw_horizontal = []
    raw_free = []

    # Extract horizontal boxes [x_min, x_max, y_min, y_max]
    if horizontal_boxes and len(horizontal_boxes[0]) > 0:
        for box in horizontal_boxes[0]:
            x_min, x_max, y_min, y_max = box
            raw_horizontal.append([float(x_min), float(x_max), float(y_min), float(y_max)])

    # Extract free-form boxes [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    if free_boxes and len(free_boxes[0]) > 0:
        for box in free_boxes[0]:
            coords = [[float(pt[0]), float(pt[1])] for pt in box]
            raw_free.append(coords)

    save_ocr_cache(cache_path, raw_horizontal, raw_free)

    print(f"Detected {len(raw_horizontal)} horizontal + {len(raw_free)} free-form text regions")
    return raw_horizontal, raw_free


def create_text_mask(
    image_shape: tuple[int, int],
    horizontal_boxes: list,
    free_boxes: list,
    expand_pixels: int = 3
) -> np.ndarray:
    """
    Create a binary mask where text regions are white (255).

    Args:
        image_shape: (height, width) of the image
        horizontal_boxes: List of [x_min, x_max, y_min, y_max]
        free_boxes: List of [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        expand_pixels: Number of pixels to expand the mask by (dilation)

    Returns:
        Binary mask (uint8) with text regions as 255
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw horizontal boxes
    for x_min, x_max, y_min, y_max in horizontal_boxes:
        x1, x2 = int(x_min), int(x_max)
        y1, y2 = int(y_min), int(y_max)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Draw free-form boxes as filled polygons
    for coords in free_boxes:
        pts = np.array([[int(x), int(y)] for x, y in coords], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Expand mask by dilation
    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return mask


def inpaint_image(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = 'ns',
    radius: int = 5
) -> np.ndarray:
    """
    Inpaint the image to remove text regions.

    Args:
        image: Input image (BGR or grayscale)
        mask: Binary mask where 255 = regions to inpaint
        method: 'ns' (Navier-Stokes) or 'telea' (Alexandru Telea)
        radius: Inpainting radius

    Returns:
        Inpainted image
    """
    if method == 'telea':
        flags = cv2.INPAINT_TELEA
    else:
        flags = cv2.INPAINT_NS

    return cv2.inpaint(image, mask, radius, flags)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect and inpaint text from images using EasyOCR and OpenCV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=DEFAULT_INPUT,
        help='Input image path'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output image path'
    )
    parser.add_argument(
        '--method',
        choices=['ns', 'telea'],
        default='ns',
        help='Inpainting method: ns (Navier-Stokes) or telea (Alexandru Telea)'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Inpainting radius in pixels'
    )
    parser.add_argument(
        '--expand',
        type=int,
        default=3,
        help='Pixels to expand text mask by (dilation)'
    )
    parser.add_argument(
        '--save-mask',
        type=Path,
        default=None,
        help='Optional path to save the text mask image'
    )
    parser.add_argument(
        '--save-debug',
        type=Path,
        default=None,
        help='Optional path to save debug image with detected boxes'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Inpainting method: {args.method} (radius={args.radius})")
    print(f"Mask expansion: {args.expand}px")
    print()

    # Load image
    image = cv2.imread(str(args.input))
    if image is None:
        print(f"ERROR: Could not load image: {args.input}")
        return 1

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    # Detect text regions
    horizontal_boxes, free_boxes = detect_text_regions(args.input)
    total_regions = len(horizontal_boxes) + len(free_boxes)

    if total_regions == 0:
        print("No text detected - copying input to output")
        cv2.imwrite(str(args.output), image)
        return 0

    # Create mask
    print(f"\nCreating text mask...")
    mask = create_text_mask((height, width), horizontal_boxes, free_boxes, args.expand)
    mask_coverage = np.count_nonzero(mask) / (width * height) * 100
    print(f"Mask coverage: {mask_coverage:.2f}%")

    # Save mask if requested
    if args.save_mask:
        cv2.imwrite(str(args.save_mask), mask)
        print(f"Saved mask to {args.save_mask}")

    # Save debug image if requested
    if args.save_debug:
        debug_img = image.copy()
        # Draw horizontal boxes in red
        for x_min, x_max, y_min, y_max in horizontal_boxes:
            cv2.rectangle(debug_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        # Draw free-form boxes in blue
        for coords in free_boxes:
            pts = np.array([[int(x), int(y)] for x, y in coords], dtype=np.int32)
            cv2.polylines(debug_img, [pts], True, (255, 0, 0), 2)
        cv2.imwrite(str(args.save_debug), debug_img)
        print(f"Saved debug image to {args.save_debug}")

    # Inpaint
    print(f"\nInpainting with {args.method} method (radius={args.radius})...")
    result = inpaint_image(image, mask, args.method, args.radius)

    # Save result
    cv2.imwrite(str(args.output), result)
    print(f"\nSaved inpainted image to {args.output}")

    output_size = args.output.stat().st_size
    print(f"Output file size: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
