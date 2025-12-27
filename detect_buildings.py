#!/usr/bin/env python3
"""
Building Footprint Detection using Traditional CV Techniques

Detects building footprints from cleaned atlas map images using edge-based
approaches. Buildings are distinguished by black outline boundaries, not fill color.

Usage:
    uv run python detect_buildings.py -i p2crop.png -o buildings
    uv run python detect_buildings.py -i p2crop.png -o buildings --threshold 80 --dilation 5

Outputs multiple SVG files for comparison:
    - buildings_threshold.svg - Dark line threshold + dilation
    - buildings_canny.svg - Canny edge + morphological closing
    - buildings_adaptive.svg - Adaptive threshold + watershed
    - buildings_hierarchy.svg - Contour hierarchy approach
    - buildings_floodfill.svg - Grid-sampled flood fill
"""

import argparse
import base64
from pathlib import Path
from typing import List, Tuple
import colorsys

import cv2
import numpy as np


# Default parameters
DEFAULT_THRESHOLD = 100  # Darkness threshold for line detection
DEFAULT_DILATION = 3     # Iterations of dilation to close gaps
DEFAULT_MIN_AREA = 500   # Minimum polygon area to keep


def load_and_encode_image(path: Path) -> Tuple[np.ndarray, str, int, int]:
    """Load image and encode as base64 for SVG embedding.

    Returns:
        Tuple of (image_bgr, base64_string, width, height)
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    height, width = image.shape[:2]

    # Encode as PNG base64
    _, buffer = cv2.imencode('.png', image)
    b64_string = base64.b64encode(buffer).decode('utf-8')

    return image, b64_string, width, height


def detect_dark_lines(gray: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Threshold to find dark pixels (lines).

    Args:
        gray: Grayscale image
        threshold: Pixels darker than this are considered lines

    Returns:
        Binary mask where 255 = line pixels
    """
    # Invert so dark pixels become white (255)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def dilate_lines(mask: np.ndarray, iterations: int = DEFAULT_DILATION,
                 kernel_size: int = 3) -> np.ndarray:
    """Morphological dilation to close gaps in hand-drawn lines.

    Args:
        mask: Binary mask of line pixels
        iterations: Number of dilation iterations
        kernel_size: Size of the structuring element

    Returns:
        Dilated binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def filter_contours_by_area(contours: List[np.ndarray],
                            min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Filter contours by minimum area."""
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
    """Simplify contour using Douglas-Peucker algorithm."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


# =============================================================================
# Detection Approaches
# =============================================================================

def detect_threshold_dilate(image: np.ndarray, threshold: int = DEFAULT_THRESHOLD,
                            dilation: int = DEFAULT_DILATION,
                            min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Approach 1: Dark line threshold + dilation + contours.

    Simple approach: threshold to find dark lines, dilate to close gaps,
    invert and find contours of enclosed regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect dark lines
    lines = detect_dark_lines(gray, threshold)

    # Dilate to close gaps
    lines_dilated = dilate_lines(lines, dilation)

    # Invert: now white = building interiors, black = lines
    inverted = cv2.bitwise_not(lines_dilated)

    # Find contours of white regions
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and simplify
    contours = filter_contours_by_area(contours, min_area)
    contours = [simplify_contour(c) for c in contours]

    return contours


def detect_canny_close(image: np.ndarray, dilation: int = DEFAULT_DILATION,
                       min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Approach 2: Canny edge detection + morphological closing.

    Uses Canny edge detection which is better at finding edges regardless of
    absolute pixel values, then morphological closing to bridge gaps.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological closing (dilate then erode) to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=dilation)

    # Additional dilation to ensure connectivity
    closed = cv2.dilate(closed, kernel, iterations=dilation)

    # Invert and find contours
    inverted = cv2.bitwise_not(closed)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and simplify
    contours = filter_contours_by_area(contours, min_area)
    contours = [simplify_contour(c) for c in contours]

    return contours


def detect_adaptive_watershed(image: np.ndarray, dilation: int = DEFAULT_DILATION,
                              min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Approach 3: Adaptive threshold + watershed segmentation.

    Adaptive threshold handles uneven paper aging/lighting. Watershed
    segmentation separates touching regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold - finds dark lines regardless of local brightness
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # Dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(adaptive, kernel, iterations=dilation)

    # Distance transform for watershed
    inverted = cv2.bitwise_not(dilated)
    dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

    # Threshold distance transform to get markers
    _, markers = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    markers = np.uint8(markers)

    # Find contours of markers (these are our building regions)
    contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and simplify
    contours = filter_contours_by_area(contours, min_area)
    contours = [simplify_contour(c) for c in contours]

    return contours


def detect_hierarchy(image: np.ndarray, threshold: int = DEFAULT_THRESHOLD,
                     dilation: int = 5, min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Approach 4: Line detection + contour hierarchy.

    Uses contour hierarchy to find enclosed regions. Buildings are "holes"
    in the line network when we invert the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect dark lines with more aggressive dilation
    lines = detect_dark_lines(gray, threshold)
    lines_dilated = dilate_lines(lines, dilation, kernel_size=5)

    # Find contours with full hierarchy
    # RETR_CCOMP retrieves all contours and organizes them into a two-level hierarchy
    contours, hierarchy = cv2.findContours(
        lines_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    # Hierarchy format: [Next, Previous, First_Child, Parent]
    # We want contours that have a parent (i.e., are holes in something)
    building_contours = []
    for i, h in enumerate(hierarchy[0]):
        # h[3] is the parent index, -1 means no parent (top-level)
        # We want contours that are children (holes in the line network)
        if h[3] != -1:  # Has a parent
            building_contours.append(contours[i])

    # If no holes found, fall back to inverted approach
    if not building_contours:
        inverted = cv2.bitwise_not(lines_dilated)
        building_contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    # Filter and simplify
    building_contours = filter_contours_by_area(building_contours, min_area)
    building_contours = [simplify_contour(c) for c in building_contours]

    return building_contours


def detect_flood_fill_grid(image: np.ndarray, threshold: int = DEFAULT_THRESHOLD,
                           dilation: int = DEFAULT_DILATION, grid_step: int = 50,
                           min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Approach 5: Edge detection + grid-sampled flood fill.

    Samples a grid of seed points and flood fills from each point that isn't
    on an edge. Each unique filled region becomes a building polygon.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Detect dark lines and dilate
    lines = detect_dark_lines(gray, threshold)
    lines_dilated = dilate_lines(lines, dilation)

    # Invert so white = interior, black = lines (floodFill fills from seed until it hits non-zero)
    fill_image = cv2.bitwise_not(lines_dilated)

    # Track which regions we've filled using a label image
    labels = np.zeros((height, width), dtype=np.int32)
    region_id = 1

    # Sample grid of seed points
    for y in range(grid_step // 2, height, grid_step):
        for x in range(grid_step // 2, width, grid_step):
            # Skip if this point is on a line (black in fill_image)
            if fill_image[y, x] == 0:
                continue
            # Skip if already labeled
            if labels[y, x] != 0:
                continue

            # Create mask for flood fill (must be h+2, w+2)
            mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

            # Flood fill - fills connected white region
            # Use FLOODFILL_MASK_ONLY to only update the mask, not the image
            cv2.floodFill(fill_image, mask, (x, y), 128,
                         flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))

            # Extract the filled region (mask has 1-pixel border)
            region = mask[1:-1, 1:-1]

            if np.any(region):
                # Label this region
                labels[region > 0] = region_id
                region_id += 1

    # Find contours for each labeled region
    all_contours = []
    for rid in range(1, region_id):
        region_mask = (labels == rid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

    # Filter and simplify
    all_contours = filter_contours_by_area(all_contours, min_area)
    all_contours = [simplify_contour(c) for c in all_contours]

    return all_contours


# =============================================================================
# SVG Generation
# =============================================================================

def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n distinct colors using HSV color wheel."""
    colors = []
    for i in range(n):
        hue = i / n
        # Use high saturation and value for vivid colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def contour_to_svg_points(contour: np.ndarray) -> str:
    """Convert OpenCV contour to SVG polygon points string."""
    points = contour.squeeze()
    if len(points.shape) == 1:
        return ""
    return " ".join(f"{p[0]},{p[1]}" for p in points)


def generate_svg_with_background(
    image_b64: str,
    width: int,
    height: int,
    contours: List[np.ndarray],
    output_path: Path,
    fill_opacity: float = 0.5,
    stroke_width: int = 2
) -> None:
    """Generate SVG with base64 image background and colored polygon overlay.

    Args:
        image_b64: Base64-encoded PNG image
        width: Image width
        height: Image height
        contours: List of contours to draw as polygons
        output_path: Path to save SVG file
        fill_opacity: Opacity of polygon fills (0-1)
        stroke_width: Width of polygon strokes
    """
    colors = generate_distinct_colors(len(contours))

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     xmlns:xlink="http://www.w3.org/1999/xlink"',
        f'     width="{width}" height="{height}"',
        f'     viewBox="0 0 {width} {height}">',
        '',
        '  <!-- Background image -->',
        f'  <image xlink:href="data:image/png;base64,{image_b64}"',
        f'         width="{width}" height="{height}"/>',
        '',
        f'  <!-- Detected buildings: {len(contours)} polygons -->',
        '  <g id="detected-buildings">',
    ]

    for idx, contour in enumerate(contours):
        points_str = contour_to_svg_points(contour)
        if not points_str:
            continue

        r, g, b = colors[idx]
        fill_color = f"rgba({r},{g},{b},{fill_opacity})"
        stroke_color = f"rgb({r},{g},{b})"

        svg_parts.append(
            f'    <polygon points="{points_str}"'
            f' fill="{fill_color}" stroke="{stroke_color}"'
            f' stroke-width="{stroke_width}" data-id="{idx}"/>'
        )

    svg_parts.extend([
        '  </g>',
        '</svg>'
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"Saved {output_path} with {len(contours)} polygons")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect building footprints from atlas maps using CV techniques.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path('p2crop.png'),
        help='Input image path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='buildings',
        help='Output prefix for SVG files'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=DEFAULT_THRESHOLD,
        help='Darkness threshold for line detection (0-255, lower = more sensitive)'
    )
    parser.add_argument(
        '-d', '--dilation',
        type=int,
        default=DEFAULT_DILATION,
        help='Dilation iterations to close gaps in lines'
    )
    parser.add_argument(
        '--min-area',
        type=float,
        default=DEFAULT_MIN_AREA,
        help='Minimum polygon area to keep'
    )
    parser.add_argument(
        '--approaches',
        type=str,
        nargs='+',
        choices=['threshold', 'canny', 'adaptive', 'hierarchy', 'floodfill', 'all'],
        default=['all'],
        help='Which detection approaches to run'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading image: {args.input}")
    image, image_b64, width, height = load_and_encode_image(args.input)
    print(f"Image size: {width}x{height}")
    print(f"Parameters: threshold={args.threshold}, dilation={args.dilation}, min_area={args.min_area}")
    print()

    # Define approaches
    approaches = {
        'threshold': (
            'Dark Line Threshold + Dilation',
            lambda: detect_threshold_dilate(image, args.threshold, args.dilation, args.min_area)
        ),
        'canny': (
            'Canny Edge + Morphological Closing',
            lambda: detect_canny_close(image, args.dilation, args.min_area)
        ),
        'adaptive': (
            'Adaptive Threshold + Watershed',
            lambda: detect_adaptive_watershed(image, args.dilation, args.min_area)
        ),
        'hierarchy': (
            'Contour Hierarchy',
            lambda: detect_hierarchy(image, args.threshold, args.dilation + 2, args.min_area)
        ),
        'floodfill': (
            'Grid-Sampled Flood Fill',
            lambda: detect_flood_fill_grid(image, args.threshold, args.dilation, 50, args.min_area)
        ),
    }

    # Determine which approaches to run
    if 'all' in args.approaches:
        to_run = list(approaches.keys())
    else:
        to_run = args.approaches

    # Run each approach
    for name in to_run:
        desc, detect_fn = approaches[name]
        print(f"Running: {desc}...")

        try:
            contours = detect_fn()
            output_path = Path(f"{args.output}_{name}.svg")
            generate_svg_with_background(image_b64, width, height, contours, output_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
