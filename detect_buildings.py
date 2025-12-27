#!/usr/bin/env python3
"""
Building Footprint Detection using Edge Detection Techniques

Detects building footprints from cleaned atlas map images using Canny edge
detection with Hough line reinforcement.

Usage:
    uv run python detect_buildings.py -i p2crop.png -o buildings
    uv run python detect_buildings.py -i p2crop.png -o buildings --dilation 5
"""

import argparse
import base64
from pathlib import Path
from typing import List, Tuple
import colorsys
import math

import cv2
import numpy as np


# Default parameters
DEFAULT_DILATION = 3
DEFAULT_MIN_AREA = 100  # Lowered since we filter slivers by width, not area
DEFAULT_MIN_WIDTH = 12.0  # Minimum width to be considered a building (not a sliver)
DEFAULT_MIN_LINE_LENGTH = 40


def load_and_encode_image(path: Path) -> Tuple[np.ndarray, str, int, int]:
    """Load image and encode as base64 for SVG embedding."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    height, width = image.shape[:2]
    _, buffer = cv2.imencode('.png', image)
    b64_string = base64.b64encode(buffer).decode('utf-8')

    return image, b64_string, width, height


def filter_contours_by_area(contours: List[np.ndarray],
                            min_area: float = DEFAULT_MIN_AREA,
                            max_area: float = None) -> List[np.ndarray]:
    """Filter contours by area range."""
    result = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        result.append(c)
    return result


def filter_by_minimum_width(contours: List[np.ndarray],
                            min_width: float = 8.0) -> List[np.ndarray]:
    """Filter out contours that are too thin (slivers).

    Uses the rotated minimum-area bounding rectangle to find the true
    minimum width of each contour. This catches slivers regardless of
    their length - a 3px wide sliver is filtered whether it's 50px or 200px long.

    Small compact buildings (e.g., 15×15) pass because their minimum
    dimension is still >= min_width.
    """
    result = []
    for c in contours:
        # Get minimum area rotated bounding rectangle
        rect = cv2.minAreaRect(c)
        width, height = rect[1]  # (center, (w, h), angle)
        min_dim = min(width, height)

        if min_dim >= min_width:
            result.append(c)

    return result


def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
    """Simplify contour using Douglas-Peucker algorithm."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


# =============================================================================
# Denoising Options
# =============================================================================

def denoise_none(gray: np.ndarray) -> np.ndarray:
    """No denoising."""
    return gray


def denoise_bilateral(gray: np.ndarray, d: int = 9,
                      sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral filter - smooths while preserving edges.

    d: Diameter of pixel neighborhood
    sigma_color: Filter sigma in color space (larger = more colors mixed)
    sigma_space: Filter sigma in coordinate space (larger = farther pixels influence)
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def denoise_nlm(gray: np.ndarray, h: float = 10,
                template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Non-local means denoising - good for removing noise while keeping edges.

    h: Filter strength (higher = more denoising but may lose detail)
    """
    return cv2.fastNlMeansDenoising(gray, None, h, template_window, search_window)


def denoise_gaussian(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def denoise_morph_open(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological opening on the image to reduce noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


# =============================================================================
# Edge Detection
# =============================================================================

def edges_canny(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detection."""
    return cv2.Canny(gray, low, high)


def remove_small_components(edges: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove small connected components from edge map (noise removal)."""
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)

    # Create output mask
    result = np.zeros_like(edges)

    # Keep only components larger than min_size
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255

    return result


# =============================================================================
# Line Detection
# =============================================================================

def detect_lines_hough(edges: np.ndarray, min_length: int = 40,
                       max_gap: int = 5) -> List[Tuple[int, int, int, int]]:
    """Detect line segments using Probabilistic Hough Transform.

    max_gap: Maximum gap between line segments to treat as single line.
             Keep small (5) to avoid connecting dotted/dashed lines.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    if lines is None:
        return []

    return [tuple(line[0]) for line in lines]


def draw_lines_on_mask(mask: np.ndarray, lines: List[Tuple[int, int, int, int]],
                       thickness: int = 2) -> np.ndarray:
    """Draw line segments onto a mask."""
    result = mask.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(result, (x1, y1), (x2, y2), 255, thickness)
    return result


# =============================================================================
# Contour Smoothing
# =============================================================================

def smooth_contour(contour: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    """Smooth contour by fitting to a slightly simplified polygon."""
    epsilon = smoothing * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


# =============================================================================
# Main Detection Pipeline
# =============================================================================

def detect_buildings(
    image: np.ndarray,
    denoise_fn=None,
    canny_low: int = 50,
    canny_high: int = 150,
    remove_noise_components: bool = False,
    min_component_size: int = 50,
    use_hough: bool = True,
    min_line_length: int = 40,
    max_line_gap: int = 5,
    dilation: int = DEFAULT_DILATION,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float = None,
    min_width: float = DEFAULT_MIN_WIDTH,
    contour_smoothing: float = 0.005
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Detect buildings with configurable pipeline.

    Returns (contours, edge_map)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Denoising
    if denoise_fn:
        gray = denoise_fn(gray)

    # Step 2: Edge detection
    edges = edges_canny(gray, canny_low, canny_high)

    # Step 3: Remove small noise components
    if remove_noise_components:
        edges = remove_small_components(edges, min_component_size)

    # Step 4: Hough line reinforcement
    if use_hough:
        lines = detect_lines_hough(edges, min_line_length, max_line_gap)
        edges = draw_lines_on_mask(edges, lines, thickness=2)

    # Step 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=dilation)
    dilated = cv2.dilate(closed, kernel, iterations=dilation)

    # Step 6: Find contours
    inverted = cv2.bitwise_not(dilated)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Filter by area
    contours = filter_contours_by_area(contours, min_area, max_area)

    # Step 8: Filter out slivers (shapes too thin to be buildings)
    contours = filter_by_minimum_width(contours, min_width)

    # Step 9: Smooth contours
    contours = [smooth_contour(c, contour_smoothing) for c in contours]

    return contours, edges


# =============================================================================
# SVG Generation
# =============================================================================

def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n distinct colors using HSV color wheel."""
    if n == 0:
        return []
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def contour_to_svg_points(contour: np.ndarray) -> str:
    """Convert OpenCV contour to SVG polygon points string."""
    points = contour.squeeze()
    if len(points.shape) == 1:
        return ""
    return " ".join(f"{p[0]},{p[1]}" for p in points)


def generate_svg(
    width: int,
    height: int,
    contours: List[np.ndarray],
    output_path: Path,
    image_b64: str = None,
    fill_opacity: float = 0.5,
    stroke_width: int = 2
) -> None:
    """Generate SVG with colored polygon overlay.

    Args:
        width: SVG width
        height: SVG height
        contours: List of contours to draw
        output_path: Output file path
        image_b64: Optional base64-encoded background image (None = no background)
        fill_opacity: Polygon fill opacity
        stroke_width: Polygon stroke width
    """
    colors = generate_distinct_colors(len(contours))

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
    ]

    if image_b64:
        svg_parts.append(f'     xmlns:xlink="http://www.w3.org/1999/xlink"')

    svg_parts.extend([
        f'     width="{width}" height="{height}"',
        f'     viewBox="0 0 {width} {height}">',
        '',
    ])

    if image_b64:
        svg_parts.extend([
            '  <!-- Background image -->',
            f'  <image xlink:href="data:image/png;base64,{image_b64}"',
            f'         width="{width}" height="{height}"/>',
            '',
        ])

    svg_parts.extend([
        f'  <!-- Detected buildings: {len(contours)} polygons -->',
        '  <g id="detected-buildings">',
    ])

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

    print(f"  -> {output_path.name}: {len(contours)} polygons")


# =============================================================================
# Batch Tests
# =============================================================================

DEFAULT_MAX_AREA = 60000  # Filter out large polygons (streets/background)


def run_batch_tests(image: np.ndarray, image_b64: str, width: int, height: int,
                    output_prefix: str, dilation: int, min_area: float,
                    max_area: float = DEFAULT_MAX_AREA,
                    min_width: float = DEFAULT_MIN_WIDTH,
                    include_background: bool = False) -> None:
    """Run focused set of tests - baseline, NLM smoothing, and min_width variations.

    Args:
        include_background: If True, embed image in SVG. If False, polygons only.
    """

    bg = image_b64 if include_background else None
    filter_params = dict(min_area=min_area, max_area=max_area, min_width=min_width)

    # ==========================================================================
    # Baseline: No denoising
    # ==========================================================================
    print("\n[Baseline]")
    contours, edges = detect_buildings(
        image, denoise_fn=None, use_hough=True,
        min_line_length=40, max_line_gap=5,
        dilation=dilation, **filter_params
    )
    generate_svg(width, height, contours, Path(f"{output_prefix}_baseline.svg"), image_b64=bg)
    cv2.imwrite(f"{output_prefix}_baseline_edges.png", edges)

    # ==========================================================================
    # NLM h5: Smoother contours
    # ==========================================================================
    print("\n[NLM h5 - smoother contours]")
    contours, edges = detect_buildings(
        image, denoise_fn=lambda g: denoise_nlm(g, 5), use_hough=True,
        min_line_length=40, max_line_gap=5,
        dilation=dilation, **filter_params
    )
    generate_svg(width, height, contours, Path(f"{output_prefix}_nlm_h5.svg"), image_b64=bg)

    # ==========================================================================
    # Test min_width variations (the sliver filter)
    # ==========================================================================
    print("\n[Min Width Variations]")
    for mw in [4, 6, 8, 10, 12]:
        test_params = dict(min_area=min_area, max_area=max_area, min_width=mw)
        contours, _ = detect_buildings(
            image, denoise_fn=None, use_hough=True,
            min_line_length=40, max_line_gap=5,
            dilation=dilation, **test_params
        )
        generate_svg(width, height, contours,
                    Path(f"{output_prefix}_minwidth{mw}.svg"), image_b64=bg)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Detect building footprints with denoising options.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', type=Path, default=Path('p2crop.png'))
    parser.add_argument('-o', '--output', type=str, default='buildings')
    parser.add_argument('-d', '--dilation', type=int, default=DEFAULT_DILATION)
    parser.add_argument('--min-area', type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument('--max-area', type=float, default=DEFAULT_MAX_AREA,
                        help='Maximum polygon area (filter out large regions)')
    parser.add_argument('--min-width', type=float, default=DEFAULT_MIN_WIDTH,
                        help='Minimum width to be a building (filters thin slivers)')
    parser.add_argument('--with-background', action='store_true',
                        help='Include base64 image in SVG (larger files)')

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading image: {args.input}")
    image, image_b64, width, height = load_and_encode_image(args.input)
    print(f"Image size: {width}x{height}")
    print(f"Parameters: dilation={args.dilation}, min_area={args.min_area}, "
          f"max_area={args.max_area}, min_width={args.min_width}")
    print(f"Background image in SVG: {args.with_background}")

    print("\nGenerating batch tests...")
    run_batch_tests(image, image_b64, width, height,
                    args.output, args.dilation, args.min_area,
                    max_area=args.max_area,
                    min_width=args.min_width,
                    include_background=args.with_background)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
