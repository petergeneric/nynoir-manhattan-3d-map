#!/usr/bin/env python3
"""
Building Footprint Detection using Edge Detection Techniques

Detects building footprints from cleaned atlas map images using edge detection
with line reinforcement for faint straight lines.

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
DEFAULT_MIN_AREA = 500
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
                            min_area: float = DEFAULT_MIN_AREA) -> List[np.ndarray]:
    """Filter contours by minimum area."""
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
    """Simplify contour using Douglas-Peucker algorithm."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


# =============================================================================
# Edge Detection
# =============================================================================

def edges_canny(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detection."""
    return cv2.Canny(gray, low, high)


# =============================================================================
# Line Detection and Reinforcement
# =============================================================================

def detect_lines_hough(edges: np.ndarray, min_length: int = 40,
                       max_gap: int = 10) -> List[Tuple[int, int, int, int]]:
    """Detect line segments using Probabilistic Hough Transform.

    Returns list of (x1, y1, x2, y2) line segments.
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


def detect_lines_lsd(gray: np.ndarray, min_length: int = 40) -> List[Tuple[int, int, int, int]]:
    """Detect line segments using Line Segment Detector (LSD).

    LSD is often better than Hough for detecting faint lines.
    Returns list of (x1, y1, x2, y2) line segments.
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray)

    if lines is None:
        return []

    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length >= min_length:
            result.append((int(x1), int(y1), int(x2), int(y2)))

    return result


def draw_lines_on_mask(mask: np.ndarray, lines: List[Tuple[int, int, int, int]],
                       thickness: int = 2) -> np.ndarray:
    """Draw line segments onto a mask."""
    result = mask.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(result, (x1, y1), (x2, y2), 255, thickness)
    return result


def extend_line_segment(x1: int, y1: int, x2: int, y2: int,
                        extension: int = 5) -> Tuple[int, int, int, int]:
    """Extend a line segment by `extension` pixels on each end."""
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0:
        return (x1, y1, x2, y2)

    # Unit direction vector
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    # Extend both ends
    new_x1 = int(x1 - dx * extension)
    new_y1 = int(y1 - dy * extension)
    new_x2 = int(x2 + dx * extension)
    new_y2 = int(y2 + dy * extension)

    return (new_x1, new_y1, new_x2, new_y2)


def extend_lines(lines: List[Tuple[int, int, int, int]],
                 extension: int = 5) -> List[Tuple[int, int, int, int]]:
    """Extend all line segments."""
    return [extend_line_segment(*line, extension) for line in lines]


def find_nearby_endpoints(lines: List[Tuple[int, int, int, int]],
                          max_distance: int = 20,
                          max_angle_diff: float = 15.0) -> List[Tuple[int, int, int, int]]:
    """Find pairs of line endpoints that are close and roughly collinear.

    Returns new line segments that bridge the gaps.
    """
    if not lines:
        return []

    # Extract all endpoints with their line's angle
    endpoints = []  # (x, y, angle, line_idx, is_start)
    for idx, (x1, y1, x2, y2) in enumerate(lines):
        angle = math.atan2(y2 - y1, x2 - x1)
        endpoints.append((x1, y1, angle, idx, True))
        endpoints.append((x2, y2, angle, idx, False))

    bridges = []

    # Check each pair of endpoints from different lines
    for i, (x1, y1, angle1, idx1, _) in enumerate(endpoints):
        for j, (x2, y2, angle2, idx2, _) in enumerate(endpoints):
            if idx1 >= idx2:  # Same line or already checked
                continue

            # Check distance
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist > max_distance or dist < 1:
                continue

            # Check angle similarity (lines should be roughly parallel)
            angle_diff = abs(angle1 - angle2)
            # Normalize to 0-90 range (lines can point opposite directions)
            angle_diff = min(angle_diff, math.pi - angle_diff)
            angle_diff_deg = math.degrees(angle_diff)

            if angle_diff_deg <= max_angle_diff:
                bridges.append((x1, y1, x2, y2))

    return bridges


# =============================================================================
# Combined Detection Approaches
# =============================================================================

def detect_with_line_reinforcement(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    min_line_length: int = 40,
    line_extension: int = 5,
    use_lsd: bool = False,
    bridge_gaps: bool = False,
    bridge_distance: int = 20,
    dilation: int = DEFAULT_DILATION,
    min_area: float = DEFAULT_MIN_AREA
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Detect buildings with line reinforcement for faint straight lines.

    1. Run Canny edge detection
    2. Detect straight line segments (Hough or LSD)
    3. Optionally extend lines to bridge small gaps
    4. Optionally connect nearby collinear endpoints
    5. Combine edges with reinforced lines
    6. Find contours

    Returns (contours, final_edge_map)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Canny edge detection
    edges = edges_canny(gray, canny_low, canny_high)

    # Step 2: Detect line segments
    if use_lsd:
        lines = detect_lines_lsd(gray, min_line_length)
    else:
        lines = detect_lines_hough(edges, min_line_length)

    # Step 3: Extend lines if requested
    if line_extension > 0:
        lines = extend_lines(lines, line_extension)

    # Step 4: Bridge gaps between nearby endpoints
    if bridge_gaps:
        bridges = find_nearby_endpoints(lines, bridge_distance)
        lines = lines + bridges

    # Step 5: Draw lines onto edge map
    reinforced = draw_lines_on_mask(edges, lines, thickness=2)

    # Step 6: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(reinforced, cv2.MORPH_CLOSE, kernel, iterations=dilation)
    dilated = cv2.dilate(closed, kernel, iterations=dilation)

    # Step 7: Find contours
    inverted = cv2.bitwise_not(dilated)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_contours_by_area(contours, min_area)
    contours = [simplify_contour(c) for c in contours]

    return contours, reinforced


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


def generate_svg_with_background(
    image_b64: str,
    width: int,
    height: int,
    contours: List[np.ndarray],
    output_path: Path,
    fill_opacity: float = 0.5,
    stroke_width: int = 2
) -> None:
    """Generate SVG with base64 image background and colored polygon overlay."""
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

    print(f"  -> {output_path.name}: {len(contours)} polygons")


# =============================================================================
# Batch Test Generation
# =============================================================================

def run_batch_tests(image: np.ndarray, image_b64: str, width: int, height: int,
                    output_prefix: str, dilation: int, min_area: float) -> None:
    """Run batch of tests with line reinforcement variations."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ==========================================================================
    # Baseline: Canny 50/150 (the best from previous tests)
    # ==========================================================================
    print("\n[Baseline: Canny 50/150]")
    edges = edges_canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=dilation)
    dilated = cv2.dilate(closed, kernel, iterations=dilation)
    inverted = cv2.bitwise_not(dilated)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_by_area(contours, min_area)
    contours = [simplify_contour(c) for c in contours]
    generate_svg_with_background(image_b64, width, height, contours,
                                 Path(f"{output_prefix}_baseline.svg"))

    # ==========================================================================
    # Hough line reinforcement with different min lengths
    # ==========================================================================
    print("\n[Hough Line Reinforcement]")
    for min_len in [30, 40, 50, 60]:
        for ext in [0, 5, 10]:
            name = f"hough_len{min_len}_ext{ext}"
            contours, _ = detect_with_line_reinforcement(
                image, 50, 150, min_line_length=min_len, line_extension=ext,
                use_lsd=False, bridge_gaps=False, dilation=dilation, min_area=min_area
            )
            generate_svg_with_background(image_b64, width, height, contours,
                                         Path(f"{output_prefix}_{name}.svg"))

    # ==========================================================================
    # Hough with gap bridging
    # ==========================================================================
    print("\n[Hough + Gap Bridging]")
    for bridge_dist in [15, 20, 30]:
        name = f"hough_bridge{bridge_dist}"
        contours, _ = detect_with_line_reinforcement(
            image, 50, 150, min_line_length=40, line_extension=5,
            use_lsd=False, bridge_gaps=True, bridge_distance=bridge_dist,
            dilation=dilation, min_area=min_area
        )
        generate_svg_with_background(image_b64, width, height, contours,
                                     Path(f"{output_prefix}_{name}.svg"))

    # ==========================================================================
    # LSD line detection (often better for faint lines)
    # ==========================================================================
    print("\n[LSD Line Detection]")
    for min_len in [30, 40, 50]:
        for ext in [0, 5, 10]:
            name = f"lsd_len{min_len}_ext{ext}"
            contours, _ = detect_with_line_reinforcement(
                image, 50, 150, min_line_length=min_len, line_extension=ext,
                use_lsd=True, bridge_gaps=False, dilation=dilation, min_area=min_area
            )
            generate_svg_with_background(image_b64, width, height, contours,
                                         Path(f"{output_prefix}_{name}.svg"))

    # ==========================================================================
    # LSD with gap bridging
    # ==========================================================================
    print("\n[LSD + Gap Bridging]")
    for bridge_dist in [15, 20, 30]:
        name = f"lsd_bridge{bridge_dist}"
        contours, _ = detect_with_line_reinforcement(
            image, 50, 150, min_line_length=40, line_extension=5,
            use_lsd=True, bridge_gaps=True, bridge_distance=bridge_dist,
            dilation=dilation, min_area=min_area
        )
        generate_svg_with_background(image_b64, width, height, contours,
                                     Path(f"{output_prefix}_{name}.svg"))

    # ==========================================================================
    # Visualize detected lines (for debugging)
    # ==========================================================================
    print("\n[Line Detection Visualization]")

    # Hough lines on blank canvas
    edges = edges_canny(gray, 50, 150)
    hough_lines = detect_lines_hough(edges, min_length=40)
    line_img = np.zeros_like(gray)
    line_img = draw_lines_on_mask(line_img, hough_lines, thickness=2)
    cv2.imwrite(f"{output_prefix}_hough_lines.png", line_img)
    print(f"  -> {output_prefix}_hough_lines.png: {len(hough_lines)} lines")

    # LSD lines on blank canvas
    lsd_lines = detect_lines_lsd(gray, min_length=40)
    line_img = np.zeros_like(gray)
    line_img = draw_lines_on_mask(line_img, lsd_lines, thickness=2)
    cv2.imwrite(f"{output_prefix}_lsd_lines.png", line_img)
    print(f"  -> {output_prefix}_lsd_lines.png: {len(lsd_lines)} lines")

    # Canny + Hough reinforced
    edges = edges_canny(gray, 50, 150)
    hough_lines = detect_lines_hough(edges, min_length=40)
    hough_lines = extend_lines(hough_lines, 5)
    reinforced = draw_lines_on_mask(edges, hough_lines, thickness=2)
    cv2.imwrite(f"{output_prefix}_canny_plus_hough.png", reinforced)
    print(f"  -> {output_prefix}_canny_plus_hough.png")

    # Canny + LSD reinforced
    lsd_lines = detect_lines_lsd(gray, min_length=40)
    lsd_lines = extend_lines(lsd_lines, 5)
    reinforced = draw_lines_on_mask(edges, lsd_lines, thickness=2)
    cv2.imwrite(f"{output_prefix}_canny_plus_lsd.png", reinforced)
    print(f"  -> {output_prefix}_canny_plus_lsd.png")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect building footprints with line reinforcement.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', type=Path, default=Path('p2crop.png'))
    parser.add_argument('-o', '--output', type=str, default='buildings')
    parser.add_argument('-d', '--dilation', type=int, default=DEFAULT_DILATION)
    parser.add_argument('--min-area', type=float, default=DEFAULT_MIN_AREA)

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading image: {args.input}")
    image, image_b64, width, height = load_and_encode_image(args.input)
    print(f"Image size: {width}x{height}")
    print(f"Parameters: dilation={args.dilation}, min_area={args.min_area}")

    print("\nGenerating batch tests with line reinforcement...")
    run_batch_tests(image, image_b64, width, height,
                    args.output, args.dilation, args.min_area)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
