"""SVG output utilities for building polygon generation."""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2


def contour_to_svg_points(contour: np.ndarray) -> str:
    """Convert OpenCV contour to SVG polygon points string."""
    points = contour.squeeze()
    if len(points.shape) == 1:
        return ""
    return " ".join(f"{p[0]},{p[1]}" for p in points)


def generate_svg_from_contours(
    contours: List[np.ndarray],
    width: int,
    height: int,
    colors: List[Tuple[int, int, int]] = None,
    fill_opacity: float = 0.3,
    stroke_width: int = 2,
) -> str:
    """Generate SVG content from a list of contours.

    Args:
        contours: List of OpenCV contours (np.ndarray)
        width: Image width
        height: Image height
        colors: Optional list of RGB colors for each contour
        fill_opacity: Fill opacity (0-1)
        stroke_width: Stroke width in pixels

    Returns:
        SVG content as string
    """
    # Generate colors if not provided
    if colors is None:
        np.random.seed(42)
        colors = [
            (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
            for _ in range(len(contours))
        ]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '  <!-- Building polygons extracted from historical map -->',
        f'  <!-- Total polygons: {len(contours)} -->',
        '',
    ]

    for idx, contour in enumerate(contours):
        points_str = contour_to_svg_points(contour)
        if not points_str:
            continue

        r, g, b = colors[idx % len(colors)]
        fill_color = f"rgba({r},{g},{b},{fill_opacity})"
        stroke_color = f"rgb({r},{g},{b})"

        svg_parts.append(
            f'  <polygon points="{points_str}" '
            f'fill="{fill_color}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width}" '
            f'data-polygon-id="{idx}"/>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def save_svg(
    contours: List[np.ndarray],
    width: int,
    height: int,
    output_path: Path,
    **kwargs
) -> None:
    """Save contours as SVG file.

    Args:
        contours: List of OpenCV contours
        width: Image width
        height: Image height
        output_path: Path to save SVG
        **kwargs: Additional arguments passed to generate_svg_from_contours
    """
    svg_content = generate_svg_from_contours(contours, width, height, **kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(svg_content)

    print(f"Saved SVG with {len(contours)} polygons: {output_path}")


def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    """Simplify a contour using Douglas-Peucker algorithm.

    Args:
        contour: OpenCV contour
        epsilon_factor: Simplification factor (relative to arc length)

    Returns:
        Simplified contour
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def filter_contours_by_area(
    contours: List[np.ndarray],
    min_area: float = 100,
    max_area: float = None,
) -> List[np.ndarray]:
    """Filter contours by area.

    Args:
        contours: List of contours
        min_area: Minimum area to keep
        max_area: Maximum area to keep (None = no limit)

    Returns:
        Filtered list of contours
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered.append(contour)
    return filtered
