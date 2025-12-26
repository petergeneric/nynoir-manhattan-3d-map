#!/usr/bin/env python3
"""Experiment 3: Edge Detection + Morphological Filtering

Use Canny edge detection with morphological operations to extract
building boundary lines while filtering out text.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from utils.svg_output import save_svg, filter_contours_by_area, simplify_contour
from utils.visualization import overlay_contours, save_comparison


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for edge detection.

    Args:
        image: Input image (BGR)

    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """Detect edges using Canny algorithm.

    Args:
        image: Preprocessed grayscale image
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny

    Returns:
        Binary edge map
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def morphological_cleanup(
    edges: np.ndarray,
    close_size: int = 5,
    close_iterations: int = 2,
    open_size: int = 3,
    open_iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operations to clean up edges.

    Args:
        edges: Binary edge map
        close_size: Kernel size for closing
        close_iterations: Number of closing iterations
        open_size: Kernel size for opening
        open_iterations: Number of opening iterations

    Returns:
        Cleaned edge map
    """
    # Closing: Connect broken lines
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    # Opening: Remove small noise (text-like thin strokes)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)

    return opened


def fill_closed_regions(edges: np.ndarray) -> np.ndarray:
    """Fill closed regions in edge map to create solid building regions.

    Args:
        edges: Binary edge map

    Returns:
        Filled binary mask
    """
    # Create a slightly larger canvas to handle edge cases
    h, w = edges.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = edges

    # Flood fill from corners (background)
    filled = padded.copy()
    cv2.floodFill(filled, None, (0, 0), 255)

    # Invert to get filled regions
    filled_inv = cv2.bitwise_not(filled)

    # Combine with original edges
    result = cv2.bitwise_or(edges, filled_inv[1:-1, 1:-1])

    return result


def filter_by_shape(
    contours: list,
    min_area: float = 500,
    min_solidity: float = 0.3,
    max_aspect_ratio: float = 10.0,
) -> list:
    """Filter contours by shape characteristics.

    Args:
        contours: List of contours
        min_area: Minimum contour area
        min_solidity: Minimum solidity (area / convex hull area)
        max_aspect_ratio: Maximum aspect ratio (filters thin text-like shapes)

    Returns:
        Filtered list of contours
    """
    filtered = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Calculate bounding rect aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w / h, h / w) if min(w, h) > 0 else float('inf')
        if aspect_ratio > max_aspect_ratio:
            continue

        # Calculate solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        filtered.append(contour)

    return filtered


def main():
    # Paths
    input_image = Path("/Users/pwright/workspace/atlas/media/map-example.png")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image: {input_image}")
    image = cv2.imread(str(input_image))
    if image is None:
        print(f"Error: Could not load image {input_image}")
        return 1

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")

    # Preprocessing
    print("\nPreprocessing...")
    preprocessed = preprocess_image(image)
    cv2.imwrite(str(output_dir / "preprocessed.png"), preprocessed)

    # Try multiple edge detection configurations
    configs = [
        # (name, canny_low, canny_high, close_size, open_size)
        ("standard", 50, 150, 5, 3),
        ("sensitive", 30, 100, 7, 3),
        ("strict", 80, 200, 3, 2),
    ]

    best_contours = None
    best_name = None
    best_edges = None

    for name, canny_low, canny_high, close_size, open_size in configs:
        print(f"\nTrying config '{name}': Canny({canny_low},{canny_high}), close={close_size}, open={open_size}")

        # Edge detection
        edges = detect_edges(preprocessed, canny_low, canny_high)
        cv2.imwrite(str(output_dir / f"edges_{name}_raw.png"), edges)

        # Morphological cleanup
        cleaned = morphological_cleanup(edges, close_size, 2, open_size, 1)
        cv2.imwrite(str(output_dir / f"edges_{name}_cleaned.png"), cleaned)

        # Try to fill closed regions
        filled = fill_closed_regions(cleaned)
        cv2.imwrite(str(output_dir / f"edges_{name}_filled.png"), filled)

        # Find contours
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"  Raw contours: {len(contours)}")

        # Filter by shape
        filtered = filter_by_shape(contours, min_area=500)
        print(f"  After shape filter: {len(filtered)}")

        if best_contours is None or len(filtered) > len(best_contours):
            best_contours = filtered
            best_name = name
            best_edges = cleaned

    print(f"\nBest config: '{best_name}' with {len(best_contours)} contours")

    # Save best results
    cv2.imwrite(str(output_dir / "edges_raw.png"), detect_edges(preprocessed, 50, 150))
    cv2.imwrite(str(output_dir / "edges_cleaned.png"), best_edges)

    # Simplify contours
    simplified = [simplify_contour(c, 0.005) for c in best_contours]

    # Create visualization
    contour_viz = overlay_contours(image, simplified, fill=True, fill_alpha=0.4)
    cv2.imwrite(str(output_dir / "contours.png"), contour_viz)

    # Save SVG
    save_svg(simplified, w, h, output_dir / "buildings.svg")

    # Alternative approach: use edges directly for line detection
    print("\nAlternative: Direct line-based extraction...")

    # Use Hough lines for straight edge detection
    raw_edges = detect_edges(preprocessed, 50, 150)
    lines = cv2.HoughLinesP(
        raw_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10,
    )

    if lines is not None:
        print(f"Detected {len(lines)} line segments")
        line_viz = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / "hough_lines.png"), line_viz)
    else:
        print("No lines detected")

    # Create comparison
    edges_bgr = cv2.cvtColor(best_edges, cv2.COLOR_GRAY2BGR)
    preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    save_comparison(
        [image, preprocessed_bgr, edges_bgr, contour_viz],
        ["Original", "Preprocessed", "Edges", "Contours"],
        output_dir / "comparison.png",
        max_width=2400,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: Edge Detection + Morphology")
    print("=" * 50)
    print(f"Best edge config: {best_name}")
    print(f"Building contours extracted: {len(simplified)}")
    if lines is not None:
        print(f"Hough line segments: {len(lines)}")
    total_polygon_points = sum(len(c) for c in simplified)
    print(f"Total polygon points: {total_polygon_points}")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - edges_raw.png: Raw Canny edges")
    print("  - edges_cleaned.png: Morphologically cleaned edges")
    print("  - contours.png: Extracted contours overlaid")
    print("  - buildings.svg: SVG with building polygons")
    print("  - hough_lines.png: Hough line detection")
    print("  - comparison.png: Side-by-side comparison")

    return 0


if __name__ == "__main__":
    exit(main())
