#!/usr/bin/env python3
"""Experiment 4: Hybrid Approach - Combining Text Masking + Color + Edge Detection

This combines the best techniques from experiments 1-3:
1. Detect text regions (don't inpaint, just mask)
2. Use color to identify potential building regions
3. Apply edge detection only within building-colored regions
4. Filter out text-overlapping contours
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import easyocr
from utils.svg_output import save_svg, filter_contours_by_area, simplify_contour
from utils.visualization import overlay_contours, save_comparison


def detect_text_mask(image_path: Path, reader: easyocr.Reader, dilation: int = 10) -> np.ndarray:
    """Detect text and return a binary mask of text regions."""
    print("Detecting text regions...")
    results = reader.readtext(str(image_path), detail=1)
    print(f"Found {len(results)} text regions")

    h, w = cv2.imread(str(image_path)).shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for detection in results:
        bbox = detection[0]
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    # Dilate to ensure full text coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def extract_building_color_mask(image: np.ndarray) -> np.ndarray:
    """Extract regions that have building-like pink/salmon coloring."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Pink/salmon buildings: low hue (red-orange), some saturation, high value
    # Also include areas that are light (could be faded buildings)
    lower_pink = np.array([0, 20, 150])
    upper_pink = np.array([25, 200, 255])
    mask1 = cv2.inRange(hsv, lower_pink, upper_pink)

    # Also try to capture the slightly darker building interiors
    lower_dark = np.array([0, 30, 120])
    upper_dark = np.array([20, 180, 200])
    mask2 = cv2.inRange(hsv, lower_dark, upper_dark)

    combined = cv2.bitwise_or(mask1, mask2)

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def extract_dark_lines(image: np.ndarray) -> np.ndarray:
    """Extract dark line work (building boundaries, streets)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold to get dark lines
    _, dark_mask = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned


def find_building_contours(
    building_mask: np.ndarray,
    line_mask: np.ndarray,
    text_mask: np.ndarray,
    min_area: float = 1000,
    max_text_overlap: float = 0.3,
) -> list:
    """Find building contours by combining color regions with line boundaries.

    Args:
        building_mask: Mask of building-colored regions
        line_mask: Mask of dark lines (boundaries)
        text_mask: Mask of text regions to exclude
        min_area: Minimum contour area
        max_text_overlap: Maximum fraction of contour that can overlap text

    Returns:
        List of filtered building contours
    """
    # Combine building color with line boundaries
    # The idea: buildings are colored regions bounded by dark lines
    # Subtract lines from building regions to get interior areas
    interior = cv2.bitwise_and(building_mask, cv2.bitwise_not(line_mask))

    # Fill holes in the interior regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    filled = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} raw contours from combined mask")

    # Filter by area and text overlap
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Check text overlap
        contour_mask = np.zeros_like(text_mask)
        cv2.fillPoly(contour_mask, [contour], 255)
        overlap = cv2.bitwise_and(contour_mask, text_mask)
        overlap_ratio = np.sum(overlap > 0) / area if area > 0 else 1.0

        if overlap_ratio > max_text_overlap:
            continue

        # Filter by aspect ratio (buildings aren't super elongated)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        if aspect > 8:
            continue

        filtered.append(contour)

    print(f"After filtering: {len(filtered)} building contours")
    return filtered


def alternative_flood_fill_approach(
    image: np.ndarray,
    line_mask: np.ndarray,
    text_mask: np.ndarray,
    min_area: float = 500,
) -> list:
    """Alternative: Use flood fill to find enclosed regions.

    This approach treats dark lines as boundaries and floods from seed points
    to find enclosed building regions.
    """
    # Dilate lines to make them solid boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = cv2.dilate(line_mask, kernel, iterations=2)

    # Invert so enclosed regions are white, boundaries are black
    regions = cv2.bitwise_not(boundary)

    # Remove text regions
    regions = cv2.bitwise_and(regions, cv2.bitwise_not(text_mask))

    # Find contours of enclosed regions
    contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Flood fill approach found {len(contours)} raw regions")

    # Filter
    filtered = filter_contours_by_area(contours, min_area=min_area, max_area=None)
    print(f"After area filter: {len(filtered)} regions")

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

    # Initialize EasyOCR
    print("\nInitializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)

    # Step 1: Detect text mask
    text_mask = detect_text_mask(input_image, reader, dilation=15)
    cv2.imwrite(str(output_dir / "text_mask.png"), text_mask)
    print(f"Text mask covers {np.sum(text_mask > 0) / (w*h) * 100:.1f}% of image")

    # Step 2: Extract building color regions
    print("\nExtracting building-colored regions...")
    color_mask = extract_building_color_mask(image)
    cv2.imwrite(str(output_dir / "color_mask.png"), color_mask)
    print(f"Color mask covers {np.sum(color_mask > 0) / (w*h) * 100:.1f}% of image")

    # Step 3: Extract dark lines
    print("\nExtracting dark line work...")
    line_mask = extract_dark_lines(image)
    cv2.imwrite(str(output_dir / "line_mask.png"), line_mask)

    # Step 4: Find building contours (Method 1: Color + Lines)
    print("\nMethod 1: Color regions bounded by lines...")
    contours_method1 = find_building_contours(
        color_mask, line_mask, text_mask,
        min_area=800, max_text_overlap=0.4
    )

    # Step 5: Alternative method - flood fill within boundaries
    print("\nMethod 2: Flood fill within line boundaries...")
    contours_method2 = alternative_flood_fill_approach(
        image, line_mask, text_mask, min_area=500
    )

    # Simplify contours
    simplified_m1 = [simplify_contour(c, 0.003) for c in contours_method1]
    simplified_m2 = [simplify_contour(c, 0.003) for c in contours_method2]

    # Visualizations
    viz_m1 = overlay_contours(image, simplified_m1, fill=True, fill_alpha=0.5)
    cv2.imwrite(str(output_dir / "contours_method1.png"), viz_m1)

    viz_m2 = overlay_contours(image, simplified_m2, fill=True, fill_alpha=0.5)
    cv2.imwrite(str(output_dir / "contours_method2.png"), viz_m2)

    # Save SVGs
    save_svg(simplified_m1, w, h, output_dir / "buildings_method1.svg")
    save_svg(simplified_m2, w, h, output_dir / "buildings_method2.svg")

    # Combined visualization
    text_viz = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
    color_viz = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    save_comparison(
        [image, text_viz, color_viz, viz_m1],
        ["Original", "Text Mask", "Color Mask", "Buildings (M1)"],
        output_dir / "comparison.png",
        max_width=2400,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Hybrid Approach Results")
    print("=" * 60)
    print(f"\nMethod 1 (Color + Line boundaries):")
    print(f"  Building contours: {len(simplified_m1)}")
    print(f"  Total polygon points: {sum(len(c) for c in simplified_m1)}")

    print(f"\nMethod 2 (Flood fill enclosed regions):")
    print(f"  Building contours: {len(simplified_m2)}")
    print(f"  Total polygon points: {sum(len(c) for c in simplified_m2)}")

    print(f"\nOutputs saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
