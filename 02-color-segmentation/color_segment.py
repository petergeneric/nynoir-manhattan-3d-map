#!/usr/bin/env python3
"""Experiment 2: Color-Based Building Segmentation

Extract building regions using HSV color detection for pink/salmon building fills,
then extract contours and generate SVG polygons.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from utils.svg_output import save_svg, filter_contours_by_area, simplify_contour
from utils.visualization import overlay_contours, save_comparison, create_debug_visualization


def extract_pink_regions(
    image: np.ndarray,
    h_range: tuple = (0, 25),
    s_range: tuple = (30, 255),
    v_range: tuple = (100, 255),
) -> np.ndarray:
    """Extract pink/salmon colored building regions using HSV thresholds.

    Args:
        image: Input image (BGR)
        h_range: Hue range (0-180 in OpenCV)
        s_range: Saturation range (0-255)
        v_range: Value range (0-255)

    Returns:
        Binary mask of pink regions
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for pink/salmon colors
    lower = np.array([h_range[0], s_range[0], v_range[0]])
    upper = np.array([h_range[1], s_range[1], v_range[1]])
    mask = cv2.inRange(hsv, lower, upper)

    return mask


def clean_mask(
    mask: np.ndarray,
    close_kernel_size: int = 5,
    close_iterations: int = 2,
    open_kernel_size: int = 3,
    open_iterations: int = 1,
) -> np.ndarray:
    """Clean up the mask using morphological operations.

    Args:
        mask: Binary mask
        close_kernel_size: Kernel size for closing (fill holes)
        close_iterations: Number of closing iterations
        open_kernel_size: Kernel size for opening (remove noise)
        open_iterations: Number of opening iterations

    Returns:
        Cleaned binary mask
    """
    # Closing: Fill small holes in buildings
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    # Opening: Remove small noise
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (open_kernel_size, open_kernel_size)
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)

    return opened


def extract_building_contours(
    mask: np.ndarray,
    min_area: float = 500,
    simplify_epsilon: float = 0.005,
) -> list:
    """Extract and simplify building contours from mask.

    Args:
        mask: Binary mask of building regions
        min_area: Minimum contour area to keep
        simplify_epsilon: Epsilon factor for contour simplification

    Returns:
        List of simplified contours
    """
    # Find contours
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    print(f"Found {len(contours)} raw contours")

    # Filter by area
    filtered = filter_contours_by_area(contours, min_area=min_area)
    print(f"After area filter (>{min_area}): {len(filtered)} contours")

    # Simplify contours
    simplified = [simplify_contour(c, simplify_epsilon) for c in filtered]

    return simplified


def analyze_hsv_distribution(image: np.ndarray, output_dir: Path):
    """Analyze HSV distribution to help tune color thresholds.

    Args:
        image: Input image (BGR)
        output_dir: Directory to save analysis
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Create visualizations
    h_viz = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
    s_viz = cv2.applyColorMap(s, cv2.COLORMAP_VIRIDIS)
    v_viz = cv2.applyColorMap(v, cv2.COLORMAP_VIRIDIS)

    cv2.imwrite(str(output_dir / "hsv_hue.png"), h_viz)
    cv2.imwrite(str(output_dir / "hsv_saturation.png"), s_viz)
    cv2.imwrite(str(output_dir / "hsv_value.png"), v_viz)

    # Print statistics
    print("\nHSV Statistics:")
    print(f"  Hue: min={h.min()}, max={h.max()}, mean={h.mean():.1f}")
    print(f"  Saturation: min={s.min()}, max={s.max()}, mean={s.mean():.1f}")
    print(f"  Value: min={v.min()}, max={v.max()}, mean={v.mean():.1f}")


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

    # Analyze HSV distribution
    print("\nAnalyzing HSV distribution...")
    analyze_hsv_distribution(image, output_dir)

    # Try multiple threshold configurations
    configs = [
        # (name, h_range, s_range, v_range)
        ("narrow", (0, 15), (40, 255), (120, 255)),
        ("medium", (0, 20), (30, 255), (100, 255)),
        ("wide", (0, 25), (20, 255), (80, 255)),
    ]

    best_contours = None
    best_name = None

    for name, h_range, s_range, v_range in configs:
        print(f"\nTrying config '{name}': H={h_range}, S={s_range}, V={v_range}")

        # Extract pink regions
        mask = extract_pink_regions(image, h_range, s_range, v_range)
        cv2.imwrite(str(output_dir / f"mask_{name}.png"), mask)

        # Clean mask
        cleaned_mask = clean_mask(mask)
        cv2.imwrite(str(output_dir / f"mask_{name}_cleaned.png"), cleaned_mask)

        # Extract contours
        contours = extract_building_contours(cleaned_mask, min_area=300)

        print(f"  Extracted {len(contours)} building contours")

        if best_contours is None or len(contours) > len(best_contours):
            best_contours = contours
            best_name = name

    print(f"\nBest config: '{best_name}' with {len(best_contours)} contours")

    # Use best result for final outputs
    # Re-run with best config
    if best_name == "narrow":
        h_range, s_range, v_range = (0, 15), (40, 255), (120, 255)
    elif best_name == "medium":
        h_range, s_range, v_range = (0, 20), (30, 255), (100, 255)
    else:
        h_range, s_range, v_range = (0, 25), (20, 255), (80, 255)

    final_mask = extract_pink_regions(image, h_range, s_range, v_range)
    final_cleaned = clean_mask(final_mask)
    final_contours = extract_building_contours(final_cleaned, min_area=300)

    # Save final mask
    cv2.imwrite(str(output_dir / "hsv_mask.png"), final_mask)
    cv2.imwrite(str(output_dir / "building_regions.png"), final_cleaned)

    # Create contour visualization
    contour_viz = overlay_contours(image, final_contours, fill=True, fill_alpha=0.4)
    cv2.imwrite(str(output_dir / "contours.png"), contour_viz)

    # Save SVG
    save_svg(final_contours, w, h, output_dir / "buildings.svg")

    # Create comparison
    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(final_cleaned, cv2.COLOR_GRAY2BGR)
    save_comparison(
        [image, mask_bgr, cleaned_bgr, contour_viz],
        ["Original", "HSV Mask", "Cleaned", "Contours"],
        output_dir / "comparison.png",
        max_width=2400,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Color-Based Building Segmentation")
    print("=" * 50)
    print(f"Best color config: {best_name}")
    print(f"Building contours extracted: {len(final_contours)}")
    total_polygon_points = sum(len(c) for c in final_contours)
    print(f"Total polygon points: {total_polygon_points}")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - hsv_mask.png: Raw HSV color mask")
    print("  - building_regions.png: Cleaned building regions")
    print("  - contours.png: Contours overlaid on image")
    print("  - buildings.svg: SVG with building polygons")
    print("  - comparison.png: Side-by-side comparison")

    return 0


if __name__ == "__main__":
    exit(main())
