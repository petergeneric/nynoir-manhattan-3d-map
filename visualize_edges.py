#!/usr/bin/env python3
"""
Visualize Edge Detection Outputs

Generates images showing raw edge detection results for visual comparison.
Outputs PNG files showing detected edges overlaid on the source image.
"""

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load image."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def edges_canny(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    """Canny edge detection."""
    return cv2.Canny(gray, low, high)


def edges_sobel(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Sobel edge detection."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / magnitude.max())
    _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    return edges


def edges_scharr(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Scharr edge detection."""
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(scharrx**2 + scharry**2)
    magnitude = np.uint8(255 * magnitude / magnitude.max())
    _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    return edges


def create_edge_overlay(image: np.ndarray, edges: np.ndarray,
                        edge_color: tuple = (0, 255, 0)) -> np.ndarray:
    """Overlay edges on original image in a bright color."""
    result = image.copy()
    # Make edges visible as colored lines
    result[edges > 0] = edge_color
    return result


def create_edge_only(edges: np.ndarray) -> np.ndarray:
    """Create white-on-black edge image."""
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def image_to_base64(image: np.ndarray) -> str:
    """Encode image as base64 PNG."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def generate_comparison_svg(images: list, labels: list,
                           output_path: Path, cols: int = 4) -> None:
    """Generate SVG with multiple edge detection results for comparison.

    Args:
        images: List of (image, label) tuples
        labels: List of labels
        output_path: Path to save SVG
        cols: Number of columns in grid
    """
    if not images:
        return

    # Get dimensions from first image
    h, w = images[0].shape[:2]

    # Calculate grid
    n = len(images)
    rows = (n + cols - 1) // cols

    # Scale factor to keep SVG reasonable size
    scale = 0.25  # 25% of original size
    sw, sh = int(w * scale), int(h * scale)

    # Total SVG dimensions
    total_w = sw * cols
    total_h = (sh + 30) * rows  # +30 for labels

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     xmlns:xlink="http://www.w3.org/1999/xlink"',
        f'     width="{total_w}" height="{total_h}"',
        f'     viewBox="0 0 {total_w} {total_h}">',
        '',
        '  <style>',
        '    text { font-family: monospace; font-size: 14px; fill: white; }',
        '    rect.bg { fill: #333; }',
        '  </style>',
        f'  <rect class="bg" width="{total_w}" height="{total_h}"/>',
        '',
    ]

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x = col * sw
        y = row * (sh + 30)

        # Resize image
        resized = cv2.resize(img, (sw, sh))
        b64 = image_to_base64(resized)

        svg_parts.append(f'  <!-- {label} -->')
        svg_parts.append(f'  <image xlink:href="data:image/png;base64,{b64}"')
        svg_parts.append(f'         x="{x}" y="{y + 20}" width="{sw}" height="{sh}"/>')
        svg_parts.append(f'  <text x="{x + 5}" y="{y + 15}">{label}</text>')
        svg_parts.append('')

    svg_parts.append('</svg>')

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize edge detection outputs')
    parser.add_argument('-i', '--input', type=Path, default=Path('p2crop.png'))
    parser.add_argument('-o', '--output', type=str, default='edges')
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    image = load_image(args.input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Also try with slight blur
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    print("Generating edge detection comparisons...")

    # ==========================================================================
    # 1. Canny threshold sweep (raw edges, white on black)
    # ==========================================================================
    print("\n[Canny Threshold Sweep]")
    canny_images = []
    canny_labels = []
    for low, high in [(10, 30), (20, 60), (30, 90), (40, 120), (50, 150),
                       (60, 180), (75, 200), (100, 250), (125, 300)]:
        edges = edges_canny(gray, low, high)
        canny_images.append(create_edge_only(edges))
        canny_labels.append(f"Canny {low}/{high}")
        print(f"  Canny {low}/{high}: {np.count_nonzero(edges)} edge pixels")

    generate_comparison_svg(canny_images, canny_labels,
                           Path(f"{args.output}_canny_sweep.svg"), cols=3)

    # ==========================================================================
    # 2. Canny with blur preprocessing
    # ==========================================================================
    print("\n[Canny + Blur Threshold Sweep]")
    canny_blur_images = []
    canny_blur_labels = []
    for low, high in [(10, 30), (20, 60), (30, 90), (40, 120), (50, 150),
                       (60, 180), (75, 200), (100, 250), (125, 300)]:
        edges = edges_canny(gray_blur, low, high)
        canny_blur_images.append(create_edge_only(edges))
        canny_blur_labels.append(f"Canny+Blur {low}/{high}")
        print(f"  Canny+Blur {low}/{high}: {np.count_nonzero(edges)} edge pixels")

    generate_comparison_svg(canny_blur_images, canny_blur_labels,
                           Path(f"{args.output}_canny_blur_sweep.svg"), cols=3)

    # ==========================================================================
    # 3. Sobel threshold sweep
    # ==========================================================================
    print("\n[Sobel Threshold Sweep]")
    sobel_images = []
    sobel_labels = []
    for threshold in [10, 15, 20, 25, 30, 40, 50, 60, 75]:
        edges = edges_sobel(gray, threshold)
        sobel_images.append(create_edge_only(edges))
        sobel_labels.append(f"Sobel t={threshold}")
        print(f"  Sobel t={threshold}: {np.count_nonzero(edges)} edge pixels")

    generate_comparison_svg(sobel_images, sobel_labels,
                           Path(f"{args.output}_sobel_sweep.svg"), cols=3)

    # ==========================================================================
    # 4. Scharr threshold sweep
    # ==========================================================================
    print("\n[Scharr Threshold Sweep]")
    scharr_images = []
    scharr_labels = []
    for threshold in [10, 15, 20, 25, 30, 40, 50, 60, 75]:
        edges = edges_scharr(gray, threshold)
        scharr_images.append(create_edge_only(edges))
        scharr_labels.append(f"Scharr t={threshold}")
        print(f"  Scharr t={threshold}: {np.count_nonzero(edges)} edge pixels")

    generate_comparison_svg(scharr_images, scharr_labels,
                           Path(f"{args.output}_scharr_sweep.svg"), cols=3)

    # ==========================================================================
    # 5. Edge overlay comparison (best candidates on original image)
    # ==========================================================================
    print("\n[Edge Overlays on Original]")
    overlay_images = []
    overlay_labels = []

    candidates = [
        ("Canny 20/60", edges_canny(gray, 20, 60)),
        ("Canny 30/90", edges_canny(gray, 30, 90)),
        ("Canny 50/150", edges_canny(gray, 50, 150)),
        ("Canny 75/200", edges_canny(gray, 75, 200)),
        ("Sobel t=20", edges_sobel(gray, 20)),
        ("Sobel t=30", edges_sobel(gray, 30)),
        ("Sobel t=50", edges_sobel(gray, 50)),
        ("Scharr t=30", edges_scharr(gray, 30)),
        ("Scharr t=50", edges_scharr(gray, 50)),
    ]

    for label, edges in candidates:
        overlay = create_edge_overlay(image, edges, (0, 255, 0))
        overlay_images.append(overlay)
        overlay_labels.append(label)

    generate_comparison_svg(overlay_images, overlay_labels,
                           Path(f"{args.output}_overlays.svg"), cols=3)

    # ==========================================================================
    # 6. Individual full-resolution edge outputs (PNG)
    # ==========================================================================
    print("\n[Full Resolution PNGs]")
    for label, edges in candidates:
        safe_name = label.lower().replace(" ", "_").replace("/", "_").replace("=", "")
        # Save raw edges
        cv2.imwrite(f"{args.output}_{safe_name}.png", edges)
        print(f"  Saved: {args.output}_{safe_name}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
