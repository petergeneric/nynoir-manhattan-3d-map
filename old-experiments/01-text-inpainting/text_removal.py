#!/usr/bin/env python3
"""Experiment 1: Text Detection and Inpainting

Detect text regions using EasyOCR, create masks, and inpaint to remove text
while preserving building boundary lines.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import easyocr
from utils.visualization import save_comparison, draw_text_boxes


def detect_text_regions(image_path: Path, reader: easyocr.Reader) -> list:
    """Detect text regions using EasyOCR.

    Args:
        image_path: Path to input image
        reader: EasyOCR reader instance

    Returns:
        List of detection results with bounding boxes and text
    """
    print(f"Detecting text in {image_path}...")
    results = reader.readtext(str(image_path), detail=1)
    print(f"Found {len(results)} text regions")
    return results


def create_text_mask(
    image_shape: tuple,
    detections: list,
    dilation_kernel_size: int = 5,
    dilation_iterations: int = 2,
) -> np.ndarray:
    """Create a binary mask of text regions.

    Args:
        image_shape: (height, width) of the image
        detections: EasyOCR detection results
        dilation_kernel_size: Kernel size for mask dilation
        dilation_iterations: Number of dilation iterations

    Returns:
        Binary mask where text regions are white (255)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for detection in detections:
        # EasyOCR returns: [bbox, text, confidence]
        bbox = detection[0]  # List of 4 corner points
        text = detection[1]
        confidence = detection[2]

        # Convert bbox to numpy array of points
        points = np.array(bbox, dtype=np.int32)

        # Fill the polygon
        cv2.fillPoly(mask, [points], 255)

    # Dilate mask to capture full text strokes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size)
    )
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    return mask


def inpaint_text_regions(
    image: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int = 3,
    method: str = "ns",
) -> np.ndarray:
    """Inpaint text regions to remove text.

    Args:
        image: Input image (BGR)
        mask: Binary mask of text regions
        inpaint_radius: Radius for inpainting
        method: "ns" for Navier-Stokes, "telea" for Telea

    Returns:
        Inpainted image with text removed
    """
    flags = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA

    print(f"Inpainting with method={method}, radius={inpaint_radius}...")
    result = cv2.inpaint(image, mask, inpaint_radius, flags)

    return result


def visualize_detections(
    image: np.ndarray,
    detections: list,
) -> np.ndarray:
    """Create visualization of detected text regions.

    Args:
        image: Input image
        detections: EasyOCR detection results

    Returns:
        Image with text boxes drawn
    """
    result = image.copy()

    for detection in detections:
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]

        points = np.array(bbox, dtype=np.int32)

        # Draw polygon
        cv2.polylines(result, [points], True, (0, 255, 0), 2)

        # Add text label
        x, y = int(points[0][0]), int(points[0][1]) - 5
        label = f"{text[:20]}... ({confidence:.2f})" if len(text) > 20 else f"{text} ({confidence:.2f})"
        cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return result


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
    print("Initializing EasyOCR (this may take a moment on first run)...")
    reader = easyocr.Reader(['en'], gpu=True)

    # Detect text
    detections = detect_text_regions(input_image, reader)

    # Visualize detections
    detection_viz = visualize_detections(image, detections)
    cv2.imwrite(str(output_dir / "detected_text.png"), detection_viz)
    print(f"Saved: {output_dir / 'detected_text.png'}")

    # Create text mask
    print("Creating text mask...")
    text_mask = create_text_mask(
        image.shape,
        detections,
        dilation_kernel_size=7,  # Slightly larger to capture full strokes
        dilation_iterations=2,
    )
    cv2.imwrite(str(output_dir / "text_mask.png"), text_mask)
    print(f"Saved: {output_dir / 'text_mask.png'}")

    # Inpaint to remove text
    # Try both methods
    cleaned_ns = inpaint_text_regions(image, text_mask, inpaint_radius=5, method="ns")
    cv2.imwrite(str(output_dir / "cleaned_image_ns.png"), cleaned_ns)
    print(f"Saved: {output_dir / 'cleaned_image_ns.png'}")

    cleaned_telea = inpaint_text_regions(image, text_mask, inpaint_radius=5, method="telea")
    cv2.imwrite(str(output_dir / "cleaned_image_telea.png"), cleaned_telea)
    print(f"Saved: {output_dir / 'cleaned_image_telea.png'}")

    # Create comparison
    save_comparison(
        [image, detection_viz, text_mask, cleaned_ns],
        ["Original", "Detected Text", "Text Mask", "Cleaned (NS)"],
        output_dir / "comparison.png",
        max_width=2400,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Text Detection + Inpainting")
    print("=" * 50)
    print(f"Text regions detected: {len(detections)}")
    print(f"Mask pixels: {np.sum(text_mask > 0)}")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - detected_text.png: Visualization of detected text")
    print("  - text_mask.png: Binary mask of text regions")
    print("  - cleaned_image_ns.png: Text removed (Navier-Stokes)")
    print("  - cleaned_image_telea.png: Text removed (Telea)")
    print("  - comparison.png: Side-by-side comparison")

    return 0


if __name__ == "__main__":
    exit(main())
