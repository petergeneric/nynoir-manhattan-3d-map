#!/usr/bin/env python3
"""Segment Anything Model tool - segments images and outputs PNG/SVG with polygon overlays"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Configuration
MODEL_TYPE = "vit_h"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"


def download_checkpoint(checkpoint_path: Path) -> None:
    """Download SAM checkpoint if not present."""
    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        return

    print(f"Downloading SAM checkpoint ({MODEL_TYPE})...")
    print(f"This is a 2.56GB download and may take a few minutes.")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint_path, reporthook=report_progress)
    print("\nDownload complete!")


def get_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Note: MPS has float64 compatibility issues with SAM, using CPU instead
    else:
        return torch.device("cpu")


def generate_svg(sam_result: list, width: int, height: int, output_path: Path) -> None:
    """Generate SVG file with polygon outlines from SAM masks."""
    # Generate consistent colors for each segment
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
              for _ in range(len(sam_result))]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    ]

    for idx, mask_data in enumerate(sam_result):
        mask = mask_data["segmentation"].astype(np.uint8)

        # Find contours for this mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        r, g, b = colors[idx]
        fill_color = f"rgba({r},{g},{b},0.5)"
        stroke_color = f"rgb({r},{g},{b})"

        for contour in contours:
            if len(contour) < 3:
                continue

            # Convert contour to SVG path points
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue

            points_str = " ".join(f"{p[0]},{p[1]}" for p in points)
            svg_parts.append(
                f'  <polygon points="{points_str}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="1"/>'
            )

    svg_parts.append("</svg>")

    with open(output_path, "w") as f:
        f.write("\n".join(svg_parts))

    print(f"Saved SVG: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment images using SAM")
    parser.add_argument("input", nargs="?", default="example.png", help="Input image path")
    parser.add_argument("-o", "--output", help="Output PNG path (default: <input>_segmented.png)")
    parser.add_argument("-s", "--svg", help="Output SVG path (default: <input>_segmented.svg)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / CHECKPOINT_NAME
    input_path = Path(args.input)

    # Generate output paths
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_segmented.png"

    if args.svg:
        svg_path = Path(args.svg)
    else:
        svg_path = input_path.parent / f"{input_path.stem}_segmented.svg"

    # Check input image exists
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return 1

    # Download checkpoint if needed
    download_checkpoint(checkpoint_path)

    # Detect device
    device = get_device()
    print(f"Using device: {device}")

    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint_path))
    sam.to(device=device)

    # Load image
    print(f"Loading image: {input_path}")
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        print(f"Error: Could not read image: {input_path}")
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Create mask generator with tuned parameters
    print("Generating masks...")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # Generate masks
    sam_result = mask_generator.generate(image_rgb)
    print(f"Found {len(sam_result)} segments")

    # Convert to supervision detections and annotate with 50% opacity
    detections = sv.Detections.from_sam(sam_result=sam_result)
    mask_annotator = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.5,  # 50% translucency
    )

    # Annotate image
    annotated_image = mask_annotator.annotate(
        scene=image_bgr.copy(),
        detections=detections,
    )

    # Save PNG output
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Saved segmented image: {output_path}")

    # Save SVG output
    height, width = image_bgr.shape[:2]
    generate_svg(sam_result, width, height, svg_path)

    return 0


if __name__ == "__main__":
    exit(main())
