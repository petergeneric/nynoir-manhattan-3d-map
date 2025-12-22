#!/usr/bin/env python3
"""Segment Anything Model tool - segments images and outputs PNG/SVG with polygon overlays.

Supports recursive segmentation for atlas plates:
1. First pass: Identify city blocks from full plates
2. Second pass: Run detailed segmentation on each extracted block
"""

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Model Configuration
MODEL_TYPE = "vit_h"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"

# Segmentation Configuration
FIRST_PASS_CONFIG = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 22500,  # 150x150 pixels minimum for city blocks
}

SECOND_PASS_CONFIG = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,  # Finer detail for individual blocks
}


# Data Structures
@dataclass
class BoundingBox:
    """Bounding box for a detected region."""
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_slice(self) -> Tuple[slice, slice]:
        """Return numpy slice for extracting region (row_slice, col_slice)."""
        return (slice(self.y, self.y + self.height),
                slice(self.x, self.x + self.width))


@dataclass
class Block:
    """Represents a city block detected in first-pass segmentation."""
    id: str                          # "0001", "0002", etc.
    bbox: BoundingBox
    contours: List[np.ndarray]
    color: Tuple[int, int, int]


@dataclass
class PlateSegmentation:
    """Results from first-pass (plate-level) segmentation."""
    volume: str
    plate_id: str
    image_width: int
    image_height: int
    blocks: List[Block]


@dataclass
class BlockSegmentation:
    """Results from second-pass (block-level) segmentation."""
    block_id: str
    parent_bbox: BoundingBox
    local_width: int
    local_height: int
    contours: List[Tuple[np.ndarray, Tuple[int, int, int]]]  # (contour, color)


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


def get_device(force_mps: bool = False) -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif force_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    # Note: MPS has float64 compatibility issues with SAM, using CPU instead
    else:
        return torch.device("cpu")


def load_image(image_path: Path) -> np.ndarray:
    """Load image from path, supporting JP2, PNG, JPG formats.

    Returns:
        numpy array in BGR format (OpenCV convention)
    """
    # First try OpenCV (works for many formats)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is not None:
        return image

    # Fallback for JP2: use Pillow
    try:
        from PIL import Image
        pil_image = Image.open(image_path)
        rgb_array = np.array(pil_image)
        if len(rgb_array.shape) == 2:  # Grayscale
            return cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
        elif rgb_array.shape[2] == 4:  # RGBA
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise RuntimeError(f"Could not load image {image_path}: {e}")


def extract_region(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Extract a rectangular region from an image."""
    y_slice, x_slice = bbox.to_slice()
    return image[y_slice, x_slice].copy()


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


# First-Pass Segmentation (City Blocks)
def create_mask_generator(sam_model, config: dict) -> SamAutomaticMaskGenerator:
    """Create a mask generator with specified configuration."""
    return SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=config["points_per_side"],
        pred_iou_thresh=config["pred_iou_thresh"],
        stability_score_thresh=config["stability_score_thresh"],
        crop_n_layers=config["crop_n_layers"],
        crop_n_points_downscale_factor=config["crop_n_points_downscale_factor"],
        min_mask_region_area=config["min_mask_region_area"],
    )


def filter_blocks_by_size(
    sam_result: list,
    image_width: int,
    image_height: int,
    min_width: int = 150,
    min_height: int = 150,
    max_area_ratio: float = 0.7
) -> list:
    """Filter SAM results to only include masks meeting size requirements.

    Args:
        sam_result: SAM mask generator output
        image_width: Width of the source image
        image_height: Height of the source image
        min_width: Minimum bounding box width
        min_height: Minimum bounding box height
        max_area_ratio: Maximum mask area as ratio of image (0.7 = 70%)
    """
    image_area = image_width * image_height
    max_area = image_area * max_area_ratio

    filtered = []
    for mask_data in sam_result:
        bbox = mask_data["bbox"]  # SAM provides [x, y, w, h]
        mask_area = mask_data["area"]

        # Check minimum size
        if bbox[2] < min_width or bbox[3] < min_height:
            continue

        # Check maximum area (skip if >= 70% of image)
        if mask_area >= max_area:
            continue

        filtered.append(mask_data)
    return filtered


def segment_plate(
    image_rgb: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    min_block_width: int = 150,
    min_block_height: int = 150,
    max_area_ratio: float = 0.7
) -> List[Block]:
    """First-pass segmentation to identify city blocks."""
    height, width = image_rgb.shape[:2]

    # Generate masks
    sam_result = mask_generator.generate(image_rgb)

    # Filter by size (min dimensions and max area ratio)
    filtered = filter_blocks_by_size(
        sam_result, width, height,
        min_block_width, min_block_height, max_area_ratio
    )

    # Sort by area (largest first) for consistent ordering
    filtered.sort(key=lambda x: x["area"], reverse=True)

    # Convert to Block objects
    np.random.seed(42)  # Consistent colors
    blocks = []
    for idx, mask_data in enumerate(filtered):
        bbox_raw = mask_data["bbox"]  # [x, y, w, h]
        bbox = BoundingBox(
            x=int(bbox_raw[0]),
            y=int(bbox_raw[1]),
            width=int(bbox_raw[2]),
            height=int(bbox_raw[3])
        )

        mask = mask_data["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )

        blocks.append(Block(
            id=f"{idx + 1:04d}",  # "0001", "0002", etc.
            bbox=bbox,
            contours=list(contours),
            color=color
        ))

    return blocks


# Second-Pass Segmentation (Block Details)
def segment_block(
    block_image_rgb: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    block_id: str,
    parent_bbox: BoundingBox
) -> BlockSegmentation:
    """Second-pass segmentation on an individual block region."""
    # Generate masks for this block region
    sam_result = mask_generator.generate(block_image_rgb)

    # Extract contours with colors
    np.random.seed(int(block_id))  # Consistent per-block colors
    contours_with_colors = []

    for mask_data in sam_result:
        mask = mask_data["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )

        for contour in contours:
            contours_with_colors.append((contour, color))

    height, width = block_image_rgb.shape[:2]

    return BlockSegmentation(
        block_id=block_id,
        parent_bbox=parent_bbox,
        local_width=width,
        local_height=height,
        contours=contours_with_colors
    )


# SVG Generators
def generate_plate_svg(plate: PlateSegmentation, output_path: Path) -> None:
    """Generate plate-level SVG showing city block outlines with labels."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{plate.image_width}" height="{plate.image_height}" '
        f'viewBox="0 0 {plate.image_width} {plate.image_height}">',
        '  <!-- Plate-level city block segmentation -->',
        '  <style>',
        '    .block-label { font-family: sans-serif; font-size: 14px; font-weight: bold; }',
        '  </style>',
    ]

    for block in plate.blocks:
        r, g, b = block.color
        fill_color = f"rgba({r},{g},{b},0.3)"  # Lighter fill for overview
        stroke_color = f"rgb({r},{g},{b})"
        filename = f"b-{block.id}.svg"

        for contour in block.contours:
            if len(contour) < 3:
                continue
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue

            points_str = " ".join(f"{p[0]},{p[1]}" for p in points)
            svg_parts.append(
                f'  <polygon points="{points_str}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" '
                f'data-block-id="{block.id}"/>'
            )

        # Add label at center of bounding box
        label_x = block.bbox.x + block.bbox.width // 2
        label_y = block.bbox.y + block.bbox.height // 2
        svg_parts.append(
            f'  <text x="{label_x}" y="{label_y}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'class="block-label" fill="{stroke_color}" stroke="white" stroke-width="3" paint-order="stroke">'
            f'{filename}</text>'
        )

    svg_parts.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"Saved plate SVG: {output_path}")


def generate_block_svg(block_seg: BlockSegmentation, output_path: Path) -> None:
    """Generate detailed SVG for an individual block."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{block_seg.local_width}" height="{block_seg.local_height}" '
        f'viewBox="0 0 {block_seg.local_width} {block_seg.local_height}" '
        f'data-block-id="{block_seg.block_id}" '
        f'data-plate-x="{block_seg.parent_bbox.x}" '
        f'data-plate-y="{block_seg.parent_bbox.y}">',
        f'  <!-- Block {block_seg.block_id} detailed segmentation -->',
    ]

    for contour, (r, g, b) in block_seg.contours:
        if len(contour) < 3:
            continue
        points = contour.squeeze()
        if len(points.shape) == 1:
            continue

        points_str = " ".join(f"{p[0]},{p[1]}" for p in points)
        svg_parts.append(
            f'  <polygon points="{points_str}" '
            f'fill="rgba({r},{g},{b},0.5)" stroke="rgb({r},{g},{b})" stroke-width="1"/>'
        )

    svg_parts.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"  Saved block SVG: {output_path}")


def generate_combined_svg(plate: PlateSegmentation, output_path: Path) -> None:
    """Generate combined SVG that references individual block SVGs."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{plate.image_width}" height="{plate.image_height}" '
        f'viewBox="0 0 {plate.image_width} {plate.image_height}">',
        '  <!-- Combined segmentation view -->',
        '  <!-- References individual block SVG files -->',
        '',
        '  <!-- Block outline layer (from plate.svg) -->',
        '  <g id="block-outlines" opacity="0.3">',
    ]

    # Add block outlines as a base layer
    for block in plate.blocks:
        r, g, b = block.color
        stroke_color = f"rgb({r},{g},{b})"

        for contour in block.contours:
            if len(contour) < 3:
                continue
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue

            points_str = " ".join(f"{p[0]},{p[1]}" for p in points)
            svg_parts.append(
                f'    <polygon points="{points_str}" '
                f'fill="none" stroke="{stroke_color}" stroke-width="3" '
                f'stroke-dasharray="10,5"/>'
            )

    svg_parts.append('  </g>')
    svg_parts.append('')
    svg_parts.append('  <!-- Detailed block layers -->')

    # Reference each block SVG
    for block in plate.blocks:
        block_svg_name = f"b-{block.id}.svg"
        svg_parts.append(
            f'  <g id="block-{block.id}" '
            f'transform="translate({block.bbox.x},{block.bbox.y})">'
        )
        svg_parts.append(
            f'    <!-- Block {block.id}: {block.bbox.width}x{block.bbox.height} '
            f'at ({block.bbox.x},{block.bbox.y}) -->'
        )
        svg_parts.append(
            f'    <image href="{block_svg_name}" '
            f'width="{block.bbox.width}" height="{block.bbox.height}"/>'
        )
        svg_parts.append('  </g>')

    svg_parts.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"Saved combined SVG: {output_path}")


# Output Management
def parse_source_file(source_path: Path) -> str:
    """Read volume identifier from SOURCE.txt."""
    if source_path.exists():
        return source_path.read_text().strip()
    return "unknown"


def get_output_structure(
    input_path: Path,
    output_base: Path
) -> Tuple[str, str, Path]:
    """Determine output directory structure from input file.

    Returns:
        Tuple of (volume_id, plate_id, output_directory)
    """
    # Get volume from SOURCE.txt in same directory
    source_file = input_path.parent / "SOURCE.txt"
    volume = parse_source_file(source_file)

    # Get plate ID from filename (without extension)
    plate_id = input_path.stem  # e.g., "p37" from "p37.jp2"

    output_dir = output_base / volume / plate_id

    return volume, plate_id, output_dir


# Main Recursive Workflow
def process_plate_recursive(
    input_path: Path,
    output_base: Path,
    sam_model,
    min_block_width: int = 150,
    min_block_height: int = 150,
    max_area_ratio: float = 0.7
) -> None:
    """Complete recursive segmentation workflow for a plate."""
    # Determine output structure
    volume, plate_id, output_dir = get_output_structure(input_path, output_base)

    print(f"Processing plate: {volume}/{plate_id}")
    print(f"Output directory: {output_dir}")

    # Load full plate image
    print("Loading plate image...")
    image_bgr = load_image(input_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]
    print(f"Image size: {width}x{height}")

    # First pass: Identify city blocks
    print("First pass: Identifying city blocks...")
    first_pass_generator = create_mask_generator(sam_model, FIRST_PASS_CONFIG)
    blocks = segment_plate(
        image_rgb,
        first_pass_generator,
        min_block_width,
        min_block_height,
        max_area_ratio
    )
    print(f"Found {len(blocks)} city blocks (filtered by min {min_block_width}x{min_block_height}px, max {max_area_ratio*100:.0f}% area)")

    # Create plate segmentation result
    plate = PlateSegmentation(
        volume=volume,
        plate_id=plate_id,
        image_width=width,
        image_height=height,
        blocks=blocks
    )

    # Generate plate-level SVG
    plate_svg_path = output_dir / "plate.svg"
    generate_plate_svg(plate, plate_svg_path)

    # Second pass: Process each block
    total_blocks = len(blocks)
    print(f"Second pass: Processing {total_blocks} individual blocks...")
    second_pass_generator = create_mask_generator(sam_model, SECOND_PASS_CONFIG)

    for idx, block in enumerate(blocks, 1):
        print(f"  [{idx}/{total_blocks}] Processing block {block.id} ({block.bbox.width}x{block.bbox.height})...")

        # Extract block region from original image
        block_image_bgr = extract_region(image_bgr, block.bbox)
        block_image_rgb = cv2.cvtColor(block_image_bgr, cv2.COLOR_BGR2RGB)

        # Segment this block
        block_seg = segment_block(
            block_image_rgb,
            second_pass_generator,
            block.id,
            block.bbox
        )

        # Generate block SVG
        block_svg_path = output_dir / f"b-{block.id}.svg"
        generate_block_svg(block_seg, block_svg_path)

        print(f"    Found {len(block_seg.contours)} segments in block {block.id}")

    # Generate combined SVG
    print("Generating combined segmentation.svg...")
    segmentation_svg_path = output_dir / "segmentation.svg"
    generate_combined_svg(plate, segmentation_svg_path)

    print(f"Complete! Output files in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment images using SAM. Default mode is recursive segmentation for atlas plates."
    )
    parser.add_argument("input", nargs="?", default="example.png", help="Input image path")
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Base output directory for recursive mode (default: output)"
    )
    parser.add_argument(
        "--min-block-width",
        type=int,
        default=150,
        help="Minimum block width in pixels for first-pass detection (default: 150)"
    )
    parser.add_argument(
        "--min-block-height",
        type=int,
        default=150,
        help="Minimum block height in pixels for first-pass detection (default: 150)"
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.7,
        help="Maximum block area as ratio of image (default: 0.7 = 70%%)"
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Run original single-pass segmentation (outputs PNG + SVG)"
    )
    # Legacy arguments for single-pass mode
    parser.add_argument(
        "--output-png",
        help="[Single-pass only] Output PNG path (default: <input>_segmented.png)"
    )
    parser.add_argument(
        "-s", "--svg",
        help="[Single-pass only] Output SVG path (default: <input>_segmented.svg)"
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Force use of MPS (Apple Silicon GPU) instead of CPU"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / CHECKPOINT_NAME
    input_path = Path(args.input)

    # Check input image exists
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return 1

    # Download checkpoint if needed
    download_checkpoint(checkpoint_path)

    # Detect device
    device = get_device(force_mps=args.mps)
    print(f"Using device: {device}")

    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint_path))
    sam.to(device=device)

    if args.single_pass:
        # Original single-pass mode for comparison
        # Generate output paths
        if args.output_png:
            output_path = Path(args.output_png)
        else:
            output_path = input_path.parent / f"{input_path.stem}_segmented.png"

        if args.svg:
            svg_path = Path(args.svg)
        else:
            svg_path = input_path.parent / f"{input_path.stem}_segmented.svg"

        # Load image
        print(f"Loading image: {input_path}")
        image_bgr = load_image(input_path)
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
    else:
        # Recursive segmentation (default mode)
        output_base = Path(args.output_dir)
        process_plate_recursive(
            input_path,
            output_base,
            sam,
            args.min_block_width,
            args.min_block_height,
            args.max_area_ratio
        )

    return 0


if __name__ == "__main__":
    exit(main())
