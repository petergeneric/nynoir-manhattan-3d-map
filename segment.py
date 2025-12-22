#!/usr/bin/env python3
"""Segment Anything Model tool - segments images and outputs PNG/SVG with polygon overlays.

Supports recursive segmentation for atlas plates:
1. First pass: Identify city blocks from full plates
2. Second pass: Run detailed segmentation on each extracted block
"""

import argparse
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

# Model Configuration
MODEL_TYPE = "vit_h"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"

# SVG Metadata Configuration
ATLAS_NAMESPACE = "http://example.com/atlas"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"

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

    def expand(self, factor: float) -> 'BoundingBox':
        """Return a new BoundingBox expanded by the given factor.

        Args:
            factor: Expansion factor (e.g., 0.1 for 10% expansion on each edge)

        Returns:
            New BoundingBox expanded by factor on all sides
        """
        expand_x = int(self.width * factor)
        expand_y = int(self.height * factor)
        return BoundingBox(
            x=self.x - expand_x,
            y=self.y - expand_y,
            width=self.width + 2 * expand_x,
            height=self.height + 2 * expand_y
        )


@dataclass
class Block:
    """Represents a city block detected in first-pass segmentation."""
    id: str                          # "0001", "0002", etc.
    bbox: BoundingBox
    contours: List[np.ndarray]
    color: Tuple[int, int, int]
    name: Optional[str] = None       # Optional name from data-name attribute

    @property
    def svg_filename(self) -> str:
        """Return the SVG filename for this block."""
        if self.name:
            return f"{self.name}.svg"
        return f"b-{self.id}.svg"

    @property
    def image_filename(self) -> str:
        """Return the image filename for this block."""
        if self.name:
            return f"{self.name}.png"
        return f"b-{self.id}.png"


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


@dataclass
class PlateMetadata:
    """Metadata extracted from plate.svg for Stage 2 processing."""
    jp2_path: Path
    volume: str
    plate_id: str
    image_width: int
    image_height: int


def bbox_from_polygon_points(points: List[Tuple[int, int]]) -> BoundingBox:
    """Compute bounding box from polygon vertices."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BoundingBox(
        x=min(xs),
        y=min(ys),
        width=max(xs) - min(xs),
        height=max(ys) - min(ys)
    )


def parse_polygon_points(points_str: str) -> List[Tuple[int, int]]:
    """Parse SVG polygon points string into list of (x, y) tuples."""
    points = []
    for pair in points_str.strip().split():
        x, y = pair.split(',')
        points.append((int(float(x)), int(float(y))))
    return points


def parse_plate_svg(svg_path: Path) -> Tuple[PlateMetadata, List[Block]]:
    """Parse plate.svg to extract metadata and block definitions.

    Returns:
        Tuple of (PlateMetadata, List[Block])
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Define namespaces for parsing
    ns = {
        "svg": SVG_NAMESPACE,
        "atlas": ATLAS_NAMESPACE,
    }

    # Extract metadata from atlas:source element
    source_elem = root.find(".//atlas:source", ns)
    if source_elem is None:
        raise ValueError(f"No atlas:source metadata found in {svg_path}")

    jp2_path_elem = source_elem.find("atlas:jp2-path", ns)
    volume_elem = source_elem.find("atlas:volume", ns)
    plate_id_elem = source_elem.find("atlas:plate-id", ns)
    width_elem = source_elem.find("atlas:image-width", ns)
    height_elem = source_elem.find("atlas:image-height", ns)

    if any(e is None for e in [jp2_path_elem, volume_elem, plate_id_elem, width_elem, height_elem]):
        raise ValueError(f"Incomplete metadata in {svg_path}")

    # Handle JP2 path - strip file:// prefix and resolve relative paths
    jp2_path_str = jp2_path_elem.text
    if jp2_path_str.startswith("file://"):
        jp2_path_str = jp2_path_str[7:]  # Strip "file://"

    jp2_path = Path(jp2_path_str)
    if not jp2_path.is_absolute():
        # Resolve relative path from working directory
        jp2_path = Path.cwd() / jp2_path

    metadata = PlateMetadata(
        jp2_path=jp2_path,
        volume=volume_elem.text,
        plate_id=plate_id_elem.text,
        image_width=int(width_elem.text),
        image_height=int(height_elem.text),
    )

    # Extract blocks from polygons with data-block-id attribute
    # Need to handle the default namespace for SVG elements
    blocks = []
    np.random.seed(42)  # Consistent colors

    # Find all polygon elements (handle both namespaced and non-namespaced)
    for polygon in root.iter():
        if polygon.tag.endswith('polygon') and polygon.get('data-block-id'):
            block_id = polygon.get('data-block-id')
            block_name = polygon.get('data-name')  # Optional name attribute
            points_str = polygon.get('points')
            if not points_str:
                continue

            points = parse_polygon_points(points_str)
            if len(points) < 3:
                continue

            bbox = bbox_from_polygon_points(points)

            # Convert points back to contour format for consistency
            contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )

            blocks.append(Block(
                id=block_id,
                bbox=bbox,
                contours=[contour],
                color=color,
                name=block_name
            ))

    # Sort by area (largest first) for consistent ordering
    blocks.sort(key=lambda b: b.bbox.area, reverse=True)

    return metadata, blocks


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
    """Extract a rectangular region from an image.

    Simple extraction using numpy slicing. Does not handle out-of-bounds.
    """
    y_slice, x_slice = bbox.to_slice()
    return image[y_slice, x_slice].copy()


def extract_region_with_bounds(
    image: np.ndarray,
    bbox: BoundingBox,
    expansion_factor: float = 0.1
) -> np.ndarray:
    """Extract a rectangular region with expansion and bounds handling.

    Expands the bounding box by the given factor, handles cases where the
    expanded box extends beyond image boundaries by padding with transparent
    pixels, and returns an RGBA image with transparency outside the expanded box.

    Args:
        image: Source image (BGR or BGRA format)
        bbox: Original bounding box to extract
        expansion_factor: How much to expand the bbox (0.1 = 10% on each edge)

    Returns:
        RGBA image with the extracted region, transparent padding where needed
    """
    img_height, img_width = image.shape[:2]

    # Expand the bounding box
    expanded = bbox.expand(expansion_factor)

    # Calculate the intersection with image bounds
    src_x1 = max(0, expanded.x)
    src_y1 = max(0, expanded.y)
    src_x2 = min(img_width, expanded.x + expanded.width)
    src_y2 = min(img_height, expanded.y + expanded.height)

    # Calculate where to place the extracted region in the output
    dst_x1 = src_x1 - expanded.x
    dst_y1 = src_y1 - expanded.y
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Create output RGBA image (transparent by default)
    output = np.zeros((expanded.height, expanded.width, 4), dtype=np.uint8)

    # Extract the valid region from source image
    if src_x2 > src_x1 and src_y2 > src_y1:
        region = image[src_y1:src_y2, src_x1:src_x2]

        # Convert to RGBA if needed
        if len(region.shape) == 2:
            # Grayscale
            region_rgba = cv2.cvtColor(region, cv2.COLOR_GRAY2RGBA)
        elif region.shape[2] == 3:
            # BGR -> BGRA
            region_rgba = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
        else:
            # Already BGRA
            region_rgba = region.copy()

        # Place the region in the output with full opacity
        region_rgba[:, :, 3] = 255
        output[dst_y1:dst_y2, dst_x1:dst_x2] = region_rgba

    return output


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


def filter_contours_outside_block(
    block_seg: BlockSegmentation,
    block_contour: np.ndarray,
    block_bbox: BoundingBox
) -> BlockSegmentation:
    """Filter out contours that lie entirely outside the block polygon.

    Args:
        block_seg: The segmentation result with contours in local coordinates
        block_contour: The block polygon contour in plate coordinates
        block_bbox: The bounding box used to extract the block region

    Returns:
        A new BlockSegmentation with only contours that intersect the block polygon
    """
    # Convert block contour from plate coordinates to local coordinates
    block_points = block_contour.squeeze()
    if len(block_points.shape) == 1:
        # Single point, can't form a polygon
        return block_seg

    local_block_points = [(p[0] - block_bbox.x, p[1] - block_bbox.y) for p in block_points]

    # Create Shapely polygon for the block boundary
    if len(local_block_points) < 3:
        return block_seg

    try:
        block_polygon = ShapelyPolygon(local_block_points)
        if not block_polygon.is_valid:
            block_polygon = make_valid(block_polygon)
    except Exception:
        # If we can't create a valid polygon, return unfiltered
        return block_seg

    # Filter contours
    filtered_contours = []
    for contour, color in block_seg.contours:
        if len(contour) < 3:
            continue

        points = contour.squeeze()
        if len(points.shape) == 1:
            continue

        try:
            seg_polygon = ShapelyPolygon([(p[0], p[1]) for p in points])
            if not seg_polygon.is_valid:
                seg_polygon = make_valid(seg_polygon)

            # Keep contour if it intersects the block polygon
            if block_polygon.intersects(seg_polygon):
                filtered_contours.append((contour, color))
        except Exception:
            # If we can't process this contour, skip it
            continue

    return BlockSegmentation(
        block_id=block_seg.block_id,
        parent_bbox=block_seg.parent_bbox,
        local_width=block_seg.local_width,
        local_height=block_seg.local_height,
        contours=filtered_contours
    )


# SVG Polygon Culling System
# Add new cull filter functions here. Each filter takes:
#   - seg_polygon: ShapelyPolygon of the segment
#   - block_polygon: ShapelyPolygon of the block boundary
#   - block_bbox: BoundingBox of the block
# Returns True to keep the polygon, False to cull it.

def cull_outside_block(
    seg_polygon: ShapelyPolygon,
    block_polygon: ShapelyPolygon,
    block_bbox: BoundingBox
) -> bool:
    """Keep polygons that intersect the block boundary."""
    return block_polygon.intersects(seg_polygon)


# Registry of all cull filters to apply (add new filters here)
CULL_FILTERS = [
    ("outside_block", cull_outside_block),
]


def parse_svg_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse SVG polygon points string into list of (x, y) tuples."""
    points = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            points.append((float(x), float(y)))
    return points


def format_svg_polygon_points(points: List[Tuple[float, float]]) -> str:
    """Format list of (x, y) tuples back to SVG points string."""
    return " ".join(f"{x},{y}" for x, y in points)


def apply_cull_filters_to_block_svg(
    block_svg_content: str,
    block_contour: np.ndarray,
    block_bbox: BoundingBox
) -> Tuple[str, int, int]:
    """Apply all cull filters to polygons in a block SVG.

    Args:
        block_svg_content: The SVG file content as a string
        block_contour: The block polygon contour in plate coordinates
        block_bbox: The bounding box of the block

    Returns:
        Tuple of (filtered_svg_content, original_count, filtered_count)
    """
    # Convert block contour to local coordinates and create Shapely polygon
    block_points = block_contour.squeeze()
    if len(block_points.shape) == 1 or len(block_points) < 3:
        # Can't form a valid block polygon, return unchanged
        return block_svg_content, 0, 0

    local_block_points = [(p[0] - block_bbox.x, p[1] - block_bbox.y) for p in block_points]

    try:
        block_polygon = ShapelyPolygon(local_block_points)
        if not block_polygon.is_valid:
            block_polygon = make_valid(block_polygon)
    except Exception:
        return block_svg_content, 0, 0

    # Find all polygon elements
    polygon_pattern = re.compile(r'<polygon\s+points="([^"]+)"([^/]*)/>')
    matches = list(polygon_pattern.finditer(block_svg_content))
    original_count = len(matches)

    # Filter polygons
    kept_polygons = []
    for match in matches:
        points_str = match.group(1)
        rest_attrs = match.group(2)

        points = parse_svg_polygon_points(points_str)
        if len(points) < 3:
            continue

        try:
            seg_polygon = ShapelyPolygon(points)
            if not seg_polygon.is_valid:
                seg_polygon = make_valid(seg_polygon)

            # Apply all cull filters - polygon must pass ALL filters to be kept
            keep = True
            for filter_name, filter_func in CULL_FILTERS:
                if not filter_func(seg_polygon, block_polygon, block_bbox):
                    keep = False
                    break

            if keep:
                kept_polygons.append(match.group(0))
        except Exception:
            # If we can't process, skip the polygon
            continue

    filtered_count = len(kept_polygons)

    # Rebuild the SVG with only kept polygons
    # Extract SVG header and closing tag
    header_match = re.match(r'(<svg[^>]*>)', block_svg_content)
    if not header_match:
        return block_svg_content, original_count, filtered_count

    header = header_match.group(1)

    # Extract comment if present
    comment_match = re.search(r'(<!--[^>]*-->)', block_svg_content)
    comment = comment_match.group(1) if comment_match else ""

    # Rebuild SVG
    parts = [header]
    if comment:
        parts.append(f"\n  {comment}")
    for polygon in kept_polygons:
        parts.append(f"\n  {polygon}")
    parts.append("\n</svg>")

    return "".join(parts), original_count, filtered_count


# SVG Generators
def generate_plate_svg(
    plate: PlateSegmentation,
    output_path: Path,
    source_jp2_path: Optional[Path] = None
) -> None:
    """Generate plate-level SVG showing city block outlines with labels.

    Args:
        plate: Plate segmentation results
        output_path: Where to save the SVG
        source_jp2_path: Optional path to source JP2 (for Stage 1 metadata)
    """
    # Build SVG with namespaces for metadata
    svg_parts = [
        f'<svg xmlns="{SVG_NAMESPACE}" '
        f'xmlns:xlink="{XLINK_NAMESPACE}" '
        f'width="{plate.image_width}" height="{plate.image_height}" '
        f'viewBox="0 0 {plate.image_width} {plate.image_height}">',
        '  <!-- Plate-level city block segmentation -->',
    ]

    # Add metadata if source JP2 path provided (Stage 1 output)
    if source_jp2_path is not None:
        abs_jp2_path = source_jp2_path.resolve()
        svg_parts.extend([
            '  <metadata>',
            f'    <atlas:source xmlns:atlas="{ATLAS_NAMESPACE}">',
            f'      <atlas:jp2-path>{abs_jp2_path}</atlas:jp2-path>',
            f'      <atlas:volume>{plate.volume}</atlas:volume>',
            f'      <atlas:plate-id>{plate.plate_id}</atlas:plate-id>',
            f'      <atlas:image-width>{plate.image_width}</atlas:image-width>',
            f'      <atlas:image-height>{plate.image_height}</atlas:image-height>',
            '    </atlas:source>',
            '  </metadata>',
            '',
            '  <!-- Background: source JP2 for ground truth reference -->',
            f'  <image id="background" href="file://{abs_jp2_path}" '
            f'width="{plate.image_width}" height="{plate.image_height}" opacity="0.5"/>',
            '',
        ])

    svg_parts.extend([
        '  <style>',
        '    .block-label { font-family: sans-serif; font-size: 14px; font-weight: bold; }',
        '  </style>',
        '',
        '  <!-- Block polygons (editable) -->',
        '  <g id="blocks">',
    ])

    for block in plate.blocks:
        r, g, b = block.color
        fill_color = f"rgba({r},{g},{b},0.3)"  # Lighter fill for overview
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
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" '
                f'data-block-id="{block.id}"/>'
            )

        # Add label at center of bounding box
        label_x = block.bbox.x + block.bbox.width // 2
        label_y = block.bbox.y + block.bbox.height // 2
        svg_parts.append(
            f'    <text x="{label_x}" y="{label_y}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'class="block-label" fill="{stroke_color}" stroke="white" stroke-width="3" paint-order="stroke">'
            f'{block.svg_filename}</text>'
        )

    svg_parts.append('  </g>')
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
    """Generate combined SVG with inlined block content.

    Inlines the polygon content from each block SVG file directly into
    the combined SVG for maximum compatibility with applications like
    Affinity Designer that don't support external SVG references.
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{plate.image_width}" height="{plate.image_height}" '
        f'viewBox="0 0 {plate.image_width} {plate.image_height}">',
        '  <!-- Combined segmentation view -->',
        '  <!-- Block content inlined for compatibility -->',
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

    # Inline content from each block SVG
    for block in plate.blocks:
        svg_parts.append(
            f'  <g id="block-{block.id}" '
            f'transform="translate({block.bbox.x},{block.bbox.y})">'
        )
        svg_parts.append(
            f'    <!-- Block {block.id}: {block.bbox.width}x{block.bbox.height} '
            f'at ({block.bbox.x},{block.bbox.y}) -->'
        )

        # Read and inline the block SVG content
        block_svg_path = output_path.parent / block.svg_filename
        if block_svg_path.exists():
            with open(block_svg_path, 'r') as f:
                block_svg_content = f.read()

            # Extract polygon elements from the block SVG
            # Parse out just the polygon tags (skip the svg wrapper)
            polygons = re.findall(r'<polygon[^>]+/>', block_svg_content)
            for polygon in polygons:
                svg_parts.append(f'    {polygon}')
        else:
            svg_parts.append(f'    <!-- Block SVG not found: {block.svg_filename} -->')

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


# Stage 1: Block Detection
def process_stage1(
    input_path: Path,
    output_base: Path,
    sam_model,
    min_block_width: int = 150,
    min_block_height: int = 150,
    max_area_ratio: float = 0.7
) -> None:
    """Stage 1: Detect blocks and output plate.svg with metadata.

    This generates a plate.svg file that can be manually edited before
    running Stage 2 to process individual blocks.
    """
    # Determine output structure
    volume, plate_id, output_dir = get_output_structure(input_path, output_base)

    print(f"Stage 1: Processing plate {volume}/{plate_id}")
    print(f"Output directory: {output_dir}")

    # Load full plate image
    print("Loading plate image...")
    image_bgr = load_image(input_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]
    print(f"Image size: {width}x{height}")

    # First pass: Identify city blocks
    print("Detecting city blocks...")
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

    # Generate plate-level SVG with metadata
    plate_svg_path = output_dir / "plate.svg"
    generate_plate_svg(plate, plate_svg_path, source_jp2_path=input_path)

    print(f"\nStage 1 complete!")
    print(f"Output: {plate_svg_path}")
    print(f"\nNext steps:")
    print(f"  1. Open {plate_svg_path} in an SVG editor")
    print(f"  2. Delete or merge blocks as needed")
    print(f"  3. Run: uv run python segment.py stage2 {plate_svg_path}")


# Stage 2: Block Segmentation
def process_stage2(svg_path: Path, sam_model) -> None:
    """Stage 2: Read plate.svg, extract blocks, run SAM on each.

    Reads metadata from the plate.svg to find the source JP2, then
    processes each block polygon through SAM for detailed segmentation.
    """
    print(f"Stage 2: Processing {svg_path}")

    # Parse plate.svg to get metadata and blocks
    print("Parsing plate.svg...")
    metadata, blocks = parse_plate_svg(svg_path)
    print(f"Found {len(blocks)} blocks in SVG")
    print(f"Source JP2: {metadata.jp2_path}")
    print(f"Volume: {metadata.volume}, Plate: {metadata.plate_id}")

    # Verify source JP2 exists
    if not metadata.jp2_path.exists():
        print(f"Error: Source JP2 not found: {metadata.jp2_path}")
        return

    # Load source image
    print("Loading source image...")
    image_bgr = load_image(metadata.jp2_path)
    print(f"Image size: {metadata.image_width}x{metadata.image_height}")

    # Determine output directory (same as plate.svg location)
    output_dir = svg_path.parent

    # Renumber blocks sequentially (preserve name if present)
    renumbered_blocks = []
    for idx, block in enumerate(blocks, 1):
        new_id = f"{idx:04d}"
        renumbered_blocks.append(Block(
            id=new_id,
            bbox=block.bbox,
            contours=block.contours,
            color=block.color,
            name=block.name
        ))

    # Update plate with renumbered blocks for combined SVG
    plate = PlateSegmentation(
        volume=metadata.volume,
        plate_id=metadata.plate_id,
        image_width=metadata.image_width,
        image_height=metadata.image_height,
        blocks=renumbered_blocks
    )

    # Process each block through SAM
    total_blocks = len(renumbered_blocks)
    print(f"\nRunning SAM on {total_blocks} blocks...")
    second_pass_generator = create_mask_generator(sam_model, SECOND_PASS_CONFIG)

    for idx, block in enumerate(renumbered_blocks, 1):
        print(f"  [{idx}/{total_blocks}] Block {block.id} ({block.bbox.width}x{block.bbox.height})...")

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

        # Write unfiltered block SVG (culling is done by process_combine)
        block_svg_path = output_dir / block.svg_filename
        generate_block_svg(block_seg, block_svg_path)

        print(f"    {len(block_seg.contours)} raw segments -> {block.svg_filename}")

    # Apply cull filters and generate combined SVG
    print("\n" + "=" * 50)
    process_combine(svg_path, apply_culling=True)

    print(f"\nStage 2 complete!")
    print(f"Output files in: {output_dir}")


def process_extract(svg_path: Path) -> Path:
    """Extract block images from the source JP2 based on plate.svg.

    Reads plate.svg metadata and block definitions, then extracts each
    block region as a PNG image file.

    Args:
        svg_path: Path to plate.svg file

    Returns:
        Path to the output directory containing extracted images
    """
    print(f"Extract: Processing {svg_path}")

    # Parse plate.svg to get metadata and blocks
    print("Parsing plate.svg...")
    metadata, blocks = parse_plate_svg(svg_path)
    print(f"Found {len(blocks)} blocks in SVG")
    print(f"Source JP2: {metadata.jp2_path}")

    # Verify source JP2 exists
    if not metadata.jp2_path.exists():
        print(f"Error: Source JP2 not found: {metadata.jp2_path}")
        raise FileNotFoundError(f"Source JP2 not found: {metadata.jp2_path}")

    # Load source image
    print("Loading source image...")
    image_bgr = load_image(metadata.jp2_path)
    print(f"Image size: {metadata.image_width}x{metadata.image_height}")

    # Determine output directory (same as plate.svg location)
    output_dir = svg_path.parent

    # Renumber blocks sequentially (matching stage2 behavior)
    renumbered_blocks = []
    for idx, block in enumerate(blocks, 1):
        new_id = f"{idx:04d}"
        renumbered_blocks.append(Block(
            id=new_id,
            bbox=block.bbox,
            contours=block.contours,
            color=block.color,
            name=block.name
        ))

    # Extract each block as an image
    total_blocks = len(renumbered_blocks)
    print(f"\nExtracting {total_blocks} block images...")

    for idx, block in enumerate(renumbered_blocks, 1):
        # Convert contours to Shapely polygon for buffering
        # Use the first (main) contour
        if not block.contours:
            print(f"  [{idx}/{total_blocks}] {block.image_filename} - SKIPPED (no contours)")
            continue

        contour = block.contours[0]
        points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
        if len(points) < 3:
            print(f"  [{idx}/{total_blocks}] {block.image_filename} - SKIPPED (insufficient points)")
            continue

        original_poly = ShapelyPolygon(points)
        if not original_poly.is_valid:
            original_poly = make_valid(original_poly)

        # Calculate buffer distance based on average bbox dimension
        avg_dimension = (block.bbox.width + block.bbox.height) / 2
        buffer_10pct = avg_dimension * 0.10
        buffer_5pct = avg_dimension * 0.05

        # Expand polygon by 10% for canvas bounds
        canvas_poly = original_poly.buffer(buffer_10pct)
        # Expand polygon by 5% for mask (visible region)
        mask_poly = original_poly.buffer(buffer_5pct)

        # Get bounding box of 10% expanded polygon for extraction
        minx, miny, maxx, maxy = canvas_poly.bounds
        canvas_bbox = BoundingBox(
            x=int(minx),
            y=int(miny),
            width=int(maxx - minx),
            height=int(maxy - miny)
        )

        # Extract region based on canvas bounds
        block_image_rgba = extract_region_with_bounds(image_bgr, canvas_bbox, expansion_factor=0.0)

        # Create mask from 5% expanded polygon
        mask = np.zeros((canvas_bbox.height, canvas_bbox.width), dtype=np.uint8)

        # Convert mask polygon to local coordinates and draw
        if mask_poly.geom_type == 'Polygon':
            mask_coords = np.array(mask_poly.exterior.coords, dtype=np.int32)
            # Translate to local coordinates
            mask_coords[:, 0] -= canvas_bbox.x
            mask_coords[:, 1] -= canvas_bbox.y
            cv2.fillPoly(mask, [mask_coords], 255)
        elif mask_poly.geom_type == 'MultiPolygon':
            for poly in mask_poly.geoms:
                mask_coords = np.array(poly.exterior.coords, dtype=np.int32)
                mask_coords[:, 0] -= canvas_bbox.x
                mask_coords[:, 1] -= canvas_bbox.y
                cv2.fillPoly(mask, [mask_coords], 255)

        # Apply mask to alpha channel (0 = transparent, 255 = opaque)
        block_image_rgba[:, :, 3] = mask

        # Save as PNG (with alpha channel for transparency)
        image_path = output_dir / block.image_filename
        cv2.imwrite(str(image_path), block_image_rgba)

        print(f"  [{idx}/{total_blocks}] {block.image_filename} ({canvas_bbox.width}x{canvas_bbox.height})")

    print(f"\nExtract complete!")
    print(f"Output: {output_dir}")

    return output_dir


def process_combine(svg_path: Path, apply_culling: bool = True) -> None:
    """Combine existing block SVGs into segmentation.svg without running SAM.

    Reads plate.svg metadata and block definitions, applies cull filters
    to each block SVG, rewrites them, then combines into segmentation.svg.

    Args:
        svg_path: Path to plate.svg file
        apply_culling: Whether to apply cull filters to block SVGs (default True)
    """
    print(f"Combine: Processing {svg_path}")

    # Parse plate.svg to get metadata and blocks
    print("Parsing plate.svg...")
    metadata, blocks = parse_plate_svg(svg_path)
    print(f"Found {len(blocks)} blocks in SVG")

    # Determine output directory (same as plate.svg location)
    output_dir = svg_path.parent

    # Renumber blocks sequentially (matching stage2 behavior)
    renumbered_blocks = []
    missing_blocks = []
    total_culled = 0

    print("\nProcessing block SVGs...")
    for idx, block in enumerate(blocks, 1):
        new_id = f"{idx:04d}"
        new_block = Block(
            id=new_id,
            bbox=block.bbox,
            contours=block.contours,
            color=block.color,
            name=block.name
        )
        renumbered_blocks.append(new_block)

        # Check if block SVG exists
        block_svg_path = output_dir / new_block.svg_filename
        if block_svg_path.exists():
            if apply_culling and new_block.contours:
                # Apply cull filters
                with open(block_svg_path, 'r') as f:
                    block_svg_content = f.read()

                filtered_content, orig_count, filt_count = apply_cull_filters_to_block_svg(
                    block_svg_content,
                    new_block.contours[0],
                    new_block.bbox
                )

                culled = orig_count - filt_count
                total_culled += culled

                # Rewrite the block SVG with filtered content
                with open(block_svg_path, 'w') as f:
                    f.write(filtered_content)

                if culled > 0:
                    print(f"  {new_block.svg_filename}: {filt_count} polygons (culled {culled})")
                else:
                    print(f"  {new_block.svg_filename}: {filt_count} polygons")
            else:
                print(f"  {new_block.svg_filename}: found")
        else:
            missing_blocks.append(new_block.svg_filename)
            print(f"  {new_block.svg_filename}: MISSING")

    if missing_blocks:
        print(f"\nWarning: {len(missing_blocks)} block SVG(s) not found on disk.")
        print("These blocks will show as empty in the combined SVG.")

    if apply_culling and total_culled > 0:
        print(f"\nCulled {total_culled} total polygons across all blocks.")

    # Create plate structure for combined SVG
    plate = PlateSegmentation(
        volume=metadata.volume,
        plate_id=metadata.plate_id,
        image_width=metadata.image_width,
        image_height=metadata.image_height,
        blocks=renumbered_blocks
    )

    # Generate combined SVG
    print("\nGenerating combined segmentation.svg...")
    segmentation_svg_path = output_dir / "segmentation.svg"
    generate_combined_svg(plate, segmentation_svg_path)

    print(f"\nCombine complete!")
    print(f"Output: {segmentation_svg_path}")


# Main Recursive Workflow (Legacy)
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


def load_sam_model(force_mps: bool = False):
    """Load SAM model and return it along with checkpoint path."""
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / CHECKPOINT_NAME

    # Download checkpoint if needed
    download_checkpoint(checkpoint_path)

    # Detect device
    device = get_device(force_mps=force_mps)
    print(f"Using device: {device}")

    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint_path))
    sam.to(device=device)

    return sam


def main():
    parser = argparse.ArgumentParser(
        description="Segment images using SAM. Supports stage1/stage2 workflow for manual editing."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Stage 1 subcommand
    stage1_parser = subparsers.add_parser(
        "stage1",
        help="Detect blocks from plate image (run this first)"
    )
    stage1_parser.add_argument("input", help="Input JP2/image path")
    stage1_parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Base output directory (default: output)"
    )
    stage1_parser.add_argument(
        "--min-block-width",
        type=int,
        default=150,
        help="Minimum block width in pixels (default: 150)"
    )
    stage1_parser.add_argument(
        "--min-block-height",
        type=int,
        default=150,
        help="Minimum block height in pixels (default: 150)"
    )
    stage1_parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.7,
        help="Maximum block area as ratio of image (default: 0.7 = 70%%)"
    )
    stage1_parser.add_argument(
        "--mps",
        action="store_true",
        help="Force use of MPS (Apple Silicon GPU)"
    )

    # Stage 2 subcommand
    stage2_parser = subparsers.add_parser(
        "stage2",
        help="Process blocks from edited plate.svg"
    )
    stage2_parser.add_argument("svg", help="Path to plate.svg file")
    stage2_parser.add_argument(
        "--mps",
        action="store_true",
        help="Force use of MPS (Apple Silicon GPU)"
    )

    # Combine subcommand
    combine_parser = subparsers.add_parser(
        "combine",
        help="Regenerate segmentation.svg from existing block SVGs (no SAM)"
    )
    combine_parser.add_argument("svg", help="Path to plate.svg file")
    combine_parser.add_argument(
        "--no-cull",
        action="store_true",
        help="Skip polygon culling (just combine existing SVGs as-is)"
    )

    # Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract block images from source JP2 (no SAM)"
    )
    extract_parser.add_argument("svg", help="Path to plate.svg file")

    # Legacy mode (no subcommand) - runs both stages
    parser.add_argument("legacy_input", nargs="?", help="Input image (legacy mode: runs both stages)")
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Base output directory (default: output)"
    )
    parser.add_argument(
        "--min-block-width",
        type=int,
        default=150,
        help="Minimum block width in pixels (default: 150)"
    )
    parser.add_argument(
        "--min-block-height",
        type=int,
        default=150,
        help="Minimum block height in pixels (default: 150)"
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
    parser.add_argument(
        "--output-png",
        help="[Single-pass only] Output PNG path"
    )
    parser.add_argument(
        "-s", "--svg",
        help="[Single-pass only] Output SVG path"
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Force use of MPS (Apple Silicon GPU)"
    )

    args = parser.parse_args()

    # Handle subcommands
    if args.command == "stage1":
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input image not found: {input_path}")
            return 1

        sam = load_sam_model(force_mps=args.mps)
        process_stage1(
            input_path,
            Path(args.output_dir),
            sam,
            args.min_block_width,
            args.min_block_height,
            args.max_area_ratio
        )
        return 0

    elif args.command == "stage2":
        svg_path = Path(args.svg)
        if not svg_path.exists():
            print(f"Error: SVG file not found: {svg_path}")
            return 1

        sam = load_sam_model(force_mps=args.mps)
        process_stage2(svg_path, sam)
        return 0

    elif args.command == "combine":
        svg_path = Path(args.svg)
        if not svg_path.exists():
            print(f"Error: SVG file not found: {svg_path}")
            return 1

        process_combine(svg_path, apply_culling=not args.no_cull)
        return 0

    elif args.command == "extract":
        svg_path = Path(args.svg)
        if not svg_path.exists():
            print(f"Error: SVG file not found: {svg_path}")
            return 1

        process_extract(svg_path)
        return 0

    # Legacy mode (no subcommand)
    if args.legacy_input is None:
        parser.print_help()
        print("\nExamples:")
        print("  Stage 1: uv run python segment.py stage1 /path/to/image.jp2")
        print("  Stage 2: uv run python segment.py stage2 output/vol1/p37/plate.svg")
        print("  Combine: uv run python segment.py combine output/vol1/p37/plate.svg")
        print("  Extract: uv run python segment.py extract output/vol1/p37/plate.svg")
        print("  Both:    uv run python segment.py /path/to/image.jp2")
        return 0

    input_path = Path(args.legacy_input)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return 1

    sam = load_sam_model(force_mps=args.mps)

    if args.single_pass:
        # Original single-pass mode for comparison
        if args.output_png:
            output_path = Path(args.output_png)
        else:
            output_path = input_path.parent / f"{input_path.stem}_segmented.png"

        if args.svg:
            svg_path = Path(args.svg)
        else:
            svg_path = input_path.parent / f"{input_path.stem}_segmented.svg"

        print(f"Loading image: {input_path}")
        image_bgr = load_image(input_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

        sam_result = mask_generator.generate(image_rgb)
        print(f"Found {len(sam_result)} segments")

        detections = sv.Detections.from_sam(sam_result=sam_result)
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.5,
        )

        annotated_image = mask_annotator.annotate(
            scene=image_bgr.copy(),
            detections=detections,
        )

        cv2.imwrite(str(output_path), annotated_image)
        print(f"Saved segmented image: {output_path}")

        height, width = image_bgr.shape[:2]
        generate_svg(sam_result, width, height, svg_path)
    else:
        # Recursive segmentation (both stages)
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
