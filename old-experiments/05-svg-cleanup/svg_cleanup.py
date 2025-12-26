#!/usr/bin/env python3
"""Experiment 5: SVG Post-Processing for Building Extraction

This script post-processes an Inkscape-traced SVG to separate building outlines from text by:
1. Splitting the single traced path into separate subpaths
2. Using OCR bounding boxes to identify and remove text-related paths
3. Outputting a clean SVG with only building polygons
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from lxml import etree
from svgpathtools import parse_path, Path as SVGPath

# Try to import EasyOCR for text detection
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    print("Warning: EasyOCR not available, will use cached OCR data if available")


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if min(self.width, self.height) == 0:
            return float('inf')
        return max(self.width, self.height) / min(self.width, self.height)

    def overlap_ratio(self, other: 'BoundingBox') -> float:
        """Calculate what fraction of this box overlaps with another."""
        x1 = max(self.x_min, other.x_min)
        y1 = max(self.y_min, other.y_min)
        x2 = min(self.x_max, other.x_max)
        y2 = min(self.y_max, other.y_max)

        if x2 < x1 or y2 < y1:
            return 0.0  # No overlap

        intersection = (x2 - x1) * (y2 - y1)
        return intersection / self.area if self.area > 0 else 0.0


@dataclass
class Subpath:
    """A single closed subpath extracted from the SVG."""
    path_data: str
    bbox: BoundingBox
    index: int
    is_text: bool = False


def parse_svg_path_data(svg_path: Path) -> Tuple[str, dict]:
    """Parse SVG file and extract the main path data.

    Returns:
        Tuple of (path_data_string, svg_metadata_dict)
    """
    tree = etree.parse(str(svg_path))
    root = tree.getroot()

    # Get SVG namespace
    nsmap = {'svg': 'http://www.w3.org/2000/svg'}

    # Find the path element
    paths = root.xpath('//svg:path', namespaces=nsmap)
    if not paths:
        # Try without namespace
        paths = root.xpath('//*[local-name()="path"]')

    if not paths:
        raise ValueError("No path elements found in SVG")

    # Get the main traced path (usually the last/largest one)
    path_elem = paths[-1]
    path_data = path_elem.get('d', '')

    # Get SVG metadata
    width = root.get('width', '0')
    height = root.get('height', '0')
    viewbox = root.get('viewBox', '0 0 0 0')

    metadata = {
        'width': width,
        'height': height,
        'viewBox': viewbox,
    }

    return path_data, metadata


def split_path_into_subpaths(path_data: str) -> List[str]:
    """Split SVG path data into individual closed subpaths.

    Each subpath starts with 'm' or 'M' and ends with 'z' or 'Z'.
    """
    # Normalize whitespace
    path_data = ' '.join(path_data.split())

    # Split on 'z' followed by 'm' or 'M' (new subpath start)
    # Also handle uppercase Z
    pattern = r'([zZ])\s*([mM])'

    # Replace pattern with a delimiter we can split on
    marked = re.sub(pattern, r'\1|||SPLIT|||\2', path_data)

    # Split and filter empty strings
    parts = marked.split('|||SPLIT|||')

    subpaths = []
    for part in parts:
        part = part.strip()
        if part:
            # Ensure path starts with move command
            if not part.lower().startswith('m'):
                part = 'm ' + part
            subpaths.append(part)

    return subpaths


def calculate_path_bbox(path_data: str) -> Optional[BoundingBox]:
    """Calculate bounding box for an SVG path string."""
    try:
        path = parse_path(path_data)
        if len(path) == 0:
            return None
        bbox = path.bbox()  # Returns (xmin, xmax, ymin, ymax)
        return BoundingBox(
            x_min=bbox[0],
            y_min=bbox[2],
            x_max=bbox[1],
            y_max=bbox[3]
        )
    except Exception as e:
        # Path parsing can fail for degenerate paths
        return None


def get_overall_bbox(subpaths: List[Subpath]) -> BoundingBox:
    """Calculate the overall bounding box of all subpaths."""
    x_min = min(sp.bbox.x_min for sp in subpaths if sp.bbox)
    y_min = min(sp.bbox.y_min for sp in subpaths if sp.bbox)
    x_max = max(sp.bbox.x_max for sp in subpaths if sp.bbox)
    y_max = max(sp.bbox.y_max for sp in subpaths if sp.bbox)
    return BoundingBox(x_min, y_min, x_max, y_max)


def run_ocr_detection(image_path: Path) -> List[BoundingBox]:
    """Run EasyOCR to detect text regions, returning bounding boxes in pixel coordinates."""
    if not HAS_EASYOCR:
        raise RuntimeError("EasyOCR is not available")

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)

    print(f"Running OCR on {image_path}...")
    results = reader.readtext(str(image_path), detail=1)
    print(f"Found {len(results)} text regions")

    boxes = []
    for detection in results:
        bbox_points = detection[0]
        # bbox_points is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        x_coords = [p[0] for p in bbox_points]
        y_coords = [p[1] for p in bbox_points]
        boxes.append(BoundingBox(
            x_min=min(x_coords),
            y_min=min(y_coords),
            x_max=max(x_coords),
            y_max=max(y_coords)
        ))

    return boxes


def load_cached_ocr(cache_path: Path) -> List[BoundingBox]:
    """Load OCR results from a cached JSON file."""
    with open(cache_path) as f:
        data = json.load(f)

    boxes = []
    for item in data:
        boxes.append(BoundingBox(
            x_min=item['x_min'],
            y_min=item['y_min'],
            x_max=item['x_max'],
            y_max=item['y_max']
        ))
    return boxes


def save_ocr_cache(boxes: List[BoundingBox], cache_path: Path):
    """Save OCR results to a JSON cache file."""
    data = [
        {
            'x_min': float(b.x_min),
            'y_min': float(b.y_min),
            'x_max': float(b.x_max),
            'y_max': float(b.y_max)
        }
        for b in boxes
    ]
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)


def transform_ocr_boxes(
    ocr_boxes: List[BoundingBox],
    image_width: int,
    image_height: int,
    svg_bbox: BoundingBox
) -> List[BoundingBox]:
    """Transform OCR bounding boxes from pixel coordinates to SVG coordinates.

    Args:
        ocr_boxes: Bounding boxes in pixel coordinates
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        svg_bbox: The overall bounding box of the SVG paths

    Returns:
        Bounding boxes transformed to SVG coordinate space
    """
    # Calculate scale factors
    svg_width = svg_bbox.x_max - svg_bbox.x_min
    svg_height = svg_bbox.y_max - svg_bbox.y_min

    scale_x = svg_width / image_width
    scale_y = svg_height / image_height

    print(f"Coordinate transformation:")
    print(f"  Image: {image_width}x{image_height} pixels")
    print(f"  SVG paths extent: {svg_width:.2f}x{svg_height:.2f} units")
    print(f"  Scale: {scale_x:.4f} x {scale_y:.4f}")
    print(f"  SVG offset: ({svg_bbox.x_min:.2f}, {svg_bbox.y_min:.2f})")

    transformed = []
    for box in ocr_boxes:
        transformed.append(BoundingBox(
            x_min=box.x_min * scale_x + svg_bbox.x_min,
            y_min=box.y_min * scale_y + svg_bbox.y_min,
            x_max=box.x_max * scale_x + svg_bbox.x_min,
            y_max=box.y_max * scale_y + svg_bbox.y_min
        ))

    return transformed


def filter_text_paths(
    subpaths: List[Subpath],
    text_boxes: List[BoundingBox],
    overlap_threshold: float = 0.8,
    min_area: float = 100,
    max_aspect_ratio: float = 10.0
) -> Tuple[List[Subpath], List[Subpath]]:
    """Filter subpaths to remove text-related paths.

    Args:
        subpaths: List of all subpaths
        text_boxes: OCR-detected text bounding boxes (in SVG coordinates)
        overlap_threshold: Remove if overlap > this fraction (default 80%)
        min_area: Remove paths smaller than this area
        max_aspect_ratio: Remove paths more elongated than this

    Returns:
        Tuple of (building_paths, text_paths)
    """
    buildings = []
    text_paths = []

    for sp in subpaths:
        if sp.bbox is None:
            continue

        # Check area (too small = noise or text artifact)
        if sp.bbox.area < min_area:
            sp.is_text = True
            text_paths.append(sp)
            continue

        # Check aspect ratio (very elongated = likely text)
        if sp.bbox.aspect_ratio > max_aspect_ratio:
            sp.is_text = True
            text_paths.append(sp)
            continue

        # Check overlap with text boxes
        max_overlap = 0.0
        for text_box in text_boxes:
            overlap = sp.bbox.overlap_ratio(text_box)
            max_overlap = max(max_overlap, overlap)

        if max_overlap > overlap_threshold:
            sp.is_text = True
            text_paths.append(sp)
        else:
            buildings.append(sp)

    return buildings, text_paths


def generate_clean_svg(
    subpaths: List[Subpath],
    svg_bbox: BoundingBox,
    output_path: Path
):
    """Generate a clean SVG with only the building paths."""
    # Create SVG with proper viewBox
    width = svg_bbox.x_max - svg_bbox.x_min
    height = svg_bbox.y_max - svg_bbox.y_min

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.2f}"
     height="{height:.2f}"
     viewBox="{svg_bbox.x_min:.2f} {svg_bbox.y_min:.2f} {width:.2f} {height:.2f}">
  <!-- Building polygons extracted from traced SVG -->
  <!-- Total paths: {len(subpaths)} -->
  <g id="buildings" fill="none" stroke="black" stroke-width="0.5">
'''

    for sp in subpaths:
        svg_content += f'    <path d="{sp.path_data}" data-path-id="{sp.index}"/>\n'

    svg_content += '''  </g>
</svg>
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content)
    print(f"Saved clean SVG: {output_path}")


def main():
    # Paths
    traced_svg = Path("/Users/pwright/workspace/atlas/media/map-example-traced.svg")
    original_image = Path("/Users/pwright/workspace/atlas/media/map-example.png")
    output_dir = Path(__file__).parent / "results"
    ocr_cache = output_dir / "ocr_cache.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Image dimensions (from the original PNG)
    IMAGE_WIDTH = 2236
    IMAGE_HEIGHT = 1318

    print("=" * 60)
    print("EXPERIMENT 5: SVG Post-Processing for Building Extraction")
    print("=" * 60)

    # Step 1: Parse SVG and extract path data
    print("\n[Step 1] Parsing SVG...")
    path_data, metadata = parse_svg_path_data(traced_svg)
    print(f"  SVG viewBox: {metadata['viewBox']}")
    print(f"  Path data length: {len(path_data):,} characters")

    # Step 2: Split into subpaths
    print("\n[Step 2] Splitting into subpaths...")
    path_strings = split_path_into_subpaths(path_data)
    print(f"  Found {len(path_strings):,} subpaths")

    # Step 3: Calculate bounding boxes for each subpath
    print("\n[Step 3] Calculating bounding boxes...")
    subpaths = []
    for i, ps in enumerate(path_strings):
        bbox = calculate_path_bbox(ps)
        if bbox is not None:
            subpaths.append(Subpath(path_data=ps, bbox=bbox, index=i))

    print(f"  Valid subpaths with bounding boxes: {len(subpaths):,}")

    # Calculate overall SVG bounds
    overall_bbox = get_overall_bbox(subpaths)
    print(f"  Overall SVG extent: ({overall_bbox.x_min:.2f}, {overall_bbox.y_min:.2f}) to ({overall_bbox.x_max:.2f}, {overall_bbox.y_max:.2f})")

    # Step 4: Get OCR text regions
    print("\n[Step 4] Loading/running OCR text detection...")

    if ocr_cache.exists():
        print(f"  Using cached OCR data from {ocr_cache}")
        ocr_boxes_pixels = load_cached_ocr(ocr_cache)
    elif HAS_EASYOCR:
        ocr_boxes_pixels = run_ocr_detection(original_image)
        save_ocr_cache(ocr_boxes_pixels, ocr_cache)
        print(f"  Saved OCR cache to {ocr_cache}")
    else:
        # Try to load from experiment 1
        exp1_cache = Path(__file__).parent.parent / "01-text-inpainting" / "results" / "text_regions.json"
        if exp1_cache.exists():
            print(f"  Using OCR data from experiment 1: {exp1_cache}")
            ocr_boxes_pixels = load_cached_ocr(exp1_cache)
        else:
            print("  ERROR: No OCR data available. Run with EasyOCR or provide cached data.")
            return 1

    print(f"  OCR text regions: {len(ocr_boxes_pixels)}")

    # Step 5: Transform OCR coordinates to SVG space
    print("\n[Step 5] Transforming OCR coordinates to SVG space...")
    ocr_boxes_svg = transform_ocr_boxes(
        ocr_boxes_pixels, IMAGE_WIDTH, IMAGE_HEIGHT, overall_bbox
    )

    # Step 6: Filter out text paths
    print("\n[Step 6] Filtering text paths...")
    buildings, text_paths = filter_text_paths(
        subpaths,
        ocr_boxes_svg,
        overlap_threshold=0.8,
        min_area=100,
        max_aspect_ratio=10.0
    )

    print(f"  Building paths: {len(buildings):,}")
    print(f"  Text paths removed: {len(text_paths):,}")

    # Step 7: Generate output SVG
    print("\n[Step 7] Generating clean SVG...")
    generate_clean_svg(buildings, overall_bbox, output_dir / "buildings.svg")

    # Also save a debug SVG with text paths highlighted
    if text_paths:
        generate_debug_svg(buildings, text_paths, overall_bbox, output_dir / "debug" / "text_highlight.svg")

    # Save analysis data
    analysis = {
        'total_subpaths': len(path_strings),
        'valid_subpaths': len(subpaths),
        'building_paths': len(buildings),
        'text_paths': len(text_paths),
        'ocr_regions': len(ocr_boxes_pixels),
        'svg_extent': {
            'x_min': overall_bbox.x_min,
            'y_min': overall_bbox.y_min,
            'x_max': overall_bbox.x_max,
            'y_max': overall_bbox.y_max,
        }
    }

    with open(output_dir / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total subpaths parsed: {len(path_strings):,}")
    print(f"Valid subpaths: {len(subpaths):,}")
    print(f"Building paths retained: {len(buildings):,}")
    print(f"Text paths removed: {len(text_paths):,}")
    print(f"\nOutput: {output_dir / 'buildings.svg'}")

    return 0


def generate_debug_svg(
    buildings: List[Subpath],
    text_paths: List[Subpath],
    svg_bbox: BoundingBox,
    output_path: Path
):
    """Generate a debug SVG showing buildings (green) and text (red)."""
    width = svg_bbox.x_max - svg_bbox.x_min
    height = svg_bbox.y_max - svg_bbox.y_min

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.2f}"
     height="{height:.2f}"
     viewBox="{svg_bbox.x_min:.2f} {svg_bbox.y_min:.2f} {width:.2f} {height:.2f}">
  <!-- Debug visualization: green=buildings, red=text -->
  <g id="text_paths" fill="rgba(255,0,0,0.3)" stroke="red" stroke-width="0.3">
'''

    for sp in text_paths:
        svg_content += f'    <path d="{sp.path_data}"/>\n'

    svg_content += '''  </g>
  <g id="buildings" fill="rgba(0,255,0,0.3)" stroke="green" stroke-width="0.3">
'''

    for sp in buildings:
        svg_content += f'    <path d="{sp.path_data}"/>\n'

    svg_content += '''  </g>
</svg>
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content)
    print(f"Saved debug SVG: {output_path}")


if __name__ == "__main__":
    exit(main())
