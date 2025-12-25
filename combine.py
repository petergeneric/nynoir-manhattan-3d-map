#!/usr/bin/env python3
"""
Combine multiple segmentation SVG files into a single output SVG.

Reads a segmentation.json file that lists SVG files and their transformation
metadata (rotation, scale, position), applies the transforms, and combines
all polygons into one output file.

Usage:
    uv run python combine.py segmentation.json -o combined.svg
"""

import argparse
import base64
import json5
import math
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree

# The metadata scale factor is designed for thumbnail images (10% of original).
# For full-size plate coordinates, we need this correction factor.
THUMBNAIL_SCALE_CORRECTION = 0.1


@dataclass
class PlateEntry:
    """An entry from segmentation.json listing a plate SVG and its metadata."""
    svg_path: Path
    metadata_path: Path


@dataclass
class PlateMetadata:
    """Transformation metadata from *.metadata.json files."""
    angle: float       # Rotation angle in degrees
    scale: float       # Scale factor
    pos_x: float       # Final X position on reference canvas
    pos_y: float       # Final Y position on reference canvas


@dataclass
class BackgroundImage:
    """Background image configuration."""
    path: Path         # Path to the image file
    width: int         # Image width in pixels
    height: int        # Image height in pixels
    angle: float       # Rotation angle in degrees
    data_uri: str      # Base64 data URI for embedding


@dataclass
class PolygonData:
    """A polygon with its attributes."""
    points: List[Tuple[float, float]]  # Local block coordinates
    fill: str
    stroke: str
    data_id: Optional[str] = None


@dataclass
class BlockData:
    """A block containing polygons."""
    block_id: str
    translate_x: int
    translate_y: int
    polygons: List[PolygonData] = field(default_factory=list)

    def filter_outline(self) -> 'BlockData':
        """Return a new BlockData with the largest polygon (block outline) removed.

        The block outline is assumed to be the polygon with the largest area,
        which typically encompasses all the building polygons within the block.
        """
        if len(self.polygons) <= 1:
            return self  # Can't filter if only one polygon

        # Calculate area for each polygon
        def polygon_area(poly: PolygonData) -> float:
            points = poly.points
            n = len(points)
            if n < 3:
                return 0
            area = 0
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            return abs(area) / 2

        # Find the largest polygon (the outline)
        areas = [(i, polygon_area(p)) for i, p in enumerate(self.polygons)]
        max_idx = max(areas, key=lambda x: x[1])[0]

        # Create new BlockData without the largest polygon
        filtered_polygons = [p for i, p in enumerate(self.polygons) if i != max_idx]

        return BlockData(
            block_id=self.block_id,
            translate_x=self.translate_x,
            translate_y=self.translate_y,
            polygons=filtered_polygons
        )


@dataclass
class PlateData:
    """Data parsed from a segmentation.svg file."""
    plate_id: str
    width: int
    height: int
    blocks: Dict[str, BlockData] = field(default_factory=dict)


@dataclass
class CombinedPolygon:
    """A polygon with transformed coordinates and source info."""
    points: List[Tuple[float, float]]  # Transformed global coordinates
    fill: str
    stroke: str
    source_plate: str
    source_block: str
    data_id: Optional[str] = None


def parse_segmentation_json(json_path: Path) -> List[PlateEntry]:
    """Parse the segmentation.json file.

    Uses JSON5 which tolerates trailing commas and unquoted keys.
    """
    content = json_path.read_text()
    data = json5.loads(content)

    entries = []
    base_dir = json_path.parent

    for item in data:
        svg_path = (base_dir / item["file"]).resolve()
        metadata_path = (base_dir / item["scale"]).resolve()
        entries.append(PlateEntry(svg_path=svg_path, metadata_path=metadata_path))

    return entries


def parse_svg_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse SVG polygon points string into list of (x, y) tuples."""
    points = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            points.append((float(x), float(y)))
    return points


def parse_segmentation_svg(svg_path: Path) -> PlateData:
    """Parse a segmentation.svg file.

    Extracts plate dimensions, block groups, and polygons with attributes.
    """
    content = svg_path.read_text()

    # Extract plate ID from path (e.g., "p20" from ".../p20/segmentation.svg")
    plate_id = svg_path.parent.name

    # Extract plate dimensions from root SVG
    svg_dims = re.search(r'<svg[^>]*width="(\d+)"[^>]*height="(\d+)"', content)
    if not svg_dims:
        raise ValueError(f"Could not find SVG dimensions in {svg_path}")
    width, height = int(svg_dims.group(1)), int(svg_dims.group(2))

    plate = PlateData(plate_id=plate_id, width=width, height=height)

    # Find all block groups: <g id="block-XXXX" transform="translate(x,y)">
    block_pattern = re.compile(
        r'<g\s+id="block-(\d+)"\s+transform="translate\((\d+),(\d+)\)"[^>]*>'
        r'(.*?)</g>',
        re.DOTALL
    )

    for match in block_pattern.finditer(content):
        block_id = match.group(1)
        translate_x = int(match.group(2))
        translate_y = int(match.group(3))
        block_content = match.group(4)

        block = BlockData(
            block_id=block_id,
            translate_x=translate_x,
            translate_y=translate_y
        )

        # Extract polygons with all attributes
        polygon_pattern = re.compile(r'<polygon\s+([^>]+)/>')

        for poly_match in polygon_pattern.finditer(block_content):
            attrs = poly_match.group(1)

            # Extract individual attributes
            points_match = re.search(r'points="([^"]+)"', attrs)
            fill_match = re.search(r'fill="([^"]*)"', attrs)
            stroke_match = re.search(r'stroke="([^"]*)"', attrs)
            data_id_match = re.search(r'data-id="([^"]*)"', attrs)

            if points_match:
                points = parse_svg_polygon_points(points_match.group(1))
                if len(points) >= 3:
                    block.polygons.append(PolygonData(
                        points=points,
                        fill=fill_match.group(1) if fill_match else "none",
                        stroke=stroke_match.group(1) if stroke_match else "none",
                        data_id=data_id_match.group(1) if data_id_match else None
                    ))

        plate.blocks[block_id] = block

    return plate


def load_metadata(metadata_path: Path) -> PlateMetadata:
    """Load transformation metadata from a .metadata.json file."""
    with open(metadata_path) as f:
        data = json5.load(f)

    return PlateMetadata(
        angle=data["angle"],
        scale=data["scale"],
        pos_x=data["pos"][0],
        pos_y=data["pos"][1]
    )


def load_background_image(image_path: Path) -> Optional[BackgroundImage]:
    """Load background image and its metadata.

    Expects a .metadata.json file alongside the image with rotation angle.
    Embeds the image as a base64 data URI for Affinity Designer compatibility.
    """
    if not image_path.exists():
        print(f"Warning: Background image not found: {image_path}")
        return None

    # Load image and convert to JPEG for embedding (widely supported)
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Convert to RGB if necessary (e.g., for RGBA or palette images)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Encode as JPEG to base64
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            b64_data = base64.b64encode(buffer.getvalue()).decode('ascii')
            data_uri = f"data:image/jpeg;base64,{b64_data}"

    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return None

    # Load metadata (angle)
    metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
    angle = 0.0
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                data = json5.load(f)
                angle = data.get("angle", 0.0)
        except Exception as e:
            print(f"Warning: Could not read metadata {metadata_path}: {e}")

    return BackgroundImage(
        path=image_path,
        width=width,
        height=height,
        angle=angle,
        data_uri=data_uri
    )


def transform_point(
    x: float, y: float,
    plate_center_x: float, plate_center_y: float,
    angle_deg: float, scale: float,
    final_pos_x: float, final_pos_y: float,
    output_scale: float = 1.0
) -> Tuple[float, float]:
    """Transform a single point using plate metadata.

    The metadata (angle, scale, pos) is designed for thumbnail images which
    are 10% of the original plate size. For full-size plate coordinates,
    we apply the THUMBNAIL_SCALE_CORRECTION.

    Args:
        output_scale: Factor to scale the final output coordinates.
                     Use 1/(reference_scale * THUMBNAIL_SCALE_CORRECTION) to
                     preserve original plate coordinate scale.

    Transformation order (matches CSS transform in align.py):
    1. Translate point relative to plate center (subtract center)
    2. Apply effective scale (scale * THUMBNAIL_SCALE_CORRECTION)
    3. Rotate around origin
    4. Translate to final position
    5. Apply output scale to preserve original coordinate magnitudes
    """
    # Step 1: Move to origin (plate center becomes 0,0)
    px = x - plate_center_x
    py = y - plate_center_y

    # Step 2: Scale (corrected for full-size plate coordinates)
    # The metadata scale is for thumbnails (10% size), so we multiply by 0.1
    effective_scale = scale * THUMBNAIL_SCALE_CORRECTION
    px *= effective_scale
    py *= effective_scale

    # Step 3: Rotate (convert degrees to radians)
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rx = px * cos_a - py * sin_a
    ry = px * sin_a + py * cos_a

    # Step 4: Translate to final position (in thumbnail-aligned space)
    fx = rx + final_pos_x
    fy = ry + final_pos_y

    # Step 5: Scale output to preserve original coordinate magnitudes
    fx *= output_scale
    fy *= output_scale

    return (fx, fy)


def transform_polygon(
    points: List[Tuple[float, float]],
    block_translate_x: int,
    block_translate_y: int,
    plate_width: int,
    plate_height: int,
    metadata: PlateMetadata,
    output_scale: float = 1.0
) -> List[Tuple[float, float]]:
    """Transform all points of a polygon.

    First converts from local block coords to plate coords,
    then applies the plate transformation.

    Args:
        output_scale: Factor to scale output coordinates to preserve original magnitudes.
    """
    plate_center_x = plate_width / 2.0
    plate_center_y = plate_height / 2.0

    transformed = []
    for x, y in points:
        # Convert local block coords to plate coords
        plate_x = x + block_translate_x
        plate_y = y + block_translate_y

        # Apply plate transformation
        tx, ty = transform_point(
            plate_x, plate_y,
            plate_center_x, plate_center_y,
            metadata.angle, metadata.scale,
            metadata.pos_x, metadata.pos_y,
            output_scale
        )
        transformed.append((tx, ty))

    return transformed


def combine_plates(
    entries: List[PlateEntry],
    skip_outlines: bool = False
) -> Tuple[List[CombinedPolygon], Tuple[float, float, float, float]]:
    """Process all plates and return combined polygons with bounding box.

    All plates are scaled relative to the first plate's scale factor to ensure
    consistent polygon areas for downstream filtering thresholds.

    Args:
        entries: List of plate entries to process
        skip_outlines: If True, filter out the largest polygon in each block
                       (assumed to be the block outline)

    Returns:
        (list of CombinedPolygon, (min_x, min_y, max_x, max_y))
    """
    all_polygons: List[CombinedPolygon] = []
    all_x: List[float] = []
    all_y: List[float] = []

    # Reference scale from the first valid plate (for output scaling)
    reference_scale: Optional[float] = None
    output_scale: float = 1.0

    for entry in entries:
        # Check files exist
        if not entry.svg_path.exists():
            print(f"Warning: SVG not found: {entry.svg_path}")
            continue
        if not entry.metadata_path.exists():
            print(f"Warning: Metadata not found: {entry.metadata_path}")
            continue

        # Load SVG and metadata
        plate_data = parse_segmentation_svg(entry.svg_path)
        metadata = load_metadata(entry.metadata_path)

        # Use first plate's scale to calculate output scaling
        # This preserves original plate coordinate magnitudes for consistent area filtering
        if reference_scale is None:
            reference_scale = metadata.scale
            # Scale output back up to original plate coordinates
            # effective_scale = scale * 0.1 shrinks to ~14%, so we multiply by 1/(ref*0.1) to undo
            output_scale = 1.0 / (reference_scale * THUMBNAIL_SCALE_CORRECTION)
            print(f"Using {plate_data.plate_id} scale ({metadata.scale:.4f}) as reference")
            print(f"Output scale factor: {output_scale:.4f} (preserves original coordinate magnitudes)")

        print(f"Processing {plate_data.plate_id}: {len(plate_data.blocks)} blocks, "
              f"angle={metadata.angle:.1f}, scale={metadata.scale:.3f}, "
              f"pos=({metadata.pos_x:.0f}, {metadata.pos_y:.0f})")

        polygon_count = 0
        outlines_skipped = 0
        for block_id, block in plate_data.blocks.items():
            # Optionally filter out block outline (largest polygon)
            if skip_outlines:
                filtered_block = block.filter_outline()
                outlines_skipped += len(block.polygons) - len(filtered_block.polygons)
                block = filtered_block

            for polygon in block.polygons:
                # Transform points (with output scaling to preserve original coordinate magnitudes)
                transformed_points = transform_polygon(
                    polygon.points,
                    block.translate_x,
                    block.translate_y,
                    plate_data.width,
                    plate_data.height,
                    metadata,
                    output_scale
                )

                # Track bounding box
                for x, y in transformed_points:
                    all_x.append(x)
                    all_y.append(y)

                all_polygons.append(CombinedPolygon(
                    points=transformed_points,
                    fill=polygon.fill,
                    stroke=polygon.stroke,
                    source_plate=plate_data.plate_id,
                    source_block=block_id,
                    data_id=polygon.data_id
                ))
                polygon_count += 1

        if skip_outlines and outlines_skipped > 0:
            print(f"  -> {polygon_count} polygons transformed ({outlines_skipped} block outlines skipped)")
        else:
            print(f"  -> {polygon_count} polygons transformed")

    if not all_x:
        return all_polygons, (0, 0, 100, 100)

    # Calculate bounding box
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_polygons, bbox


def generate_output_svg(
    polygons: List[CombinedPolygon],
    bbox: Tuple[float, float, float, float],
    output_path: Path,
    margin: float = 50.0,
    background: Optional[BackgroundImage] = None
):
    """Generate the combined output SVG file."""
    min_x, min_y, max_x, max_y = bbox

    # Add margin
    view_min_x = min_x - margin
    view_min_y = min_y - margin
    view_width = (max_x - min_x) + 2 * margin
    view_height = (max_y - min_y) + 2 * margin

    # Group polygons by source plate for structure
    by_plate: Dict[str, Dict[str, List[CombinedPolygon]]] = {}
    for poly in polygons:
        if poly.source_plate not in by_plate:
            by_plate[poly.source_plate] = {}
        if poly.source_block not in by_plate[poly.source_plate]:
            by_plate[poly.source_plate][poly.source_block] = []
        by_plate[poly.source_plate][poly.source_block].append(poly)

    # Generate SVG with xlink namespace for Affinity compatibility
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{view_width:.0f}" height="{view_height:.0f}" '
        f'viewBox="{view_min_x:.2f} {view_min_y:.2f} {view_width:.2f} {view_height:.2f}">',
        '',
        f'  <!-- Combined from {len(by_plate)} plates, {len(polygons)} polygons -->',
        '',
    ]

    # Add background image if provided (embedded as base64 for Affinity compatibility)
    if background:
        # The background image is rotated around its center
        cx = background.width / 2
        cy = background.height / 2

        lines.append('  <!-- Background reference map (embedded) -->')
        lines.append(f'  <image xlink:href="{background.data_uri}" '
                     f'x="0" y="0" '
                     f'width="{background.width}" height="{background.height}" '
                     f'transform="rotate({background.angle} {cx} {cy})" />')
        lines.append('')

    for plate_id in sorted(by_plate.keys()):
        plate_blocks = by_plate[plate_id]
        lines.append(f'  <g id="plate-{plate_id}" data-source="{plate_id}">')

        for block_id in sorted(plate_blocks.keys()):
            block_polygons = plate_blocks[block_id]
            lines.append(f'    <g id="{plate_id}-block-{block_id}">')

            for poly in block_polygons:
                points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in poly.points)
                attrs = [f'points="{points_str}"']
                if poly.fill and poly.fill != "none":
                    attrs.append(f'fill="{poly.fill}"')
                if poly.stroke and poly.stroke != "none":
                    attrs.append(f'stroke="{poly.stroke}"')
                if poly.data_id:
                    attrs.append(f'data-id="{poly.data_id}"')
                attrs.append(f'data-source-plate="{poly.source_plate}"')
                attrs.append(f'data-source-block="{poly.source_block}"')

                lines.append(f'      <polygon {" ".join(attrs)} />')

            lines.append('    </g>')

        lines.append('  </g>')

    lines.append('</svg>')

    output_path.write_text("\n".join(lines))
    print(f"\nWrote {len(polygons)} polygons to {output_path}")
    print(f"Bounding box: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
    print(f"Canvas size: {view_width:.0f} x {view_height:.0f}")


def test_cross_plate_overlap(
    polygons: List[CombinedPolygon],
    max_overlap_percent: float = 5.0
) -> Tuple[bool, Dict[str, dict]]:
    """Test that polygons from different plates don't overlap more than threshold.

    Args:
        polygons: List of combined polygons with source plate info
        max_overlap_percent: Maximum allowed percentage of polygons that can overlap

    Returns:
        (passed, results_dict) where results_dict has per-plate overlap statistics
    """
    # Group polygons by source plate
    by_plate: Dict[str, List[Tuple[int, CombinedPolygon]]] = {}
    for idx, poly in enumerate(polygons):
        if poly.source_plate not in by_plate:
            by_plate[poly.source_plate] = []
        by_plate[poly.source_plate].append((idx, poly))

    plates = list(by_plate.keys())
    if len(plates) < 2:
        return True, {"message": "Less than 2 plates, no cross-plate overlap possible"}

    results: Dict[str, dict] = {}
    all_passed = True

    # For each plate, count how many of its polygons overlap with any other plate's polygons
    for plate_id in plates:
        plate_polygons = by_plate[plate_id]
        other_polygons: List[Tuple[int, CombinedPolygon]] = []
        for other_plate in plates:
            if other_plate != plate_id:
                other_polygons.extend(by_plate[other_plate])

        if not other_polygons:
            results[plate_id] = {
                "total": len(plate_polygons),
                "overlapping": 0,
                "percent": 0.0,
                "passed": True
            }
            continue

        # Build spatial index for other plates' polygons
        other_shapely = []
        for idx, poly in other_polygons:
            if len(poly.points) >= 3:
                try:
                    sp = ShapelyPolygon(poly.points)
                    if sp.is_valid and sp.area > 0:
                        other_shapely.append(sp)
                except:
                    pass

        if not other_shapely:
            results[plate_id] = {
                "total": len(plate_polygons),
                "overlapping": 0,
                "percent": 0.0,
                "passed": True
            }
            continue

        tree = STRtree(other_shapely)

        # Count overlapping polygons from this plate
        overlapping_count = 0
        for idx, poly in plate_polygons:
            if len(poly.points) < 3:
                continue
            try:
                sp = ShapelyPolygon(poly.points)
                if not sp.is_valid or sp.area <= 0:
                    continue

                # Query spatial index for potential overlaps
                candidates = tree.query(sp)
                for candidate in candidates:
                    if sp.intersects(candidate):
                        intersection = sp.intersection(candidate)
                        # Only count as overlap if intersection area is significant (>1% of polygon area)
                        if intersection.area > sp.area * 0.01:
                            overlapping_count += 1
                            break
            except:
                pass

        percent = (overlapping_count / len(plate_polygons) * 100) if plate_polygons else 0
        passed = percent <= max_overlap_percent

        results[plate_id] = {
            "total": len(plate_polygons),
            "overlapping": overlapping_count,
            "percent": round(percent, 2),
            "passed": passed
        }

        if not passed:
            all_passed = False

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple segmentation SVG files into a single output SVG."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to segmentation.json file listing SVGs and metadata"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("combined_segmentation.svg"),
        help="Output SVG file path (default: combined_segmentation.svg)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=50.0,
        help="Margin around the bounding box (default: 50)"
    )
    parser.add_argument(
        "--background",
        type=Path,
        help="Background image (JP2/JPEG/PNG) to include with xlink:href"
    )
    parser.add_argument(
        "--skip-outlines",
        action="store_true",
        help="Skip block outline polygons (the largest polygon in each block)"
    )
    args = parser.parse_args()

    # Validate input file
    if not args.input_json.exists():
        print(f"Error: Input file not found: {args.input_json}")
        return 1

    # Load background image if specified
    background = None
    if args.background:
        print(f"Loading background image: {args.background}")
        background = load_background_image(args.background)
        if background:
            print(f"  Size: {background.width}x{background.height}, rotation: {background.angle:.1f}°")

    # Parse input
    print(f"Reading {args.input_json}")
    entries = parse_segmentation_json(args.input_json)
    print(f"Found {len(entries)} plates to combine\n")

    # Process and combine
    polygons, bbox = combine_plates(entries, skip_outlines=args.skip_outlines)

    if not polygons:
        print("Error: No polygons found to combine")
        return 1

    # Generate output
    generate_output_svg(polygons, bbox, args.output, args.margin, background)

    # Run overlap test
    print("\n--- Cross-plate overlap test ---")
    passed, results = test_cross_plate_overlap(polygons, max_overlap_percent=5.0)

    for plate_id, stats in sorted(results.items()):
        if isinstance(stats, dict) and "total" in stats:
            status = "PASS" if stats["passed"] else "FAIL"
            print(f"  {plate_id}: {stats['overlapping']}/{stats['total']} "
                  f"({stats['percent']:.1f}%) overlapping with other plates [{status}]")

    if passed:
        print("\nOverlap test PASSED: All plates have <5% cross-plate overlap")
        return 0
    else:
        print("\nOverlap test FAILED: Some plates have >5% cross-plate overlap")
        return 1


if __name__ == "__main__":
    exit(main())
