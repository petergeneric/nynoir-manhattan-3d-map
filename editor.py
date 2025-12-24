#!/usr/bin/env python3
"""
SVG Editor - Web-based GUI for editing block polygons.

This tool allows you to:
- View Stage 1 SVG outputs (plate.svg) with JPEG background
- View Stage 2 SVG outputs (segmentation.svg) with detailed polygons
- Select, delete, and merge block polygons
- Save edited SVGs (changes to segmentation.svg also update individual block files)

Usage:
    uv run python editor.py [--port PORT] [--output-dir DIR] [--media-dir DIR]
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_file, send_from_directory
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

app = Flask(__name__)

# Configuration (set via command line args)
OUTPUT_DIR = Path("output")
MEDIA_DIR = Path("../media/v1")

# SVG namespace
ATLAS_NS = "http://example.com/atlas"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', XLINK_NS)
ET.register_namespace('atlas', ATLAS_NS)


def parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points string into list of (x, y) tuples."""
    points = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            points.append((float(x), float(y)))
    return points


def points_to_svg_string(points: list[tuple[float, float]]) -> str:
    """Convert list of (x, y) tuples to SVG polygon points string."""
    return ' '.join(f"{int(x)},{int(y)}" for x, y in points)


def merge_polygons(polygon_points_list: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """
    Merge multiple polygons into their geometric union.

    Returns a list of polygon point lists (may return multiple if union is disjoint).
    """
    shapely_polygons = []

    for points in polygon_points_list:
        if len(points) >= 3:
            try:
                poly = Polygon(points)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid and not poly.is_empty:
                    shapely_polygons.append(poly)
            except Exception as e:
                print(f"Warning: Could not create polygon: {e}")
                continue

    if not shapely_polygons:
        return []

    # Compute union
    try:
        union_result = unary_union(shapely_polygons)
    except Exception as e:
        print(f"Warning: Union failed: {e}")
        return polygon_points_list  # Return original if union fails

    # Extract polygon(s) from result
    result_polygons = []

    if union_result.is_empty:
        return []
    elif isinstance(union_result, Polygon):
        # Single polygon result
        coords = list(union_result.exterior.coords)
        if coords:
            result_polygons.append(coords[:-1])  # Remove duplicate closing point
    elif isinstance(union_result, MultiPolygon):
        # Multiple disjoint polygons
        for poly in union_result.geoms:
            coords = list(poly.exterior.coords)
            if coords:
                result_polygons.append(coords[:-1])
    else:
        # Handle GeometryCollection or other types
        if hasattr(union_result, 'geoms'):
            for geom in union_result.geoms:
                if isinstance(geom, Polygon) and not geom.is_empty:
                    coords = list(geom.exterior.coords)
                    if coords:
                        result_polygons.append(coords[:-1])

    return result_polygons


def get_plates() -> list[dict]:
    """Get list of available plates with their SVG files."""
    plates = []

    if not OUTPUT_DIR.exists():
        return plates

    for volume_dir in sorted(OUTPUT_DIR.iterdir()):
        if not volume_dir.is_dir():
            continue

        volume = volume_dir.name

        for plate_dir in sorted(volume_dir.iterdir()):
            if not plate_dir.is_dir():
                continue

            plate_id = plate_dir.name
            plate_svg = plate_dir / "plate.svg"
            segmentation_svg = plate_dir / "segmentation.svg"

            # Try to find corresponding JPEG
            jpeg_path = MEDIA_DIR / f"{plate_id}.jpeg"
            has_jpeg = jpeg_path.exists()

            # Add plate.svg entry if exists
            if plate_svg.exists():
                plates.append({
                    "volume": volume,
                    "plate_id": plate_id,
                    "svg_path": str(plate_svg),
                    "jpeg_path": str(jpeg_path) if has_jpeg else None,
                    "has_jpeg": has_jpeg,
                    "svg_type": "plate"
                })

            # Add segmentation.svg entry if exists
            if segmentation_svg.exists():
                plates.append({
                    "volume": volume,
                    "plate_id": plate_id,
                    "svg_path": str(segmentation_svg),
                    "jpeg_path": str(jpeg_path) if has_jpeg else None,
                    "has_jpeg": has_jpeg,
                    "svg_type": "segmentation"
                })

    return plates


def load_svg_data(svg_path: str) -> dict:
    """Load plate.svg and extract polygon data."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get dimensions
    width = root.get('width', '0')
    height = root.get('height', '0')

    # Extract metadata
    metadata = {}
    metadata_elem = root.find(f'.//{{{SVG_NS}}}metadata')
    if metadata_elem is None:
        metadata_elem = root.find('.//metadata')

    if metadata_elem is not None:
        source_elem = metadata_elem.find(f'.//{{{ATLAS_NS}}}source')
        if source_elem is not None:
            for child in source_elem:
                tag = child.tag.replace(f'{{{ATLAS_NS}}}', '')
                metadata[tag] = child.text

    # Extract polygons
    polygons = []
    blocks_group = root.find(f'.//{{{SVG_NS}}}g[@id="blocks"]')
    if blocks_group is None:
        blocks_group = root.find('.//g[@id="blocks"]')

    if blocks_group is not None:
        for elem in blocks_group:
            tag = elem.tag.replace(f'{{{SVG_NS}}}', '')
            if tag == 'polygon':
                block_id = elem.get('data-block-id', '')
                points_str = elem.get('points', '')
                fill = elem.get('fill', 'rgba(100,100,200,0.3)')
                stroke = elem.get('stroke', 'rgb(100,100,200)')
                name = elem.get('data-name', '')

                poly_data = {
                    "id": block_id,
                    "points": points_str,
                    "fill": fill,
                    "stroke": stroke
                }
                if name:
                    poly_data["name"] = name

                polygons.append(poly_data)

    return {
        "width": width,
        "height": height,
        "metadata": metadata,
        "polygons": polygons,
        "svg_type": "plate"
    }


def load_segmentation_svg_data(svg_path: str) -> dict:
    """
    Load segmentation.svg and extract flattened polygon data.

    Blocks with multiple child polygons are treated as "container blocks" (cb-####).
    The container block is conceptually removed, leaving only the child polygons
    with names like "b-0001/p-001".

    Blocks with a single polygon are treated as top-level buildings and kept as-is.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get dimensions
    width = root.get('width', '0')
    height = root.get('height', '0')

    # Collect all polygons in a flattened structure
    polygons = []

    # Track container blocks (blocks with >1 polygon)
    container_blocks = []

    for elem in root:
        tag = elem.tag.replace(f'{{{SVG_NS}}}', '')
        if tag == 'g':
            group_id = elem.get('id', '')
            if group_id.startswith('block-') and group_id != 'block-outlines':
                # Extract block ID (e.g., "0001" from "block-0001")
                block_id = group_id.replace('block-', '')
                block_name = f"b-{block_id}"

                # Parse transform to get position
                transform = elem.get('transform', '')
                plate_x, plate_y = 0, 0
                if transform.startswith('translate('):
                    coords = transform.replace('translate(', '').replace(')', '')
                    parts = coords.split(',')
                    if len(parts) == 2:
                        plate_x = int(parts[0])
                        plate_y = int(parts[1])

                # Extract all polygons in this block
                block_polygons = []
                poly_counter = 1
                for child in elem:
                    child_tag = child.tag.replace(f'{{{SVG_NS}}}', '')
                    if child_tag == 'polygon':
                        points_str = child.get('points', '')
                        fill = child.get('fill', 'rgba(100,100,200,0.3)')
                        stroke = child.get('stroke', 'rgb(100,100,200)')

                        # Get or assign data-id for the polygon
                        poly_id = child.get('data-id', '')
                        if not poly_id:
                            poly_id = f"p-{poly_counter:03d}"
                        poly_counter += 1

                        block_polygons.append({
                            "points": points_str,
                            "fill": fill,
                            "stroke": stroke,
                            "poly_id": poly_id,
                            "block_id": block_id,
                            "plate_x": plate_x,
                            "plate_y": plate_y
                        })

                # Determine if this is a container block or a standalone building
                if len(block_polygons) > 1:
                    # Container block: rename to cb-#### and add all child polygons
                    container_blocks.append({
                        "block_id": block_id,
                        "container_name": f"cb-{block_id}",
                        "plate_x": plate_x,
                        "plate_y": plate_y,
                        "polygon_count": len(block_polygons)
                    })

                    # Add each polygon with its full name
                    for poly in block_polygons:
                        polygons.append({
                            "id": f"{block_name}/{poly['poly_id']}",
                            "name": f"{block_name}/{poly['poly_id']}",
                            "points": poly["points"],
                            "fill": poly["fill"],
                            "stroke": poly["stroke"],
                            "block_id": block_id,
                            "poly_id": poly["poly_id"],
                            "plate_x": plate_x,
                            "plate_y": plate_y,
                            "is_container_child": True
                        })
                elif len(block_polygons) == 1:
                    # Single polygon - treat as top-level building
                    poly = block_polygons[0]
                    polygons.append({
                        "id": block_name,
                        "name": block_name,
                        "points": poly["points"],
                        "fill": poly["fill"],
                        "stroke": poly["stroke"],
                        "block_id": block_id,
                        "poly_id": None,
                        "plate_x": plate_x,
                        "plate_y": plate_y,
                        "is_container_child": False
                    })

    # Also extract block outlines from plate.svg layer
    block_outlines = []
    outlines_group = root.find(f'.//{{{SVG_NS}}}g[@id="block-outlines"]')
    if outlines_group is None:
        outlines_group = root.find('.//g[@id="block-outlines"]')

    if outlines_group is not None:
        for elem in outlines_group:
            tag = elem.tag.replace(f'{{{SVG_NS}}}', '')
            if tag == 'polygon':
                points_str = elem.get('points', '')
                stroke = elem.get('stroke', 'rgb(102,179,92)')
                block_outlines.append({
                    "points": points_str,
                    "stroke": stroke
                })

    return {
        "width": width,
        "height": height,
        "polygons": polygons,
        "container_blocks": container_blocks,
        "block_outlines": block_outlines,
        "svg_type": "segmentation"
    }


def save_svg_data(svg_path: str, polygons: list[dict]) -> bool:
    """Save updated polygon data to SVG file."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Find blocks group
        blocks_group = root.find(f'.//{{{SVG_NS}}}g[@id="blocks"]')
        if blocks_group is None:
            blocks_group = root.find('.//g[@id="blocks"]')

        if blocks_group is None:
            return False

        # Clear existing content
        for child in list(blocks_group):
            blocks_group.remove(child)

        # Add updated polygons with renumbered IDs
        for idx, poly_data in enumerate(polygons, 1):
            block_id = f"{idx:04d}"

            # Get custom name or default to block ID format
            custom_name = poly_data.get('name', '')
            display_name = custom_name if custom_name else f"b-{block_id}"

            # Create polygon element
            polygon = ET.SubElement(blocks_group, 'polygon')
            polygon.set('points', poly_data['points'])
            polygon.set('fill', poly_data.get('fill', 'rgba(102,179,92,0.3)'))
            polygon.set('stroke', poly_data.get('stroke', 'rgb(102,179,92)'))
            polygon.set('stroke-width', '2')
            polygon.set('data-block-id', block_id)
            if custom_name:
                polygon.set('data-name', custom_name)

            # Calculate centroid for label
            points = parse_polygon_points(poly_data['points'])
            if points:
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)

                # Create text label
                text = ET.SubElement(blocks_group, 'text')
                text.set('x', str(int(cx)))
                text.set('y', str(int(cy)))
                text.set('text-anchor', 'middle')
                text.set('dominant-baseline', 'middle')
                text.set('class', 'block-label')
                text.set('fill', poly_data.get('stroke', 'rgb(102,179,92)'))
                text.set('stroke', 'white')
                text.set('stroke-width', '3')
                text.set('paint-order', 'stroke')
                text.text = display_name

        # Write back
        tree.write(svg_path, encoding='unicode', xml_declaration=True)
        return True

    except Exception as e:
        print(f"Error saving SVG: {e}")
        return False


def save_segmentation_svg_data(svg_path: str, polygons: list[dict]) -> dict:
    """
    Save updated polygon data to both segmentation.svg and individual block files.

    Accepts flattened polygon data and reconstructs the block structure for saving.

    Args:
        svg_path: Path to segmentation.svg
        polygons: List of polygon dicts with block_id, poly_id, plate_x, plate_y, etc.

    Returns:
        dict with success status and details
    """
    svg_dir = Path(svg_path).parent
    results = {"segmentation_svg": False, "block_files": {}}

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Group polygons by block_id to reconstruct block structure
        blocks_data = {}
        for poly in polygons:
            block_id = poly.get('block_id', '')
            if not block_id:
                continue

            if block_id not in blocks_data:
                blocks_data[block_id] = {
                    "plate_x": poly.get('plate_x', 0),
                    "plate_y": poly.get('plate_y', 0),
                    "polygons": []
                }

            blocks_data[block_id]["polygons"].append({
                "points": poly.get('points', ''),
                "fill": poly.get('fill', 'rgba(100,100,200,0.3)'),
                "stroke": poly.get('stroke', 'rgb(100,100,200)'),
                "data_id": poly.get('poly_id', '')
            })

        # Build a map of existing block groups
        block_groups = {}
        for elem in list(root):
            tag = elem.tag.replace(f'{{{SVG_NS}}}', '')
            if tag == 'g':
                group_id = elem.get('id', '')
                if group_id.startswith('block-') and group_id != 'block-outlines':
                    block_id = group_id.replace('block-', '')
                    block_groups[block_id] = elem

        # Process each block
        updated_block_ids = set()
        for block_id, block_data in blocks_data.items():
            updated_block_ids.add(block_id)
            plate_x = block_data.get('plate_x', 0)
            plate_y = block_data.get('plate_y', 0)
            block_polygons = block_data.get('polygons', [])

            # Update or create block group in segmentation.svg
            if block_id in block_groups:
                block_group = block_groups[block_id]
                # Clear existing polygons but keep comments
                for child in list(block_group):
                    child_tag = child.tag.replace(f'{{{SVG_NS}}}', '')
                    if child_tag == 'polygon':
                        block_group.remove(child)
            else:
                # Create new block group
                block_group = ET.SubElement(root, 'g')
                block_group.set('id', f'block-{block_id}')
                block_group.set('transform', f'translate({plate_x},{plate_y})')

            # Add polygons to block group with data-id attributes
            for poly in block_polygons:
                polygon = ET.SubElement(block_group, 'polygon')
                polygon.set('points', poly.get('points', ''))
                polygon.set('fill', poly.get('fill', 'rgba(100,100,200,0.3)'))
                polygon.set('stroke', poly.get('stroke', 'rgb(100,100,200)'))
                polygon.set('stroke-width', '1')
                if poly.get('data_id'):
                    polygon.set('data-id', poly['data_id'])

            # Also update individual block SVG file
            block_svg_path = svg_dir / f"b-{block_id}.svg"
            if block_svg_path.exists():
                try:
                    save_block_svg(str(block_svg_path), block_polygons)
                    results["block_files"][block_id] = True
                except Exception as e:
                    print(f"Error saving block {block_id}: {e}")
                    results["block_files"][block_id] = False

        # Remove block groups that are no longer in the data
        for block_id, group in block_groups.items():
            if block_id not in updated_block_ids:
                root.remove(group)
                # Optionally delete the individual block file
                block_svg_path = svg_dir / f"b-{block_id}.svg"
                if block_svg_path.exists():
                    try:
                        block_svg_path.unlink()
                        results["block_files"][block_id] = "deleted"
                    except Exception as e:
                        print(f"Error deleting block file {block_id}: {e}")

        # Write back segmentation.svg
        tree.write(svg_path, encoding='unicode', xml_declaration=True)
        results["segmentation_svg"] = True

    except Exception as e:
        print(f"Error saving segmentation SVG: {e}")
        results["error"] = str(e)

    return results


def save_block_svg(block_svg_path: str, polygons: list[dict]) -> bool:
    """
    Save updated polygon data to an individual block SVG file.

    Preserves the root element attributes (data-block-id, data-plate-x, etc.)
    and updates only the polygon content.
    """
    try:
        tree = ET.parse(block_svg_path)
        root = tree.getroot()

        # Remove existing polygons
        for child in list(root):
            child_tag = child.tag.replace(f'{{{SVG_NS}}}', '')
            if child_tag == 'polygon':
                root.remove(child)

        # Add updated polygons
        for poly in polygons:
            polygon = ET.SubElement(root, 'polygon')
            polygon.set('points', poly.get('points', ''))
            polygon.set('fill', poly.get('fill', 'rgba(100,100,200,0.3)'))
            polygon.set('stroke', poly.get('stroke', 'rgb(100,100,200)'))
            polygon.set('stroke-width', '1')
            # Preserve data-id if present
            if poly.get('data_id'):
                polygon.set('data-id', poly['data_id'])

        # Write back
        tree.write(block_svg_path, encoding='unicode', xml_declaration=True)
        return True

    except Exception as e:
        print(f"Error saving block SVG {block_svg_path}: {e}")
        return False


# Flask routes

@app.route('/')
def index():
    """Serve the main editor HTML page."""
    return send_file('editor.html')


@app.route('/api/plates')
def api_plates():
    """Get list of available plates."""
    return jsonify(get_plates())


@app.route('/api/plate/<volume>/<plate_id>')
def api_plate(volume, plate_id):
    """Get SVG data for a specific plate (plate.svg)."""
    svg_path = OUTPUT_DIR / volume / plate_id / "plate.svg"

    if not svg_path.exists():
        return jsonify({"error": "Plate not found"}), 404

    data = load_svg_data(str(svg_path))
    data["volume"] = volume
    data["plate_id"] = plate_id

    return jsonify(data)


@app.route('/api/segmentation/<volume>/<plate_id>')
def api_segmentation(volume, plate_id):
    """Get SVG data for a segmentation file (segmentation.svg)."""
    svg_path = OUTPUT_DIR / volume / plate_id / "segmentation.svg"

    if not svg_path.exists():
        return jsonify({"error": "Segmentation not found"}), 404

    data = load_segmentation_svg_data(str(svg_path))
    data["volume"] = volume
    data["plate_id"] = plate_id

    return jsonify(data)


@app.route('/api/plate/<volume>/<plate_id>/save', methods=['POST'])
def api_save_plate(volume, plate_id):
    """Save updated polygon data to plate.svg."""
    svg_path = OUTPUT_DIR / volume / plate_id / "plate.svg"

    if not svg_path.exists():
        return jsonify({"error": "Plate not found"}), 404

    data = request.json
    polygons = data.get('polygons', [])

    if save_svg_data(str(svg_path), polygons):
        return jsonify({"success": True, "polygon_count": len(polygons)})
    else:
        return jsonify({"error": "Failed to save"}), 500


@app.route('/api/segmentation/<volume>/<plate_id>/save', methods=['POST'])
def api_save_segmentation(volume, plate_id):
    """Save updated polygon data to segmentation.svg and individual block files."""
    svg_path = OUTPUT_DIR / volume / plate_id / "segmentation.svg"

    if not svg_path.exists():
        return jsonify({"error": "Segmentation not found"}), 404

    data = request.json
    polygons = data.get('polygons', [])

    results = save_segmentation_svg_data(str(svg_path), polygons)

    if results.get("segmentation_svg"):
        # Count unique blocks
        block_ids = set(p.get('block_id') for p in polygons if p.get('block_id'))
        return jsonify({
            "success": True,
            "block_count": len(block_ids),
            "polygon_count": len(polygons),
            "block_files": results.get("block_files", {})
        })
    else:
        return jsonify({"error": results.get("error", "Failed to save")}), 500


@app.route('/api/merge', methods=['POST'])
def api_merge():
    """Merge multiple polygons into one."""
    data = request.json
    polygon_points_strs = data.get('polygons', [])

    # Parse all polygon point strings
    polygon_points_list = []
    for points_str in polygon_points_strs:
        points = parse_polygon_points(points_str)
        if points:
            polygon_points_list.append(points)

    if len(polygon_points_list) < 2:
        return jsonify({"error": "Need at least 2 polygons to merge"}), 400

    # Merge polygons
    merged = merge_polygons(polygon_points_list)

    # Convert back to SVG strings
    result = [points_to_svg_string(pts) for pts in merged]

    return jsonify({
        "success": True,
        "merged_polygons": result,
        "input_count": len(polygon_points_list),
        "output_count": len(result)
    })


@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve media files (JPEG images)."""
    return send_from_directory(MEDIA_DIR, filename)


def main():
    global OUTPUT_DIR, MEDIA_DIR

    parser = argparse.ArgumentParser(description='SVG Editor for plate.svg and segmentation.svg files')
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory containing plate SVGs')
    parser.add_argument('--media-dir', type=str,
                        default='../media/v1',
                        help='Media directory containing JPEG files')

    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    MEDIA_DIR = Path(args.media_dir)

    print(f"Starting SVG Editor...")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Media dir: {MEDIA_DIR}")
    print(f"  URL: http://localhost:{args.port}")
    print()

    app.run(host='0.0.0.0', port=args.port, debug=True)


if __name__ == '__main__':
    main()
