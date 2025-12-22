#!/usr/bin/env python3
"""
Stage 1 SVG Editor - Web-based GUI for editing block polygons.

This tool allows you to:
- View Stage 1 SVG outputs with JPEG background
- Select, delete, and merge block polygons
- Save edited SVGs for Stage 2 processing

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

            plate_svg = plate_dir / "plate.svg"
            if plate_svg.exists():
                plate_id = plate_dir.name

                # Try to find corresponding JPEG
                jpeg_path = MEDIA_DIR / f"{plate_id}.jpeg"
                has_jpeg = jpeg_path.exists()

                plates.append({
                    "volume": volume,
                    "plate_id": plate_id,
                    "svg_path": str(plate_svg),
                    "jpeg_path": str(jpeg_path) if has_jpeg else None,
                    "has_jpeg": has_jpeg
                })

    return plates


def load_svg_data(svg_path: str) -> dict:
    """Load SVG and extract polygon data."""
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

                polygons.append({
                    "id": block_id,
                    "points": points_str,
                    "fill": fill,
                    "stroke": stroke
                })

    return {
        "width": width,
        "height": height,
        "metadata": metadata,
        "polygons": polygons
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

            # Create polygon element
            polygon = ET.SubElement(blocks_group, 'polygon')
            polygon.set('points', poly_data['points'])
            polygon.set('fill', poly_data.get('fill', 'rgba(102,179,92,0.3)'))
            polygon.set('stroke', poly_data.get('stroke', 'rgb(102,179,92)'))
            polygon.set('stroke-width', '2')
            polygon.set('data-block-id', block_id)

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
                text.text = f"b-{block_id}.svg"

        # Write back
        tree.write(svg_path, encoding='unicode', xml_declaration=True)
        return True

    except Exception as e:
        print(f"Error saving SVG: {e}")
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
    """Get SVG data for a specific plate."""
    svg_path = OUTPUT_DIR / volume / plate_id / "plate.svg"

    if not svg_path.exists():
        return jsonify({"error": "Plate not found"}), 404

    data = load_svg_data(str(svg_path))
    data["volume"] = volume
    data["plate_id"] = plate_id

    return jsonify(data)


@app.route('/api/plate/<volume>/<plate_id>/save', methods=['POST'])
def api_save_plate(volume, plate_id):
    """Save updated polygon data."""
    svg_path = OUTPUT_DIR / volume / plate_id / "plate.svg"

    if not svg_path.exists():
        return jsonify({"error": "Plate not found"}), 404

    data = request.json
    polygons = data.get('polygons', [])

    if save_svg_data(str(svg_path), polygons):
        return jsonify({"success": True, "polygon_count": len(polygons)})
    else:
        return jsonify({"error": "Failed to save"}), 500


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

    parser = argparse.ArgumentParser(description='Stage 1 SVG Editor')
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
