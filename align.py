#!/usr/bin/env python3
"""
Plate Alignment Editor - Web-based GUI for arranging atlas plates on a map.

This tool allows you to:
- View all available plates organized by volume
- Align plates to a reference map by setting rotation, scale, and position
- Save alignment metadata to plate.svg and .metadata.json files

Usage:
    uv run python align.py [--port PORT] [--output-dir DIR] [--media-dir DIR]
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory, Response
from PIL import Image, ImageDraw

app = Flask(__name__)

# Configuration (set via command line args)
OUTPUT_DIR = Path("output")
MEDIA_DIR = Path("../media/v1")
OVERVIEW_DIR = Path("../media/overview")

# SVG namespace
ATLAS_NS = "http://example.com/atlas"
SVG_NS = "http://www.w3.org/2000/svg"

ET.register_namespace('', SVG_NS)
ET.register_namespace('atlas', ATLAS_NS)

# Constants
REFERENCE_SCALE = 600  # Reference map scale: 1 inch = 600 feet
THUMBNAIL_SCALE = 0.1  # Thumbnails at 10% of original size


def get_volumes() -> list[dict]:
    """Get list of volumes with their plates and alignment status."""
    volumes = {}

    if not OUTPUT_DIR.exists():
        return []

    for volume_dir in sorted(OUTPUT_DIR.iterdir()):
        if not volume_dir.is_dir():
            continue

        volume_name = volume_dir.name
        if volume_name not in volumes:
            volumes[volume_name] = {
                "name": volume_name,
                "plates": []
            }

        for plate_dir in sorted(volume_dir.iterdir()):
            if not plate_dir.is_dir():
                continue

            plate_id = plate_dir.name
            plate_svg = plate_dir / "plate.svg"

            if not plate_svg.exists():
                continue

            # Get JP2 path from SVG metadata (ensures correct volume path)
            jp2_path = get_jp2_path_from_svg(plate_svg)
            has_jp2 = jp2_path is not None and jp2_path.exists()

            # Check for JPEG in same directory as JP2
            has_jpeg = False
            if jp2_path is not None:
                jpeg_path = jp2_path.parent / f"{plate_id}.jpeg"
                has_jpeg = jpeg_path.exists()

            # Load alignment data
            alignment = load_plate_alignment(plate_svg, plate_id)

            # has_thumb is True if we have a JP2 (thumbnails are generated on demand)
            volumes[volume_name]["plates"].append({
                "plate_id": plate_id,
                "has_svg": True,
                "has_jpeg": has_jpeg,
                "has_thumb": has_jp2,  # Can generate thumbnail if JP2 exists
                "has_alignment": alignment.get("has_alignment", False),
                "alignment": alignment if alignment.get("has_alignment") else None
            })

    return list(volumes.values())


def load_plate_alignment(plate_svg_path: Path, plate_id: str) -> dict:
    """Load alignment metadata from plate.svg or fallback JSON."""
    result = {"has_alignment": False}

    # Try plate.svg first
    if plate_svg_path.exists():
        try:
            tree = ET.parse(plate_svg_path)
            root = tree.getroot()

            # Find metadata/source element
            metadata_elem = root.find(f'.//{{{SVG_NS}}}metadata')
            if metadata_elem is None:
                metadata_elem = root.find('.//metadata')

            if metadata_elem is not None:
                source_elem = metadata_elem.find(f'.//{{{ATLAS_NS}}}source')
                if source_elem is not None:
                    angle_elem = source_elem.find(f'{{{ATLAS_NS}}}angle')
                    scale_elem = source_elem.find(f'{{{ATLAS_NS}}}scale')
                    pos_x_elem = source_elem.find(f'{{{ATLAS_NS}}}pos-x')
                    pos_y_elem = source_elem.find(f'{{{ATLAS_NS}}}pos-y')

                    if all([angle_elem is not None and angle_elem.text,
                            scale_elem is not None and scale_elem.text,
                            pos_x_elem is not None and pos_x_elem.text,
                            pos_y_elem is not None and pos_y_elem.text]):
                        return {
                            "angle": float(angle_elem.text),
                            "scale": float(scale_elem.text),
                            "pos": [float(pos_x_elem.text), float(pos_y_elem.text)],
                            "has_alignment": True
                        }
        except Exception as e:
            print(f"Error reading plate.svg: {e}")

    # Try fallback JSON
    json_path = MEDIA_DIR / f"{plate_id}.metadata.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
                if "angle" in data and "scale" in data and "pos" in data:
                    return {
                        "angle": data["angle"],
                        "scale": data["scale"],
                        "pos": data["pos"],
                        "has_alignment": True
                    }
        except Exception as e:
            print(f"Error reading metadata.json: {e}")

    return result


def save_plate_alignment(volume: str, plate_id: str, alignment: dict) -> dict:
    """Save alignment metadata to plate.svg and fallback JSON."""
    results = {"svg_saved": False, "json_saved": False}

    plate_svg_path = OUTPUT_DIR / volume / plate_id / "plate.svg"

    # Update plate.svg
    if plate_svg_path.exists():
        try:
            tree = ET.parse(plate_svg_path)
            root = tree.getroot()

            # Find or create metadata element
            metadata_elem = root.find(f'.//{{{SVG_NS}}}metadata')
            if metadata_elem is None:
                metadata_elem = root.find('.//metadata')
            if metadata_elem is None:
                metadata_elem = ET.SubElement(root, 'metadata')

            # Find or create source element
            source_elem = metadata_elem.find(f'.//{{{ATLAS_NS}}}source')
            if source_elem is None:
                source_elem = ET.SubElement(metadata_elem, f'{{{ATLAS_NS}}}source')

            # Add or update alignment elements
            for tag, value in [
                ('angle', alignment['angle']),
                ('scale', alignment['scale']),
                ('pos-x', alignment['pos'][0]),
                ('pos-y', alignment['pos'][1])
            ]:
                elem = source_elem.find(f'{{{ATLAS_NS}}}{tag}')
                if elem is None:
                    elem = ET.SubElement(source_elem, f'{{{ATLAS_NS}}}{tag}')
                elem.text = str(value)

            tree.write(plate_svg_path, encoding='unicode', xml_declaration=True)
            results["svg_saved"] = True
        except Exception as e:
            print(f"Error saving to plate.svg: {e}")
            results["svg_error"] = str(e)

    # Also save fallback JSON
    json_path = MEDIA_DIR / f"{plate_id}.metadata.json"
    try:
        json_data = {
            "angle": alignment['angle'],
            "scale": alignment['scale'],
            "pos": alignment['pos']
        }
        if 'calibration' in alignment:
            json_data['calibration'] = alignment['calibration']

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        results["json_saved"] = True
    except Exception as e:
        print(f"Error saving metadata.json: {e}")
        results["json_error"] = str(e)

    return results


# Thumbnail generation helpers

def get_jp2_path_from_svg(plate_svg_path: Path) -> Path | None:
    """Extract JP2 path from plate.svg metadata."""
    if not plate_svg_path.exists():
        return None

    try:
        tree = ET.parse(plate_svg_path)
        root = tree.getroot()

        # Find metadata/source element
        metadata_elem = root.find(f'.//{{{SVG_NS}}}metadata')
        if metadata_elem is None:
            metadata_elem = root.find('.//metadata')

        if metadata_elem is not None:
            source_elem = metadata_elem.find(f'.//{{{ATLAS_NS}}}source')
            if source_elem is not None:
                jp2_elem = source_elem.find(f'{{{ATLAS_NS}}}jp2-path')
                if jp2_elem is not None and jp2_elem.text:
                    return Path(jp2_elem.text)
    except Exception as e:
        print(f"Error reading JP2 path from plate.svg: {e}")

    return None


def extract_polygons_from_svg(plate_svg_path: Path) -> list[list[tuple[float, float]]]:
    """Extract polygon coordinates from plate.svg."""
    polygons = []

    if not plate_svg_path.exists():
        return polygons

    try:
        tree = ET.parse(plate_svg_path)
        root = tree.getroot()

        # Find all polygon elements
        for polygon_elem in root.findall(f'.//{{{SVG_NS}}}polygon'):
            points_str = polygon_elem.get('points', '')
            if not points_str:
                continue

            # Parse points: "x1,y1 x2,y2 x3,y3 ..."
            coords = []
            for point in points_str.strip().split():
                try:
                    x, y = point.split(',')
                    coords.append((float(x), float(y)))
                except ValueError:
                    continue

            if len(coords) >= 3:
                polygons.append(coords)
    except Exception as e:
        print(f"Error extracting polygons from plate.svg: {e}")

    return polygons


def create_alpha_mask(polygons: list[list[tuple[float, float]]], width: int, height: int) -> Image.Image:
    """Create an alpha mask from polygon coordinates."""
    # Create a blank mask (all transparent)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Fill each polygon with white (opaque)
    for coords in polygons:
        if len(coords) >= 3:
            draw.polygon(coords, fill=255)

    return mask


def generate_thumbnail(plate_svg_path: Path, cache_path: Path) -> bool:
    """Generate a thumbnail with polygon alpha mask and save to cache."""
    # Get JP2 path from SVG
    jp2_path = get_jp2_path_from_svg(plate_svg_path)
    if jp2_path is None or not jp2_path.exists():
        print(f"JP2 file not found: {jp2_path}")
        return False

    try:
        # Load the JP2 image
        img = Image.open(jp2_path)
        width, height = img.size

        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Extract polygons from SVG
        polygons = extract_polygons_from_svg(plate_svg_path)

        if polygons:
            # Create alpha mask
            mask = create_alpha_mask(polygons, width, height)

            # Apply the mask as alpha channel
            img.putalpha(mask)

        # Scale down to 10%
        new_width = int(width * 0.1)
        new_height = int(height * 0.1)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG with transparency
        img.save(cache_path, 'PNG')
        print(f"Generated thumbnail: {cache_path}")
        return True

    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return False


# Flask routes

@app.route('/')
def index():
    """Serve the main editor HTML page."""
    return Response(HTML_TEMPLATE, mimetype='text/html')


@app.route('/api/volumes')
def api_volumes():
    """Get list of volumes with their plates."""
    return jsonify({"volumes": get_volumes()})


@app.route('/api/plate/<volume>/<plate_id>/save', methods=['POST'])
def api_save_plate(volume, plate_id):
    """Save alignment metadata for a plate."""
    data = request.json

    if not all(k in data for k in ['angle', 'scale', 'pos']):
        return jsonify({"error": "Missing required fields"}), 400

    alignment = {
        "angle": data['angle'],
        "scale": data['scale'],
        "pos": data['pos']
    }
    if 'calibration' in data:
        alignment['calibration'] = data['calibration']

    results = save_plate_alignment(volume, plate_id, alignment)

    if results.get("svg_saved") or results.get("json_saved"):
        return jsonify({"success": True, **results})
    else:
        return jsonify({"error": "Failed to save", **results}), 500


@app.route('/api/plate/<volume>/<plate_id>/thumb')
def api_plate_thumbnail(volume, plate_id):
    """Generate and serve a plate thumbnail with polygon alpha mask."""
    # Construct paths
    plate_svg_path = OUTPUT_DIR / volume / plate_id / "plate.svg"

    if not plate_svg_path.exists():
        return jsonify({"error": "plate.svg not found"}), 404

    # Get JP2 path from SVG metadata
    jp2_path = get_jp2_path_from_svg(plate_svg_path)
    if jp2_path is None:
        return jsonify({"error": "JP2 path not found in SVG metadata"}), 404

    if not jp2_path.exists():
        return jsonify({"error": f"JP2 file not found: {jp2_path}"}), 404

    # Cache path is alongside the JP2 file
    cache_path = jp2_path.parent / f"{plate_id}.thumb.png"

    # Check if cache is valid (exists and newer than SVG)
    regenerate = True
    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        svg_mtime = plate_svg_path.stat().st_mtime
        if cache_mtime > svg_mtime:
            regenerate = False

    # Generate if needed
    if regenerate:
        if not generate_thumbnail(plate_svg_path, cache_path):
            return jsonify({"error": "Failed to generate thumbnail"}), 500

    # Serve the cached file
    return send_file(cache_path, mimetype='image/png')


@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve media files (JPEG images, thumbnails)."""
    return send_from_directory(MEDIA_DIR, filename)


@app.route('/media/overview/<path:filename>')
def serve_overview(filename):
    """Serve overview/reference map files."""
    return send_from_directory(OVERVIEW_DIR, filename)


@app.route('/api/reference-map')
def api_get_reference_map():
    """Get reference map alignment metadata."""
    json_path = OVERVIEW_DIR / "section1.metadata.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
                return jsonify({"has_alignment": True, "angle": data.get("angle", 0)})
        except Exception as e:
            print(f"Error reading reference map metadata: {e}")
    return jsonify({"has_alignment": False})


@app.route('/api/reference-map/save', methods=['POST'])
def api_save_reference_map():
    """Save reference map alignment metadata."""
    data = request.json

    if 'angle' not in data:
        return jsonify({"error": "Missing angle field"}), 400

    json_path = OVERVIEW_DIR / "section1.metadata.json"
    try:
        with open(json_path, 'w') as f:
            json.dump({"angle": data['angle']}, f, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# HTML Template (embedded)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plate Alignment Editor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: #16213e;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            border-bottom: 1px solid #0f3460;
        }

        header h1 {
            font-size: 18px;
            font-weight: 500;
            color: #e94560;
        }

        .status {
            font-size: 13px;
            color: #888;
            padding: 0 10px;
        }

        .toolbar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: auto;
        }

        .toolbar button {
            background: #0f3460;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }

        .toolbar button:hover:not(:disabled) {
            background: #1a4a7a;
        }

        .toolbar button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .toolbar button.primary {
            background: #0a8754;
        }

        .toolbar button.primary:hover:not(:disabled) {
            background: #0fa968;
        }

        main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        .sidebar {
            width: 280px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-header {
            padding: 12px 16px;
            border-bottom: 1px solid #0f3460;
            font-size: 14px;
            font-weight: 500;
            color: #e94560;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .align-map-btn {
            background: #d4a017;
            color: #000;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            font-weight: 500;
        }

        .align-map-btn:hover {
            background: #e9b82a;
        }

        .align-map-btn.aligned {
            background: #0a8754;
            color: #fff;
        }

        .volume-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }

        .volume-group {
            margin-bottom: 8px;
        }

        .volume-header {
            padding: 8px 12px;
            background: #0f3460;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
        }

        .volume-header:hover {
            background: #1a4a7a;
        }

        .volume-header .arrow {
            transition: transform 0.2s;
        }

        .volume-header.collapsed .arrow {
            transform: rotate(-90deg);
        }

        .plate-list {
            padding-left: 16px;
            margin-top: 4px;
        }

        .plate-list.collapsed {
            display: none;
        }

        .plate-item {
            padding: 8px 12px;
            background: #1a1a2e;
            border-radius: 4px;
            margin-bottom: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.15s;
        }

        .plate-item:hover {
            background: #252542;
        }

        .plate-item.active {
            background: #2a4a6a;
            outline: 2px solid #4a8aca;
        }

        .plate-item .status-icon {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .plate-item .status-icon.aligned {
            background: #0a8754;
        }

        .plate-item .status-icon.pending {
            background: #d4a017;
        }

        .plate-item .plate-name {
            flex: 1;
        }

        .calibration-panel {
            border-top: 1px solid #0f3460;
            padding: 16px;
        }

        .calibration-panel h3 {
            font-size: 13px;
            color: #888;
            margin-bottom: 12px;
        }

        .calibration-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }

        .calibration-row label {
            color: #888;
        }

        .calibration-row .value {
            color: #fff;
            font-family: monospace;
        }

        .calibration-row .value.editable {
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 3px;
            transition: background 0.15s;
        }

        .calibration-row .value.editable:hover {
            background: #0f3460;
        }

        .calibration-row .value-input {
            width: 80px;
            background: #0f3460;
            border: 1px solid #4a8aca;
            color: #fff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 13px;
            text-align: right;
        }

        .calibration-row .value-input:focus {
            outline: none;
            border-color: #6ab0f0;
        }

        .canvas-container {
            flex: 1;
            overflow: hidden;
            position: relative;
            background: #111;
        }

        #canvas-wrapper {
            width: 100%;
            height: 100%;
            overflow: auto;
            cursor: grab;
        }

        #canvas-wrapper.dragging {
            cursor: grabbing;
        }

        #canvas-content {
            position: relative;
            transform-origin: 0 0;
        }

        #reference-map {
            display: block;
        }

        .plate-thumbnail {
            position: absolute;
            transform-origin: center center;
            pointer-events: auto;
            cursor: move;
            transition: opacity 0.15s;
        }

        .plate-thumbnail.dragging {
            opacity: 0.5;
        }

        .plate-thumbnail.locked {
            cursor: default;
            pointer-events: none;
        }

        .zoom-controls {
            position: absolute;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 8px;
            background: rgba(22, 33, 62, 0.9);
            padding: 8px;
            border-radius: 6px;
        }

        .zoom-controls button {
            background: #0f3460;
            color: #fff;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 18px;
        }

        .zoom-controls button:hover {
            background: #1a4a7a;
        }

        .zoom-level {
            display: flex;
            align-items: center;
            padding: 0 10px;
            font-size: 13px;
            color: #888;
        }

        .help-text {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(22, 33, 62, 0.9);
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 12px;
            color: #888;
            line-height: 1.6;
            max-width: 400px;
        }

        .help-text kbd {
            background: #0f3460;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            color: #ccc;
        }

        /* Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            display: none;
            flex-direction: column;
        }

        .modal-overlay.active {
            display: flex;
        }

        .modal-header {
            background: #16213e;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            border-bottom: 1px solid #0f3460;
        }

        .modal-header h2 {
            font-size: 16px;
            font-weight: 500;
            color: #e94560;
        }

        .modal-header .step-indicator {
            font-size: 14px;
            color: #888;
        }

        .modal-header .instructions {
            flex: 1;
            font-size: 14px;
            color: #fff;
        }

        .modal-header button {
            background: #8b0000;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }

        .modal-header button:hover {
            background: #a52a2a;
        }

        .modal-content {
            flex: 1;
            overflow: auto;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .modal-image-container {
            position: relative;
            display: inline-block;
        }

        .modal-image {
            max-width: calc(100vw - 40px);
            max-height: calc(100vh - 100px);  /* Account for header + padding */
            cursor: crosshair;
        }

        .calibration-marker {
            position: absolute;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #e94560;
            border: 2px solid #fff;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: #fff;
            pointer-events: none;
        }

        .calibration-line {
            position: absolute;
            background: #e94560;
            height: 2px;
            transform-origin: left center;
            pointer-events: none;
        }

        /* Input panel for scale */
        .input-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #16213e;
            padding: 24px;
            border-radius: 8px;
            border: 1px solid #0f3460;
            z-index: 1001;
            display: none;
            min-width: 300px;
        }

        .input-panel.active {
            display: block;
        }

        .input-panel h3 {
            font-size: 16px;
            margin-bottom: 16px;
            color: #e94560;
        }

        .input-group {
            margin-bottom: 16px;
        }

        .input-group label {
            display: block;
            font-size: 13px;
            color: #888;
            margin-bottom: 4px;
        }

        .input-group input {
            width: 100%;
            background: #0f3460;
            border: 1px solid #1a1a2e;
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
        }

        .input-group input:focus {
            outline: none;
            border-color: #4a8aca;
        }

        .input-panel-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }

        .input-panel-buttons button {
            background: #0f3460;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }

        .input-panel-buttons button.primary {
            background: #0a8754;
        }

        .input-panel-buttons button:hover {
            opacity: 0.9;
        }

        .toast {
            position: fixed;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: #16213e;
            color: #fff;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 14px;
            z-index: 2000;
            animation: fadeInOut 3s ease-in-out;
            border: 1px solid #0f3460;
        }

        .toast.success {
            border-color: #0a8754;
        }

        .toast.error {
            border-color: #8b0000;
        }

        @keyframes fadeInOut {
            0% { opacity: 0; transform: translateX(-50%) translateY(20px); }
            15% { opacity: 1; transform: translateX(-50%) translateY(0); }
            85% { opacity: 1; transform: translateX(-50%) translateY(0); }
            100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
        }
    </style>
</head>
<body>
    <header>
        <h1>Plate Alignment Editor</h1>
        <div class="status" id="status">Select a plate to align</div>
        <div class="toolbar">
            <button id="btn-save" class="primary" disabled>Save Alignment</button>
        </div>
    </header>

    <main>
        <div class="sidebar">
            <div class="sidebar-header">
                Volumes & Plates
                <button id="btn-align-map" class="align-map-btn" title="Align Reference Map">Align Map</button>
            </div>
            <div class="volume-list" id="volume-list"></div>
            <div class="calibration-panel" id="calibration-panel" style="display: none;">
                <h3>Current Alignment</h3>
                <div class="calibration-row">
                    <label>Angle:</label>
                    <span class="value editable" id="cal-angle" title="Click to edit">--</span>
                </div>
                <div class="calibration-row">
                    <label>Scale:</label>
                    <span class="value editable" id="cal-scale" title="Click to edit">--</span>
                </div>
                <div class="calibration-row">
                    <label>Position:</label>
                    <span class="value" id="cal-pos">--</span>
                </div>
            </div>
        </div>

        <div class="canvas-container">
            <div id="canvas-wrapper">
                <div id="canvas-content">
                    <img id="reference-map" src="/media/overview/section1.jpeg" alt="Reference Map" draggable="false">
                    <div id="thumbnails-layer"></div>
                </div>
            </div>

            <div class="zoom-controls">
                <button id="btn-zoom-out">-</button>
                <div class="zoom-level"><span id="zoom-level">100%</span></div>
                <button id="btn-zoom-in">+</button>
                <button id="btn-zoom-fit">Fit</button>
            </div>

            <div class="help-text" id="help-text">
                Click a <span style="color: #d4a017;">pending plate</span> to begin alignment.
                <br><kbd>[</kbd> / <kbd>]</kbd> Scale &nbsp; <kbd>-</kbd> / <kbd>=</kbd> Rotate
                <br><kbd>,</kbd> / <kbd>.</kbd> Opacity &nbsp; <kbd>Ctrl+S</kbd> Save
            </div>
        </div>
    </main>

    <!-- Modal for calibration -->
    <div class="modal-overlay" id="modal">
        <div class="modal-header">
            <h2 id="modal-title">Calibration</h2>
            <span class="step-indicator" id="step-indicator">Step 1 of 4</span>
            <span class="instructions" id="modal-instructions">Click the SOUTH point on the compass</span>
            <button id="btn-cancel-modal">Cancel</button>
        </div>
        <div class="modal-content" id="modal-content">
            <div class="modal-image-container" id="modal-image-container">
                <img class="modal-image" id="modal-image" alt="Plate Image">
            </div>
        </div>
    </div>

    <!-- Input panel for scale values -->
    <div class="input-panel" id="input-panel">
        <h3>Enter Scale Parameters</h3>
        <div class="input-group">
            <label>Distance between points (feet):</label>
            <input type="number" id="input-distance" value="300" min="1">
        </div>
        <div class="input-group">
            <label>Plate scale (1 inch = X feet):</label>
            <input type="number" id="input-scale" value="80" min="1">
        </div>
        <div class="input-panel-buttons">
            <button id="btn-cancel-input">Cancel</button>
            <button id="btn-confirm-input" class="primary">Confirm</button>
        </div>
    </div>

    <script>
        // Constants
        const REFERENCE_SCALE = 600;  // Reference map: 1 inch = 600 feet
        const THUMBNAIL_SCALE = 10.0;  // Scale factor for thumbnails (already 10% size)

        // State
        let volumes = [];
        let currentPlate = null;
        let workflowStep = 0;  // 0=idle, 1-6=plate alignment, 10-11=reference map alignment
        let referenceMapAngle = 0;
        let referenceMapAligned = false;
        let calibrationData = {
            anglePoints: [],      // [{x, y}, {x, y}] south->north
            scalePoints: [],      // [{x, y}, {x, y}] for distance
            realDistanceFeet: 300,
            imageScaleFeetPerInch: 80,
            pixelDistance: 0,
            angle: 0,
            scaleFactor: 0,
            position: {x: 0, y: 0},
            opacity: 1.0
        };
        let zoom = 0.5;
        let isDragging = false;
        let dragStart = {x: 0, y: 0};
        let dragOffset = {x: 0, y: 0};

        // DOM elements
        const volumeList = document.getElementById('volume-list');
        const canvasWrapper = document.getElementById('canvas-wrapper');
        const canvasContent = document.getElementById('canvas-content');
        const referenceMap = document.getElementById('reference-map');
        const thumbnailsLayer = document.getElementById('thumbnails-layer');
        const btnSave = document.getElementById('btn-save');
        const statusEl = document.getElementById('status');
        const helpText = document.getElementById('help-text');
        const calibrationPanel = document.getElementById('calibration-panel');
        const btnAlignMap = document.getElementById('btn-align-map');

        // Modal elements
        const modal = document.getElementById('modal');
        const modalTitle = document.getElementById('modal-title');
        const modalInstructions = document.getElementById('modal-instructions');
        const stepIndicator = document.getElementById('step-indicator');
        const modalImage = document.getElementById('modal-image');
        const modalImageContainer = document.getElementById('modal-image-container');
        const btnCancelModal = document.getElementById('btn-cancel-modal');

        // Input panel elements
        const inputPanel = document.getElementById('input-panel');
        const inputDistance = document.getElementById('input-distance');
        const inputScale = document.getElementById('input-scale');
        const btnCancelInput = document.getElementById('btn-cancel-input');
        const btnConfirmInput = document.getElementById('btn-confirm-input');

        // Initialize
        async function init() {
            await loadReferenceMapAlignment();
            await loadVolumes();
            setupEventListeners();
            fitToView();
        }

        async function loadReferenceMapAlignment() {
            try {
                const response = await fetch('/api/reference-map');
                const data = await response.json();
                if (data.has_alignment) {
                    referenceMapAngle = data.angle;
                    referenceMapAligned = true;
                    applyReferenceMapRotation();
                    updateAlignMapButton();
                } else {
                    // Prompt user to align reference map
                    referenceMapAligned = false;
                    showToast('Reference map needs alignment - click "Align Map" in sidebar', 'warning');
                }
            } catch (error) {
                console.error('Failed to load reference map alignment:', error);
            }
        }

        function updateAlignMapButton() {
            if (referenceMapAligned) {
                btnAlignMap.classList.add('aligned');
                btnAlignMap.textContent = 'Map Aligned';
            } else {
                btnAlignMap.classList.remove('aligned');
                btnAlignMap.textContent = 'Align Map';
            }
        }

        function applyReferenceMapRotation() {
            if (referenceMapAngle !== 0) {
                referenceMap.style.transformOrigin = 'center center';
                referenceMap.style.transform = `rotate(${referenceMapAngle}deg)`;
            }
        }

        function startReferenceMapAlignment() {
            workflowStep = 10;  // Reference map alignment mode
            calibrationData.anglePoints = [];

            // Show modal with reference map image
            modalTitle.textContent = 'Align Reference Map';
            modalImage.src = '/media/overview/section1.jpeg';
            modal.classList.add('active');

            stepIndicator.textContent = 'Step 1 of 2';
            modalInstructions.textContent = 'Click the SOUTH point on the compass';
            statusEl.textContent = 'Aligning reference map...';
        }

        async function saveReferenceMapAlignment() {
            try {
                const response = await fetch('/api/reference-map/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ angle: referenceMapAngle })
                });

                const result = await response.json();
                if (result.success) {
                    referenceMapAligned = true;
                    applyReferenceMapRotation();
                    updateAlignMapButton();
                    showToast('Reference map alignment saved', 'success');
                } else {
                    showToast('Failed to save: ' + result.error, 'error');
                }
            } catch (error) {
                showToast('Failed to save: ' + error.message, 'error');
            }
        }

        async function loadVolumes() {
            try {
                const response = await fetch('/api/volumes');
                const data = await response.json();
                volumes = data.volumes;
                renderVolumeList();
                renderAlignedThumbnails();
            } catch (error) {
                showToast('Failed to load volumes', 'error');
            }
        }

        function renderVolumeList() {
            volumeList.innerHTML = '';

            volumes.forEach(volume => {
                const group = document.createElement('div');
                group.className = 'volume-group';

                const header = document.createElement('div');
                header.className = 'volume-header';
                header.innerHTML = `<span class="arrow">▼</span> ${volume.name} (${volume.plates.length})`;

                header.addEventListener('click', () => {
                    header.classList.toggle('collapsed');
                    plateList.classList.toggle('collapsed');
                });

                const plateList = document.createElement('div');
                plateList.className = 'plate-list';

                volume.plates.forEach(plate => {
                    const item = document.createElement('div');
                    item.className = 'plate-item';
                    item.dataset.volume = volume.name;
                    item.dataset.plateId = plate.plate_id;

                    const statusClass = plate.has_alignment ? 'aligned' : 'pending';
                    item.innerHTML = `
                        <div class="status-icon ${statusClass}"></div>
                        <span class="plate-name">${plate.plate_id}</span>
                    `;

                    item.addEventListener('click', () => selectPlate(volume.name, plate));
                    plateList.appendChild(item);
                });

                group.appendChild(header);
                group.appendChild(plateList);
                volumeList.appendChild(group);
            });
        }

        function renderAlignedThumbnails() {
            thumbnailsLayer.innerHTML = '';

            volumes.forEach(volume => {
                volume.plates.forEach(plate => {
                    if (plate.has_alignment && plate.alignment && plate.has_thumb) {
                        createThumbnail(volume.name, plate);
                    }
                });
            });
        }

        function createThumbnail(volumeName, plate) {
            const img = document.createElement('img');
            img.className = 'plate-thumbnail';
            if (plate.plate_id !== currentPlate?.plate_id || workflowStep === 0) {
                img.classList.add('locked');
            }
            img.id = `thumb-${plate.plate_id}`;
            img.src = `/api/plate/${volumeName}/${plate.plate_id}/thumb`;
            img.dataset.volume = volumeName;
            img.dataset.plateId = plate.plate_id;

            img.onload = () => {
                updateThumbnailTransform(img, plate.alignment);
            };

            thumbnailsLayer.appendChild(img);
            return img;
        }

        function updateThumbnailTransform(img, alignment) {
            if (!alignment) return;

            const { angle, scale, pos } = alignment;
            const width = img.naturalWidth || 498;
            const height = img.naturalHeight || 336;

            // Transform: translate to center at pos, then rotate, then scale
            // We apply translation to position, then use transform for rotation/scale
            const halfW = (width * scale) / 2;
            const halfH = (height * scale) / 2;

            img.style.left = `${pos[0]}px`;
            img.style.top = `${pos[1]}px`;
            img.style.transform = `translate(-50%, -50%) rotate(${angle}deg) scale(${scale})`;
        }

        function selectPlate(volumeName, plate) {
            // Deselect previous
            document.querySelectorAll('.plate-item.active').forEach(el => el.classList.remove('active'));

            // Select new
            const item = document.querySelector(`.plate-item[data-plate-id="${plate.plate_id}"]`);
            if (item) item.classList.add('active');

            currentPlate = { ...plate, volume: volumeName };

            if (plate.has_alignment) {
                // Show existing alignment
                showAlignmentInfo(plate.alignment);
                calibrationData = {
                    ...calibrationData,
                    angle: plate.alignment.angle,
                    scaleFactor: plate.alignment.scale,
                    position: { x: plate.alignment.pos[0], y: plate.alignment.pos[1] }
                };
                workflowStep = 6;  // Fine adjust mode
                unlockCurrentThumbnail();
                updateHelpText('Drag to adjust. <kbd>[</kbd>/<kbd>]</kbd> Scale <kbd>-</kbd>/<kbd>=</kbd> Rotate <kbd>,</kbd>/<kbd>.</kbd> Opacity');
                btnSave.disabled = false;
            } else {
                // Start alignment workflow
                startAlignment(volumeName, plate);
            }
        }

        function startAlignment(volumeName, plate) {
            currentPlate = { ...plate, volume: volumeName };
            calibrationData = {
                anglePoints: [],
                scalePoints: [],
                realDistanceFeet: 300,
                imageScaleFeetPerInch: 80,
                pixelDistance: 0,
                angle: 0,
                scaleFactor: 0,
                position: {x: 0, y: 0},
                opacity: 1.0
            };
            workflowStep = 1;

            // Clear any existing markers
            clearMarkers();

            // Show modal with plate image
            modalTitle.textContent = `Align ${plate.plate_id}`;
            modalImage.src = `/media/${plate.plate_id}.jpeg`;
            modal.classList.add('active');

            updateModalStep();
        }

        function updateModalStep() {
            switch (workflowStep) {
                case 1:
                    stepIndicator.textContent = 'Step 1 of 4';
                    modalInstructions.textContent = 'Click the SOUTH point on the compass';
                    break;
                case 2:
                    stepIndicator.textContent = 'Step 2 of 4';
                    modalInstructions.textContent = 'Click the NORTH point on the compass';
                    break;
                case 3:
                    stepIndicator.textContent = 'Step 3 of 4';
                    modalInstructions.textContent = 'Click first point for scale measurement';
                    break;
                case 4:
                    stepIndicator.textContent = 'Step 4 of 4';
                    modalInstructions.textContent = 'Click second point for scale measurement';
                    break;
            }
        }

        function handleModalClick(e) {
            // Handle both plate alignment (1-4) and reference map alignment (10-11)
            if (!((workflowStep >= 1 && workflowStep <= 4) || (workflowStep >= 10 && workflowStep <= 11))) return;

            const rect = modalImage.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Convert to image coordinates
            const imgX = (x / rect.width) * modalImage.naturalWidth;
            const imgY = (y / rect.height) * modalImage.naturalHeight;

            switch (workflowStep) {
                case 1:  // South point (plate)
                    calibrationData.anglePoints[0] = {x: imgX, y: imgY};
                    addMarker(x, y, 'S', rect);
                    workflowStep = 2;
                    updateModalStep();
                    break;

                case 2:  // North point (plate)
                    calibrationData.anglePoints[1] = {x: imgX, y: imgY};
                    addMarker(x, y, 'N', rect);
                    drawLine(calibrationData.anglePoints[0], calibrationData.anglePoints[1], rect);
                    calibrationData.angle = calculateAngle(
                        calibrationData.anglePoints[0],
                        calibrationData.anglePoints[1]
                    );
                    workflowStep = 3;
                    updateModalStep();
                    break;

                case 3:  // Scale point 1
                    calibrationData.scalePoints[0] = {x: imgX, y: imgY};
                    addMarker(x, y, '1', rect);
                    workflowStep = 4;
                    updateModalStep();
                    break;

                case 4:  // Scale point 2
                    calibrationData.scalePoints[1] = {x: imgX, y: imgY};
                    addMarker(x, y, '2', rect);
                    drawLine(calibrationData.scalePoints[0], calibrationData.scalePoints[1], rect);
                    calibrationData.pixelDistance = calculateDistance(
                        calibrationData.scalePoints[0],
                        calibrationData.scalePoints[1]
                    );
                    // Show input panel for distance/scale values
                    showInputPanel();
                    break;

                case 10:  // South point (reference map)
                    calibrationData.anglePoints[0] = {x: imgX, y: imgY};
                    addMarker(x, y, 'S', rect);
                    workflowStep = 11;
                    stepIndicator.textContent = 'Step 2 of 2';
                    modalInstructions.textContent = 'Click the NORTH point on the compass';
                    break;

                case 11:  // North point (reference map)
                    calibrationData.anglePoints[1] = {x: imgX, y: imgY};
                    addMarker(x, y, 'N', rect);
                    drawLine(calibrationData.anglePoints[0], calibrationData.anglePoints[1], rect);
                    referenceMapAngle = calculateAngle(
                        calibrationData.anglePoints[0],
                        calibrationData.anglePoints[1]
                    );
                    // Close modal and save
                    modal.classList.remove('active');
                    clearMarkers();
                    saveReferenceMapAlignment();
                    workflowStep = 0;
                    statusEl.textContent = 'Select a plate to align';
                    break;
            }
        }

        function addMarker(x, y, label, rect) {
            const marker = document.createElement('div');
            marker.className = 'calibration-marker';
            marker.textContent = label;
            marker.style.left = `${x}px`;
            marker.style.top = `${y}px`;
            modalImageContainer.appendChild(marker);
        }

        function drawLine(p1, p2, rect) {
            // Convert image coords to display coords
            const x1 = (p1.x / modalImage.naturalWidth) * rect.width;
            const y1 = (p1.y / modalImage.naturalHeight) * rect.height;
            const x2 = (p2.x / modalImage.naturalWidth) * rect.width;
            const y2 = (p2.y / modalImage.naturalHeight) * rect.height;

            const dx = x2 - x1;
            const dy = y2 - y1;
            const length = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;

            const line = document.createElement('div');
            line.className = 'calibration-line';
            line.style.left = `${x1}px`;
            line.style.top = `${y1}px`;
            line.style.width = `${length}px`;
            line.style.transform = `rotate(${angle}deg)`;
            modalImageContainer.appendChild(line);
        }

        function clearMarkers() {
            modalImageContainer.querySelectorAll('.calibration-marker, .calibration-line').forEach(el => el.remove());
        }

        function calculateAngle(south, north) {
            // Calculate angle from south to north
            // SVG/screen: Y increases downward
            const dx = north.x - south.x;
            const dy = south.y - north.y;  // Inverted for screen coords

            // atan2 gives angle from positive X axis
            // We want rotation needed to align north-up (negate for CSS rotation direction)
            let angle = -Math.atan2(dx, dy) * (180 / Math.PI);

            return angle;
        }

        function calculateDistance(p1, p2) {
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            return Math.sqrt(dx * dx + dy * dy);
        }

        function showInputPanel() {
            inputDistance.value = calibrationData.realDistanceFeet;
            inputScale.value = calibrationData.imageScaleFeetPerInch;
            inputPanel.classList.add('active');
        }

        function confirmScaleInput() {
            calibrationData.realDistanceFeet = parseFloat(inputDistance.value) || 300;
            calibrationData.imageScaleFeetPerInch = parseFloat(inputScale.value) || 80;

            // Calculate scale factor
            // Plate scale: 1 inch = imageScaleFeetPerInch feet
            // Reference: 1 inch = 600 feet
            // Thumbnail is already 10% scale
            calibrationData.scaleFactor = (calibrationData.imageScaleFeetPerInch / REFERENCE_SCALE) * THUMBNAIL_SCALE;

            inputPanel.classList.remove('active');
            modal.classList.remove('active');
            clearMarkers();

            workflowStep = 5;
            updateHelpText('Click on the map to place the plate center');
            statusEl.textContent = `Placing ${currentPlate.plate_id}...`;
        }

        function handleCanvasClick(e) {
            if (workflowStep !== 5) return;

            const rect = canvasContent.getBoundingClientRect();
            const x = (e.clientX - rect.left) / zoom;
            const y = (e.clientY - rect.top) / zoom;

            calibrationData.position = { x, y };

            // Create or update thumbnail
            let thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
            if (!thumb) {
                thumb = document.createElement('img');
                thumb.className = 'plate-thumbnail';
                thumb.id = `thumb-${currentPlate.plate_id}`;
                thumb.src = `/api/plate/${currentPlate.volume}/${currentPlate.plate_id}/thumb`;
                thumb.dataset.volume = currentPlate.volume;
                thumb.dataset.plateId = currentPlate.plate_id;
                thumbnailsLayer.appendChild(thumb);
            }

            thumb.classList.remove('locked');
            thumb.onload = () => {
                updateThumbnailTransform(thumb, {
                    angle: calibrationData.angle,
                    scale: calibrationData.scaleFactor,
                    pos: [calibrationData.position.x, calibrationData.position.y]
                });
            };
            if (thumb.complete) {
                updateThumbnailTransform(thumb, {
                    angle: calibrationData.angle,
                    scale: calibrationData.scaleFactor,
                    pos: [calibrationData.position.x, calibrationData.position.y]
                });
            }

            workflowStep = 6;
            showAlignmentInfo({
                angle: calibrationData.angle,
                scale: calibrationData.scaleFactor,
                pos: [calibrationData.position.x, calibrationData.position.y]
            });
            updateHelpText('Drag to adjust position. Press [ or ] to scale. Save when done.');
            btnSave.disabled = false;
        }

        function unlockCurrentThumbnail() {
            if (!currentPlate) return;
            const thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
            if (thumb) {
                thumb.classList.remove('locked');
            }
        }

        function showAlignmentInfo(alignment) {
            calibrationPanel.style.display = 'block';
            document.getElementById('cal-angle').textContent = `${alignment.angle.toFixed(1)}°`;
            document.getElementById('cal-scale').textContent = alignment.scale.toFixed(4);
            document.getElementById('cal-pos').textContent = `(${Math.round(alignment.pos[0])}, ${Math.round(alignment.pos[1])})`;
            statusEl.textContent = `Editing ${currentPlate.plate_id}`;
        }

        function updateHelpText(text) {
            helpText.innerHTML = text;
        }

        // Thumbnail dragging
        function startDrag(e, thumb) {
            if (thumb.classList.contains('locked')) return;
            if (workflowStep !== 6) return;

            isDragging = true;
            thumb.classList.add('dragging');

            const rect = canvasContent.getBoundingClientRect();
            dragStart = {
                x: e.clientX,
                y: e.clientY
            };
            dragOffset = {
                x: calibrationData.position.x,
                y: calibrationData.position.y
            };

            e.preventDefault();
        }

        function doDrag(e) {
            if (!isDragging) return;

            const dx = (e.clientX - dragStart.x) / zoom;
            const dy = (e.clientY - dragStart.y) / zoom;

            calibrationData.position.x = dragOffset.x + dx;
            calibrationData.position.y = dragOffset.y + dy;

            const thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
            if (thumb) {
                updateThumbnailTransform(thumb, {
                    angle: calibrationData.angle,
                    scale: calibrationData.scaleFactor,
                    pos: [calibrationData.position.x, calibrationData.position.y]
                });
            }

            showAlignmentInfo({
                angle: calibrationData.angle,
                scale: calibrationData.scaleFactor,
                pos: [calibrationData.position.x, calibrationData.position.y]
            });
        }

        function endDrag() {
            if (!isDragging) return;
            isDragging = false;

            const thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
            if (thumb) {
                thumb.classList.remove('dragging');
            }
        }

        async function saveAlignment() {
            if (!currentPlate || workflowStep < 6) return;

            const alignment = {
                angle: calibrationData.angle,
                scale: calibrationData.scaleFactor,
                pos: [calibrationData.position.x, calibrationData.position.y],
                calibration: {
                    pixel_distance: calibrationData.pixelDistance,
                    real_distance_feet: calibrationData.realDistanceFeet,
                    image_scale_feet_per_inch: calibrationData.imageScaleFeetPerInch
                }
            };

            try {
                const response = await fetch(`/api/plate/${currentPlate.volume}/${currentPlate.plate_id}/save`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(alignment)
                });

                const result = await response.json();

                if (result.success) {
                    // Update local data
                    const volume = volumes.find(v => v.name === currentPlate.volume);
                    if (volume) {
                        const plate = volume.plates.find(p => p.plate_id === currentPlate.plate_id);
                        if (plate) {
                            plate.has_alignment = true;
                            plate.alignment = {
                                angle: alignment.angle,
                                scale: alignment.scale,
                                pos: alignment.pos
                            };
                        }
                    }

                    // Lock thumbnail and reset opacity
                    const thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
                    if (thumb) {
                        thumb.classList.add('locked');
                        thumb.style.opacity = 1.0;
                    }

                    // Update sidebar
                    const item = document.querySelector(`.plate-item[data-plate-id="${currentPlate.plate_id}"] .status-icon`);
                    if (item) {
                        item.classList.remove('pending');
                        item.classList.add('aligned');
                    }

                    workflowStep = 0;
                    btnSave.disabled = true;
                    showToast(`Saved alignment for ${currentPlate.plate_id}`, 'success');
                } else {
                    showToast('Save failed: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showToast('Save failed: ' + error.message, 'error');
            }
        }

        function cancelWorkflow() {
            modal.classList.remove('active');
            inputPanel.classList.remove('active');
            clearMarkers();

            // Remove thumbnail if not saved
            if (currentPlate && !currentPlate.has_alignment) {
                const thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
                if (thumb) thumb.remove();
            }

            workflowStep = 0;
            currentPlate = null;
            calibrationPanel.style.display = 'none';
            btnSave.disabled = true;
            statusEl.textContent = 'Select a plate to align';
            updateHelpText('Click a <span style="color: #d4a017;">pending plate</span> to begin alignment.');

            document.querySelectorAll('.plate-item.active').forEach(el => el.classList.remove('active'));
        }

        // Zoom controls
        function applyZoom() {
            canvasContent.style.transform = `scale(${zoom})`;
            canvasContent.style.transformOrigin = '0 0';
            document.getElementById('zoom-level').textContent = `${Math.round(zoom * 100)}%`;
        }

        function zoomIn() {
            zoom = Math.min(zoom * 1.25, 4);
            applyZoom();
        }

        function zoomOut() {
            zoom = Math.max(zoom / 1.25, 0.05);
            applyZoom();
        }

        function fitToView() {
            const containerWidth = canvasWrapper.clientWidth;
            const containerHeight = canvasWrapper.clientHeight;
            const imgWidth = referenceMap.naturalWidth || 4980;
            const imgHeight = referenceMap.naturalHeight || 3364;

            const scaleX = containerWidth / imgWidth;
            const scaleY = containerHeight / imgHeight;
            zoom = Math.min(scaleX, scaleY) * 0.95;
            applyZoom();
        }

        // Pan controls
        let isPanning = false;
        let panStartX, panStartY;
        let scrollStartX, scrollStartY;

        function setupEventListeners() {
            // Zoom buttons
            document.getElementById('btn-zoom-in').addEventListener('click', zoomIn);
            document.getElementById('btn-zoom-out').addEventListener('click', zoomOut);
            document.getElementById('btn-zoom-fit').addEventListener('click', fitToView);

            // Save button
            btnSave.addEventListener('click', saveAlignment);

            // Editable calibration values
            document.getElementById('cal-angle').addEventListener('click', () => {
                if (workflowStep !== 6) return;
                makeEditable('cal-angle', calibrationData.angle, (val) => {
                    calibrationData.angle = parseFloat(val) || 0;
                    updateCurrentThumbnail();
                });
            });

            document.getElementById('cal-scale').addEventListener('click', () => {
                if (workflowStep !== 6) return;
                makeEditable('cal-scale', calibrationData.scaleFactor, (val) => {
                    calibrationData.scaleFactor = Math.max(0.01, parseFloat(val) || 0.1);
                    updateCurrentThumbnail();
                });
            });

            // Align Map button
            btnAlignMap.addEventListener('click', startReferenceMapAlignment);

            // Modal
            modalImage.addEventListener('click', handleModalClick);
            btnCancelModal.addEventListener('click', cancelWorkflow);

            // Input panel
            btnCancelInput.addEventListener('click', () => {
                inputPanel.classList.remove('active');
                cancelWorkflow();
            });
            btnConfirmInput.addEventListener('click', confirmScaleInput);

            // Canvas click for placement
            canvasWrapper.addEventListener('click', (e) => {
                if (e.target === referenceMap || e.target === thumbnailsLayer || e.target === canvasContent) {
                    handleCanvasClick(e);
                }
            });

            // Thumbnail dragging
            thumbnailsLayer.addEventListener('mousedown', (e) => {
                if (e.target.classList.contains('plate-thumbnail')) {
                    startDrag(e, e.target);
                }
            });

            window.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    doDrag(e);
                } else if (isPanning) {
                    const dx = e.clientX - panStartX;
                    const dy = e.clientY - panStartY;
                    canvasWrapper.scrollLeft = scrollStartX - dx;
                    canvasWrapper.scrollTop = scrollStartY - dy;
                }
            });

            window.addEventListener('mouseup', () => {
                endDrag();
                isPanning = false;
                canvasWrapper.classList.remove('dragging');
            });

            // Pan
            canvasWrapper.addEventListener('mousedown', (e) => {
                if (e.target === canvasWrapper || e.target === referenceMap) {
                    if (workflowStep !== 5) {  // Don't pan when placing
                        isPanning = true;
                        panStartX = e.clientX;
                        panStartY = e.clientY;
                        scrollStartX = canvasWrapper.scrollLeft;
                        scrollStartY = canvasWrapper.scrollTop;
                        canvasWrapper.classList.add('dragging');
                    }
                }
            });

            // Wheel zoom
            canvasWrapper.addEventListener('wheel', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    let delta = e.deltaY;
                    if (e.deltaMode === 1) delta *= 16;
                    if (e.deltaMode === 2) delta *= 100;
                    delta = Math.max(-100, Math.min(100, delta));

                    const zoomFactor = 1 - delta * 0.003;
                    const oldZoom = zoom;
                    const newZoom = Math.max(0.05, Math.min(4, zoom * zoomFactor));

                    if (newZoom !== oldZoom) {
                        const rect = canvasWrapper.getBoundingClientRect();
                        const mouseX = e.clientX - rect.left;
                        const mouseY = e.clientY - rect.top;

                        const svgX = (canvasWrapper.scrollLeft + mouseX) / oldZoom;
                        const svgY = (canvasWrapper.scrollTop + mouseY) / oldZoom;

                        zoom = newZoom;
                        applyZoom();

                        canvasWrapper.scrollLeft = svgX * newZoom - mouseX;
                        canvasWrapper.scrollTop = svgY * newZoom - mouseY;
                    }
                }
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT') return;

                // Zoom-dependent adjustments: more precise at higher zoom
                // At zoom 0.5: scale ±5%, angle ±1°
                // At zoom 1.7: scale ±0.01, angle ±0.1°
                const scaleStep = Math.max(0.01, 0.05 / zoom);
                const angleStep = Math.max(0.1, 1.0 / zoom);

                if (e.key === '[' && workflowStep === 6) {
                    calibrationData.scaleFactor = Math.max(0.01, calibrationData.scaleFactor - scaleStep);
                    updateCurrentThumbnail();
                } else if (e.key === ']' && workflowStep === 6) {
                    calibrationData.scaleFactor += scaleStep;
                    updateCurrentThumbnail();
                } else if (e.key === '-' && workflowStep === 6) {
                    calibrationData.angle -= angleStep;
                    updateCurrentThumbnail();
                } else if (e.key === '=' && workflowStep === 6) {
                    calibrationData.angle += angleStep;
                    updateCurrentThumbnail();
                } else if (e.key === ',' && workflowStep === 6) {
                    calibrationData.opacity = Math.max(0.1, calibrationData.opacity - 0.1);
                    updateCurrentThumbnail();
                } else if (e.key === '.' && workflowStep === 6) {
                    calibrationData.opacity = Math.min(1.0, calibrationData.opacity + 0.1);
                    updateCurrentThumbnail();
                } else if (e.key === 'Escape') {
                    cancelWorkflow();
                } else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                    e.preventDefault();
                    saveAlignment();
                }
            });

            // Reference map load
            referenceMap.onload = () => {
                fitToView();
            };
        }

        function updateCurrentThumbnail() {
            if (!currentPlate) return;
            const thumb = document.getElementById(`thumb-${currentPlate.plate_id}`);
            if (thumb) {
                updateThumbnailTransform(thumb, {
                    angle: calibrationData.angle,
                    scale: calibrationData.scaleFactor,
                    pos: [calibrationData.position.x, calibrationData.position.y]
                });
                thumb.style.opacity = calibrationData.opacity;
            }
            showAlignmentInfo({
                angle: calibrationData.angle,
                scale: calibrationData.scaleFactor,
                pos: [calibrationData.position.x, calibrationData.position.y]
            });
        }

        function makeEditable(elementId, currentValue, onSave) {
            const span = document.getElementById(elementId);
            if (!span || span.querySelector('input')) return;  // Already editing

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'value-input';
            input.value = typeof currentValue === 'number' ? currentValue.toFixed(4) : currentValue;

            const originalText = span.textContent;
            span.textContent = '';
            span.appendChild(input);
            input.focus();
            input.select();

            const finishEdit = (save) => {
                if (save) {
                    onSave(input.value);
                }
                // Restore span display (will be updated by showAlignmentInfo)
                span.textContent = originalText;
                showAlignmentInfo({
                    angle: calibrationData.angle,
                    scale: calibrationData.scaleFactor,
                    pos: [calibrationData.position.x, calibrationData.position.y]
                });
            };

            input.addEventListener('blur', () => finishEdit(true));
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    input.blur();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    finishEdit(false);
                }
            });
        }

        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            document.body.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, 3000);
        }

        // Start
        init();
    </script>
</body>
</html>
'''


def main():
    global OUTPUT_DIR, MEDIA_DIR, OVERVIEW_DIR

    parser = argparse.ArgumentParser(description='Plate Alignment Editor')
    parser.add_argument('--port', type=int, default=5002, help='Port to run on')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory containing plate SVGs')
    parser.add_argument('--media-dir', type=str, default='../media/v1',
                        help='Media directory containing JPEG/thumbnail files')
    parser.add_argument('--overview-dir', type=str, default='../media/overview',
                        help='Overview directory containing reference map')

    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    MEDIA_DIR = Path(args.media_dir)
    OVERVIEW_DIR = Path(args.overview_dir)

    print(f"Starting Plate Alignment Editor...")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Media dir: {MEDIA_DIR}")
    print(f"  Overview dir: {OVERVIEW_DIR}")
    print(f"  URL: http://localhost:{args.port}")
    print()

    app.run(host='0.0.0.0', port=args.port, debug=True)


if __name__ == '__main__':
    main()
