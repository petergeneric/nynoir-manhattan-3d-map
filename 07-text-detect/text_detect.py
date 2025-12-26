#!/usr/bin/env python3
"""
Text Detection Experiment using EasyOCR/CRAFT

Detects text regions in an image using EasyOCR's CRAFT-based detector
and outputs bounding polygons as a new 'text' group in an SVG.

Uses low link_threshold to get character-level boxes rather than word-level.
"""

import sys
from pathlib import Path
from lxml import etree
import numpy as np
from PIL import Image
import easyocr

# Paths
INPUT_SVG = Path(__file__).parent.parent.parent / "media" / "p2.svg"
OUTPUT_SVG = Path(__file__).parent / "text_detected.svg"

# SVG namespace
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS, 'xlink': XLINK_NS}

# Detection parameters - low link_threshold for character-level detection
TEXT_THRESHOLD = 0.7      # Confidence for character regions
LINK_THRESHOLD = 0.1      # Low value = less linking = more individual boxes
LOW_TEXT = 0.4            # Minimum text confidence


def extract_image_path_from_svg(svg_path: Path) -> Path:
    """Extract the background image path from SVG."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(svg_path), parser)
    root = tree.getroot()

    # Find the image element
    image_elem = root.find(f'.//{{{SVG_NS}}}image')
    if image_elem is None:
        raise ValueError("No <image> element found in SVG")

    # Get href (try both xlink:href and href)
    href = image_elem.get(f'{{{XLINK_NS}}}href') or image_elem.get('href')
    if not href:
        raise ValueError("No href attribute on image element")

    # Resolve relative path
    image_path = svg_path.parent / href
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def box_to_polygon_str(box) -> str:
    """Convert EasyOCR box (4 corner points) to SVG points string."""
    # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    points = []
    for pt in box:
        points.append(f"{pt[0]:.1f},{pt[1]:.1f}")
    return " ".join(points)


def main():
    print(f"Loading SVG: {INPUT_SVG}")

    # Parse SVG
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(INPUT_SVG), parser)
    root = tree.getroot()

    # Get image path
    image_path = extract_image_path_from_svg(INPUT_SVG)
    print(f"Image path: {image_path}")

    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"Image size: {width} x {height}")

    # Initialize EasyOCR reader (detection only, no recognition needed)
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR initialized")

    # Run detection only (no OCR recognition)
    print(f"Running text detection (text_threshold={TEXT_THRESHOLD}, link_threshold={LINK_THRESHOLD})...")
    horizontal_boxes, free_boxes = reader.detect(
        str(image_path),
        text_threshold=TEXT_THRESHOLD,
        link_threshold=LINK_THRESHOLD,
        low_text=LOW_TEXT,
        canvas_size=max(width, height),  # Process at full resolution
    )

    # horizontal_boxes format: [[x_min, x_max, y_min, y_max], ...]
    # free_boxes format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]

    print(f"Detected {len(horizontal_boxes[0]) if horizontal_boxes else 0} horizontal boxes")
    print(f"Detected {len(free_boxes[0]) if free_boxes else 0} free-form boxes")

    # Create text group in SVG
    text_group = etree.SubElement(root, f'{{{SVG_NS}}}g', id='text')

    # Add horizontal boxes (convert to polygon format)
    box_count = 0
    if horizontal_boxes and len(horizontal_boxes[0]) > 0:
        for i, box in enumerate(horizontal_boxes[0]):
            # box is [x_min, x_max, y_min, y_max]
            x_min, x_max, y_min, y_max = box
            # Convert to 4 corners
            points_str = f"{x_min:.1f},{y_min:.1f} {x_max:.1f},{y_min:.1f} {x_max:.1f},{y_max:.1f} {x_min:.1f},{y_max:.1f}"
            poly_attribs = {
                'points': points_str,
                'fill': 'rgba(255,0,0,0.2)',
                'stroke': 'red',
                'stroke-width': '1',
                'data-text-id': str(box_count),
                'data-type': 'horizontal',
            }
            etree.SubElement(text_group, f'{{{SVG_NS}}}polygon', **poly_attribs)
            box_count += 1

    # Add free-form boxes
    if free_boxes and len(free_boxes[0]) > 0:
        for i, box in enumerate(free_boxes[0]):
            points_str = box_to_polygon_str(box)
            poly_attribs = {
                'points': points_str,
                'fill': 'rgba(255,0,0,0.2)',
                'stroke': 'red',
                'stroke-width': '1',
                'data-text-id': str(box_count),
                'data-type': 'free',
            }
            etree.SubElement(text_group, f'{{{SVG_NS}}}polygon', **poly_attribs)
            box_count += 1

    print(f"Added {box_count} text region polygons to 'text' group")

    # Write output
    print(f"\nWriting output: {OUTPUT_SVG}")
    tree.write(str(OUTPUT_SVG), encoding='utf-8', xml_declaration=True, pretty_print=True)

    # Report file size
    output_size = OUTPUT_SVG.stat().st_size
    print(f"Output file size: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
