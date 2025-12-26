#!/usr/bin/env python3
"""
Block Trace Extraction Experiment

Clips an autotrace path by block polygons, producing one path per block
containing only the trace segments within that block's boundary.

Input: SVG file with:
  - Background image
  - Autotrace <path> element
  - <g id="blocks"> containing <polygon> elements

Output: SVG file with additional <g id="block-traces"> group containing
        one <path> per block with clipped trace segments.
"""

import sys
from pathlib import Path
from lxml import etree
from svgpathtools import parse_path, Path as SvgPath
from shapely.geometry import Polygon, Point

# Paths
INPUT_SVG = Path(__file__).parent.parent.parent / "media" / "p2.svg"
OUTPUT_SVG = Path(__file__).parent / "blocktraced.svg"

# SVG namespace
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}


def parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points attribute to coordinate tuples."""
    coords = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            coords.append((float(x), float(y)))
    return coords


def segment_in_polygon(segment, polygon: Polygon) -> bool:
    """Test if segment's endpoints are inside polygon."""
    start = Point(segment.start.real, segment.start.imag)
    end = Point(segment.end.real, segment.end.imag)
    return polygon.contains(start) and polygon.contains(end)


def clip_path_to_polygon(path: SvgPath, polygon: Polygon) -> list:
    """Return segments where both endpoints are inside polygon."""
    return [seg for seg in path if segment_in_polygon(seg, polygon)]


def segments_to_path_string(segments: list) -> str:
    """Convert segment list back to SVG path d string.

    Groups contiguous segments into subpaths and uses svgpathtools' native
    d() method to preserve original segment types (curves, arcs, etc).
    """
    if not segments:
        return ""

    # Group contiguous segments into subpaths
    subpaths = []
    current_subpath = [segments[0]]

    for i in range(1, len(segments)):
        prev_end = segments[i-1].end
        curr_start = segments[i].start

        # Check if segments are contiguous (endpoints match within tolerance)
        if abs(prev_end - curr_start) < 0.5:
            current_subpath.append(segments[i])
        else:
            subpaths.append(current_subpath)
            current_subpath = [segments[i]]

    subpaths.append(current_subpath)

    # Build path string using svgpathtools Path objects to preserve curves
    d_parts = []
    for subpath_segments in subpaths:
        # Create a Path object from segments and get its d string
        subpath = SvgPath(*subpath_segments)
        d_parts.append(subpath.d())

    return " ".join(d_parts)


def main():
    print(f"Loading SVG: {INPUT_SVG}")

    # Parse SVG
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(INPUT_SVG), parser)
    root = tree.getroot()

    # Find autotrace path (first <path> element)
    path_elem = root.find(f'.//{{{SVG_NS}}}path')
    if path_elem is None:
        print("ERROR: No <path> element found in SVG")
        sys.exit(1)

    path_d = path_elem.get('d')
    print(f"Parsing autotrace path ({len(path_d):,} characters)...")
    full_path = parse_path(path_d)
    print(f"  Total segments: {len(full_path):,}")

    # Find blocks group
    blocks_group = root.find(f'.//*[@id="blocks"]')
    if blocks_group is None:
        print("ERROR: No <g id='blocks'> group found")
        sys.exit(1)

    # Find all polygons in blocks group
    polygons = blocks_group.findall(f'.//{{{SVG_NS}}}polygon')
    print(f"Found {len(polygons)} block polygons")

    # Create block-traces group
    traces_group = etree.SubElement(root, f'{{{SVG_NS}}}g', id='block-traces')

    # Process each block
    total_clipped = 0
    blocks_with_traces = 0

    for poly_elem in polygons:
        block_id = poly_elem.get('data-block-id', 'unknown')
        block_name = poly_elem.get('data-name', block_id)
        stroke_color = poly_elem.get('stroke', '#000000')

        # Parse polygon points
        points_str = poly_elem.get('points')
        if not points_str:
            print(f"  Block {block_id}: No points attribute, skipping")
            continue

        coords = parse_polygon_points(points_str)
        if len(coords) < 3:
            print(f"  Block {block_id}: Not enough points ({len(coords)}), skipping")
            continue

        # Create Shapely polygon
        polygon = Polygon(coords)

        # Clip path to polygon
        clipped_segments = clip_path_to_polygon(full_path, polygon)

        if clipped_segments:
            blocks_with_traces += 1
            total_clipped += len(clipped_segments)

            # Create path element
            path_string = segments_to_path_string(clipped_segments)
            path_attribs = {
                'd': path_string,
                'fill': 'none',
                'stroke': stroke_color,
                'stroke-width': '2',
                'data-block-id': block_id,
                'data-block-name': block_name,
            }
            etree.SubElement(traces_group, f'{{{SVG_NS}}}path', **path_attribs)

            print(f"  Block {block_id} ({block_name}): {len(clipped_segments):,} segments")
        else:
            print(f"  Block {block_id} ({block_name}): 0 segments (empty)")

    print(f"\nSummary:")
    print(f"  Blocks with traces: {blocks_with_traces}/{len(polygons)}")
    print(f"  Total clipped segments: {total_clipped:,}")

    # Write output
    print(f"\nWriting output: {OUTPUT_SVG}")
    tree.write(str(OUTPUT_SVG), encoding='utf-8', xml_declaration=True, pretty_print=True)

    # Report file size
    output_size = OUTPUT_SVG.stat().st_size
    print(f"Output file size: {output_size:,} bytes ({output_size/1024/1024:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
