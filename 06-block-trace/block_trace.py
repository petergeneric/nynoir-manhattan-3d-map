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
from svgpathtools import parse_path, Path as SvgPath, Line
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import split

# Paths
INPUT_SVG = Path(__file__).parent.parent.parent / "media" / "p2.svg"
OUTPUT_SVG = Path(__file__).parent / "blocktraced.svg"

# SVG namespace
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}

# Number of points to sample along curves for intersection testing
CURVE_SAMPLES = 20


def parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points attribute to coordinate tuples."""
    coords = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            coords.append((float(x), float(y)))
    return coords


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Test if a point is inside the polygon."""
    return polygon.contains(Point(x, y))


def segment_to_linestring(segment, num_samples: int = CURVE_SAMPLES) -> LineString:
    """Convert an SVG path segment to a Shapely LineString.

    For curves, samples points along the curve to approximate it.
    """
    points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        pt = segment.point(t)
        points.append((pt.real, pt.imag))
    return LineString(points)


def clip_segment_to_polygon(segment, polygon: Polygon) -> list:
    """Clip a single segment to a polygon.

    Returns a list of Line segments representing the clipped result.
    For segments fully inside, returns the original segment.
    For segments crossing the boundary, returns clipped line segments.
    For segments fully outside, returns empty list.
    """
    start = Point(segment.start.real, segment.start.imag)
    end = Point(segment.end.real, segment.end.imag)

    start_inside = polygon.contains(start)
    end_inside = polygon.contains(end)

    # Case 1: Both endpoints inside - keep original segment
    if start_inside and end_inside:
        # But we need to check if the segment crosses outside and back
        # Sample points along the segment to check
        all_inside = True
        for i in range(1, CURVE_SAMPLES):
            t = i / CURVE_SAMPLES
            pt = segment.point(t)
            if not polygon.contains(Point(pt.real, pt.imag)):
                all_inside = False
                break

        if all_inside:
            return [segment]
        # Falls through to geometric clipping below

    # Case 2: Both endpoints outside - might still cross through
    # Case 3: One inside, one outside - definitely crosses
    # Use geometric intersection for all crossing cases

    # Convert segment to LineString
    ls = segment_to_linestring(segment)

    # Intersect with polygon
    intersection = ls.intersection(polygon)

    if intersection.is_empty:
        return []

    # Convert intersection result to Line segments
    result = []

    if intersection.geom_type == 'LineString':
        coords = list(intersection.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            result.append(Line(complex(x1, y1), complex(x2, y2)))

    elif intersection.geom_type == 'MultiLineString':
        for line in intersection.geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                result.append(Line(complex(x1, y1), complex(x2, y2)))

    elif intersection.geom_type == 'Point':
        # Single point intersection - skip
        pass

    elif intersection.geom_type == 'MultiPoint':
        # Multiple point intersections - skip
        pass

    elif intersection.geom_type == 'GeometryCollection':
        # Mixed results - extract LineStrings
        for geom in intersection.geoms:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    result.append(Line(complex(x1, y1), complex(x2, y2)))

    return result


def clip_path_to_polygon(path: SvgPath, polygon: Polygon) -> list:
    """Clip all segments in path to polygon with proper geometric clipping."""
    clipped = []
    for segment in path:
        clipped.extend(clip_segment_to_polygon(segment, polygon))
    return clipped


def segments_to_path_string(segments: list) -> str:
    """Convert segment list back to SVG path d string.

    Groups contiguous segments into subpaths and uses svgpathtools' native
    d() method to preserve original segment types (curves, arcs, etc).
    Closes subpaths that form closed loops.
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
        d_str = subpath.d()

        # Check if subpath forms a closed loop (end ≈ start)
        first_start = subpath_segments[0].start
        last_end = subpath_segments[-1].end
        if abs(last_end - first_start) < 0.5:
            # It's a closed loop - add Z command
            d_str += " Z"

        d_parts.append(d_str)

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
