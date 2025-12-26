#!/usr/bin/env python3
"""
Block Trace Extraction with Text Filtering

Clips an autotrace path by block polygons, producing one path per block
containing only the trace segments within that block's boundary.
Uses EasyOCR text detection to separate text paths from building paths.

Input: SVG file with:
  - Background image
  - Autotrace <path> element
  - <g id="blocks"> containing <polygon> elements

Output: SVG file with hierarchical structure:
  <g id="block-traces">
    <g id="b-XXXX">
      <path id="autotrace" .../>
      <g id="text" visibility="hidden">
        <path id="o0001" .../>
        ...
      </g>
    </g>
  </g>
"""

import sys
from pathlib import Path
from lxml import etree
from svgpathtools import parse_path, Path as SvgPath, Line
from shapely.geometry import Polygon, Point, LineString, box as shapely_box
from shapely.ops import split
from PIL import Image
import easyocr

# Paths
INPUT_SVG = Path(__file__).parent.parent.parent / "media" / "p2.svg"
OUTPUT_SVG = Path(__file__).parent / "blocktraced.svg"

# SVG namespace
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS}

# Number of points to sample along curves for intersection testing
CURVE_SAMPLES = 20

# Text detection parameters
TEXT_THRESHOLD = 0.7
LINK_THRESHOLD = 0.1  # Low value for character-level detection
LOW_TEXT = 0.4

# Morphological filtering parameters
MAX_TEXT_AREA = 5000  # pixels^2 - paths larger than this are not text
MAX_TEXT_ASPECT = 10  # paths more elongated than this are not text


def parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points attribute to coordinate tuples."""
    coords = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            coords.append((float(x), float(y)))
    return coords


def extract_image_path_from_svg(svg_path: Path, root) -> Path:
    """Extract the background image path from SVG."""
    image_elem = root.find(f'.//{{{SVG_NS}}}image')
    if image_elem is None:
        raise ValueError("No <image> element found in SVG")

    href = image_elem.get(f'{{{XLINK_NS}}}href') or image_elem.get('href')
    if not href:
        raise ValueError("No href attribute on image element")

    image_path = svg_path.parent / href
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def detect_text_regions(image_path: Path) -> list[Polygon]:
    """Run EasyOCR detection and return text bounding boxes as Shapely polygons."""
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)

    with Image.open(image_path) as img:
        width, height = img.size

    print(f"Running text detection (link_threshold={LINK_THRESHOLD})...")
    horizontal_boxes, free_boxes = reader.detect(
        str(image_path),
        text_threshold=TEXT_THRESHOLD,
        link_threshold=LINK_THRESHOLD,
        low_text=LOW_TEXT,
        canvas_size=max(width, height),
    )

    text_polygons = []

    # Convert horizontal boxes [x_min, x_max, y_min, y_max] to polygons
    if horizontal_boxes and len(horizontal_boxes[0]) > 0:
        for box in horizontal_boxes[0]:
            x_min, x_max, y_min, y_max = box
            text_polygons.append(shapely_box(x_min, y_min, x_max, y_max))

    # Convert free-form boxes [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to polygons
    if free_boxes and len(free_boxes[0]) > 0:
        for box in free_boxes[0]:
            coords = [(pt[0], pt[1]) for pt in box]
            text_polygons.append(Polygon(coords))

    print(f"Detected {len(text_polygons)} text regions")
    return text_polygons


def segment_to_linestring(segment, num_samples: int = CURVE_SAMPLES) -> LineString:
    """Convert an SVG path segment to a Shapely LineString."""
    points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        pt = segment.point(t)
        points.append((pt.real, pt.imag))
    return LineString(points)


def clip_segment_to_polygon(segment, polygon: Polygon) -> list:
    """Clip a single segment to a polygon with proper geometric clipping."""
    start = Point(segment.start.real, segment.start.imag)
    end = Point(segment.end.real, segment.end.imag)

    start_inside = polygon.contains(start)
    end_inside = polygon.contains(end)

    if start_inside and end_inside:
        all_inside = True
        for i in range(1, CURVE_SAMPLES):
            t = i / CURVE_SAMPLES
            pt = segment.point(t)
            if not polygon.contains(Point(pt.real, pt.imag)):
                all_inside = False
                break
        if all_inside:
            return [segment]

    ls = segment_to_linestring(segment)
    intersection = ls.intersection(polygon)

    if intersection.is_empty:
        return []

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

    elif intersection.geom_type == 'GeometryCollection':
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


def group_segments_into_subpaths(segments: list) -> list[list]:
    """Group contiguous segments into subpaths."""
    if not segments:
        return []

    subpaths = []
    current_subpath = [segments[0]]

    for i in range(1, len(segments)):
        prev_end = segments[i-1].end
        curr_start = segments[i].start

        if abs(prev_end - curr_start) < 0.5:
            current_subpath.append(segments[i])
        else:
            subpaths.append(current_subpath)
            current_subpath = [segments[i]]

    subpaths.append(current_subpath)
    return subpaths


def get_subpath_bounds(subpath_segments: list) -> Polygon:
    """Get bounding box of a subpath as a Shapely box."""
    xs = [seg.start.real for seg in subpath_segments] + [subpath_segments[-1].end.real]
    ys = [seg.start.imag for seg in subpath_segments] + [subpath_segments[-1].end.imag]
    return shapely_box(min(xs), min(ys), max(xs), max(ys))


def is_fully_contained_in_text_box(subpath_segments: list, text_boxes: list[Polygon]) -> bool:
    """Test if subpath is fully contained within any EasyOCR text detection box.

    Only returns True if the subpath is completely isolated within a text box,
    ensuring we don't incorrectly classify connected paths (like building outlines
    that pass through text regions) as text.
    """
    subpath_box = get_subpath_bounds(subpath_segments)

    for text_box in text_boxes:
        # Check if the text box fully contains the subpath
        if text_box.contains(subpath_box):
            return True

        # Also accept if >95% contained (allows for minor boundary overlap)
        if subpath_box.intersects(text_box):
            intersection_area = subpath_box.intersection(text_box).area
            if subpath_box.area > 0 and intersection_area / subpath_box.area > 0.95:
                return True

    return False


def has_text_morphology(subpath_segments: list) -> bool:
    """Check if path has text-like morphological properties."""
    bounds = get_subpath_bounds(subpath_segments)
    x_min, y_min, x_max, y_max = bounds.bounds
    width = x_max - x_min
    height = y_max - y_min
    area = bounds.area

    # Too large to be a single character
    if area > MAX_TEXT_AREA:
        return False

    # Too elongated (probably a line, not text)
    min_dim = min(width, height)
    if min_dim > 0:
        aspect = max(width, height) / min_dim
        if aspect > MAX_TEXT_ASPECT:
            return False

    return True


def is_text_path(subpath_segments: list, text_boxes: list[Polygon]) -> bool:
    """Determine if a subpath is text based on detection boxes and morphology.

    A subpath is classified as text only if:
    1. It is fully contained within a text detection box (isolated from other paths)
    2. It has text-like morphological properties (small, not too elongated)
    """
    if not is_fully_contained_in_text_box(subpath_segments, text_boxes):
        return False
    return has_text_morphology(subpath_segments)


def subpath_to_path_string(subpath_segments: list) -> str:
    """Convert a single subpath to SVG path d string."""
    subpath = SvgPath(*subpath_segments)
    d_str = subpath.d()

    # Close if it forms a loop
    first_start = subpath_segments[0].start
    last_end = subpath_segments[-1].end
    if abs(last_end - first_start) < 0.5:
        d_str += " Z"

    return d_str


def subpaths_to_path_string(subpaths: list[list]) -> str:
    """Convert multiple subpaths to a single SVG path d string."""
    if not subpaths:
        return ""
    return " ".join(subpath_to_path_string(sp) for sp in subpaths)


def main():
    print(f"Loading SVG: {INPUT_SVG}")

    # Parse SVG
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(INPUT_SVG), parser)
    root = tree.getroot()

    # Extract image and run text detection
    image_path = extract_image_path_from_svg(INPUT_SVG, root)
    print(f"Image path: {image_path}")
    text_boxes = detect_text_regions(image_path)

    # Find autotrace path
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

    polygons = blocks_group.findall(f'.//{{{SVG_NS}}}polygon')
    print(f"Found {len(polygons)} block polygons")

    # Create block-traces group
    traces_group = etree.SubElement(root, f'{{{SVG_NS}}}g', id='block-traces')

    # Process each block
    total_autotrace = 0
    total_text = 0
    blocks_processed = 0

    for poly_elem in polygons:
        block_id = poly_elem.get('data-block-id', 'unknown')
        block_name = poly_elem.get('data-name', block_id)
        stroke_color = poly_elem.get('stroke', '#000000')

        points_str = poly_elem.get('points')
        if not points_str:
            continue

        coords = parse_polygon_points(points_str)
        if len(coords) < 3:
            continue

        polygon = Polygon(coords)

        # Clip path to block polygon
        clipped_segments = clip_path_to_polygon(full_path, polygon)

        if not clipped_segments:
            continue

        # Group into subpaths
        subpaths = group_segments_into_subpaths(clipped_segments)

        # Classify subpaths
        autotrace_subpaths = []
        text_subpaths = []

        for subpath in subpaths:
            if is_text_path(subpath, text_boxes):
                text_subpaths.append(subpath)
            else:
                autotrace_subpaths.append(subpath)

        blocks_processed += 1
        total_autotrace += len(autotrace_subpaths)
        total_text += len(text_subpaths)

        # Create block group
        block_group = etree.SubElement(traces_group, f'{{{SVG_NS}}}g', id=block_name)

        # Add autotrace path
        if autotrace_subpaths:
            autotrace_d = subpaths_to_path_string(autotrace_subpaths)
            path_attribs = {
                'id': 'autotrace',
                'd': autotrace_d,
                'fill': 'none',
                'stroke': stroke_color,
                'stroke-width': '2',
            }
            etree.SubElement(block_group, f'{{{SVG_NS}}}path', **path_attribs)

        # Create hidden text group
        text_group = etree.SubElement(block_group, f'{{{SVG_NS}}}g', id='text', visibility='hidden')

        # Add text paths
        for i, text_subpath in enumerate(text_subpaths):
            text_d = subpath_to_path_string(text_subpath)
            text_attribs = {
                'id': f'o{i+1:04d}',
                'd': text_d,
                'fill': 'none',
                'stroke': stroke_color,
                'stroke-width': '2',
            }
            etree.SubElement(text_group, f'{{{SVG_NS}}}path', **text_attribs)

        print(f"  Block {block_name}: {len(autotrace_subpaths)} autotrace, {len(text_subpaths)} text subpaths")

    print(f"\nSummary:")
    print(f"  Blocks processed: {blocks_processed}/{len(polygons)}")
    print(f"  Total autotrace subpaths: {total_autotrace:,}")
    print(f"  Total text subpaths: {total_text:,}")

    # Write output
    print(f"\nWriting output: {OUTPUT_SVG}")
    tree.write(str(OUTPUT_SVG), encoding='utf-8', xml_declaration=True, pretty_print=True)

    output_size = OUTPUT_SVG.stat().st_size
    print(f"Output file size: {output_size:,} bytes ({output_size/1024/1024:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
