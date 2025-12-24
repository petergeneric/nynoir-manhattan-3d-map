#!/usr/bin/env python3
"""
Generate an STL file by extruding building shapes from an SVG.

Supports two input formats:
1. Legacy SVG with <path> elements (original mode)
2. Segmentation files from ../segment-anything with <polygon> elements

Heights are randomized between 2-5 stories, weighted towards 2.
Block context is tracked for each shape to enable context-aware height generation.
"""

import re
import random
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from stl import mesh

# Story height in the same units as the SVG (pixels)
STORY_HEIGHT = 10  # Each story is 10 units tall

# Area thresholds for height rules
MIN_SKIP_AREA = 400         # Skip polygons smaller than this
MIN_AREA_THRESHOLD = 2200   # Small polygons - bias toward lower heights
EDGE_SMALL_AREA = 3000      # Edge objects smaller than this = 1 story
MAX_AREA_THRESHOLD = 200000 # Skip polygons larger than this

# Edge detection tolerance
EDGE_OVERLAP_TOLERANCE = 5  # Pixels of overlap allowed when checking for blocking shapes

# Polygon simplification
DEFAULT_SIMPLIFY_TOLERANCE = 2.0  # Max deviation in pixels for polygon simplification


@dataclass
class HeightConfig:
    """Configuration for height generation."""
    min_stories: float = 1.0
    max_stories: float = 7.0
    story_height: float = STORY_HEIGHT


@dataclass
class BlockContext:
    """Context information about a city block from segmentation."""
    block_id: str
    plate_x: int  # Block's X offset in plate coordinates
    plate_y: int  # Block's Y offset in plate coordinates
    width: int
    height: int
    source_file: Path

    def __repr__(self):
        return f"BlockContext(id={self.block_id}, offset=({self.plate_x},{self.plate_y}), size={self.width}x{self.height})"


@dataclass
class Shape:
    """A 2D shape with its block context."""
    points: List[Tuple[float, float]]
    block: Optional[BlockContext] = None
    local_bounds: Optional[Tuple[float, float, float, float]] = None  # min_x, min_y, max_x, max_y

    def __post_init__(self):
        if self.points and self.local_bounds is None:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.local_bounds = (min(xs), min(ys), max(xs), max(ys))

    @property
    def area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.points)
        if n < 3:
            return 0
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2

    def to_plate_coords(self) -> List[Tuple[float, float]]:
        """Convert local coordinates to plate coordinates."""
        if self.block is None:
            return self.points
        return [(p[0] + self.block.plate_x, p[1] + self.block.plate_y) for p in self.points]


@dataclass
class SegmentationData:
    """Container for all shapes loaded from segmentation files."""
    blocks: Dict[str, BlockContext] = field(default_factory=dict)
    shapes: List[Shape] = field(default_factory=list)
    source_dir: Optional[Path] = None

    def shapes_in_block(self, block_id: str) -> List[Shape]:
        """Get all shapes belonging to a specific block."""
        return [s for s in self.shapes if s.block and s.block.block_id == block_id]

    def get_block_for_shape(self, shape: Shape) -> Optional[BlockContext]:
        """Get the block context for a shape."""
        return shape.block


def ranges_overlap(a_min: float, a_max: float, b_min: float, b_max: float,
                   tolerance: float = EDGE_OVERLAP_TOLERANCE) -> bool:
    """Check if two 1D ranges overlap by more than tolerance pixels."""
    overlap = min(a_max, b_max) - max(a_min, b_min)
    return overlap > tolerance


def is_at_block_edge(shape: Shape, seg_data: Optional[SegmentationData] = None) -> bool:
    """Check if shape can reach any block boundary without intersecting other shapes.

    A shape is considered at the edge if it can "see" a block boundary in at least
    one cardinal direction (north/south/east/west) without another shape blocking
    the path (with >5px overlap tolerance).

    Args:
        shape: The shape to check
        seg_data: Segmentation data containing all shapes in the block

    Returns:
        True if shape is at block edge in any direction
    """
    if shape.block is None or seg_data is None:
        return False

    block = shape.block
    bounds = shape.local_bounds
    if bounds is None:
        return False

    min_x, min_y, max_x, max_y = bounds

    # Get all other shapes in the same block
    block_shapes = seg_data.shapes_in_block(block.block_id)
    other_shapes = [s for s in block_shapes if s is not shape and s.local_bounds is not None]

    # Check each cardinal direction
    # North: Can we reach y=0?
    north_clear = True
    for other in other_shapes:
        o_min_x, o_min_y, o_max_x, o_max_y = other.local_bounds
        # Does this shape block our path north? (overlaps horizontally AND is north of us)
        if ranges_overlap(min_x, max_x, o_min_x, o_max_x) and o_min_y < min_y:
            north_clear = False
            break

    if north_clear:
        return True

    # South: Can we reach y=block.height?
    south_clear = True
    for other in other_shapes:
        o_min_x, o_min_y, o_max_x, o_max_y = other.local_bounds
        if ranges_overlap(min_x, max_x, o_min_x, o_max_x) and o_max_y > max_y:
            south_clear = False
            break

    if south_clear:
        return True

    # West: Can we reach x=0?
    west_clear = True
    for other in other_shapes:
        o_min_x, o_min_y, o_max_x, o_max_y = other.local_bounds
        if ranges_overlap(min_y, max_y, o_min_y, o_max_y) and o_min_x < min_x:
            west_clear = False
            break

    if west_clear:
        return True

    # East: Can we reach x=block.width?
    east_clear = True
    for other in other_shapes:
        o_min_x, o_min_y, o_max_x, o_max_y = other.local_bounds
        if ranges_overlap(min_y, max_y, o_min_y, o_max_y) and o_max_x > max_x:
            east_clear = False
            break

    return east_clear


def parse_svg_paths(svg_content: str) -> List[str]:
    """Extract all path 'd' attributes from the SVG (legacy format)."""
    path_pattern = r'<path\s+d="([^"]+)"'
    return re.findall(path_pattern, svg_content)


def parse_svg_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse SVG polygon points string into list of (x, y) tuples."""
    points = []
    for pair in points_str.strip().split():
        if ',' in pair:
            x, y = pair.split(',')
            points.append((float(x), float(y)))
    return points


def parse_block_svg(svg_path: Path,
                    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE
                    ) -> Tuple[Optional[BlockContext], List[Shape]]:
    """Parse a block SVG file from segment-anything output.

    Args:
        svg_path: Path to the block SVG file
        simplify_tolerance: Tolerance for polygon simplification (0 to disable)

    Returns:
        Tuple of (BlockContext, list of Shapes with that context)
    """
    with open(svg_path, 'r') as f:
        content = f.read()

    # Extract block metadata from SVG attributes
    # Format: <svg ... data-block-id="0001" data-plate-x="123" data-plate-y="456">
    svg_match = re.search(
        r'<svg[^>]*'
        r'width="(\d+)"[^>]*'
        r'height="(\d+)"[^>]*'
        r'data-block-id="([^"]+)"[^>]*'
        r'data-plate-x="(\d+)"[^>]*'
        r'data-plate-y="(\d+)"',
        content, re.DOTALL
    )

    # Try alternate attribute order
    if not svg_match:
        svg_match = re.search(
            r'<svg[^>]*'
            r'data-block-id="([^"]+)"[^>]*'
            r'data-plate-x="(\d+)"[^>]*'
            r'data-plate-y="(\d+)"[^>]*'
            r'width="(\d+)"[^>]*'
            r'height="(\d+)"',
            content, re.DOTALL
        )
        if svg_match:
            block_id, plate_x, plate_y, width, height = svg_match.groups()
        else:
            # Try to get just width/height for non-block SVGs
            dim_match = re.search(r'<svg[^>]*width="(\d+)"[^>]*height="(\d+)"', content)
            if dim_match:
                width, height = dim_match.groups()
                block_id, plate_x, plate_y = None, 0, 0
            else:
                return None, []
    else:
        width, height, block_id, plate_x, plate_y = svg_match.groups()

    block = None
    if block_id:
        block = BlockContext(
            block_id=block_id,
            plate_x=int(plate_x),
            plate_y=int(plate_y),
            width=int(width),
            height=int(height),
            source_file=svg_path
        )

    # Extract all polygon elements
    polygon_pattern = r'<polygon\s+points="([^"]+)"'
    shapes = []

    for points_str in re.findall(polygon_pattern, content):
        points = parse_svg_polygon_points(points_str)
        if len(points) >= 3:
            if simplify_tolerance > 0:
                points = simplify_polygon(points, simplify_tolerance)
            shapes.append(Shape(points=points, block=block))

    return block, shapes


def load_segmentation_directory(seg_dir: Path,
                                simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE
                                ) -> SegmentationData:
    """Load all block SVG files from a segmentation output directory.

    Expected structure (from segment-anything):
        seg_dir/
            plate.svg       (optional, plate-level overview)
            b-0001.svg      (block detail files)
            b-0002.svg
            ...
            or
            {name}.svg      (named block files)

    Args:
        seg_dir: Path to segmentation output directory (e.g., output/vol1/p1/)
        simplify_tolerance: Tolerance for polygon simplification (0 to disable)

    Returns:
        SegmentationData with all blocks and shapes loaded
    """
    data = SegmentationData(source_dir=seg_dir)

    # Find all block SVG files (b-XXXX.svg or custom named)
    svg_files = list(seg_dir.glob("b-*.svg"))

    # Also include any other SVGs that aren't plate.svg or segmentation.svg
    for svg_file in seg_dir.glob("*.svg"):
        if svg_file.name not in ("plate.svg", "segmentation.svg"):
            if svg_file not in svg_files:
                svg_files.append(svg_file)

    print(f"Found {len(svg_files)} block SVG files in {seg_dir}")

    for svg_file in sorted(svg_files):
        block, shapes = parse_block_svg(svg_file, simplify_tolerance)
        if block:
            data.blocks[block.block_id] = block
        data.shapes.extend(shapes)

    print(f"Loaded {len(data.shapes)} shapes from {len(data.blocks)} blocks")
    return data

def parse_path_to_polygon(d_attr):
    """
    Parse a simple SVG path (M, L, Z commands only) into a list of (x, y) points.
    Returns None if the path is too complex.
    """
    points = []

    # Tokenize: split on commands while keeping them, handle numbers with commas
    # Insert spaces before commands to make splitting easier
    d_normalized = re.sub(r'([MLHVZmlhvz])', r' \1 ', d_attr)
    # Replace commas with spaces
    d_normalized = d_normalized.replace(',', ' ')
    # Split on whitespace
    tokens = d_normalized.split()

    i = 0
    current_x, current_y = 0, 0
    current_cmd = None

    while i < len(tokens):
        token = tokens[i]

        # Check if it's a command
        if token in 'MLHVZmlhvz':
            current_cmd = token
            i += 1
            continue

        # Otherwise it's a number, process based on current command
        if current_cmd == 'M':
            # Move to absolute
            current_x = float(tokens[i])
            current_y = float(tokens[i + 1])
            points.append((current_x, current_y))
            i += 2
            # After first M coord pair, implicit L
            current_cmd = 'L'
        elif current_cmd == 'm':
            # Move to relative
            current_x += float(tokens[i])
            current_y += float(tokens[i + 1])
            points.append((current_x, current_y))
            i += 2
            current_cmd = 'l'
        elif current_cmd == 'L':
            # Line to absolute
            current_x = float(tokens[i])
            current_y = float(tokens[i + 1])
            points.append((current_x, current_y))
            i += 2
        elif current_cmd == 'l':
            # Line to relative
            current_x += float(tokens[i])
            current_y += float(tokens[i + 1])
            points.append((current_x, current_y))
            i += 2
        elif current_cmd == 'H':
            # Horizontal line to absolute
            current_x = float(tokens[i])
            points.append((current_x, current_y))
            i += 1
        elif current_cmd == 'h':
            # Horizontal line to relative
            current_x += float(tokens[i])
            points.append((current_x, current_y))
            i += 1
        elif current_cmd == 'V':
            # Vertical line to absolute
            current_y = float(tokens[i])
            points.append((current_x, current_y))
            i += 1
        elif current_cmd == 'v':
            # Vertical line to relative
            current_y += float(tokens[i])
            points.append((current_x, current_y))
            i += 1
        elif current_cmd in ('Z', 'z'):
            # Close path - should not have numbers
            i += 1
        else:
            # Unknown, skip
            i += 1

    # Remove duplicate last point if it matches first
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]

    return points if len(points) >= 3 else None


def simplify_polygon(points: List[Tuple[float, float]],
                     tolerance: float = 2.0) -> List[Tuple[float, float]]:
    """Simplify a polygon using the Ramer-Douglas-Peucker algorithm.

    Args:
        points: List of (x, y) tuples forming the polygon
        tolerance: Maximum distance a point can deviate from the simplified line.
                   Higher values = more aggressive simplification.

    Returns:
        Simplified list of (x, y) tuples. Returns original points if
        simplification fails or produces fewer than 3 points.
    """
    if len(points) < 3 or tolerance <= 0:
        return points

    try:
        poly = ShapelyPolygon(points)
        if not poly.is_valid:
            # Try to fix invalid polygons
            poly = poly.buffer(0)

        simplified = poly.simplify(tolerance, preserve_topology=True)

        # Extract coordinates (excluding the closing point which duplicates the first)
        coords = list(simplified.exterior.coords)[:-1]

        if len(coords) >= 3:
            return [(float(x), float(y)) for x, y in coords]
        else:
            return points
    except Exception:
        # If simplification fails, return original
        return points


def get_weighted_random_stories():
    """
    Return a random number of stories between 2-5, weighted towards 2.
    Weights: 2 stories (50%), 3 stories (25%), 4 stories (15%), 5 stories (10%)
    Legacy function - kept for backwards compatibility with legacy SVG mode.
    """
    choices = [2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
    return random.choice(choices)


def get_random_stories(min_stories: float = 1.0, max_stories: float = 7.0,
                       bias_low: bool = False) -> float:
    """Generate a random story count with weighted distribution.

    Heights are generated in 0.25 increments. Stories above 5 are less likely.

    Args:
        min_stories: Minimum number of stories (default 1.0)
        max_stories: Maximum number of stories (default 7.0)
        bias_low: If True, bias toward lower heights (for small areas)

    Returns:
        Number of stories as a float (e.g., 2.25, 3.5)
    """
    # Generate possible values in 0.25 increments
    values = []
    weights = []

    current = min_stories
    while current <= max_stories + 0.001:  # Small epsilon for float comparison
        values.append(current)

        # Base weight calculation
        if current <= 3.0:
            weight = 1.0
        elif current <= 5.0:
            weight = 0.8
        else:
            # Exponential decay for heights > 5
            weight = 0.3 * (0.7 ** (current - 5.0))

        # Apply low bias for small areas
        if bias_low:
            if current <= 2.0:
                weight *= 1.5
            elif current <= 3.0:
                weight *= 1.0
            else:
                weight *= 0.5

        weights.append(weight)
        current += 0.25

    if not values:
        return min_stories

    # Normalize weights and select
    total = sum(weights)
    weights = [w / total for w in weights]

    return random.choices(values, weights=weights, k=1)[0]

def compute_normal(v0, v1, v2):
    """Compute the normal vector for a triangle."""
    # Edge vectors
    u = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    v = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])

    # Cross product
    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]

    # Normalize
    length = (nx*nx + ny*ny + nz*nz) ** 0.5
    if length > 0:
        nx, ny, nz = nx/length, ny/length, nz/length

    return (nx, ny, nz)

def triangulate_polygon(points):
    """
    Simple ear-clipping triangulation for a 2D polygon.
    Returns a list of triangle indices.
    """
    if len(points) < 3:
        return []

    # Copy indices
    indices = list(range(len(points)))
    triangles = []

    def signed_area(p0, p1, p2):
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

    def point_in_triangle(p, a, b, c):
        d1 = signed_area(p, a, b)
        d2 = signed_area(p, b, c)
        d3 = signed_area(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # Determine winding order
    total = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        total += (points[j][0] - points[i][0]) * (points[j][1] + points[i][1])
    ccw = total < 0

    max_iterations = len(points) * 3
    iteration = 0

    while len(indices) > 3 and iteration < max_iterations:
        iteration += 1
        ear_found = False

        for i in range(len(indices)):
            prev_i = (i - 1) % len(indices)
            next_i = (i + 1) % len(indices)

            p0 = points[indices[prev_i]]
            p1 = points[indices[i]]
            p2 = points[indices[next_i]]

            # Check if this is a convex vertex
            cross = signed_area(p0, p1, p2)
            if (ccw and cross <= 0) or (not ccw and cross >= 0):
                continue

            # Check if any other point is inside this triangle
            is_ear = True
            for j in range(len(indices)):
                if j in (prev_i, i, next_i):
                    continue
                if point_in_triangle(points[indices[j]], p0, p1, p2):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((indices[prev_i], indices[i], indices[next_i]))
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles

def extrude_polygon_to_mesh(points, height):
    """
    Extrude a 2D polygon to 3D, returning a list of triangles.
    Each triangle is ((v0, v1, v2), normal) where v is (x, y, z).
    The polygon is extruded along the Z axis.
    """
    triangles = []
    n = len(points)

    # Create bottom and top vertices
    # SVG Y is inverted (down is positive), so we flip it
    bottom = [(p[0], -p[1], 0) for p in points]
    top = [(p[0], -p[1], height) for p in points]

    # Triangulate the top and bottom faces
    tri_indices = triangulate_polygon(points)

    # Bottom face (reverse winding for correct normal)
    for t in tri_indices:
        v0, v1, v2 = bottom[t[0]], bottom[t[2]], bottom[t[1]]
        normal = compute_normal(v0, v1, v2)
        triangles.append(((v0, v1, v2), normal))

    # Top face
    for t in tri_indices:
        v0, v1, v2 = top[t[0]], top[t[1]], top[t[2]]
        normal = compute_normal(v0, v1, v2)
        triangles.append(((v0, v1, v2), normal))

    # Side faces
    for i in range(n):
        j = (i + 1) % n

        # Two triangles per side
        b0, b1 = bottom[i], bottom[j]
        t0, t1 = top[i], top[j]

        # Triangle 1
        v0, v1, v2 = b0, b1, t1
        normal = compute_normal(v0, v1, v2)
        triangles.append(((v0, v1, v2), normal))

        # Triangle 2
        v0, v1, v2 = b0, t1, t0
        normal = compute_normal(v0, v1, v2)
        triangles.append(((v0, v1, v2), normal))

    return triangles

def create_stl_mesh(triangles):
    """Create an STL mesh from triangles using numpy-stl."""
    stl_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))

    for i, (verts, _) in enumerate(triangles):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[j]

    return stl_mesh

def get_height_for_shape(shape: Shape, seg_data: Optional[SegmentationData] = None,
                         config: Optional[HeightConfig] = None) -> Optional[float]:
    """Determine extrusion height for a shape based on elevation rules.

    Rules applied:
    1. Skip polygons with area < 400
    2. Skip polygons with area > 200,000
    3. Edge objects with area < 3000 get 1 story
    4. Edge objects with area >= 3000 get at least 2 stories
    5. Small polygons (area < 2200) bias toward lower heights
    6. Heights above 5 stories are less likely

    Args:
        shape: The shape to get height for
        seg_data: Optional segmentation data for context-aware height generation
        config: Optional height configuration (min/max stories)

    Returns:
        Height in SVG units, or None if shape should be skipped
    """
    if config is None:
        config = HeightConfig()

    area = shape.area

    # Rule 1: Skip small polygons
    if area < MIN_SKIP_AREA:
        return None

    # Rule 2: Skip large polygons
    if area > MAX_AREA_THRESHOLD:
        return None

    # Check if shape is at block edge
    at_edge = is_at_block_edge(shape, seg_data)

    # Rule 2 & 3: Edge object rules
    if at_edge:
        if area < EDGE_SMALL_AREA:
            # Small edge objects get 1 story
            return 1.0 * config.story_height
        else:
            # Larger edge objects get at least 2 stories
            min_stories = max(2.0, config.min_stories)
            bias_low = area < MIN_AREA_THRESHOLD
            stories = get_random_stories(min_stories, config.max_stories, bias_low=bias_low)
            return stories * config.story_height

    # Rule 4: Small polygon bias
    bias_low = area < MIN_AREA_THRESHOLD

    # Generate random height with appropriate bias
    stories = get_random_stories(config.min_stories, config.max_stories, bias_low=bias_low)
    return stories * config.story_height


def process_legacy_svg(svg_path: Path, output_path: Path,
                       simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE) -> None:
    """Process legacy SVG with <path> elements."""
    with open(svg_path, 'r') as f:
        svg_content = f.read()

    path_data = parse_svg_paths(svg_content)
    print(f"Found {len(path_data)} paths in SVG")

    all_triangles = []
    successful_buildings = 0

    for i, d in enumerate(path_data):
        points = parse_path_to_polygon(d)
        if points and len(points) >= 3:
            if simplify_tolerance > 0:
                points = simplify_polygon(points, simplify_tolerance)
            shape = Shape(points=points)
            height = get_height_for_shape(shape)

            try:
                triangles = extrude_polygon_to_mesh(points, height)
                all_triangles.extend(triangles)
                successful_buildings += 1
            except Exception as e:
                print(f"Warning: Failed to process building {i}: {e}")

    print(f"Successfully processed {successful_buildings} buildings")
    print(f"Total triangles: {len(all_triangles)}")

    stl_mesh = create_stl_mesh(all_triangles)
    stl_mesh.save(str(output_path))
    print(f"Wrote STL file to: {output_path}")


def process_segmentation(seg_dir: Path, output_path: Path, use_plate_coords: bool = True,
                         config: Optional[HeightConfig] = None,
                         simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE) -> None:
    """Process segmentation files from segment-anything.

    Args:
        seg_dir: Directory containing block SVG files (e.g., output/vol1/p1/)
        output_path: Path for output STL file
        use_plate_coords: If True, place shapes in plate coordinates; if False, use local block coords
        config: Optional height configuration (min/max stories)
        simplify_tolerance: Tolerance for polygon simplification (0 to disable)
    """
    seg_data = load_segmentation_directory(seg_dir, simplify_tolerance)

    if not seg_data.shapes:
        print("No shapes found in segmentation directory")
        return

    all_triangles = []
    successful_buildings = 0
    skipped_buildings = 0

    for i, shape in enumerate(seg_data.shapes):
        # Get coordinates (plate or local)
        if use_plate_coords:
            points = shape.to_plate_coords()
        else:
            points = shape.points

        if len(points) < 3:
            continue

        height = get_height_for_shape(shape, seg_data, config)

        # Skip shapes with None height (e.g., too large)
        if height is None:
            skipped_buildings += 1
            continue

        try:
            triangles = extrude_polygon_to_mesh(points, height)
            all_triangles.extend(triangles)
            successful_buildings += 1
        except Exception as e:
            block_info = f" (block {shape.block.block_id})" if shape.block else ""
            print(f"Warning: Failed to process shape {i}{block_info}: {e}")

    print(f"Successfully processed {successful_buildings} buildings")
    if skipped_buildings > 0:
        print(f"Skipped {skipped_buildings} shapes (area < {MIN_SKIP_AREA} or > {MAX_AREA_THRESHOLD})")
    print(f"Total triangles: {len(all_triangles)}")

    stl_mesh = create_stl_mesh(all_triangles)
    stl_mesh.save(str(output_path))
    print(f"Wrote STL file to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate STL files by extruding 2D shapes from SVG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy mode (SVG with <path> elements)
  uv run python generate_stl.py --svg src/block.svg

  # Segmentation mode (from segment-anything output)
  uv run python generate_stl.py --segmentation ../segment-anything/output/vol1/p1/

  # Process single block SVG
  uv run python generate_stl.py --block ../segment-anything/output/vol1/p1/b-0001.svg
        """
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--svg",
        type=Path,
        help="Path to legacy SVG file with <path> elements"
    )
    input_group.add_argument(
        "--segmentation", "--seg",
        type=Path,
        dest="segmentation",
        help="Path to segmentation directory (e.g., output/vol1/p1/)"
    )
    input_group.add_argument(
        "--block",
        type=Path,
        help="Path to single block SVG file"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output STL file path (default: auto-generated based on input)"
    )
    parser.add_argument(
        "--local-coords",
        action="store_true",
        help="Use local block coordinates instead of plate coordinates"
    )
    parser.add_argument(
        "--min-stories",
        type=float,
        default=1.0,
        help="Minimum building height in stories (default: 1.0)"
    )
    parser.add_argument(
        "--max-stories",
        type=float,
        default=7.0,
        help="Maximum building height in stories (default: 7.0)"
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=DEFAULT_SIMPLIFY_TOLERANCE,
        metavar="TOLERANCE",
        help=f"Polygon simplification tolerance in pixels (default: {DEFAULT_SIMPLIFY_TOLERANCE}, 0 to disable)"
    )

    args = parser.parse_args()

    # Create height configuration from CLI args
    height_config = HeightConfig(
        min_stories=args.min_stories,
        max_stories=args.max_stories
    )

    # Determine input mode and output path
    if args.segmentation:
        seg_dir = args.segmentation
        if not seg_dir.is_dir():
            print(f"Error: Not a directory: {seg_dir}")
            return 1
        output_path = args.output or (seg_dir / "extruded.stl")
        process_segmentation(seg_dir, output_path, use_plate_coords=not args.local_coords,
                           config=height_config, simplify_tolerance=args.simplify)

    elif args.block:
        block_path = args.block
        if not block_path.exists():
            print(f"Error: File not found: {block_path}")
            return 1
        output_path = args.output or block_path.with_suffix(".stl")

        # Create a temporary segmentation data with just this block
        block, shapes = parse_block_svg(block_path, args.simplify)
        if not shapes:
            print(f"No shapes found in {block_path}")
            return 1

        seg_data = SegmentationData(source_dir=block_path.parent)
        if block:
            seg_data.blocks[block.block_id] = block
        seg_data.shapes = shapes

        all_triangles = []
        successful = 0
        skipped = 0
        for i, shape in enumerate(shapes):
            points = shape.points if args.local_coords else shape.to_plate_coords()
            if len(points) < 3:
                continue
            height = get_height_for_shape(shape, seg_data, height_config)
            if height is None:
                skipped += 1
                continue
            try:
                triangles = extrude_polygon_to_mesh(points, height)
                all_triangles.extend(triangles)
                successful += 1
            except Exception as e:
                print(f"Warning: Failed to process shape {i}: {e}")

        print(f"Successfully processed {successful} shapes from block")
        if skipped > 0:
            print(f"Skipped {skipped} shapes (area < {MIN_SKIP_AREA} or > {MAX_AREA_THRESHOLD})")
        print(f"Total triangles: {len(all_triangles)}")

        stl_mesh = create_stl_mesh(all_triangles)
        stl_mesh.save(str(output_path))
        print(f"Wrote STL file to: {output_path}")

    elif args.svg:
        svg_path = args.svg
        if not svg_path.exists():
            print(f"Error: File not found: {svg_path}")
            return 1
        output_path = args.output or svg_path.with_suffix(".stl")
        process_legacy_svg(svg_path, output_path, args.simplify)

    else:
        # Default: use legacy mode with default input file
        svg_path = Path(__file__).parent / 'src' / 'nyn block test.svg'
        output_path = args.output or (Path(__file__).parent / 'nyc_block.stl')

        if not svg_path.exists():
            parser.print_help()
            print(f"\nNote: Default input file not found: {svg_path}")
            return 1

        process_legacy_svg(svg_path, output_path, args.simplify)

    return 0


if __name__ == '__main__':
    exit(main())
