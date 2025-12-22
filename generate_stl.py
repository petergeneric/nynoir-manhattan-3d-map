#!/usr/bin/env python3
"""
Generate an STL file by extruding building shapes from an SVG.
Heights are randomized between 2-5 stories, weighted towards 2.
"""

import re
import random
from pathlib import Path

import numpy as np
from stl import mesh

# Story height in the same units as the SVG (pixels)
STORY_HEIGHT = 10  # Each story is 10 units tall

def parse_svg_paths(svg_content):
    """Extract all path 'd' attributes from the SVG."""
    path_pattern = r'<path\s+d="([^"]+)"'
    return re.findall(path_pattern, svg_content)

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

def get_weighted_random_stories():
    """
    Return a random number of stories between 2-5, weighted towards 2.
    Weights: 2 stories (50%), 3 stories (25%), 4 stories (15%), 5 stories (10%)
    """
    choices = [2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
    return random.choice(choices)

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

def main():
    # Read SVG file
    svg_path = Path(__file__).parent / 'src' / 'nyn block test.svg'
    with open(svg_path, 'r') as f:
        svg_content = f.read()

    # Parse all paths
    path_data = parse_svg_paths(svg_content)
    print(f"Found {len(path_data)} paths in SVG")

    all_triangles = []
    successful_buildings = 0

    for i, d in enumerate(path_data):
        points = parse_path_to_polygon(d)
        if points and len(points) >= 3:
            stories = get_weighted_random_stories()
            height = stories * STORY_HEIGHT

            try:
                triangles = extrude_polygon_to_mesh(points, height)
                all_triangles.extend(triangles)
                successful_buildings += 1
            except Exception as e:
                print(f"Warning: Failed to process building {i}: {e}")

    print(f"Successfully processed {successful_buildings} buildings")
    print(f"Total triangles: {len(all_triangles)}")

    # Write STL
    output_path = Path(__file__).parent / 'nyc_block.stl'
    stl_mesh = create_stl_mesh(all_triangles)
    stl_mesh.save(str(output_path))
    print(f"Wrote STL file to: {output_path}")

if __name__ == '__main__':
    main()
