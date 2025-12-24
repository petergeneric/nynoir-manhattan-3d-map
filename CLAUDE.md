# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an SVG-to-STL extruder that converts 2D building footprints from SVG files into 3D models. It supports two input formats:

1. **Legacy SVG** - Files with `<path>` elements (original format)
2. **Segmentation output** - Files from `../segment-anything` with `<polygon>` elements

The extruder parses shapes, triangulates polygons using ear-clipping, and extrudes them to random heights (2-5 stories) to create 3D city block models.

## Commands

### Setup
```bash
uv sync  # Install dependencies using uv package manager
```

### Run

**Segmentation mode** (recommended for segment-anything output):
```bash
# Process combined segmentation.svg file
uv run python generate_stl.py --segmentation ../segment-anything/output/vol1/p1/segmentation.svg

# Process single block
uv run python generate_stl.py --block ../segment-anything/output/vol1/p1/b-0001.svg

# Use local block coordinates instead of plate coordinates
uv run python generate_stl.py --segmentation ../segment-anything/output/vol1/p1/segmentation.svg --local-coords
```

**Legacy mode** (for SVG with `<path>` elements):
```bash
uv run python generate_stl.py --svg src/block.svg

# Or use default file
uv run python generate_stl.py
uv run extrude  # Uses the script entry point
```

**Common options**:
```bash
-o, --output PATH    # Specify output STL file path
--local-coords       # Use local block coordinates (default: plate coordinates)
```

### View Output
Open `webviewer/index.html` in a browser (requires the STL file to be copied to the webviewer directory, or serve via HTTP to avoid CORS issues).

## Architecture

**generate_stl.py** - Main implementation containing:

### Data Classes
- `BlockContext` - Tracks city block metadata (id, plate offset, dimensions, source file)
- `Shape` - A 2D polygon with optional block context, supports coordinate conversion
- `SegmentationData` - Container for all blocks and shapes from segmentation files

### Parsing Functions
- `parse_svg_paths()` - Extracts `<path d="...">` attributes (legacy format)
- `parse_svg_polygon_points()` - Parses SVG polygon points strings
- `parse_segmentation_svg()` - Parses combined segmentation.svg file with all blocks
- `parse_block_svg()` - Parses individual block SVG files with metadata attributes
- `load_segmentation_directory()` - Loads all block SVGs from a directory (legacy)
- `parse_path_to_polygon()` - Converts SVG path commands to polygon points

### Geometry Functions
- `triangulate_polygon()` - Ear-clipping triangulation for 2D polygons
- `extrude_polygon_to_mesh()` - Extrudes 2D polygon to 3D with top/bottom/sides
- `compute_normal()` - Calculates triangle normal vectors

### Height Generation
- `get_height_for_shape()` - Determines extrusion height (extensible for context-aware generation)
- `get_weighted_random_stories()` - Returns 2-5 stories with weighted distribution

**webviewer/index.html** - Three.js-based STL viewer for previewing generated models

## Segmentation File Format

### Combined segmentation.svg (recommended)

The combined format has all blocks in a single file:

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="4980" height="3334" viewBox="0 0 4980 3334">
  <g id="block-outlines" opacity="0.3">...</g>  <!-- Block outlines (ignored) -->
  <g id="block-0001" transform="translate(544,322)">
    <polygon points="x1,y1 x2,y2 ..." fill="..." stroke="..." data-id="p-001" />
    ...
  </g>
  <g id="block-0002" transform="translate(547,325)">
    ...
  </g>
</svg>
```

Key structure:
- `transform="translate(x,y)"` - Block's plate offset coordinates
- `id="block-XXXX"` - Block identifier
- Polygon points are in local block coordinates

### Individual block SVG files (legacy)

```xml
<svg xmlns="http://www.w3.org/2000/svg"
     width="328" height="366"
     viewBox="0 0 328 366"
     data-block-id="0001"
     data-plate-x="2417"
     data-plate-y="2109">
  <polygon points="x1,y1 x2,y2 ..." fill="..." stroke="..."/>
  ...
</svg>
```

Key attributes:
- `data-block-id` - Block identifier
- `data-plate-x`, `data-plate-y` - Offset to convert local coords to plate coords
- `width`, `height` - Block dimensions in pixels

## Key Constants

- `STORY_HEIGHT = 10` - Height per story in SVG units
- Building heights: 2-5 stories with weighted random distribution (50% 2-story, 25% 3-story, 15% 4-story, 10% 5-story)

## Input/Output

**Segmentation input** (from ../segment-anything):
- Combined file: `../segment-anything/output/{volume}/{plate}/segmentation.svg`
- Output: `segmentation.stl` in same directory (or custom path with `-o`)

**Legacy input**:
- Input: `src/nyn block test.svg` (SVG with `<path d="...">` elements)
- Output: `nyc_block.stl` (binary STL file)

## Future Extensions

The `get_height_for_shape()` function accepts `SegmentationData` to enable context-aware height generation:
- Use block context to vary heights by block
- Use shape area/position for height decisions
- Query neighboring shapes within a block
