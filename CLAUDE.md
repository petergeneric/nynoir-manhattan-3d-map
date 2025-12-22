# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an SVG-to-STL extruder that converts 2D building footprints from SVG files into 3D models. It parses SVG path elements, triangulates the polygons, and extrudes them to random heights (2-5 stories) to create a 3D city block model.

## Commands

### Setup
```bash
uv sync  # Install dependencies using uv package manager
```

### Run
```bash
uv run python generate_stl.py  # Generate STL from SVG
# or
uv run extrude  # Uses the script entry point
```

### View Output
Open `webviewer/index.html` in a browser (requires the STL file to be copied to the webviewer directory, or serve via HTTP to avoid CORS issues).

## Architecture

**generate_stl.py** - Single-file implementation containing:
- SVG path parsing (M, L, H, V, Z commands, both absolute and relative)
- Ear-clipping polygon triangulation for 2D shapes
- 3D extrusion creating top/bottom faces and side walls
- STL mesh generation using numpy-stl

**webviewer/index.html** - Three.js-based STL viewer for previewing generated models

## Key Constants

- `STORY_HEIGHT = 10` - Height per story in SVG units
- Building heights: 2-5 stories with weighted random distribution (50% 2-story, 25% 3-story, 15% 4-story, 10% 5-story)

## Input/Output

- Input: `src/nyn block test.svg` (SVG with `<path d="...">` elements)
- Output: `nyc_block.stl` (binary STL file)
