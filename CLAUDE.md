# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Remove text from 1920s atlas maps of dense urban cities to produce clean building footprint outlines for 3D extrusion. The pipeline uses OCR-based text detection combined with geometric analysis to classify SVG path segments as either "text" or "building outline."

## Commands

```bash
# Run with default settings (input: ../media/p2.svg, output: ./blocktraced.svg)
uv run python block_trace.py

# Custom input/output
uv run python block_trace.py -i /path/to/input.svg -o /path/to/output.svg

# Adjust minimum path size filter (default: 40px)
uv run python block_trace.py --min-size 50

# Disable size filtering to keep all paths
uv run python block_trace.py --no-size-filter

# Smoothing options (removes text remnants from walls)
uv run python block_trace.py --buffer-radius 3.0 --simplify-tolerance 2.0  # More aggressive
uv run python block_trace.py --no-smoothing  # Disable smoothing

# Install dependencies
uv sync
```

## Architecture

### Input Format
The input SVG contains:
- A JPEG reference image of the scanned atlas page
- A single large `<path>` element (8M+ characters) from autotracing the map
- A `<g id="blocks">` group with polygon elements defining city block boundaries

### Processing Pipeline

1. **Parse SVG** - Extract image reference, autotrace path, and block polygons
2. **Text Detection** - Run EasyOCR on the image (results cached to `{input}.ocr_cache.json`)
3. **Block Clipping** - Clip autotrace path segments to each block polygon boundary
4. **Subpath Grouping** - Group contiguous segments into disconnected subpaths
5. **Smoothing** - Buffer shrink-expand + Douglas-Peucker to remove text remnants from walls
6. **Text Classification** - Two-phase system:
   - Spatial: Is subpath 95%+ contained within an OCR text box?
   - Morphological: Is it small (<5000px²) with moderate aspect ratio (<10)?
7. **Size Filtering** - Remove paths smaller than threshold (default 40px)
8. **SVG Generation** - Output hierarchical structure with text paths hidden

### Output Structure
```xml
<g id="block-traces">
  <g id="b-{block-name}">
    <path id="p{n}" />           <!-- building outlines -->
    <g id="text" visibility="hidden">
      <path id="o{n}" />         <!-- detected text -->
    </g>
  </g>
</g>
<g id="ocr-text-bounds" visibility="hidden">
  <!-- OCR bounding boxes for reference -->
</g>
```

### Key Modules
- `block_trace.py` - Main script with CLI, text detection, path clipping, classification
- `utils/svg_output.py` - SVG generation from contours
- `utils/visualization.py` - Debug visualization and image overlays

### Key Dependencies
- **EasyOCR + CRAFT** - Text region detection with GPU support (MPS/CUDA)
- **Shapely** - Geometric operations (clipping, containment testing)
- **svgpathtools** - SVG path parsing and manipulation
- **lxml** - XML/SVG parsing for large files

### Configuration Constants (in block_trace.py)
- `TEXT_THRESHOLD = 0.7` - OCR confidence threshold
- `LINK_THRESHOLD = 0.1` - Low value for character-level detection
- `MAX_TEXT_AREA = 5000` - Maximum area for text classification
- `MAX_TEXT_ASPECT = 10` - Maximum aspect ratio for text
- `DEFAULT_MIN_PATH_SIZE = 40` - Minimum path size filter

### Smoothing Parameters (CLI)
- `--buffer-radius` (default: 2.0) - Morphological opening removes features < 2×radius pixels
- `--simplify-tolerance` (default: 1.0) - Douglas-Peucker vertex reduction tolerance
- `--no-smoothing` - Disable smoothing entirely
