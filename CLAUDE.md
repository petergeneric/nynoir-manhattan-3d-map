# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This tool segments atlas plate images and outputs SVG with polygon overlays. It uses a **hybrid approach**:
- **Stage 1**: Uses Meta's SAM (Segment Anything Model, vit_h) for city block detection
- **Stage 2**: Uses traditional CV (Canny edge detection + Hough line reinforcement) for building detection within blocks

The tool supports a **two-stage workflow** for atlas plates with manual editing between stages:
1. **Stage 1**: Detect city blocks from full plates using SAM, output `plate.svg` with embedded metadata
2. **Manual Editing**: Edit `plate.svg` to fix/merge/delete blocks
3. **Stage 2**: Process edited blocks through traditional CV for detailed building segmentation

## Input Files

Input files can be found at: `../media/v1/*.jp2`. They are very large JP2 images.

- `SOURCE.txt` in the input directory contains the volume identifier (e.g., "vol1")
- Plate ID is derived from the filename (e.g., "p37" from "p37.jp2")

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Stage 1: Detect blocks (outputs plate.svg with JP2 background)
uv run python segment.py stage1 /path/to/p37.jp2

# Stage 2: Process blocks from edited plate.svg
uv run python segment.py stage2 output/vol1/p37/plate.svg

# Legacy: Run both stages together (no manual editing)
uv run python segment.py /path/to/p37.jp2

# Adjust block detection thresholds (Stage 1)
uv run python segment.py stage1 /path/to/p37.jp2 --min-block-width 200 --max-area-ratio 0.5

# Force MPS (Apple Silicon GPU) - experimental
uv run python segment.py stage1 /path/to/p37.jp2 --mps

# Run original single-pass mode (for comparison)
uv run python segment.py /path/to/p37.jp2 --single-pass

# Scrub text from source image (removes text via DBNet++ + LaMa inpainting)
uv run python segment.py scrubtext output/vol1/p37/plate.svg

# Stage 2 (text scrubbing is on by default)
uv run python segment.py stage2 output/vol1/p37/plate.svg

# Stage 2 without text scrubbing
uv run python segment.py stage2 output/vol1/p37/plate.svg --no-scrub-text

# Adjust text detection threshold (0-1, lower = more aggressive)
uv run python segment.py scrubtext output/vol1/p37/plate.svg --threshold 0.2
```

## Workflow

### Stage 1: Block Detection
```bash
uv run python segment.py stage1 /path/to/p37.jp2 -o output/
```

Outputs:
```
output/vol1/p37/
└── plate.svg    # Block outlines with JP2 background + metadata
```

The `plate.svg` includes:
- JP2 image as background layer (50% opacity) for ground truth reference
- Metadata for Stage 2 (source JP2 path, volume, plate ID, dimensions)
- Block polygons with `data-block-id` attributes

### Manual Editing (Web Editor)

Use the built-in web editor for SVG cleanup. The editor supports both:
- **plate.svg** (Stage 1 output) - Edit block outlines before Stage 2 processing
- **segmentation.svg** (Stage 2 output) - Edit detailed segmentation polygons

```bash
# Start the editor (opens at http://localhost:5001)
uv run python editor.py

# Custom port and directories
uv run python editor.py --port 8080 --output-dir output --media-dir /path/to/media
```

The dropdown shows both `[plate]` and `[segmentation]` entries for each processed plate.

**plate.svg mode** provides:
- Visual display of block polygons over JPEG background image
- Click to select, Shift+Click for multi-select
- Delete selected blocks
- Merge selected blocks (geometric union using Shapely)
- Save changes back to `plate.svg`

**segmentation.svg mode** provides:
- Visual display of all detailed segmentation polygons
- Block outlines shown with dashed lines
- Select/delete entire blocks (with all their polygons)
- Save changes to both `segmentation.svg` AND individual `b-XXXX.svg` files

Keyboard shortcuts:
- `Delete` / `Backspace`: Delete selected
- `M`: Merge selected (plate mode only)
- `U`: Undo last delete
- `H`: Hide selected
- `S`: Show all
- `I`: Isolate small polygons (hide all with area > threshold)
- `X`: Delete all visible polygons
- `L`: Toggle labels
- `F`: Flash all visible polygons on/off (press again to stop)
- `Ctrl+S`: Save changes
- `Ctrl+A`: Select all
- `Escape`: Deselect all
- `+` / `-` / `0`: Zoom in / out / fit

The area threshold input (default 100) filters polygons by SVG area for the Isolate command.

The complexity threshold input (default 10) filters by polygon complexity (area / number of points). The Select button isolates polygons matching both area ≤ threshold AND complexity ≤ threshold.

Alternatively, open `plate.svg` in an SVG editor (Inkscape, Illustrator, etc.) and:
- Delete incorrect block detections
- Merge overlapping blocks
- Adjust polygon boundaries

### Stage 2: Building Detection (CV)
```bash
uv run python segment.py stage2 output/vol1/p37/plate.svg
```

Reads metadata from `plate.svg`, runs traditional CV building detection on each block, and outputs:
```
output/vol1/p37/
├── plate.svg           # Original (manually edited)
├── b-0001.svg          # Detailed segmentation of block 1
├── b-0002.svg          # Detailed segmentation of block 2
├── ...
└── segmentation.svg    # Combined view referencing b-####.svg
```

Blocks are renumbered sequentially based on remaining polygons after editing.

## CLI Options

### Stage 1 Options
| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | `output` | Base output directory |
| `--min-block-width` | 150 | Minimum block width in pixels |
| `--min-block-height` | 150 | Minimum block height in pixels |
| `--max-area-ratio` | 0.7 | Maximum block area as ratio of image |
| `--mps` | false | Force MPS (Apple Silicon GPU) |

### Stage 2 Options
| Option | Default | Description |
|--------|---------|-------------|
| `--no-scrub-text` | false | Skip text removal (text scrubbing is enabled by default) |
| `--text-threshold` | 0.1 | Text detection threshold (0-1, lower = more aggressive) |
| `--text-expand` | 0 | Pixels to expand text mask by |

Note: Stage 2 uses traditional CV (no GPU acceleration needed), so `--mps` is not applicable.

### Scrubtext Options
| Option | Default | Description |
|--------|---------|-------------|
| `-t, --threshold` | 0.1 | Text detection threshold (0-1, lower = more aggressive) |
| `--expand` | 0 | Pixels to expand text mask by |

## Architecture

Single-file tool (`segment.py`) organized into sections:

1. **Data Structures**: `BoundingBox`, `Block`, `PlateSegmentation`, `BlockSegmentation`, `PlateMetadata`, `TextRegion`, `ScrubResult`
2. **Configuration**: `FIRST_PASS_CONFIG` (SAM params), `CV_DEFAULT_*` (CV detection params), `DEFAULT_TEXT_THRESHOLD` (0.1)
3. **SVG Parsing**: `parse_plate_svg`, `parse_polygon_points`, `bbox_from_polygon_points`
4. **Image Utilities**: `load_image` (with JP2 fallback via Pillow), `extract_region`
5. **Text Detection/Inpainting**: `detect_text_regions` (DBNet++), `expand_mask`, `inpaint_with_lama`, `scrub_text_from_image`
6. **First-Pass Segmentation (SAM)**: `segment_plate`, `filter_blocks_by_size`, `create_mask_generator`
7. **Second-Pass Segmentation (CV)**: `detect_buildings_cv`, `segment_block_cv`, `cv_edges_canny`, `cv_detect_lines_hough`
8. **SVG Generators**: `generate_plate_svg` (with metadata), `generate_block_svg`, `generate_combined_svg`
9. **Output Management**: `parse_source_file`, `get_output_structure`
10. **Stage Processors**: `process_stage1`, `process_stage2`, `process_scrubtext`, `process_plate_recursive` (legacy)

Key dependencies:
- `segment-anything`: Meta's SAM library (installed from GitHub)
- `supervision`: For mask annotation visualization (single-pass mode)
- `torch`/`torchvision`: PyTorch for model inference
- `opencv-python`: Image I/O and contour detection
- `Pillow`: JP2 fallback support
- `flask`: Web server for the SVG editor
- `shapely`: Polygon merging (geometric union)
- `python-doctr`: Text detection using DBNet++ model
- `simple-lama-inpainting`: LaMa model for text inpainting

## Device Selection

- CUDA is used if available
- MPS (Apple Silicon) can be forced with `--mps` flag (may have float64 compatibility issues)
- Falls back to CPU by default

## Block Filtering

First-pass segmentation filters detected masks:
- **Minimum size**: Blocks must be at least `min_block_width` x `min_block_height` pixels
- **Maximum area**: Blocks larger than `max_area_ratio` of the image are excluded (filters background/full-plate masks)

## Plate SVG Metadata Format

The `plate.svg` includes XML metadata for Stage 2 processing:

```xml
<metadata>
  <atlas:source xmlns:atlas="http://example.com/atlas">
    <atlas:jp2-path>/absolute/path/to/p37.jp2</atlas:jp2-path>
    <atlas:volume>vol1</atlas:volume>
    <atlas:plate-id>p37</atlas:plate-id>
    <atlas:image-width>W</atlas:image-width>
    <atlas:image-height>H</atlas:image-height>
    <!-- Optional: text regions detected during scrubbing (JSON array) -->
    <atlas:text-regions>[{"x":100,"y":200,"width":50,"height":20,"center_x":125,"center_y":210},...]</atlas:text-regions>
  </atlas:source>
</metadata>
```

## Text Scrubbing

The `scrubtext` subcommand removes text from source images before CV building detection:

1. **Detection**: Uses DBNet++ (via python-doctr) for pixel-level text detection
2. **Thresholding**: Binarizes the probability map (default threshold: 0.1)
3. **Mask Expansion**: Dilates the mask to ensure text is fully covered (default: 0px)
4. **Inpainting**: Uses LaMa (Large Mask Inpainting) to fill in the text regions

Text region centers are stored as JSON metadata in `segmentation.svg` for downstream use. When running `scrubtext` standalone, a `text_regions.json` file is also saved.

**Integration with Stage 2:**
- Text scrubbing is **enabled by default** in stage2
- Use `--no-scrub-text` to disable if needed
- CV building detection runs against the text-free `scrubbed.png` image
- Text regions are embedded in the combined `segmentation.svg` metadata

## Stage 2 CV Building Detection

Stage 2 uses traditional computer vision techniques instead of SAM for faster, more deterministic building detection:

1. **Denoising**: Non-local means (NLM) denoising to reduce noise while preserving edges
2. **Edge Detection**: Canny edge detection (50-150 thresholds)
3. **Line Reinforcement**: Hough line detection to strengthen building outlines
4. **Morphological Operations**: Close + dilate to connect nearby edges
5. **Contour Finding**: Extract contours from inverted edge mask
6. **Filtering**: Remove small areas (<100px²), large areas (>60000px²), and thin slivers (<12px min width)
7. **Smoothing**: Douglas-Peucker simplification for cleaner polygons

This approach is faster than SAM and produces cleaner building footprints for atlas maps.
