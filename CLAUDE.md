# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Segment Anything Model (SAM) tool that segments images and outputs SVG with polygon overlays. It uses Meta's SAM (vit_h model) via the segment-anything library.

The tool supports a **two-stage workflow** for atlas plates with manual editing between stages:
1. **Stage 1**: Detect city blocks from full plates, output `plate.svg` with embedded metadata
2. **Manual Editing**: Edit `plate.svg` to fix/merge/delete blocks
3. **Stage 2**: Process edited blocks through SAM for detailed segmentation

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

### Manual Editing

Open `plate.svg` in an SVG editor (Inkscape, Illustrator, etc.) and:
- Delete incorrect block detections
- Merge overlapping blocks
- Adjust polygon boundaries

### Stage 2: Detail Segmentation
```bash
uv run python segment.py stage2 output/vol1/p37/plate.svg
```

Reads metadata from `plate.svg` and outputs:
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
| `--mps` | false | Force MPS (Apple Silicon GPU) |

## Architecture

Single-file tool (`segment.py`) organized into sections:

1. **Data Structures**: `BoundingBox`, `Block`, `PlateSegmentation`, `BlockSegmentation`, `PlateMetadata`
2. **Configuration**: `FIRST_PASS_CONFIG` (22500 min area), `SECOND_PASS_CONFIG` (100 min area)
3. **SVG Parsing**: `parse_plate_svg`, `parse_polygon_points`, `bbox_from_polygon_points`
4. **Image Utilities**: `load_image` (with JP2 fallback via Pillow), `extract_region`
5. **First-Pass Segmentation**: `segment_plate`, `filter_blocks_by_size`
6. **Second-Pass Segmentation**: `segment_block`
7. **SVG Generators**: `generate_plate_svg` (with metadata), `generate_block_svg`, `generate_combined_svg`
8. **Output Management**: `parse_source_file`, `get_output_structure`
9. **Stage Processors**: `process_stage1`, `process_stage2`, `process_plate_recursive` (legacy)

Key dependencies:
- `segment-anything`: Meta's SAM library (installed from GitHub)
- `supervision`: For mask annotation visualization (single-pass mode)
- `torch`/`torchvision`: PyTorch for model inference
- `opencv-python`: Image I/O and contour detection
- `Pillow`: JP2 fallback support

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
  </atlas:source>
</metadata>
```
