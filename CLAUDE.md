# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Segment Anything Model (SAM) tool that segments images and outputs SVG with polygon overlays. It uses Meta's SAM (vit_h model) via the segment-anything library.

The tool supports **recursive two-pass segmentation** for atlas plates:
1. **First pass**: Identifies city blocks from full plates (configurable min/max size filters)
2. **Second pass**: Runs detailed segmentation on each extracted block

## Input Files

Input files can be found at: `../media/v1/*.jp2`. They are very large JP2 images.

- `SOURCE.txt` in the input directory contains the volume identifier (e.g., "vol1")
- Plate ID is derived from the filename (e.g., "p37" from "p37.jp2")

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run recursive segmentation (default mode)
uv run python segment.py /path/to/p37.jp2

# Run with custom output directory
uv run python segment.py /path/to/p37.jp2 -o /custom/output

# Adjust block detection thresholds
uv run python segment.py /path/to/p37.jp2 --min-block-width 200 --min-block-height 200 --max-area-ratio 0.5

# Force MPS (Apple Silicon GPU) - experimental
uv run python segment.py /path/to/p37.jp2 --mps

# Run original single-pass mode (for comparison)
uv run python segment.py /path/to/p37.jp2 --single-pass
```

## Output Structure

Recursive mode outputs to: `output/{volume}/{plate}/`

```
output/vol1/p37/
├── plate.svg           # City block outlines with labeled filenames
├── b-0001.svg          # Detailed segmentation of block 1
├── b-0002.svg          # Detailed segmentation of block 2
├── ...
└── segmentation.svg    # Combined view (references b-####.svg via <image href>)
```

- `plate.svg`: Shows block boundaries with labels indicating which file each block maps to
- `b-XXXX.svg`: Individual block detail with fine-grained segmentation
- `segmentation.svg`: Combined view that references block SVGs for modular cleanup

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | `output` | Base output directory |
| `--min-block-width` | 150 | Minimum block width in pixels |
| `--min-block-height` | 150 | Minimum block height in pixels |
| `--max-area-ratio` | 0.7 | Maximum block area as ratio of image (filters large masks) |
| `--mps` | false | Force MPS (Apple Silicon GPU) instead of CPU |
| `--single-pass` | false | Use original single-pass mode |

## Architecture

Single-file tool (`segment.py`) organized into sections:

1. **Data Structures**: `BoundingBox`, `Block`, `PlateSegmentation`, `BlockSegmentation`
2. **Configuration**: `FIRST_PASS_CONFIG` (22500 min area), `SECOND_PASS_CONFIG` (100 min area)
3. **Image Utilities**: `load_image` (with JP2 fallback via Pillow), `extract_region`
4. **First-Pass Segmentation**: `segment_plate`, `filter_blocks_by_size`
5. **Second-Pass Segmentation**: `segment_block`
6. **SVG Generators**: `generate_plate_svg`, `generate_block_svg`, `generate_combined_svg`
7. **Output Management**: `parse_source_file`, `get_output_structure`
8. **Main Workflow**: `process_plate_recursive`

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
