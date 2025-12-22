# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Segment Anything Model (SAM) tool that segments images and outputs PNG/SVG with polygon overlays. It uses Meta's SAM (vit_h model) via the segment-anything library.

## Input Files

Input files can be found at: `../media/v1/*.jp2`. They are very large

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run segmentation on an image
uv run python segment.py <input_image>

# Run with custom output paths
uv run python segment.py input.png -o output.png -s output.svg
```

## Architecture

Single-file tool (`segment.py`) that:
1. Downloads SAM checkpoint (~2.56GB) on first run if not present
2. Loads image with OpenCV
3. Runs SAM automatic mask generation
4. Outputs annotated PNG (using supervision library) and SVG with polygon outlines

Key dependencies:
- `segment-anything`: Meta's SAM library (installed from GitHub)
- `supervision`: For mask annotation visualization
- `torch`/`torchvision`: PyTorch for model inference
- `opencv-python`: Image I/O and contour detection

## Device Selection

CUDA is not available, MPS (Apple Silicon) had to be disabled because of float64 compatibility issues with the SAM model