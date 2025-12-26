# Experimental Results Summary: Historical Map Building Extraction

## Overview

Five experiments were conducted to extract building outlines from historical hand-drawn atlas maps, specifically targeting the challenge of text annotations that overlap building boundaries.

**Input:** `/Users/pwright/workspace/atlas/media/map-example.png` (2236x1318 pixels)

---

## Experiment Results

### Experiment 1: Text Detection + Inpainting
**Location:** `01-text-inpainting/`

| Metric | Value |
|--------|-------|
| Text regions detected | 377 |
| Image coverage by text | 33.5% |
| Inpainting quality | Poor |

**Findings:**
- EasyOCR successfully detected 377 text regions including street names, building labels, and numbers
- OpenCV inpainting (both Navier-Stokes and Telea) creates blotchy artifacts
- Inpainting doesn't preserve the underlying line work - building boundaries get smoothed over
- **Recommendation:** Use text detection for masking/filtering, not for image cleaning

### Experiment 2: Color-Based Building Segmentation
**Location:** `02-color-segmentation/`

| Metric | Value |
|--------|-------|
| Building contours found | 2 |
| HSV mask effectiveness | Poor |

**Findings:**
- The pink/salmon building color is not distinctive enough from the aged paper background
- HSV thresholds either capture too much (99%+ of image) or too little
- Only detected 2 large contours - not useful for individual building extraction
- **Recommendation:** Use color as a secondary filter, not primary detection

### Experiment 3: Edge Detection + Morphology
**Location:** `03-edge-morph/`

| Metric | Value |
|--------|-------|
| Building contours found | 43 |
| Hough line segments | 7,361 |
| Total polygon points | 1,203 |

**Findings:**
- Canny edge detection picks up building boundaries but also text, streets, and noise
- Morphological operations help but don't fully separate buildings from text
- The "strict" configuration (Canny 80-200) produced the best results
- **Recommendation:** Useful as part of a pipeline but needs text masking first

### Experiment 4: Hybrid Approach (Raster-Based)
**Location:** `04-hybrid/`

| Metric | Method 1 | Method 2 (Flood Fill) |
|--------|----------|----------------------|
| Building contours | 1 | **91** |
| Polygon points | 4 | 4,337 |

**Findings:**
- **Flood fill approach is most promising** - treats dark lines as boundaries
- Successfully identifies individual building footprints
- Text masking (33.5% of image) helps exclude noisy regions
- The approach works because maps use dark lines to delineate building boundaries

### Experiment 5: SVG Post-Processing (Best Results)
**Location:** `05-svg-cleanup/`

| Metric | Value |
|--------|-------|
| Total subpaths parsed | 9,065 |
| Building paths retained | **325** |
| Text paths removed | 8,740 |
| OCR regions used | 377 |

**Findings:**
- **Best approach**: Post-process Inkscape-traced SVG instead of raster image
- Inkscape Edge Detection (threshold 0.650) captures all lines and text cleanly
- `svgpathtools` library splits single path into 9,065 individual subpaths
- OCR bounding boxes transformed to SVG coordinates for text filtering
- 80% overlap threshold effectively separates text from buildings
- **Sharp vector edges** - no jaggedness compared to raster approaches
- Output is ~968KB clean SVG ready for 3D extrusion

**Key Advantage:** Vector-based approach preserves crisp edges that raster methods lose.

---

## Key Insights

### What Works
1. **Text Detection (EasyOCR)** - Excellent at finding text regions (377 detected)
2. **Dark Line Detection** - Building boundaries are drawn with dark lines
3. **Flood Fill with Line Boundaries** - Best method for finding enclosed building regions
4. **Text Masking** - Excluding text regions improves contour quality

### What Doesn't Work
1. **Color-only segmentation** - Building color blends with aged paper
2. **OpenCV Inpainting** - Creates artifacts, doesn't preserve lines
3. **Edge detection alone** - Too noisy, picks up text and streets

### Recommended Pipeline

```
Input Image
    │
    ▼
┌─────────────────┐
│ Text Detection  │ ──► Text Mask (exclude from processing)
│ (EasyOCR)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Dark Line       │ ──► Line boundaries (building edges)
│ Extraction      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Flood Fill      │ ──► Enclosed regions = buildings
│ (within bounds) │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Filter by:      │
│ - Area          │ ──► Clean building polygons
│ - Aspect ratio  │
│ - Text overlap  │
└─────────────────┘
    │
    ▼
   SVG Output
```

---

## Output Files

### Experiment 1
- `detected_text.png` - Text regions with bounding boxes
- `text_mask.png` - Binary mask of text areas
- `cleaned_image_ns.png` - Text removed (Navier-Stokes inpainting)
- `cleaned_image_telea.png` - Text removed (Telea inpainting)

### Experiment 2
- `hsv_mask.png` - HSV color detection mask
- `buildings.svg` - Extracted polygons (2 only)

### Experiment 3
- `edges_raw.png` - Raw Canny edges
- `edges_cleaned.png` - Morphologically cleaned edges
- `buildings.svg` - Extracted polygons (43)
- `hough_lines.png` - Detected line segments

### Experiment 4
- `text_mask.png` - Text regions to exclude
- `line_mask.png` - Dark line boundaries
- `contours_method2.png` - Building polygons overlaid (91 buildings)
- `buildings_method2.svg` - Best raster-based SVG output (91 polygons)

### Experiment 5 (Best)
- `buildings.svg` - **Best SVG output (325 building paths)**
- `analysis.json` - Processing statistics
- `ocr_cache.json` - Cached OCR bounding boxes
- `debug/text_highlight.svg` - Visualization (green=buildings, red=text)

---

## Next Steps

### Immediate Improvements
1. **Tune flood fill parameters** - Adjust line detection thresholds
2. **Add shape filtering** - Remove non-building shapes (streets, parks)
3. **Improve polygon simplification** - Reduce point count while preserving shape

### Advanced Options
1. **Try LaMa inpainting** - Better quality than OpenCV for text removal
2. **Train custom model** - Fine-tune SAM or similar on historical maps
3. **Use NYPL Map Vectorizer** - Existing tool for similar atlas digitization

### Integration with SAM
The flood-fill results could be used as:
1. **Pre-filtering** - Run SAM only on detected building regions
2. **Prompts** - Use flood-fill centroids as SAM point prompts
3. **Validation** - Compare SAM output against flood-fill boundaries

---

## Running the Experiments

```bash
cd /Users/pwright/workspace/atlas/experiment

# Run all experiments
uv run python 01-text-inpainting/text_removal.py
uv run python 02-color-segmentation/color_segment.py
uv run python 03-edge-morph/edge_extract.py
uv run python 04-hybrid/hybrid_extraction.py
uv run python 05-svg-cleanup/svg_cleanup.py
```

---

## Conclusion

The **SVG post-processing approach (Experiment 5)** produced the best results with **325 building paths** extracted. This method works because:

1. Inkscape's Edge Detection trace captures all lines cleanly as vectors
2. SVG paths can be split into individual subpaths programmatically
3. OCR bounding boxes effectively identify text regions for filtering
4. Vector output preserves sharp edges without rasterization artifacts

The SVG output at `05-svg-cleanup/results/buildings.svg` is ready for 3D extrusion testing.

### Comparison: Raster vs Vector Approach

| Approach | Experiment | Building Count | Edge Quality |
|----------|------------|----------------|--------------|
| Raster (flood fill) | 4 | 91 | Jagged |
| Vector (SVG cleanup) | 5 | 325 | Sharp |

The vector approach extracts ~3.5x more buildings with cleaner edges.
