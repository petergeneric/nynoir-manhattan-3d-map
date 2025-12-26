"""Visualization utilities for comparing experiment results."""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2


def overlay_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    colors: List[Tuple[int, int, int]] = None,
    thickness: int = 2,
    fill: bool = False,
    fill_alpha: float = 0.3,
) -> np.ndarray:
    """Overlay contours on an image.

    Args:
        image: Input image (BGR)
        contours: List of contours to draw
        colors: Optional list of BGR colors
        thickness: Line thickness (-1 for filled)
        fill: Whether to fill polygons with transparent color
        fill_alpha: Alpha value for fill (0-1)

    Returns:
        Image with contours overlaid
    """
    result = image.copy()

    # Generate colors if not provided
    if colors is None:
        np.random.seed(42)
        colors = [
            (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
            for _ in range(len(contours))
        ]

    if fill:
        # Create overlay for transparent fill
        overlay = result.copy()
        for idx, contour in enumerate(contours):
            color = colors[idx % len(colors)]
            cv2.fillPoly(overlay, [contour], color)
        result = cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0)

    # Draw contour outlines
    for idx, contour in enumerate(contours):
        color = colors[idx % len(colors)]
        cv2.drawContours(result, [contour], -1, color, thickness)

    return result


def save_comparison(
    images: List[np.ndarray],
    titles: List[str],
    output_path: Path,
    max_width: int = 1920,
) -> None:
    """Save a side-by-side comparison of multiple images.

    Args:
        images: List of images to compare
        titles: List of titles for each image
        output_path: Path to save comparison image
        max_width: Maximum width of combined image
    """
    if len(images) != len(titles):
        raise ValueError("Number of images must match number of titles")

    # Convert grayscale images to BGR
    images_bgr = []
    for img in images:
        if len(img.shape) == 2:
            images_bgr.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        else:
            images_bgr.append(img)
    images = images_bgr

    # Calculate target size for each image
    n_images = len(images)
    target_width = max_width // n_images

    # Resize images to same height
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        new_w = target_width
        resized_img = cv2.resize(img, (new_w, new_h))
        resized.append(resized_img)

    # Make all images same height (pad if needed)
    max_height = max(img.shape[0] for img in resized)
    padded = []
    for img in resized:
        h, w = img.shape[:2]
        if h < max_height:
            pad = np.zeros((max_height - h, w, 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    # Add titles
    titled = []
    for img, title in zip(padded, titles):
        # Add title bar
        title_bar = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            title_bar, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        titled.append(np.vstack([title_bar, img]))

    # Combine horizontally
    combined = np.hstack(titled)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    print(f"Saved comparison: {output_path}")


def create_debug_visualization(
    original: np.ndarray,
    stages: List[Tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str = "debug",
) -> None:
    """Create debug visualizations for each processing stage.

    Args:
        original: Original input image
        stages: List of (name, image) tuples for each stage
        output_dir: Directory to save debug images
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    cv2.imwrite(str(output_dir / f"{prefix}_00_original.png"), original)

    # Save each stage
    for idx, (name, image) in enumerate(stages, 1):
        # Convert grayscale to BGR for consistency
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        filename = f"{prefix}_{idx:02d}_{name}.png"
        cv2.imwrite(str(output_dir / filename), image)

    print(f"Saved {len(stages) + 1} debug images to {output_dir}")


def draw_text_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image (for text detection visualization).

    Args:
        image: Input image
        boxes: List of (x, y, w, h) bounding boxes
        color: BGR color
        thickness: Line thickness

    Returns:
        Image with boxes drawn
    """
    result = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result
