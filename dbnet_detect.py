#!/usr/bin/env python3
"""
DBNet++ Text Detection Visualization

Uses DBNet++ to detect text at the pixel level, outputting a probability
heatmap that can be thresholded to create a text mask.

Usage:
    uv run python dbnet_detect.py
    uv run python dbnet_detect.py -i /path/to/input.png -o /path/to/output.png
    uv run python dbnet_detect.py --threshold 0.1   # Very aggressive (more text)
    uv run python dbnet_detect.py --threshold 0.5   # More conservative
    uv run python dbnet_detect.py --save-heatmap heatmap.png  # Save raw probability map
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Default paths
DEFAULT_INPUT = Path(__file__).parent.parent / "media" / "p2-mono.png"
DEFAULT_OUTPUT = Path(__file__).parent / "dbnet_detected.png"

# Default threshold (low = aggressive, catches more text but may have false positives)
DEFAULT_THRESHOLD = 0.2


def get_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dbnet_model(device: str):
    """Load DBNet detection model."""
    from doctr.models import detection_predictor

    print(f"Loading DBNet model on {device}...")

    predictor = detection_predictor(
        arch='db_resnet50',
        pretrained=True,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
    )

    predictor.model = predictor.model.to(device)
    predictor.model.eval()

    return predictor.model


def get_text_heatmap(model, image: np.ndarray, device: str) -> np.ndarray:
    """
    Run DBNet and extract the text probability heatmap.

    Args:
        model: The DBNet model
        image: Input image (RGB, uint8)
        device: Device string

    Returns:
        Probability heatmap (float32, 0-1 range, same size as input)
    """
    height, width = image.shape[:2]

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Resize to make dimensions divisible by 32 (model requirement)
    # Also limit max size to avoid OOM
    max_size = 2048
    scale = min(max_size / height, max_size / width, 1.0)

    new_h = int(height * scale)
    new_w = int(width * scale)
    # Make dimensions divisible by 32
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Transform to tensor
    img_tensor = transform(resized).unsqueeze(0).to(device)

    # Run model to get probability map
    with torch.inference_mode():
        # Get features from backbone
        feats = model.feat_extractor(img_tensor)

        # Convert OrderedDict to list for FPN
        feats_list = [feats[str(i)] for i in range(len(feats))]

        # Run through FPN
        fpn_out = model.fpn(feats_list)

        # Get probability map from prob_head (returns logits)
        logits = model.prob_head(fpn_out)

        # Apply sigmoid to get probabilities
        prob_map = torch.sigmoid(logits)

    # Convert to numpy
    prob_map = prob_map.squeeze().cpu().numpy()

    # Resize back to original image size
    prob_map = cv2.resize(prob_map, (width, height), interpolation=cv2.INTER_LINEAR)

    return prob_map.astype(np.float32)


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    threshold: float,
    overlay_color: tuple = (255, 0, 255),  # Magenta
    overlay_alpha: float = 0.5
) -> np.ndarray:
    """
    Create an overlay image showing detected text pixels.

    Args:
        image: Original image (BGR)
        heatmap: Probability heatmap (0-1)
        threshold: Threshold for text detection
        overlay_color: BGR color for text highlight
        overlay_alpha: Transparency of overlay

    Returns:
        Image with text regions highlighted
    """
    # Create binary mask from threshold
    mask = (heatmap >= threshold).astype(np.uint8)

    # Create colored overlay
    overlay = image.copy()
    overlay[mask == 1] = overlay_color

    # Blend with original
    result = cv2.addWeighted(image, 1 - overlay_alpha, overlay, overlay_alpha, 0)

    # Also draw contours for clarity
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, overlay_color, 1)

    return result


def create_heatmap_visualization(heatmap: np.ndarray) -> np.ndarray:
    """Create a colorized heatmap visualization."""
    # Normalize to 0-255
    heatmap_norm = (heatmap * 255).astype(np.uint8)

    # Apply colormap (TURBO gives nice gradient from blue to red)
    colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_TURBO)

    return colored


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect text pixels using DBNet and visualize results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=DEFAULT_INPUT,
        help='Input image path'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output image path (overlay visualization)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help='Detection threshold (0-1, lower = more aggressive text detection)'
    )
    parser.add_argument(
        '--save-heatmap',
        type=Path,
        default=None,
        help='Save the raw probability heatmap to this path'
    )
    parser.add_argument(
        '--save-mask',
        type=Path,
        default=None,
        help='Save the binary mask to this path'
    )
    parser.add_argument(
        '--overlay-alpha',
        type=float,
        default=0.5,
        help='Transparency of the text overlay (0-1)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print()

    # Load image
    image_bgr = cv2.imread(str(args.input))
    if image_bgr is None:
        print(f"ERROR: Could not load image: {args.input}")
        return 1

    height, width = image_bgr.shape[:2]
    print(f"Image size: {width}x{height}")

    # Convert to RGB for the model
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Load model and get heatmap
    device = get_device()
    model = load_dbnet_model(device)

    print("Running DBNet detection...")
    heatmap = get_text_heatmap(model, image_rgb, device)

    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"Pixels above threshold: {(heatmap >= args.threshold).sum():,} ({(heatmap >= args.threshold).mean() * 100:.2f}%)")

    # Save heatmap if requested
    if args.save_heatmap:
        heatmap_vis = create_heatmap_visualization(heatmap)
        cv2.imwrite(str(args.save_heatmap), heatmap_vis)
        print(f"Saved heatmap to {args.save_heatmap}")

    # Save binary mask if requested
    if args.save_mask:
        mask = ((heatmap >= args.threshold) * 255).astype(np.uint8)
        cv2.imwrite(str(args.save_mask), mask)
        print(f"Saved mask to {args.save_mask}")

    # Create and save overlay
    print("\nCreating overlay visualization...")
    overlay = create_overlay(image_bgr, heatmap, args.threshold, overlay_alpha=args.overlay_alpha)
    cv2.imwrite(str(args.output), overlay)
    print(f"Saved overlay to {args.output}")

    output_size = args.output.stat().st_size
    print(f"Output file size: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
