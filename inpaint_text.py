#!/usr/bin/env python3
"""
Text Detection and Inpainting

Detects text regions using DBNet++ or CRAFT pixel-level detectors, then uses LaMa
(Large Mask Inpainting) to remove the text and fill in the background.

Usage:
    uv run python inpaint_text.py
    uv run python inpaint_text.py -i /path/to/input.png -o /path/to/output.png
    uv run python inpaint_text.py --expand 5      # Expand mask by 5 pixels

    # Detector selection (default: dbnet)
    uv run python inpaint_text.py --detector dbnet   # DBNet++ (recommended)
    uv run python inpaint_text.py --detector craft   # CRAFT

    # Threshold tuning (0-1, lower = more aggressive text detection)
    uv run python inpaint_text.py --threshold 0.2    # Default, balanced
    uv run python inpaint_text.py --threshold 0.1    # Aggressive (more text detected)
    uv run python inpaint_text.py --threshold 0.5    # Conservative (less text)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Default paths
DEFAULT_INPUT = Path(__file__).parent.parent / "media" / "p2-mono.png"
DEFAULT_OUTPUT = Path(__file__).parent / "inpainted.png"

# Default detector
DEFAULT_DETECTOR = "dbnet"

# Default threshold for pixel-level detectors
DEFAULT_THRESHOLD = 0.2


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_text_dbnet(image_path: Path, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Detect text using DBNet++ and return a binary mask.

    Args:
        image_path: Path to the input image
        threshold: Detection threshold (0-1, lower = more aggressive)

    Returns:
        Binary mask (uint8) with text pixels as 255
    """
    from doctr.models import detection_predictor
    from torchvision import transforms

    device = get_device()
    print(f"Loading DBNet++ model on {device}...")

    predictor = detection_predictor(
        arch='db_resnet50',
        pretrained=True,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
    )
    model = predictor.model.to(device)
    model.eval()

    # Load and prepare image
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Resize to make dimensions divisible by 32 (model requirement)
    max_size = 2048
    scale = min(max_size / height, max_size / width, 1.0)
    new_h = ((int(height * scale) + 31) // 32) * 32
    new_w = ((int(width * scale) + 31) // 32) * 32

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_tensor = transform(resized).unsqueeze(0).to(device)

    print(f"Running DBNet++ detection (threshold={threshold})...")

    with torch.inference_mode():
        feats = model.feat_extractor(img_tensor)
        feats_list = [feats[str(i)] for i in range(len(feats))]
        fpn_out = model.fpn(feats_list)
        logits = model.prob_head(fpn_out)
        prob_map = torch.sigmoid(logits)

    # Convert to numpy and resize back
    prob_map = prob_map.squeeze().cpu().numpy()
    prob_map = cv2.resize(prob_map, (width, height), interpolation=cv2.INTER_LINEAR)

    print(f"Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")

    # Apply threshold to create binary mask
    mask = ((prob_map >= threshold) * 255).astype(np.uint8)

    pixel_count = np.count_nonzero(mask)
    print(f"Detected {pixel_count:,} text pixels ({pixel_count / (width * height) * 100:.2f}%)")

    return mask


def detect_text_craft(image_path: Path, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Detect text using CRAFT and return a binary mask.

    Args:
        image_path: Path to the input image
        threshold: Detection threshold (0-1, lower = more aggressive)

    Returns:
        Binary mask (uint8) with text pixels as 255
    """
    from craft_text_detector import Craft

    device = get_device()
    device_str = str(device)
    # CRAFT uses 'cuda' string, not 'mps'
    cuda = device_str == "cuda"

    print(f"Loading CRAFT model (cuda={cuda})...")

    # Initialize CRAFT detector
    craft = Craft(
        output_dir=None,  # Don't save outputs
        crop_type="poly",
        cuda=cuda,
        text_threshold=threshold,
        link_threshold=threshold * 0.5,  # Link threshold typically lower
        low_text=threshold * 0.7,
    )

    # Load image
    image_bgr = cv2.imread(str(image_path))
    height, width = image_bgr.shape[:2]

    print(f"Running CRAFT detection (threshold={threshold})...")

    # Run detection to get heatmaps
    prediction_result = craft.detect_text(str(image_path))

    # Get the text score heatmap
    heatmaps = prediction_result["heatmaps"]
    text_score = heatmaps["text_score_heatmap"]

    # The heatmap is already normalized 0-255, convert to 0-1
    if text_score.max() > 1:
        text_score = text_score / 255.0

    # Resize to original image size if needed
    if text_score.shape[:2] != (height, width):
        text_score = cv2.resize(text_score, (width, height), interpolation=cv2.INTER_LINEAR)

    print(f"Heatmap range: [{text_score.min():.4f}, {text_score.max():.4f}]")

    # Apply threshold to create binary mask
    mask = ((text_score >= threshold) * 255).astype(np.uint8)

    # Cleanup CRAFT resources
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    pixel_count = np.count_nonzero(mask)
    print(f"Detected {pixel_count:,} text pixels ({pixel_count / (width * height) * 100:.2f}%)")

    return mask


def detect_text(
    image_path: Path,
    detector: str = DEFAULT_DETECTOR,
    threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """
    Detect text using the specified detector.

    Args:
        image_path: Path to the input image
        detector: Detector to use ('dbnet' or 'craft')
        threshold: Detection threshold (0-1, lower = more aggressive)

    Returns:
        Binary mask (uint8) with text pixels as 255
    """
    if detector == "dbnet":
        return detect_text_dbnet(image_path, threshold)
    elif detector == "craft":
        return detect_text_craft(image_path, threshold)
    else:
        raise ValueError(f"Unknown detector: {detector}. Use 'dbnet' or 'craft'.")


def expand_mask(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    """
    Expand a binary mask by dilation.

    Args:
        mask: Binary mask (uint8)
        expand_pixels: Number of pixels to expand by

    Returns:
        Expanded mask
    """
    if expand_pixels <= 0:
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
    )
    return cv2.dilate(mask, kernel)


def load_lama_model(device: torch.device):
    """Load LaMa model with proper device mapping for non-CUDA systems."""
    from simple_lama_inpainting.utils import download_model
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL

    model_path = download_model(LAMA_MODEL_URL)

    # Load with map_location to handle CUDA->CPU/MPS conversion
    model = torch.jit.load(model_path, map_location='cpu')
    model = model.to(device)
    model.eval()

    return model


def inpaint_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Inpaint using LaMa (Large Mask Inpainting) deep learning model.

    Args:
        image: Input image (BGR)
        mask: Binary mask where 255 = regions to inpaint

    Returns:
        Inpainted image (BGR)
    """
    from simple_lama_inpainting.utils import prepare_img_and_mask

    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask)

    # Load model with proper device handling
    device = get_device()
    print(f"Loading LaMa model on {device}...")
    model = load_lama_model(device)

    # Prepare inputs
    img_tensor, mask_tensor = prepare_img_and_mask(pil_image, pil_mask, device)

    # Run inference
    with torch.inference_mode():
        inpainted = model(img_tensor, mask_tensor)
        cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

    # Convert RGB back to BGR
    result_bgr = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    return result_bgr


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect and remove text from images using DBNet++/CRAFT + LaMa inpainting.',
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
        help='Output image path'
    )
    parser.add_argument(
        '--expand',
        type=int,
        default=3,
        help='Pixels to expand text mask by (dilation)'
    )

    # Detector selection
    detector_group = parser.add_argument_group('Detector options')
    detector_group.add_argument(
        '--detector',
        type=str,
        choices=['dbnet', 'craft'],
        default=DEFAULT_DETECTOR,
        help='Text detection model to use'
    )
    detector_group.add_argument(
        '-t', '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help='Detection threshold (0-1, lower = more aggressive text detection)'
    )

    # Debug options
    debug_group = parser.add_argument_group('Debug options')
    debug_group.add_argument(
        '--save-mask',
        type=Path,
        default=None,
        help='Save the text mask image to this path'
    )
    debug_group.add_argument(
        '--mask-only',
        action='store_true',
        help='Only generate the mask, skip inpainting'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Detector: {args.detector}")
    print(f"Threshold: {args.threshold}")
    print(f"Mask expansion: {args.expand}px")
    print()

    # Load image
    image = cv2.imread(str(args.input))
    if image is None:
        print(f"ERROR: Could not load image: {args.input}")
        return 1

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    # Detect text and create mask
    mask = detect_text(args.input, args.detector, args.threshold)

    # Expand mask if requested
    if args.expand > 0:
        print(f"\nExpanding mask by {args.expand}px...")
        mask = expand_mask(mask, args.expand)

    mask_coverage = np.count_nonzero(mask) / (width * height) * 100
    print(f"Final mask coverage: {mask_coverage:.2f}%")

    # Save mask if requested
    if args.save_mask:
        cv2.imwrite(str(args.save_mask), mask)
        print(f"Saved mask to {args.save_mask}")

    # If mask-only mode, we're done
    if args.mask_only:
        if not args.save_mask:
            # Default to saving mask with output path
            cv2.imwrite(str(args.output), mask)
            print(f"Saved mask to {args.output}")
        print("Done (mask-only mode)!")
        return 0

    # Check if there's anything to inpaint
    if np.count_nonzero(mask) == 0:
        print("No text detected - copying input to output")
        cv2.imwrite(str(args.output), image)
        return 0

    # Inpaint with LaMa
    print(f"\nInpainting with LaMa...")
    result = inpaint_lama(image, mask)

    # Save result
    cv2.imwrite(str(args.output), result)
    print(f"\nSaved inpainted image to {args.output}")

    output_size = args.output.stat().st_size
    print(f"Output file size: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
