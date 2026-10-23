"""Compare two cow images with the trained model and print their cosine similarity.

Example (from the project folder):
    python scripts\\compare_images.py path\\to\\cow_a.jpg path\\to\\cow_b.jpg
    python scripts\\compare_images.py a.jpg b.jpg --detect --threshold 0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cowid.config import resolve_cli_path, resolve_path  # noqa: E402
from cowid.inference import CowEmbedder  # noqa: E402
from cowid.utils.checkpoint_utils import BEST_CHECKPOINT_NAME  # noqa: E402
from cowid.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosine similarity between the hide patterns of two cow images.")
    parser.add_argument("image1", type=Path)
    parser.add_argument("image2", type=Path)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint (default: models/checkpoints/best_model.pth)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--detect", action="store_true",
                        help="Crop the largest cow with YOLO before embedding (needs ultralytics)")
    parser.add_argument("--detector-model", default="yolo11s.pt", help="YOLO weights for --detect")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional decision threshold; pick it from the evaluation's pair similarities")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    for image in (args.image1, args.image2):
        if not image.is_file():
            print(f"ERROR: image not found: {image}", file=sys.stderr)
            return 1
    checkpoint = (resolve_cli_path(args.checkpoint) if args.checkpoint
                  else resolve_path("models/checkpoints") / BEST_CHECKPOINT_NAME)

    cropper = None
    if args.detect:
        from cowid.detection.detector import CowCropper
        cropper = CowCropper(args.detector_model, device=args.device)
    try:
        embedder = CowEmbedder.from_checkpoint(checkpoint, device=args.device, cropper=cropper)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    score = embedder.compare(args.image1, args.image2)
    info = embedder.checkpoint_info
    print(f"Checkpoint : {checkpoint} (epoch {info['epoch']}, best {info['monitor']} = {info['best_metric']})")
    print(f"Image 1    : {args.image1}")
    print(f"Image 2    : {args.image2}")
    print(f"Cosine similarity: {score:.4f}")
    if args.threshold is not None:
        verdict = "SAME cow" if score >= args.threshold else "DIFFERENT cows"
        print(f"Decision (threshold {args.threshold:.2f}): {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
