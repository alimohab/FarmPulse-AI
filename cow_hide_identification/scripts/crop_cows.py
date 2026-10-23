"""Optional preprocessing: crop the largest cow out of every image with YOLO.

Mirrors data\\raw (Cow_XXX\\<view>\\...) into data\\processed. Afterwards set
`paths.dataset_dir: data/processed` in configs\\training_config.yaml. Needs `pip install ultralytics`.

Example (from the project folder):
    python scripts\\crop_cows.py --model ..\\cow-detection\\models\\yolo11m.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cowid.config import load_config, resolve_cli_path  # noqa: E402
from cowid.detection.detector import CowCropper, crop_dataset  # noqa: E402
from cowid.utils.logging_utils import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop cows from Cow_XXX/<view> images with YOLO.")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO weights with a 'cow' class")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()

    data_cfg = load_config(args.config)["data"]
    cropper = CowCropper(args.model, conf=args.conf, device=args.device)
    counts = crop_dataset(resolve_cli_path(args.input_dir), resolve_cli_path(args.output_dir), cropper,
                          data_cfg["views"], data_cfg.get("image_extensions", (".jpg", ".jpeg", ".png")),
                          overwrite=args.overwrite)
    print(f"Cropped {counts['cropped']} image(s); no cow found in {counts['no_cow']}; "
          f"{counts['skipped_existing']} already existed. Review the crops before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
