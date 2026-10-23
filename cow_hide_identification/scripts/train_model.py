"""Train the cow hide identification model.

Examples (from the project folder):
    python scripts\\train_model.py
    python scripts\\train_model.py --resume              # continue from models\\checkpoints\\last_checkpoint.pth
    python scripts\\train_model.py --resplit             # rebuild data\\splits after adding images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cowid.config import load_config, resolve_cli_path, resolve_path  # noqa: E402
from cowid.data.dataset import DatasetValidationError  # noqa: E402
from cowid.training.train import train  # noqa: E402
from cowid.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Siamese cow hide identification model.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/training_config.yaml)")
    parser.add_argument("--resume", nargs="?", const="last", default=None,
                        help="Resume training; optionally give a checkpoint path (default: last_checkpoint.pth)")
    parser.add_argument("--resplit", action="store_true", help="Regenerate the train/val/test splits")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--device", help="Override training.device (auto, cpu, cuda)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device:
        cfg["training"]["device"] = args.device
    setup_logging(resolve_path(cfg["paths"]["output_dir"]) / "train.log")

    resume = None
    if args.resume == "last":
        resume = True
    elif args.resume:
        resume = resolve_cli_path(args.resume)
    try:
        train(cfg, resume=resume, resplit=args.resplit)
    except (DatasetValidationError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
