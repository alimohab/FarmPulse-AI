"""Evaluate the best checkpoint on the test split and save the results as JSON.

Example (from the project folder):
    python scripts\\evaluate_model.py
    python scripts\\evaluate_model.py --split val --checkpoint models\\checkpoints\\last_checkpoint.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cowid.config import load_config, resolve_cli_path, resolve_path  # noqa: E402
from cowid.data.dataset import DatasetValidationError  # noqa: E402
from cowid.training.validate import evaluate_checkpoint  # noqa: E402
from cowid.utils.checkpoint_utils import BEST_CHECKPOINT_NAME  # noqa: E402
from cowid.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained cow identification checkpoint.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/training_config.yaml)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint (default: models/checkpoints/best_model.pth)")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="Split used as queries")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--output", default=None, help="Results JSON (default: outputs/evaluation_<split>.json)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    setup_logging(output_dir / "evaluate.log")
    checkpoint = (resolve_cli_path(args.checkpoint) if args.checkpoint
                  else resolve_path(cfg["paths"]["checkpoint_dir"]) / BEST_CHECKPOINT_NAME)
    try:
        results = evaluate_checkpoint(cfg, checkpoint, split=args.split, device_preference=args.device)
    except (DatasetValidationError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    output = resolve_cli_path(args.output) if args.output else output_dir / f"evaluation_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    m = results["metrics"]
    print(f"\nEvaluation on the {args.split} split ({results['num_query_images']} queries vs "
          f"{results['num_gallery_images']} gallery images, checkpoint epoch {results['checkpoint_epoch']})")
    for key, value in m.items():
        print(f"  {key:<26} {value if not isinstance(value, float) else f'{value:.4f}'}")
    ps = results["pair_similarity"]
    if ps["same_cow_mean"] is not None and ps["different_cow_mean"] is not None:
        print(f"  mean cosine similarity: same cow {ps['same_cow_mean']:.3f}, "
              f"different cows {ps['different_cow_mean']:.3f}")
    print(f"\nResults saved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
