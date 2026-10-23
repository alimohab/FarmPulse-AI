"""Configuration loading, validation and project-root-aware path resolution."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"

_REQUIRED_KEYS = {
    "paths": ("dataset_dir", "splits_dir", "checkpoint_dir", "output_dir"),
    "data": ("views", "image_size", "split_ratios"),
    "model": ("backbone", "embedding_dim"),
    "training": ("epochs", "batch_size", "samples_per_class", "learning_rate", "margin"),
    "evaluation": ("recall_k",),
}
_MONITOR_PATTERN = re.compile(r"^(top1|mAP|val_loss|recall@\d+)$")


def resolve_path(path: PathLike) -> Path:
    """Return an absolute path; relative paths are interpreted from the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def resolve_cli_path(path: PathLike) -> Path:
    """Resolve a path typed on the command line: relative to the CWD if it exists there, else the project root."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    return resolve_path(p)


def load_config(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load and validate a YAML training configuration (defaults to configs/training_config.yaml)."""
    cfg_path = resolve_cli_path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_config(cfg, source=str(cfg_path))
    return cfg


def validate_config(cfg: Dict[str, Any], source: str = "config") -> None:
    """Raise ``ValueError`` with a readable message if the configuration is incomplete or inconsistent."""
    for section, keys in _REQUIRED_KEYS.items():
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{source}: missing section '{section}'")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise ValueError(f"{source}: missing key(s) {missing} in section '{section}'")

    training = cfg["training"]
    k, batch_size = int(training["samples_per_class"]), int(training["batch_size"])
    if k < 2:
        raise ValueError(f"{source}: training.samples_per_class must be >= 2 so batches contain positive pairs")
    if batch_size < 2 * k or batch_size % k:
        raise ValueError(f"{source}: training.batch_size ({batch_size}) must be a multiple of "
                         f"samples_per_class ({k}) holding at least two groups")
    if int(training["epochs"]) < 1:
        raise ValueError(f"{source}: training.epochs must be >= 1")
    monitor = str(training.get("monitor", "top1"))
    if not _MONITOR_PATTERN.match(monitor):
        raise ValueError(f"{source}: training.monitor must be top1, mAP, val_loss or recall@K (got '{monitor}')")

    size = cfg["data"]["image_size"]
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(int(s) > 0 for s in size)):
        raise ValueError(f"{source}: data.image_size must be [height, width]")

    ratios = cfg["data"]["split_ratios"]
    if set(ratios) != {"train", "val", "test"} or any(float(v) < 0 for v in ratios.values()):
        raise ValueError(f"{source}: data.split_ratios needs non-negative train, val and test entries")
    if float(ratios["train"]) <= 0 or float(ratios["val"]) <= 0:
        raise ValueError(f"{source}: data.split_ratios train and val must be > 0 (val selects the best model)")
    if not math.isclose(sum(float(v) for v in ratios.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"{source}: data.split_ratios must sum to 1.0")
