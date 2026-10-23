"""Saving/loading checkpoints and tracking the best validation metric."""
from __future__ import annotations

import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from cowid.reid.siamese_network import SiameseNetwork

logger = logging.getLogger(__name__)

BEST_CHECKPOINT_NAME = "best_model.pth"
LAST_CHECKPOINT_NAME = "last_checkpoint.pth"
CHECKPOINT_FORMAT_VERSION = 1


class BestMetricTracker:
    """Decides whether a validation result improves on the best one seen so far.

    ``mode="max"`` for accuracy-like metrics, ``"min"`` for losses. An improvement must beat the best
    value by more than ``min_delta``; ``None``/NaN never counts as an improvement.
    """

    def __init__(self, mode: str = "max", min_delta: float = 0.0):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs = 0

    def is_improvement(self, value: Optional[float]) -> bool:
        if value is None or math.isnan(value):
            return False
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: Optional[float], epoch: int) -> bool:
        """Record an epoch's value; returns True if it is a new best."""
        if self.is_improvement(value):
            self.best, self.best_epoch, self.num_bad_epochs = float(value), epoch, 0
            return True
        self.num_bad_epochs += 1
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "min_delta": self.min_delta, "best": self.best,
                "best_epoch": self.best_epoch, "num_bad_epochs": self.num_bad_epochs}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best = state.get("best")
        self.best_epoch = state.get("best_epoch")
        self.num_bad_epochs = int(state.get("num_bad_epochs", 0))


def save_checkpoint(path: Union[str, Path], model: SiameseNetwork, epoch: int, *,
                    optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None,
                    scaler: Optional[Any] = None, best_metric: Optional[float] = None, monitor: str = "",
                    config: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a checkpoint usable for both inference and resuming training.

    The file is written to a temporary name and then atomically renamed, so an interrupted save
    never leaves a corrupted ``best_model.pth`` behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_size = (config or {}).get("data", {}).get("image_size")
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "monitor": monitor,
        "model_config": dict(model.config),
        "embedding_dim": model.embedding_dim,
        "image_size": [int(s) for s in image_size] if image_size else None,
        "training_config": config,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),  # plain str: TorchVersion is rejected by weights_only loading
    }
    payload.update(extra or {})
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dictionary (tensors + plain Python types only, so ``weights_only`` is safe)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}\n"
                                "Train a model first: python scripts\\train_model.py")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    missing = [k for k in ("model_state_dict", "model_config", "epoch") if k not in checkpoint]
    if missing:
        raise ValueError(f"{path} is not a cowid checkpoint (missing {missing})")
    return checkpoint


def build_model_from_checkpoint(checkpoint: Dict[str, Any], device: Union[str, torch.device] = "cpu"
                                ) -> SiameseNetwork:
    """Recreate the network from the stored config and load its weights (no ImageNet download)."""
    model = SiameseNetwork.from_config(checkpoint["model_config"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_model_for_inference(path: Union[str, Path], device: Union[str, torch.device] = "cpu"
                             ) -> Tuple[SiameseNetwork, Dict[str, Any]]:
    """Load a checkpoint and return ``(model in eval mode, checkpoint dict)``."""
    checkpoint = load_checkpoint(path, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    return model, checkpoint


def archive_existing_checkpoints(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Move best/last checkpoints of a previous run to ``archive/<timestamp>/`` before a fresh run."""
    checkpoint_dir = Path(checkpoint_dir)
    existing = [checkpoint_dir / n for n in (BEST_CHECKPOINT_NAME, LAST_CHECKPOINT_NAME)
                if (checkpoint_dir / n).is_file()]
    if not existing:
        return None
    archive = checkpoint_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), str(archive / p.name))
    return archive
