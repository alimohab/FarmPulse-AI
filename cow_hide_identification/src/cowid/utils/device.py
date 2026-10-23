"""Device selection and reproducibility helpers."""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Return CUDA when available (or requested), otherwise the CPU."""
    pref = (preference or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Device '%s' requested but CUDA is not available; using the CPU instead.", preference)
        return torch.device("cpu")
    return torch.device(pref)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
