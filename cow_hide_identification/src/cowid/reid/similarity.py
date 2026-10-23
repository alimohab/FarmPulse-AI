"""Cosine similarity / distance between embeddings."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity of two ``[N, D]`` (or ``[D]``) tensors, in [-1, 1]."""
    return F.cosine_similarity(a, b, dim=-1)


def pairwise_cosine_similarity(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``[N, M]`` matrix of cosine similarities between rows of ``a`` and rows of ``b`` (default ``a``)."""
    a = F.normalize(a, p=2, dim=1)
    b = a if b is None else F.normalize(b, p=2, dim=1)
    return a @ b.T


def pairwise_cosine_distance(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``1 - cosine similarity``, in [0, 2]."""
    return (1.0 - pairwise_cosine_similarity(a, b)).clamp_min(0.0)
