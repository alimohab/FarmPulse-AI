"""Triplet margin loss on cosine distance, with online semi-hard mining."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cowid.reid.similarity import cosine_similarity
from cowid.reid.triplet_mining import mine_semi_hard_triplets


def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                        margin: float = 0.2) -> torch.Tensor:
    """Per-triplet loss ``max(0, d(a,p) - d(a,n) + margin)`` with ``d = 1 - cosine similarity``."""
    d_ap = 1.0 - cosine_similarity(anchor, positive)
    d_an = 1.0 - cosine_similarity(anchor, negative)
    return F.relu(d_ap - d_an + margin)


class SemiHardTripletLoss(nn.Module):
    """Mines semi-hard triplets in the batch and averages their triplet margin loss.

    ``forward`` returns ``(loss, stats)``; ``stats["num_triplets"] == 0`` means the batch contained
    no valid triplet (the returned loss is then a zero that still supports ``backward``).
    """

    def __init__(self, margin: float = 0.2, cross_view: bool = False):
        super().__init__()
        self.margin = float(margin)
        self.cross_view = cross_view

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, views: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, int]]:
        embeddings = embeddings.float()
        triplets = mine_semi_hard_triplets(embeddings, labels, views, self.cross_view)
        if len(triplets) == 0:
            return embeddings.sum() * 0.0, {"num_triplets": 0, "num_active": 0}
        losses = triplet_margin_loss(embeddings[triplets.anchor], embeddings[triplets.positive],
                                     embeddings[triplets.negative], self.margin)
        return losses.mean(), {"num_triplets": len(triplets), "num_active": int((losses > 0).sum())}
