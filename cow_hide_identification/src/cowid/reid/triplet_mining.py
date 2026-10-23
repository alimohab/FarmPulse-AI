"""Online semi-hard triplet mining within a batch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from cowid.reid.similarity import pairwise_cosine_distance


@dataclass
class Triplets:
    """Index tensors into the batch: ``(anchor[i], positive[i], negative[i])`` is one triplet."""

    anchor: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor

    def __len__(self) -> int:
        return int(self.anchor.numel())


def build_pair_masks(labels: torch.Tensor, views: Optional[torch.Tensor] = None, cross_view: bool = False
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Boolean ``[N, N]`` masks of valid positive and negative partners for each anchor.

    Positives: same cow, different image, and the same view unless ``cross_view`` is True.
    Negatives: any image of a different cow. A same-cow image from another view is neither
    (it is ignored) while training same-view, so it is never pushed away.
    """
    same_cow = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = same_cow & ~eye
    if not cross_view and views is not None:
        positive &= views[:, None] == views[None, :]
    return positive, ~same_cow


@torch.no_grad()
def mine_semi_hard_triplets(embeddings: torch.Tensor, labels: torch.Tensor,
                            views: Optional[torch.Tensor] = None, cross_view: bool = False) -> Triplets:
    """Select one negative for every valid anchor-positive pair (FaceNet semi-hard mining).

    The chosen negative is the closest negative that is still farther from the anchor than the
    positive, i.e. ``d(a,p) < d(a,n)``; it yields a loss when ``d(a,n) < d(a,p) + margin``.
    If every negative is closer than the positive, the farthest negative is used instead (as in
    TensorFlow Addons' semi-hard loss), which avoids collapsing embeddings early in training.
    Distances are cosine distances ``1 - cos``.
    """
    empty = torch.empty(0, dtype=torch.long, device=embeddings.device)
    pos_mask, neg_mask = build_pair_masks(labels, views, cross_view)
    a_idx, p_idx = pos_mask.nonzero(as_tuple=True)
    if a_idx.numel() == 0 or not neg_mask.any():
        return Triplets(empty, empty, empty)

    dist = pairwise_cosine_distance(embeddings.float())
    d_ap = dist[a_idx, p_idx]                      # [M]
    d_an = dist[a_idx]                             # [M, N]
    negatives = neg_mask[a_idx]                    # [M, N]
    semi_hard = negatives & (d_an > d_ap[:, None])

    closest_semi_hard = d_an.masked_fill(~semi_hard, float("inf")).argmin(dim=1)
    farthest_negative = d_an.masked_fill(~negatives, float("-inf")).argmax(dim=1)
    n_idx = torch.where(semi_hard.any(dim=1), closest_semi_hard, farthest_negative)

    keep = negatives.any(dim=1)
    return Triplets(a_idx[keep], p_idx[keep], n_idx[keep])
