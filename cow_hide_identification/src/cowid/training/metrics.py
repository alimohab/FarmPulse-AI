"""Retrieval metrics for re-identification: top-1 accuracy, Recall@K and mAP."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from cowid.reid.similarity import pairwise_cosine_similarity


def retrieval_metrics(query_embeddings: torch.Tensor, query_labels: torch.Tensor,
                      gallery_embeddings: torch.Tensor, gallery_labels: torch.Tensor,
                      ks: Sequence[int] = (1, 5, 10), query_views: Optional[torch.Tensor] = None,
                      gallery_views: Optional[torch.Tensor] = None, same_view: bool = False
                      ) -> Dict[str, Optional[float]]:
    """Rank the gallery by cosine similarity for each query and score the ranking.

    * ``top1``: fraction of queries whose most similar gallery image shows the same cow.
    * ``recall@K``: fraction with the correct cow among the K most similar images; only reported
      when every query has at least K candidate gallery images.
    * ``mAP``: mean average precision over all correct gallery images.

    With ``same_view`` a query is only compared to gallery images of its own view. Queries whose cow
    has no candidate in the gallery cannot be answered and are excluded (counted separately).
    """
    if len(query_embeddings) == 0 or len(gallery_embeddings) == 0:
        raise ValueError("retrieval_metrics needs at least one query and one gallery embedding")
    sim = pairwise_cosine_similarity(query_embeddings.float(), gallery_embeddings.float())
    valid = torch.ones_like(sim, dtype=torch.bool)
    if same_view:
        if query_views is None or gallery_views is None:
            raise ValueError("same_view retrieval needs query_views and gallery_views")
        valid = query_views[:, None] == gallery_views[None, :]
    sim = sim.masked_fill(~valid, float("-inf"))

    order = sim.argsort(dim=1, descending=True)
    ranked_valid = valid.gather(1, order)
    matches = (gallery_labels[order] == query_labels[:, None]) & ranked_valid
    answerable = matches.any(dim=1)

    result: Dict[str, Optional[float]] = {"num_queries": int(answerable.sum()),
                                          "num_unanswerable_queries": int((~answerable).sum()),
                                          "top1": None, "mAP": None}
    if not answerable.any():
        return result
    matches, ranked_valid = matches[answerable], ranked_valid[answerable]
    result["top1"] = matches[:, 0].float().mean().item()
    min_candidates = int(ranked_valid.sum(dim=1).min())
    for k in sorted({int(k) for k in ks if int(k) > 1}):
        if k <= min_candidates:
            result[f"recall@{k}"] = matches[:, :k].any(dim=1).float().mean().item()

    hits = matches.float()
    ranks = torch.arange(1, hits.shape[1] + 1, dtype=torch.float32)
    precision_at_hit = hits.cumsum(dim=1) / ranks * hits
    result["mAP"] = (precision_at_hit.sum(dim=1) / hits.sum(dim=1)).mean().item()
    return result


def pair_similarity_stats(embeddings: torch.Tensor, labels: torch.Tensor,
                          views: Optional[torch.Tensor] = None, same_view: bool = False
                          ) -> Dict[str, Optional[float]]:
    """Mean/std cosine similarity of same-cow vs. different-cow image pairs (helps pick a threshold)."""
    n = len(embeddings)
    i, j = torch.triu_indices(n, n, offset=1)
    sims = pairwise_cosine_similarity(embeddings.float())[i, j]
    same = labels[i] == labels[j]
    considered = (views[i] == views[j]) if (same_view and views is not None) else torch.ones_like(same)

    def _stats(mask: torch.Tensor, prefix: str) -> Dict[str, Optional[float]]:
        values = sims[mask & considered]
        return {f"{prefix}_pairs": int(values.numel()),
                f"{prefix}_mean": values.mean().item() if values.numel() else None,
                f"{prefix}_std": values.std().item() if values.numel() > 1 else None}

    return {**_stats(same, "same_cow"), **_stats(~same, "different_cow")}
