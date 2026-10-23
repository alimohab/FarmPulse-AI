"""Embedding extraction, validation during training and checkpoint evaluation."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from cowid.config import resolve_path
from cowid.data.augmentations import build_eval_transform
from cowid.data.dataset import (DatasetValidationError, PKBatchSampler, build_label_maps,
                                make_dataloader)
from cowid.data.split_dataset import SPLITS, prepare_splits
from cowid.reid.losses import SemiHardTripletLoss
from cowid.reid.siamese_network import SiameseNetwork
from cowid.training.metrics import pair_similarity_stats, retrieval_metrics
from cowid.utils.checkpoint_utils import build_model_from_checkpoint, load_checkpoint
from cowid.utils.device import get_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_embeddings(model: SiameseNetwork, loader: DataLoader, device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed every image of a loader; returns ``(embeddings, cow_indices, view_indices)`` on the CPU."""
    model.eval()
    embeddings, labels, views = [], [], []
    for images, cow_idx, view_idx in loader:
        embeddings.append(model(images.to(device, non_blocking=True)).float().cpu())
        labels.append(cow_idx)
        views.append(view_idx)
    if not embeddings:
        empty = torch.empty(0, dtype=torch.long)
        return torch.empty(0, model.embedding_dim), empty, empty
    return torch.cat(embeddings), torch.cat(labels), torch.cat(views)


@torch.no_grad()
def compute_triplet_loss(model: SiameseNetwork, loader: DataLoader, loss_fn: SemiHardTripletLoss,
                         device: torch.device) -> Optional[float]:
    """Triplet-weighted mean loss over PK batches; ``None`` if no batch produced a triplet."""
    model.eval()
    total, count = 0.0, 0
    for images, labels, views in loader:
        loss, stats = loss_fn(model(images.to(device)), labels.to(device), views.to(device))
        total += loss.item() * stats["num_triplets"]
        count += stats["num_triplets"]
    return total / count if count else None


def try_build_loss_loader(samples, data_root, cow_to_idx, view_to_idx, transform, train_cfg: Dict[str, Any],
                          seed: int, num_workers: int = 0, pin_memory: bool = False) -> Optional[DataLoader]:
    """Deterministic PK loader for evaluation loss, or ``None`` if the split cannot form triplets."""
    try:
        sampler = PKBatchSampler(samples, int(train_cfg["batch_size"]), int(train_cfg["samples_per_class"]),
                                 bool(train_cfg.get("cross_view_positives", False)), seed=seed)
    except DatasetValidationError as exc:
        logger.warning("Triplet loss unavailable for this split: %s", exc)
        return None
    return make_dataloader(samples, data_root, cow_to_idx, view_to_idx, transform, batch_sampler=sampler,
                           num_workers=num_workers, pin_memory=pin_memory)


def validate(model: SiameseNetwork, *, query_loader: DataLoader, gallery_loader: DataLoader,
             device: torch.device, loss_fn: Optional[SemiHardTripletLoss] = None,
             loss_loader: Optional[DataLoader] = None, ks: Sequence[int] = (1, 5, 10),
             same_view: bool = True) -> Dict[str, Optional[float]]:
    """Retrieval metrics (queries vs. gallery) plus the triplet loss on the query split."""
    q_emb, q_lab, q_view = compute_embeddings(model, query_loader, device)
    g_emb, g_lab, g_view = compute_embeddings(model, gallery_loader, device)
    metrics = retrieval_metrics(q_emb, q_lab, g_emb, g_lab, ks, q_view, g_view, same_view)
    metrics["val_loss"] = (compute_triplet_loss(model, loss_loader, loss_fn, device)
                           if loss_loader is not None and loss_fn is not None else None)
    return metrics


def evaluate_checkpoint(cfg: Dict[str, Any], checkpoint_path: Union[str, Path], split: str = "test",
                        device_preference: str = "auto") -> Dict[str, Any]:
    """Evaluate a checkpoint on ``split`` (queries) against all earlier splits (gallery).

    ``test`` uses train + val as the gallery; ``val`` uses train, matching training-time validation.
    """
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    device = get_device(device_preference)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device).eval()

    splits = prepare_splits(cfg, create_if_missing=False)
    queries = splits[split]
    gallery = [s for name in SPLITS[:SPLITS.index(split)] for s in splits[name]]
    if not queries:
        raise DatasetValidationError(f"The {split} split is empty; there is nothing to evaluate.")
    if not gallery:
        raise DatasetValidationError("The gallery (training images) is empty.")

    data_cfg, train_cfg = cfg["data"], cfg["training"]
    data_root = resolve_path(cfg["paths"]["dataset_dir"])
    cow_to_idx, view_to_idx = build_label_maps([s for items in splits.values() for s in items],
                                               data_cfg.get("views", ()))
    transform = build_eval_transform(checkpoint.get("image_size") or data_cfg["image_size"])
    loader_args = dict(num_workers=int(data_cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    batch_size = int(train_cfg["batch_size"])
    query_loader = make_dataloader(queries, data_root, cow_to_idx, view_to_idx, transform,
                                   batch_size=batch_size, **loader_args)
    gallery_loader = make_dataloader(gallery, data_root, cow_to_idx, view_to_idx, transform,
                                     batch_size=batch_size, **loader_args)
    trained_with = checkpoint.get("training_config") or cfg
    loss_fn = SemiHardTripletLoss(float(trained_with["training"]["margin"]),
                                  bool(trained_with["training"].get("cross_view_positives", False)))
    loss_loader = try_build_loss_loader(queries, data_root, cow_to_idx, view_to_idx, transform, train_cfg,
                                        seed=int(cfg.get("seed", 42)), **loader_args)

    same_view = bool(cfg["evaluation"].get("same_view_retrieval", True))
    ks = cfg["evaluation"].get("recall_k", [1, 5, 10])
    q_emb, q_lab, q_view = compute_embeddings(model, query_loader, device)
    g_emb, g_lab, g_view = compute_embeddings(model, gallery_loader, device)
    metrics = retrieval_metrics(q_emb, q_lab, g_emb, g_lab, ks, q_view, g_view, same_view)
    metrics[f"{split}_loss"] = (compute_triplet_loss(model, loss_loader, loss_fn, device)
                                if loss_loader is not None else None)

    return {
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_monitor": checkpoint.get("monitor"),
        "checkpoint_best_validation_metric": checkpoint.get("best_metric"),
        "num_query_images": len(queries),
        "num_gallery_images": len(gallery),
        "num_cows": len({s.cow_id for s in queries}),
        "same_view_retrieval": same_view,
        "metrics": metrics,
        "pair_similarity": pair_similarity_stats(q_emb, q_lab, q_view, same_view),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
