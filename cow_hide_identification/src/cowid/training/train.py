"""Training loop: semi-hard triplet training with validation, early stopping and checkpointing."""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from cowid.config import resolve_path
from cowid.data.augmentations import build_eval_transform, build_train_transform
from cowid.data.dataset import (DatasetValidationError, PKBatchSampler, build_label_maps, make_dataloader,
                                validate_for_training)
from cowid.data.split_dataset import prepare_splits, splits_fingerprint, summarize_splits
from cowid.reid.losses import SemiHardTripletLoss
from cowid.reid.siamese_network import SiameseNetwork
from cowid.training.validate import try_build_loss_loader, validate
from cowid.utils.checkpoint_utils import (BEST_CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, BestMetricTracker,
                                          archive_existing_checkpoints, build_model_from_checkpoint,
                                          load_checkpoint, save_checkpoint)
from cowid.utils.device import get_device, seed_everything

logger = logging.getLogger(__name__)


def build_optimizer(model: SiameseNetwork, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with a smaller learning rate for the pretrained backbone than for the new head."""
    lr = float(train_cfg["learning_rate"])
    backbone_lr = lr * float(train_cfg.get("backbone_lr_multiplier", 1.0))
    return torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": backbone_lr},
                              {"params": model.head.parameters(), "lr": lr}],
                             weight_decay=float(train_cfg.get("weight_decay", 0.0)))


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Dict[str, Any]):
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name == "cosine":
        # Closed-form cosine decay over the *current* epoch budget, so resuming with more epochs
        # continues the curve instead of staying at the old schedule's final (zero) learning rate.
        epochs = max(1, int(train_cfg["epochs"]))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 0.5 * (1.0 + math.cos(math.pi * min(e, epochs) / epochs)))
    if name in ("none", "null", ""):
        return None
    raise ValueError(f"Unknown training.scheduler '{name}' (use cosine or none)")


def train_one_epoch(model: SiameseNetwork, loader: DataLoader, loss_fn: SemiHardTripletLoss,
                    optimizer: torch.optim.Optimizer, device: torch.device, scaler: torch.amp.GradScaler,
                    use_amp: bool = False, log_every: int = 0) -> Dict[str, float]:
    """One pass over the PK batches; batches without a valid triplet are skipped."""
    model.train()
    total_loss, steps, num_triplets, num_active, skipped = 0.0, 0, 0, 0, 0
    for step, (images, labels, views) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels, views = labels.to(device), views.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            embeddings = model(images)
        loss, stats = loss_fn(embeddings, labels, views)
        if stats["num_triplets"] == 0:
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
        num_triplets += stats["num_triplets"]
        num_active += stats["num_active"]
        if log_every and step % log_every == 0:
            logger.info("  batch %d/%d  loss %.4f  active triplets %d/%d", step, len(loader), loss.item(),
                        stats["num_active"], stats["num_triplets"])
    if steps == 0:
        raise DatasetValidationError("No training batch contained a valid triplet. Each cow/view needs at "
                                     "least 2 images and the training split needs at least 2 cows.")
    return {"train_loss": total_loss / steps, "active_triplet_fraction": num_active / max(num_triplets, 1),
            "skipped_batches": skipped}


def _resolve_resume_path(resume: Union[None, bool, str, Path], checkpoint_dir: Path) -> Optional[Path]:
    if resume in (None, False, ""):
        return None
    path = checkpoint_dir / LAST_CHECKPOINT_NAME if resume is True or str(resume) == "last" else Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume: checkpoint not found at {path}")
    return path


def _fmt(value: Optional[float], spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def _write_history(history: List[Dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(k for row in history for k in row))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def train(cfg: Dict[str, Any], resume: Union[None, bool, str, Path] = None, resplit: bool = False
          ) -> Dict[str, Any]:
    """Train the Siamese network as configured and return a summary of the run.

    ``best_model.pth`` is written only when the monitored validation metric improves;
    ``last_checkpoint.pth`` is written after every epoch for resuming (``resume=True``).
    """
    data_cfg, train_cfg, eval_cfg = cfg["data"], cfg["training"], cfg["evaluation"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = get_device(train_cfg.get("device", "auto"))
    checkpoint_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    data_root = resolve_path(cfg["paths"]["dataset_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        resume = bool(train_cfg.get("resume", False))
    resume_path = _resolve_resume_path(resume, checkpoint_dir)
    logger.info("Device: %s", device)

    # ---- data -------------------------------------------------------------------------------
    splits = prepare_splits(cfg, resplit=resplit)
    logger.info("Dataset splits:\n%s", summarize_splits(splits))
    cross_view = bool(train_cfg.get("cross_view_positives", False))
    validate_for_training(splits["train"], cross_view, "train")
    if not splits["val"]:
        raise DatasetValidationError(
            "The validation split is empty, so the best model cannot be selected. Add more images "
            "(at least 3 per cow and view) or regenerate the splits with --resplit.")

    cow_to_idx, view_to_idx = build_label_maps([s for items in splits.values() for s in items],
                                               data_cfg.get("views", ()))
    batch_size, k = int(train_cfg["batch_size"]), int(train_cfg["samples_per_class"])
    loader_args = dict(num_workers=int(data_cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    eval_tf = build_eval_transform(data_cfg["image_size"])
    train_sampler = PKBatchSampler(splits["train"], batch_size, k, cross_view,
                                   train_cfg.get("batches_per_epoch"), seed)
    train_loader = make_dataloader(splits["train"], data_root, cow_to_idx, view_to_idx,
                                   build_train_transform(data_cfg["image_size"]), batch_sampler=train_sampler,
                                   **loader_args)
    gallery_loader = make_dataloader(splits["train"], data_root, cow_to_idx, view_to_idx, eval_tf,
                                     batch_size=batch_size, **loader_args)
    val_loader = make_dataloader(splits["val"], data_root, cow_to_idx, view_to_idx, eval_tf,
                                 batch_size=batch_size, **loader_args)
    val_loss_loader = try_build_loss_loader(splits["val"], data_root, cow_to_idx, view_to_idx, eval_tf,
                                            train_cfg, seed, **loader_args)

    monitor = str(train_cfg.get("monitor", "top1"))
    if monitor == "val_loss" and val_loss_loader is None:
        raise DatasetValidationError("training.monitor is val_loss, but the validation split cannot form "
                                     "triplets. Use monitor: top1 or add more validation images.")

    # ---- model, optimisation, state -------------------------------------------------------------
    checkpoint = load_checkpoint(resume_path, map_location=device) if resume_path else None
    if checkpoint is not None:
        model = build_model_from_checkpoint(checkpoint, device)
        changed = {key: (checkpoint["model_config"].get(key), cfg["model"].get(key))
                   for key in ("backbone", "embedding_dim", "hidden_dim")
                   if checkpoint["model_config"].get(key) != cfg["model"].get(key)}
        if changed:
            logger.warning("Resuming with the checkpoint's architecture; config differs: %s", changed)
    else:
        archived = archive_existing_checkpoints(checkpoint_dir)
        if archived:
            logger.info("Fresh run: previous checkpoints moved to %s", archived)
        model = SiameseNetwork.from_config(cfg["model"]).to(device)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = SemiHardTripletLoss(float(train_cfg["margin"]), cross_view)
    early_cfg = train_cfg.get("early_stopping") or {}
    tracker = BestMetricTracker("min" if monitor == "val_loss" else "max", float(early_cfg.get("min_delta", 0.0)))
    fingerprint = splits_fingerprint(splits)
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # The optimizer stored the old schedule's last LR; re-derive it for the current epoch budget.
            for group, base_lr, lr_fn in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
                group["lr"] = base_lr * lr_fn(scheduler.last_epoch)
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("monitor") == monitor and checkpoint.get("tracker_state"):
            tracker.load_state_dict(checkpoint["tracker_state"])
        else:
            logger.warning("Monitored metric changed (%s -> %s); best-metric tracking restarts.",
                           checkpoint.get("monitor"), monitor)
        if checkpoint.get("splits_fingerprint") != fingerprint:
            logger.warning("The dataset splits differ from those used before this checkpoint; "
                           "metrics are not directly comparable with earlier epochs.")
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from %s (epoch %d, best %s = %s)", resume_path, checkpoint["epoch"], monitor,
                    _fmt(tracker.best))

    # ---- epochs ------------------------------------------------------------------------------
    epochs = int(train_cfg["epochs"])
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    patience = int(early_cfg.get("patience", 0)) if early_cfg.get("enabled", False) else 0
    best_path, last_path = checkpoint_dir / BEST_CHECKPOINT_NAME, checkpoint_dir / LAST_CHECKPOINT_NAME
    history_path = output_dir / "training_history.csv"
    ks = eval_cfg.get("recall_k", [1, 5, 10])
    same_view = bool(eval_cfg.get("same_view_retrieval", True))
    stopped_early = False
    if start_epoch > epochs:
        logger.info("Checkpoint already reached epoch %d of %d; raise training.epochs to continue.",
                    start_epoch - 1, epochs)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        model.set_backbone_trainable(epoch > freeze_epochs)
        train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[-1]["lr"]
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler, use_amp,
                                      int(train_cfg.get("log_every", 0)))
        val_metrics = validate(model, query_loader=val_loader, gallery_loader=gallery_loader, device=device,
                               loss_fn=loss_fn, loss_loader=val_loss_loader, ks=ks, same_view=same_view)
        if scheduler is not None:
            scheduler.step()

        current = val_metrics.get(monitor)
        improved = tracker.update(current, epoch)
        row = {"epoch": epoch, "lr": lr, "backbone_frozen": epoch <= freeze_epochs, **train_stats,
               **val_metrics, "seconds": round(time.time() - started, 1), "is_best": improved}
        history.append(row)
        _write_history(history, history_path)

        recall_text = "  ".join(f"{key} {_fmt(v, '.3f')}" for key, v in val_metrics.items() if key.startswith("recall@"))
        logger.info("Epoch %d/%d | train_loss %s | val_loss %s | top1 %s | %s | mAP %s | lr %.2e | %.1fs%s",
                    epoch, epochs, _fmt(train_stats["train_loss"]), _fmt(val_metrics["val_loss"]),
                    _fmt(val_metrics["top1"], ".3f"), recall_text or "recall@K n/a", _fmt(val_metrics["mAP"], ".3f"),
                    lr, row["seconds"], "  * new best" if improved else "")
        if current is None:
            logger.warning("Monitored metric '%s' could not be computed this epoch.", monitor)

        extra = {"tracker_state": tracker.state_dict(), "history": history, "val_metrics": val_metrics,
                 "splits_fingerprint": fingerprint, "cow_ids": sorted(cow_to_idx)}
        save_args = dict(optimizer=optimizer, scheduler=scheduler, scaler=scaler, best_metric=tracker.best,
                         monitor=monitor, config=cfg, extra=extra)
        if improved:
            save_checkpoint(best_path, model, epoch, **save_args)
            logger.info("Saved best model (%s = %s) to %s", monitor, _fmt(current), best_path)
        save_checkpoint(last_path, model, epoch, **save_args)

        if patience and tracker.num_bad_epochs >= patience:
            logger.info("Early stopping: %s has not improved for %d epochs.", monitor, patience)
            stopped_early = True
            break

    if tracker.best is None:
        logger.warning("No epoch produced a valid '%s'; best_model.pth was not written.", monitor)
    summary = {"monitor": monitor, "best_metric": tracker.best, "best_epoch": tracker.best_epoch,
               "epochs_completed": history[-1]["epoch"] if history else 0, "stopped_early": stopped_early,
               "best_checkpoint": str(best_path) if best_path.is_file() else None,
               "last_checkpoint": str(last_path) if last_path.is_file() else None,
               "history_csv": str(history_path), "device": str(device)}
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training finished. Best %s = %s at epoch %s. Best checkpoint: %s", monitor,
                _fmt(tracker.best), tracker.best_epoch, summary["best_checkpoint"])
    return summary
