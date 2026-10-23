"""Reproducible train/val/test splits, grouped by recording session when sessions are known.

Splits are *closed-set*: every cow appears in train (the gallery) and its remaining images are
divided over val and test (the queries). Consecutive video frames are near-duplicates, so whenever
a cow has enough known sessions, whole sessions are assigned to one split to avoid leakage.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from cowid.config import resolve_path
from cowid.data.dataset import (DEFAULT_EXTENSIONS, DEFAULT_VIEWS, DatasetValidationError, Sample,
                                scan_dataset)

logger = logging.getLogger(__name__)

SPLITS: Tuple[str, ...] = ("train", "val", "test")
CSV_FIELDS: Tuple[str, ...] = ("image", "cow_id", "view", "session")


def apply_session_metadata(samples: Sequence[Sample], csv_path: Union[str, Path]) -> List[Sample]:
    """Override sessions from a CSV with columns ``image,session`` (image relative to the dataset root)."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetValidationError(f"data.session_metadata points to a missing file: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "session"} <= set(reader.fieldnames):
            raise DatasetValidationError(f"{csv_path} must have the columns 'image' and 'session'")
        sessions = {row["image"].strip().replace("\\", "/"): row["session"].strip() for row in reader}
    updated = [replace(s, session=sessions[s.image]) if s.image in sessions else s for s in samples]
    missing = sum(s.image not in sessions for s in samples)
    if missing:
        logger.warning("%d image(s) have no entry in %s and keep their folder-based session (if any).",
                       missing, csv_path)
    return updated


def _normalized_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    total = sum(float(ratios.get(name, 0.0)) for name in SPLITS)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    return {name: float(ratios.get(name, 0.0)) / total for name in SPLITS}


def _allocate_counts(n: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Split ``n`` images into (train, val, test) counts, keeping >= 2 training images when possible."""
    n_val, n_test = round(n * val_ratio), round(n * test_ratio)
    if n >= 3:  # with 3+ images every split with a positive ratio gets at least one image
        n_val = max(n_val, 1) if val_ratio > 0 else 0
        n_test = max(n_test, 1) if test_ratio > 0 else 0
    while n - n_val - n_test < min(2, n):  # train needs a positive pair; val outranks test
        if n_test > 0 and n_test >= n_val:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n - n_val - n_test, n_val, n_test


def _assign_images(samples: Sequence[Sample], ratios: Dict[str, float], rng: random.Random
                   ) -> Dict[str, List[Sample]]:
    """Image-level split, stratified by view so every split covers the cow's views."""
    out: Dict[str, List[Sample]] = {name: [] for name in SPLITS}
    by_view: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_view[s.view].append(s)
    for view in sorted(by_view):
        items = list(by_view[view])
        rng.shuffle(items)
        n_train, n_val, _ = _allocate_counts(len(items), ratios["val"], ratios["test"])
        out["train"] += items[:n_train]
        out["val"] += items[n_train:n_train + n_val]
        out["test"] += items[n_train + n_val:]
    return out


def _assign_sessions(samples: Sequence[Sample], ratios: Dict[str, float], rng: random.Random
                     ) -> Dict[str, List[Sample]]:
    """Session-level split: each session goes entirely to one split."""
    by_session: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_session[s.session].append(s)
    remaining = sorted(by_session)
    rng.shuffle(remaining)
    out: Dict[str, List[Sample]] = {name: [] for name in SPLITS}
    active = [name for name in SPLITS if ratios[name] > 0]
    for name in active:  # every split with a positive ratio gets one session first (train first)
        if remaining:
            out[name].extend(by_session[remaining.pop(0)])
    for session in remaining:  # then fill the split that is furthest below its target size
        name = max(active, key=lambda n: ratios[n] * len(samples) - len(out[n]))
        out[name].extend(by_session[session])
    return out


def create_splits(samples: Sequence[Sample], ratios: Mapping[str, float], seed: int = 42,
                  min_sessions_for_grouping: int = 3) -> Dict[str, List[Sample]]:
    """Create reproducible per-cow splits (same samples + seed -> same splits)."""
    r = _normalized_ratios(ratios)
    by_cow: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_cow[s.cow_id].append(s)

    splits: Dict[str, List[Sample]] = {name: [] for name in SPLITS}
    image_level_cows: List[str] = []
    for cow in sorted(by_cow):
        cow_samples = sorted(by_cow[cow], key=lambda s: s.image)
        rng = random.Random(f"{seed}:{cow}")  # string seeds are hashed deterministically (sha512)
        n_sessions = len({s.session for s in cow_samples})
        if all(s.session for s in cow_samples) and n_sessions >= max(1, min_sessions_for_grouping):
            assigned = _assign_sessions(cow_samples, r, rng)
        else:
            image_level_cows.append(cow)
            assigned = _assign_images(cow_samples, r, rng)
        for name in SPLITS:
            splits[name].extend(assigned[name])

    if image_level_cows:
        logger.warning(
            "No session grouping for %d cow(s) (sessions unknown or fewer than %d): %s. Their images were "
            "split individually; near-duplicate video frames across splits make val/test scores optimistic. "
            "Put images in session folders (e.g. Cow_001/left/2024-10-15/...) to group them.",
            len(image_level_cows), min_sessions_for_grouping,
            ", ".join(image_level_cows[:10]) + (" ..." if len(image_level_cows) > 10 else ""))
    return {name: sorted(items, key=lambda s: s.image) for name, items in splits.items()}


def splits_exist(splits_dir: Union[str, Path]) -> bool:
    return all((Path(splits_dir) / f"{name}.csv").is_file() for name in SPLITS)


def save_splits(splits: Mapping[str, Sequence[Sample]], splits_dir: Union[str, Path]) -> None:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        with open(splits_dir / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)
            writer.writerows((s.image, s.cow_id, s.view, s.session) for s in splits.get(name, []))


def load_splits(splits_dir: Union[str, Path]) -> Dict[str, List[Sample]]:
    splits_dir = Path(splits_dir)
    splits: Dict[str, List[Sample]] = {}
    for name in SPLITS:
        path = splits_dir / f"{name}.csv"
        if not path.is_file():
            raise DatasetValidationError(f"Split file missing: {path}")
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not set(CSV_FIELDS) <= set(reader.fieldnames):
                raise DatasetValidationError(f"{path} must have the columns {', '.join(CSV_FIELDS)}")
            splits[name] = [Sample(row["image"], row["cow_id"], row["view"], row["session"] or "")
                            for row in reader]
    return splits


def splits_fingerprint(splits: Mapping[str, Sequence[Sample]]) -> str:
    """Short hash identifying the exact split assignment (stored in checkpoints)."""
    lines = sorted(f"{name}|{s.image}|{s.cow_id}" for name in SPLITS for s in splits.get(name, []))
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def summarize_splits(splits: Mapping[str, Sequence[Sample]]) -> str:
    rows = []
    for name in SPLITS:
        items = splits.get(name, [])
        views: Dict[str, int] = defaultdict(int)
        for s in items:
            views[s.view] += 1
        per_view = ", ".join(f"{v} {n}" for v, n in sorted(views.items())) or "-"
        rows.append(f"  {name:<5} {len(items):>6} images  {len({s.cow_id for s in items}):>4} cows  ({per_view})")
    return "\n".join(rows)


def prepare_splits(cfg: Dict[str, Any], resplit: bool = False, create_if_missing: bool = True
                   ) -> Dict[str, List[Sample]]:
    """Load the saved splits, or scan the dataset and create them (``resplit`` forces regeneration)."""
    data_cfg = cfg["data"]
    data_root = resolve_path(cfg["paths"]["dataset_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    views = data_cfg.get("views", DEFAULT_VIEWS)
    extensions = data_cfg.get("image_extensions", DEFAULT_EXTENSIONS)

    if splits_exist(splits_dir) and not resplit:
        splits = load_splits(splits_dir)
        known = {s.image for items in splits.values() for s in items}
        missing = sorted(img for img in known if not (data_root / img).is_file())
        if missing:
            raise DatasetValidationError(
                f"{len(missing)} image(s) listed in {splits_dir} no longer exist under {data_root} "
                f"(e.g. {missing[0]}). Regenerate the splits with: python scripts/train_model.py --resplit")
        new = [s for s in scan_dataset(data_root, views, extensions) if s.image not in known]
        if new:
            logger.warning("%d image(s) in %s are not in the saved splits and are ignored. "
                           "Use --resplit to include them.", len(new), data_root)
        logger.info("Loaded dataset splits from %s", splits_dir)
        return splits

    if not create_if_missing:
        raise DatasetValidationError(
            f"No dataset splits found in {splits_dir}. Run scripts/train_model.py first; it creates them.")
    samples = scan_dataset(data_root, views, extensions)
    if data_cfg.get("session_metadata"):
        samples = apply_session_metadata(samples, resolve_path(data_cfg["session_metadata"]))
    splits = create_splits(samples, data_cfg["split_ratios"], seed=int(cfg.get("seed", 42)),
                           min_sessions_for_grouping=int(data_cfg.get("min_sessions_for_grouping", 3)))
    save_splits(splits, splits_dir)
    logger.info("Created dataset splits from %d images in %s -> %s", len(samples), data_root, splits_dir)
    return splits
