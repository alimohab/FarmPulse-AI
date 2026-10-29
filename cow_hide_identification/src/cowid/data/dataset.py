"""Dataset discovery, validation and PyTorch data loading for cow hide images.

Expected layout (any number of cows, any subset of the configured views)::

    <dataset_dir>/
        Cow_001/
            left/   *.jpg
            right/  *.jpg
            top/    *.jpg
        Cow_002/
            ...

Images may also sit one level deeper in a *session* folder
(``Cow_001/left/2024-10-15_morning/*.jpg``). The session name is then used to
keep a whole recording session inside a single train/val/test split.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)

DEFAULT_VIEWS: Tuple[str, ...] = ("left", "right", "top")
DEFAULT_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp","HEIC")


class DatasetValidationError(ValueError):
    """Raised when the dataset cannot support metric-learning training or evaluation."""


@dataclass(frozen=True)
class Sample:
    """One labelled image. ``image`` is relative to the dataset root and uses '/' separators."""

    image: str
    cow_id: str
    view: str
    session: str = ""  # empty when the recording session is unknown


def scan_dataset(root: Union[str, Path], views: Sequence[str] = DEFAULT_VIEWS,
                 extensions: Sequence[str] = DEFAULT_EXTENSIONS) -> List[Sample]:
    """Collect labelled samples from the ``Cow_XXX/<view>/[<session>/]image`` folder structure."""
    root = Path(root)
    example = root / "Cow_001" / "left" / "img_0001.jpg"
    if not root.is_dir():
        raise DatasetValidationError(
            f"Dataset folder not found: {root}\n"
            f"Create it with one folder per cow and one sub-folder per view, e.g. {example}")
    allowed_views = {v.lower() for v in views}
    allowed_ext = {e.lower() for e in extensions}
    cow_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not cow_dirs:
        raise DatasetValidationError(f"No cow folders found in {root}. Expected e.g. {example}")

    samples: List[Sample] = []
    for cow_dir in cow_dirs:
        n_before, loose = len(samples), 0
        for entry in sorted(cow_dir.iterdir()):
            if entry.is_file():
                loose += entry.suffix.lower() in allowed_ext
                continue
            view = entry.name.lower()
            if view not in allowed_views:
                logger.warning("Ignoring folder %s: unknown view '%s' (expected one of: %s).",
                               entry, entry.name, ", ".join(sorted(allowed_views)))
                continue
            for img in sorted(entry.rglob("*")):
                if img.is_file() and img.suffix.lower() in allowed_ext:
                    rel = img.relative_to(root)
                    session = rel.parts[2] if len(rel.parts) > 3 else ""
                    samples.append(Sample(rel.as_posix(), cow_dir.name, view, session))
        if loose:
            logger.warning("Ignoring %d image(s) directly inside %s: move them into a view folder (%s).",
                           loose, cow_dir, ", ".join(sorted(allowed_views)))
        if len(samples) == n_before:
            logger.warning("Cow folder %s contains no images inside view folders.", cow_dir)

    if not samples:
        raise DatasetValidationError(
            f"No images ({', '.join(sorted(allowed_ext))}) found under {root}. Expected e.g. {example}")
    return samples


def group_key(sample: Sample, cross_view: bool = False) -> str:
    """Images that share a group key are valid positives for each other."""
    return sample.cow_id if cross_view else f"{sample.cow_id}/{sample.view}"


def validate_for_training(samples: Sequence[Sample], cross_view: bool = False, split_name: str = "train") -> None:
    """Ensure triplets (anchor, positive of the same cow, negative of another cow) can be formed."""
    cows = sorted({s.cow_id for s in samples})
    if len(cows) < 2:
        raise DatasetValidationError(
            f"The {split_name} split has {len(cows)} cow(s) ({', '.join(cows) or 'none'}). Triplet training "
            "needs at least 2 different cows: anchor and positive come from one cow, the negative from another.")
    counts = Counter(group_key(s, cross_view) for s in samples)
    if not any(n >= 2 for n in counts.values()):
        unit = "cow" if cross_view else "cow/view combination"
        raise DatasetValidationError(
            f"No valid triplets in the {split_name} split: no {unit} has at least 2 images, so no "
            "anchor/positive pair can be formed. Add more images per cow"
            + ("." if cross_view else " and view (positives must share the view)."))
    singles = sorted(g for g, n in counts.items() if n < 2)
    if singles:
        logger.warning("%d group(s) in the %s split have a single image and can only serve as negatives: %s",
                       len(singles), split_name, ", ".join(singles[:10]) + (" ..." if len(singles) > 10 else ""))


def build_label_maps(samples: Sequence[Sample], views: Sequence[str] = DEFAULT_VIEWS
                     ) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Map cow ids and view names to integer indices (sorted cows, configured view order first)."""
    cow_to_idx = {cow: i for i, cow in enumerate(sorted({s.cow_id for s in samples}))}
    ordered_views = list(dict.fromkeys([v.lower() for v in views] + sorted({s.view for s in samples})))
    return cow_to_idx, {v: i for i, v in enumerate(ordered_views)}


def load_image(path: Union[str, Path]) -> Image.Image:
    """Open an image as RGB, applying the EXIF orientation written by phone cameras."""
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {path}") from None
    except (UnidentifiedImageError, OSError) as exc:
        raise DatasetValidationError(f"Cannot read image {path}: {exc}") from exc


class CowHideDataset(Dataset):
    """Returns ``(image_tensor, cow_index, view_index)`` for each sample."""

    def __init__(self, samples: Sequence[Sample], root: Union[str, Path], cow_to_idx: Dict[str, int],
                 view_to_idx: Dict[str, int], transform: Optional[Callable] = None):
        self.samples = list(samples)
        self.root = Path(root)
        self.cow_to_idx = cow_to_idx
        self.view_to_idx = view_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(self.root / sample.image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.cow_to_idx[sample.cow_id], self.view_to_idx[sample.view]


class PKBatchSampler(Sampler):
    """Yields batches of P groups x K images, so every batch contains positives and negatives.

    A group is a cow/view combination (or a cow when ``cross_view`` is True). Only groups with at
    least two images are sampled as positive groups; if all chosen groups belong to one cow, images
    of other cows are appended so that negatives always exist. Groups smaller than K are sampled
    with replacement (the training augmentations make repeated images differ).
    """

    def __init__(self, samples: Sequence[Sample], batch_size: int, samples_per_group: int,
                 cross_view: bool = False, batches_per_epoch: Optional[int] = None, seed: int = 0):
        if samples_per_group < 2:
            raise ValueError("samples_per_group must be >= 2")
        if batch_size < 2 * samples_per_group or batch_size % samples_per_group:
            raise ValueError("batch_size must be a multiple of samples_per_group holding at least two groups")
        validate_for_training(samples, cross_view, split_name="sampled")
        self.k = samples_per_group
        self.p = batch_size // samples_per_group
        index: Dict[str, List[int]] = defaultdict(list)
        cow_index: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            index[group_key(s, cross_view)].append(i)
            cow_index[s.cow_id].append(i)
        self.groups = {g: idx for g, idx in sorted(index.items()) if len(idx) >= 2}
        self.group_cow = {g: samples[idx[0]].cow_id for g, idx in self.groups.items()}
        self.cow_index = dict(cow_index)
        self.batches_per_epoch = batches_per_epoch or max(1, math.ceil(len(samples) / batch_size))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the sampling order deterministically per epoch (reproducible when resuming)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.groups)
        for _ in range(self.batches_per_epoch):
            chosen = [keys[j] for j in rng.choice(len(keys), size=min(self.p, len(keys)), replace=False)]
            batch: List[int] = []
            for g in chosen:
                idx = self.groups[g]
                batch.extend(rng.choice(idx, size=self.k, replace=len(idx) < self.k).tolist())
            cows = {self.group_cow[g] for g in chosen}
            if len(cows) < 2:
                others = [i for cow, idx in self.cow_index.items() if cow not in cows for i in idx]
                batch.extend(rng.choice(others, size=min(self.k, len(others)), replace=False).tolist())
            yield batch


def make_dataloader(samples: Sequence[Sample], root: Union[str, Path], cow_to_idx: Dict[str, int],
                    view_to_idx: Dict[str, int], transform: Callable, *, batch_size: int = 32,
                    batch_sampler: Optional[Sampler] = None, num_workers: int = 0,
                    pin_memory: bool = False) -> DataLoader:
    """Build a DataLoader; with ``batch_sampler`` the batches come from it (e.g. a PKBatchSampler)."""
    dataset = CowHideDataset(samples, root, cow_to_idx, view_to_idx, transform)
    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)
