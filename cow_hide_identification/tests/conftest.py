"""Shared test fixtures.

The tests build tiny synthetic image folders inside pytest's temporary directory. They only exercise
the code paths (loading, splitting, shapes, checkpointing); they are never used as training data.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Optional, Sequence

import pytest
from PIL import Image, ImageDraw

from cowid.config import load_config


def write_test_image(path: Path, cow_seed: int, image_seed: int, size=(64, 48)) -> None:
    """A coloured background with cow-specific rectangles plus small per-image jitter."""
    cow_rng, img_rng = random.Random(cow_seed), random.Random(image_seed)
    img = Image.new("RGB", size, (200, 200, 200))
    draw = ImageDraw.Draw(img)
    for _ in range(4):
        x, y = cow_rng.randrange(size[0] - 16) + img_rng.randrange(3), cow_rng.randrange(size[1] - 12)
        draw.rectangle([x, y, x + 14, y + 10], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


@pytest.fixture
def make_dataset(tmp_path):
    def _make(cows: int = 3, views: Sequence[str] = ("left",), images_per_view: int = 6,
              sessions: Optional[int] = None) -> Path:
        root = tmp_path / "raw"
        for c in range(1, cows + 1):
            for v, view in enumerate(views):
                for i in range(images_per_view):
                    parts = [f"Cow_{c:03d}", view] + ([f"session_{i % sessions}"] if sessions else [])
                    write_test_image(root.joinpath(*parts, f"img_{i:03d}.png"), cow_seed=c * 100 + v,
                                     image_seed=c * 10_000 + v * 100 + i)
        return root
    return _make


@pytest.fixture
def small_config(tmp_path):
    """The shipped config, shrunk for fast CPU tests and pointed at temporary folders."""
    def _make(dataset_dir: Path) -> dict:
        cfg = copy.deepcopy(load_config())
        cfg["paths"] = {"dataset_dir": str(dataset_dir), "splits_dir": str(tmp_path / "splits"),
                        "checkpoint_dir": str(tmp_path / "checkpoints"), "output_dir": str(tmp_path / "outputs")}
        cfg["data"].update(image_size=[64, 64], num_workers=0)
        cfg["model"].update(backbone="resnet18", pretrained=False, embedding_dim=32, hidden_dim=64)
        cfg["training"].update(epochs=2, batch_size=8, samples_per_class=2, freeze_backbone_epochs=1,
                               log_every=0, device="cpu")
        cfg["training"]["early_stopping"]["min_delta"] = 0.0
        return cfg
    return _make
