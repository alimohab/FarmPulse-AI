"""Tests for dataset scanning, validation, splitting and batching."""
from __future__ import annotations

import pytest
import torch

from cowid.data.augmentations import build_eval_transform, build_train_transform
from cowid.data.dataset import (CowHideDataset, DatasetValidationError, PKBatchSampler, Sample,
                                build_label_maps, scan_dataset, validate_for_training)
from cowid.data.split_dataset import SPLITS, create_splits, load_splits, save_splits

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def test_scan_loads_cows_views_and_labels(make_dataset):
    root = make_dataset(cows=3, views=("left", "right", "top"), images_per_view=4)
    samples = scan_dataset(root)
    assert len(samples) == 3 * 3 * 4
    assert {s.cow_id for s in samples} == {"Cow_001", "Cow_002", "Cow_003"}
    assert {s.view for s in samples} == {"left", "right", "top"}
    assert all((root / s.image).is_file() for s in samples)
    assert all(s.session == "" for s in samples)


def test_scan_reads_session_from_subfolder(make_dataset):
    samples = scan_dataset(make_dataset(cows=2, images_per_view=6, sessions=3))
    assert {s.session for s in samples} == {"session_0", "session_1", "session_2"}


def test_scan_ignores_unknown_view_folders(make_dataset):
    root = make_dataset(cows=2, images_per_view=3)
    (root / "Cow_001" / "front").mkdir()
    (root / "Cow_001" / "front" / "x.png").write_bytes((root / "Cow_001" / "left" / "img_000.png").read_bytes())
    assert all(s.view == "left" for s in scan_dataset(root))


def test_scan_missing_or_empty_dataset_raises(tmp_path):
    with pytest.raises(DatasetValidationError, match="not found"):
        scan_dataset(tmp_path / "does_not_exist")
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetValidationError, match="No cow folders"):
        scan_dataset(tmp_path / "empty")


def test_validation_requires_two_cows_and_positive_pairs():
    one_cow = [Sample(f"Cow_001/left/{i}.png", "Cow_001", "left") for i in range(4)]
    with pytest.raises(DatasetValidationError, match="at least 2 different cows"):
        validate_for_training(one_cow)
    singletons = [Sample("a.png", "Cow_001", "left"), Sample("b.png", "Cow_002", "left"),
                  Sample("c.png", "Cow_001", "right")]
    with pytest.raises(DatasetValidationError, match="No valid triplets"):
        validate_for_training(singletons)
    validate_for_training(singletons, cross_view=True)  # Cow_001 left+right form a cross-view pair


def test_splits_are_reproducible_disjoint_and_complete(make_dataset):
    samples = scan_dataset(make_dataset(cows=3, views=("left", "right"), images_per_view=10))
    a = create_splits(samples, RATIOS, seed=7)
    b = create_splits(samples, RATIOS, seed=7)
    assert a == b
    images = [s.image for name in SPLITS for s in a[name]]
    assert len(images) == len(set(images)) == len(samples)
    for cow in {s.cow_id for s in samples}:  # closed set: every cow has train (gallery) and val images
        assert any(s.cow_id == cow for s in a["train"]) and any(s.cow_id == cow for s in a["val"])


def test_session_grouping_keeps_sessions_in_one_split(make_dataset):
    samples = scan_dataset(make_dataset(cows=2, views=("left", "right"), images_per_view=12, sessions=4))
    splits = create_splits(samples, RATIOS, seed=1, min_sessions_for_grouping=3)
    owner = {}
    for name in SPLITS:
        for s in splits[name]:
            assert owner.setdefault((s.cow_id, s.session), name) == name
    assert all(splits[name] for name in SPLITS)


def test_save_and_load_splits_roundtrip(make_dataset, tmp_path):
    splits = create_splits(scan_dataset(make_dataset(cows=2, images_per_view=6, sessions=3)), RATIOS)
    save_splits(splits, tmp_path / "splits")
    assert load_splits(tmp_path / "splits") == splits


def test_dataset_returns_tensor_and_labels(make_dataset):
    root = make_dataset(cows=2, views=("left", "top"), images_per_view=2)
    samples = scan_dataset(root)
    cow_to_idx, view_to_idx = build_label_maps(samples)
    for transform in (build_eval_transform([64, 96]), build_train_transform([64, 96])):
        image, cow, view = CowHideDataset(samples, root, cow_to_idx, view_to_idx, transform)[0]
        assert isinstance(image, torch.Tensor) and image.shape == (3, 64, 96)
        assert cow == cow_to_idx[samples[0].cow_id] and view == view_to_idx[samples[0].view]


def test_pk_sampler_batches_contain_same_view_positives_and_negatives(make_dataset):
    samples = scan_dataset(make_dataset(cows=3, views=("left", "right"), images_per_view=5))
    sampler = PKBatchSampler(samples, batch_size=8, samples_per_group=2, seed=0)
    for batch in sampler:
        assert len(batch) == 8
        keys = [(samples[i].cow_id, samples[i].view) for i in batch]
        assert len({k[0] for k in keys}) >= 2
        assert any(keys.count(k) >= 2 for k in keys)
    sampler.set_epoch(1)
    first_epoch1 = next(iter(sampler))
    sampler.set_epoch(1)
    assert next(iter(sampler)) == first_epoch1  # reproducible per epoch
