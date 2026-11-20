"""Tests for the multi-cow crop selection workflow (no YOLO and no window needed)."""
from __future__ import annotations

import os
import shutil
import time

import pytest
from PIL import Image

from cowid.detection.detector import CowBox, crop_box
from cowid.detection.selection import (AUTO, DRAWN, PICKED, SKIPPED, collect_photos, detect_all, export_crops,
                                       load_selections, make_selection, needs_review, output_path,
                                       save_selections, siblings, summarize)


@pytest.fixture
def photo_dirs(tmp_path):
    """raw/ with two single-cow photos and excluded/ with one photo filed under two cows."""
    raw, excluded = tmp_path / "raw", tmp_path / "excluded"
    for cow, color in (("cow_1", (255, 0, 0)), ("cow_2", (0, 255, 0))):
        (raw / cow / "left").mkdir(parents=True)
        Image.new("RGB", (100, 80), color).save(raw / cow / "left" / f"{cow}_a.jpg")
    shared = Image.new("RGB", (200, 100), (0, 0, 255))
    for cow, view in (("cow_1", "top"), ("cow_2", "right")):
        (excluded / cow / view).mkdir(parents=True)
        shared.save(excluded / cow / view / "group.jpg")
    return raw, excluded


def test_collect_photos_and_siblings(photo_dirs, tmp_path):
    raw, excluded = photo_dirs
    items = collect_photos([raw, excluded, tmp_path / "missing"], project_root=tmp_path)
    assert {i.key for i in items} == {"raw/cow_1/left/cow_1_a.jpg", "raw/cow_2/left/cow_2_a.jpg",
                                      "excluded/cow_1/top/group.jpg", "excluded/cow_2/right/group.jpg"}
    sib = siblings(items)
    assert set(sib) == {"excluded/cow_1/top/group.jpg", "excluded/cow_2/right/group.jpg"}
    assert [o.cow_id for o in sib["excluded/cow_1/top/group.jpg"]] == ["cow_2"]
    with pytest.raises(ValueError, match="twice"):
        collect_photos([raw, raw], project_root=tmp_path)


def test_needs_review_rules():
    one, two = [CowBox(0, 0, 1, 1, 0.9)], [CowBox(0, 0, 1, 1, 0.9), CowBox(2, 2, 3, 3, 0.8)]
    assert not needs_review(one, is_shared=False)
    assert needs_review(two, is_shared=False)
    assert needs_review([], is_shared=False)
    assert needs_review(one, is_shared=True)
    assert needs_review(one, is_shared=False, review_all=True)


def test_detect_all_sorts_boxes_and_uses_cache(photo_dirs, tmp_path):
    items = collect_photos(list(photo_dirs), project_root=tmp_path)
    calls = []

    def fake_detect(image):
        calls.append(image.size)
        return [CowBox(50, 0, 90, 40, 0.8), CowBox(5, 0, 40, 40, 0.9)]

    cache = {}
    detections = detect_all(items, fake_detect, cache)
    assert len(calls) == len(items)
    assert all([b.x1 for b in boxes] == [5, 50] for boxes in detections.values())  # left to right
    detect_all(items, fake_detect, cache)
    assert len(calls) == len(items)  # unchanged files come from the cache
    changed = items[0].path
    Image.new("RGB", (100, 80), (9, 9, 9)).save(changed)
    os.utime(changed, (time.time() + 5, time.time() + 5))
    detect_all(items, fake_detect, cache)
    assert len(calls) == len(items) + 1


def test_selections_roundtrip(photo_dirs, tmp_path):
    items = collect_photos(list(photo_dirs), project_root=tmp_path)
    selections = {items[0].key: make_selection(items[0], PICKED, CowBox(1, 2, 30, 40, 0.5), 1),
                  items[1].key: make_selection(items[1], SKIPPED),
                  items[2].key: make_selection(items[2], DRAWN, CowBox(0, 0, 10, 10, 1.0))}
    save_selections(selections, tmp_path / "sel.csv")
    loaded = load_selections(tmp_path / "sel.csv")
    assert set(loaded) == set(selections)
    assert loaded[items[0].key].box_index == 1 and loaded[items[0].key].box.x2 == 30
    assert loaded[items[1].key].status == SKIPPED and loaded[items[1].key].box is None
    assert loaded[items[2].key].box_index is None
    with pytest.raises(ValueError, match="needs a box"):
        make_selection(items[0], PICKED)


def test_crop_box_pads_and_clips():
    image = Image.new("RGB", (100, 100))
    assert crop_box(image, CowBox(10, 10, 50, 30, 1.0), padding=0.1).size == (48, 24)
    assert crop_box(image, CowBox(0, 0, 100, 100, 1.0), padding=0.2).size == (100, 100)


def test_export_crops_writes_each_cow_and_follows_changes(photo_dirs, tmp_path):
    items = collect_photos(list(photo_dirs), project_root=tmp_path)
    by_key = {i.key: i for i in items}
    a = by_key["excluded/cow_1/top/group.jpg"]
    b = by_key["excluded/cow_2/right/group.jpg"]
    single = by_key["raw/cow_1/left/cow_1_a.jpg"]
    selections = {a.key: make_selection(a, PICKED, CowBox(0, 0, 100, 100, 1.0), 0),
                  b.key: make_selection(b, PICKED, CowBox(100, 0, 200, 50, 1.0), 1),
                  single.key: make_selection(single, AUTO, CowBox(0, 0, 50, 40, 1.0), 0)}
    out = tmp_path / "processed"
    counts = export_crops(items, selections, out, padding=0.0)
    assert counts == {"written": 3, "up_to_date": 0, "removed_skipped": 0, "not_selected": 1}
    assert Image.open(out / "cow_1" / "top" / "group.jpg").size == (100, 100)
    assert Image.open(out / "cow_2" / "right" / "group.jpg").size == (100, 50)
    assert export_crops(items, selections, out, padding=0.0)["up_to_date"] == 3

    time.sleep(1.1)  # selection timestamps have one-second resolution
    selections[a.key] = make_selection(a, SKIPPED)
    selections[b.key] = make_selection(b, PICKED, CowBox(150, 0, 200, 100, 1.0), 1)
    counts = export_crops(items, selections, out, padding=0.0)
    assert counts["written"] == 1 and counts["removed_skipped"] == 1
    assert not (out / "cow_1" / "top" / "group.jpg").exists()
    assert Image.open(out / "cow_2" / "right" / "group.jpg").size == (50, 100)
    assert summarize(items, selections) == [("cow_1", 1, 1, 0), ("cow_2", 1, 0, 1)]


def test_export_keeps_session_folders(tmp_path):
    raw = tmp_path / "raw"
    for day in ("2026-12-01", "2026-12-02"):  # same file name on two days must not collide
        (raw / "cow_1" / "left" / day).mkdir(parents=True)
        Image.new("RGB", (40, 30), (10, 20, 30) if day.endswith("1") else (90, 80, 70)).save(
            raw / "cow_1" / "left" / day / "IMG_0001.JPG")
    items = collect_photos([raw], project_root=tmp_path)
    selections = {i.key: make_selection(i, AUTO, CowBox(0, 0, 20, 20, 1.0), 0) for i in items}
    export_crops(items, selections, tmp_path / "processed")
    for day in ("2026-12-01", "2026-12-02"):
        assert (tmp_path / "processed" / "cow_1" / "left" / day / "IMG_0001.jpg").is_file()


def test_export_refuses_two_photos_with_one_output(photo_dirs, tmp_path):
    raw, excluded = photo_dirs
    (raw / "cow_1" / "top").mkdir()
    shutil.copy(excluded / "cow_1" / "top" / "group.jpg", raw / "cow_1" / "top" / "group.png")
    items = collect_photos([raw, excluded], project_root=tmp_path)
    assert output_path(items[0], tmp_path).suffix == ".jpg"
    with pytest.raises(ValueError, match="same file"):
        export_crops(items, {}, tmp_path / "processed")
