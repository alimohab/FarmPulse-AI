"""Choose which detected cow in each photo is the labelled cow, and export the chosen crops.

Used by ``scripts/pick_cows.py``. Photos with several cows (or the same photo filed under several
cows) need a person to say which box belongs to the cow named by the folder. This module holds the
parts of that workflow that need no window: finding the images, caching detections, deciding which
photos need review, saving the choices to a CSV, and cropping the chosen boxes into
``<output>/<cow>/<view>/<name>.jpg``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image

from cowid.data.dataset import DEFAULT_EXTENSIONS, DEFAULT_VIEWS, load_image, scan_dataset
from cowid.detection.detector import CowBox, crop_box

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Selection statuses. AUTO: the only detected cow; PICKED: a detected box chosen by a person;
# DRAWN: a box drawn by a person; SKIPPED: the labelled cow is not usable in this photo.
AUTO, PICKED, DRAWN, SKIPPED = "auto", "picked", "drawn", "skipped"
CROPPED_STATUSES = (AUTO, PICKED, DRAWN)
SELECTION_FIELDS = ("image", "cow_id", "view", "status", "x1", "y1", "x2", "y2", "box_index", "updated_at")


@dataclass(frozen=True)
class PhotoItem:
    """One labelled photo. ``key`` is unique (the path, relative to the project root when possible)."""

    key: str
    path: Path
    cow_id: str
    view: str
    md5: str = field(default="", compare=False)
    rel: str = field(default="", compare=False)  # path inside its input folder: <cow>/<view>/[<session>/]<name>


    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Selection:
    image: str
    cow_id: str
    view: str
    status: str
    box: Optional[CowBox] = None
    box_index: Optional[int] = None
    updated_at: str = ""


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key_for(path: Path, project_root: Optional[Path]) -> str:
    if project_root is not None:
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_posix()


def collect_photos(input_dirs: Sequence[PathLike], views: Sequence[str] = DEFAULT_VIEWS,
                   extensions: Sequence[str] = DEFAULT_EXTENSIONS,
                   project_root: Optional[PathLike] = None) -> List[PhotoItem]:
    """All ``<cow>/<view>/[...]/image`` photos in the input folders (missing folders are skipped)."""
    root = Path(project_root) if project_root else None
    items: List[PhotoItem] = []
    for input_dir in map(Path, input_dirs):
        if not input_dir.is_dir():
            logger.warning("Input folder not found, skipped: %s", input_dir)
            continue
        for s in scan_dataset(input_dir, views, extensions):
            path = input_dir / s.image
            items.append(PhotoItem(_key_for(path, root), path, s.cow_id, s.view, _file_md5(path), s.image))
    keys = [i.key for i in items]
    if len(keys) != len(set(keys)):
        raise ValueError("The same input folder was given twice")
    return items


def siblings(items: Iterable[PhotoItem]) -> Dict[str, List[PhotoItem]]:
    """key -> the other items holding the identical photo under a different cow."""
    by_md5: Dict[str, List[PhotoItem]] = defaultdict(list)
    for item in items:
        by_md5[item.md5].append(item)
    return {i.key: [o for o in group if o.cow_id != i.cow_id]
            for group in by_md5.values() for i in group if any(o.cow_id != i.cow_id for o in group)}


def needs_review(boxes: Sequence[CowBox], is_shared: bool, review_all: bool = False) -> bool:
    """A person must choose unless exactly one cow was detected in a photo filed under one cow."""
    return review_all or is_shared or len(boxes) != 1


# ---- detection cache ------------------------------------------------------------------------------

def _stamp(path: Path) -> List[float]:
    st = path.stat()
    return [st.st_size, round(st.st_mtime, 3)]


def load_detection_cache(path: PathLike) -> Dict[str, dict]:
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_detection_cache(cache: Dict[str, dict], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp.replace(path)


def detect_all(items: Sequence[PhotoItem], detect: Callable[[Image.Image], List[CowBox]],
               cache: Dict[str, dict], save: Optional[Callable[[], None]] = None,
               save_every: int = 25) -> Dict[str, List[CowBox]]:
    """Detections for every photo, reusing ``cache`` entries whose file is unchanged.

    ``detect`` gets the EXIF-rotated image, so boxes are in the same coordinates as ``load_image``.
    Boxes are stored left to right, so box numbers are stable between runs.
    """
    result: Dict[str, List[CowBox]] = {}
    todo = [i for i in items if cache.get(i.key, {}).get("stamp") != _stamp(i.path)]
    if todo:
        logger.info("Detecting cows in %d photo(s) (%d cached)...", len(todo), len(items) - len(todo))
    for n, item in enumerate(todo, start=1):
        boxes = sorted(detect(load_image(item.path)), key=lambda b: (b.x1, b.y1))
        cache[item.key] = {"stamp": _stamp(item.path),
                           "boxes": [[b.x1, b.y1, b.x2, b.y2, b.confidence] for b in boxes]}
        if n % save_every == 0 or n == len(todo):
            logger.info("  detected %d/%d", n, len(todo))
            if save:
                save()
    for item in items:
        result[item.key] = [CowBox(*b[:4], confidence=b[4]) for b in cache[item.key]["boxes"]]
    return result


# ---- selections CSV -------------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_selections(path: PathLike) -> Dict[str, Selection]:
    path = Path(path)
    if not path.is_file():
        return {}
    out: Dict[str, Selection] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(SELECTION_FIELDS) <= set(reader.fieldnames):
            raise ValueError(f"{path} is not a selections file (expected columns {', '.join(SELECTION_FIELDS)})")
        for row in reader:
            box = None
            if row["x1"] != "":
                box = CowBox(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), 1.0)
            index = int(row["box_index"]) if row["box_index"] != "" else None
            out[row["image"]] = Selection(row["image"], row["cow_id"], row["view"], row["status"], box, index,
                                          row["updated_at"])
    return out


def save_selections(selections: Dict[str, Selection], path: PathLike) -> None:
    """Write all selections (atomically, so quitting mid-write never loses earlier choices)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SELECTION_FIELDS)
        for key in sorted(selections):
            s = selections[key]
            coords = [f"{v:.1f}" for v in (s.box.x1, s.box.y1, s.box.x2, s.box.y2)] if s.box else ["", "", "", ""]
            writer.writerow([s.image, s.cow_id, s.view, s.status, *coords,
                             "" if s.box_index is None else s.box_index, s.updated_at])
    tmp.replace(path)


def make_selection(item: PhotoItem, status: str, box: Optional[CowBox] = None,
                   box_index: Optional[int] = None) -> Selection:
    if status in CROPPED_STATUSES and box is None:
        raise ValueError(f"status '{status}' needs a box")
    return Selection(item.key, item.cow_id, item.view, status, box, box_index, now_stamp())


# ---- export -----------------------------------------------------------------------------------------

def output_path(item: PhotoItem, output_dir: PathLike) -> Path:
    """Crop location; session sub-folders (``<cow>/<view>/<session>/``) are kept so splits can group by them."""
    if item.rel:
        return Path(output_dir) / Path(item.rel).with_suffix(".jpg")
    return Path(output_dir) / item.cow_id / item.view / f"{item.path.stem}.jpg"


def export_crops(items: Sequence[PhotoItem], selections: Dict[str, Selection], output_dir: PathLike,
                 padding: float = 0.05, overwrite: bool = False) -> Dict[str, int]:
    """Write one crop per selected photo to ``<output_dir>/<cow>/<view>/<name>.jpg``.

    A crop is (re)written when it is missing, older than its selection, or ``overwrite`` is set.
    Crops of photos that are now skipped are deleted, so the output always matches the CSV.
    """
    counts = {"written": 0, "up_to_date": 0, "removed_skipped": 0, "not_selected": 0}
    targets: Dict[Path, str] = {}
    for item in items:
        out = output_path(item, output_dir)
        if out in targets:
            raise ValueError(f"Two photos would be cropped to the same file {out}: {targets[out]} and {item.key}")
        targets[out] = item.key
        sel = selections.get(item.key)
        if sel is None:
            counts["not_selected"] += 1
            continue
        if sel.status == SKIPPED:
            if out.exists():
                out.unlink()
                counts["removed_skipped"] += 1
            continue
        selected_at = datetime.fromisoformat(sel.updated_at).timestamp() if sel.updated_at else 0.0
        if out.exists() and not overwrite and out.stat().st_mtime >= selected_at:
            counts["up_to_date"] += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        crop_box(load_image(item.path), sel.box, padding).save(out, "JPEG", quality=95)
        counts["written"] += 1
    return counts


def summarize(items: Sequence[PhotoItem], selections: Dict[str, Selection]) -> List[Tuple[str, int, int, int]]:
    """Per cow: (cow, cropped photos, skipped photos, photos still without a choice)."""
    rows: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for item in items:
        sel = selections.get(item.key)
        rows[item.cow_id][0 if sel and sel.status in CROPPED_STATUSES else 1 if sel else 2] += 1
    return [(cow, *rows[cow]) for cow in sorted(rows)]
