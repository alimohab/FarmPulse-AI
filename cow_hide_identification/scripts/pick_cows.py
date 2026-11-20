"""Crop the *labelled* cow out of photos that show several cows, with a person choosing the box.

crop_cows.py always keeps the largest cow, which is often the wrong one in group photos. This tool:

  1. detects every cow in each photo with YOLO (results are cached, so later runs start instantly);
  2. picks the box automatically when a photo shows exactly one cow and is filed under one cow;
  3. opens a window for every other photo: numbered boxes, and the cow you must find in the title;
  4. saves each choice to data/crop_selections.csv immediately (quit and continue any time);
  5. crops every chosen box into data/processed/<cow>/<view>/<name>.jpg.

A photo filed under several cows (data/excluded_multi_cow) is shown once per cow, so each cow gets
its own crop of the same photo.

Keys in the window:
  1-9 or click a box   this box is the cow named in the title
  d                    draw the box yourself (drag, then ENTER; c cancels)
  s                    skip: the cow is not visible or not usable in this photo
  b                    back to the previous photo (its choice is cleared)
  q / Esc / close      save and quit

Examples (from the project folder):
    python scripts\\pick_cows.py
    python scripts\\pick_cows.py --review-all          # also check the automatic single-cow picks
    python scripts\\pick_cows.py --export-only         # just (re)write the crops from the CSV
Then set `paths.dataset_dir: data/processed` in configs\\training_config.yaml and train with --resplit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cowid.config import load_config, resolve_cli_path, resolve_path  # noqa: E402
from cowid.data.dataset import load_image  # noqa: E402
from cowid.detection.detector import CowBox  # noqa: E402
from cowid.detection.selection import (AUTO, DRAWN, PICKED, SKIPPED, PhotoItem, Selection,  # noqa: E402
                                       collect_photos, detect_all, export_crops, load_detection_cache,
                                       load_selections, make_selection, needs_review, save_detection_cache,
                                       save_selections, siblings, summarize)
from cowid.utils.logging_utils import setup_logging  # noqa: E402

WINDOW = "pick_cows - which box is the labelled cow?"
HEADER_H = 92
MAX_CHOICES = 9
COLORS = [(0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 128, 255),
          (255, 255, 0), (128, 0, 255), (0, 255, 160), (160, 160, 255)]
TAKEN_COLOR = (150, 150, 150)
DEFAULT_MODEL = PROJECT_ROOT.parent / "cow-detection" / "models" / "yolo11m.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Choose the labelled cow in multi-cow photos and crop it.")
    p.add_argument("--inputs", nargs="+", default=["data/raw", "data/excluded_multi_cow"],
                   help="Folders laid out as <cow>/<view>/... (missing folders are skipped)")
    p.add_argument("--output", default="data/processed", help="Where the crops are written")
    p.add_argument("--selections", default="data/crop_selections.csv", help="CSV holding every choice")
    p.add_argument("--model", default=str(DEFAULT_MODEL if DEFAULT_MODEL.is_file() else "yolo11s.pt"),
                   help="YOLO weights with a 'cow' class")
    p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--padding", type=float, default=0.05, help="Extra margin around each crop (fraction of box)")
    p.add_argument("--review-all", action="store_true",
                   help="Also show photos that were picked automatically (single cow detected)")
    p.add_argument("--no-gui", action="store_true", help="Detect and auto-pick only; no window")
    p.add_argument("--export-only", action="store_true", help="Skip detection and picking; only write crops")
    p.add_argument("--overwrite", action="store_true", help="Rewrite every crop, even if up to date")
    p.add_argument("--max-display", nargs=2, type=int, default=[1400, 820], metavar=("W", "H"),
                   help="Largest size of the photo on screen")
    p.add_argument("--config", default=None)
    return p.parse_args()


# ---- picker window --------------------------------------------------------------------------------

class Picker:
    """Shows one photo at a time and records the user's choice in ``selections``."""

    def __init__(self, pending: Sequence[PhotoItem], detections: Dict[str, List[CowBox]],
                 sibling_map: Dict[str, List[PhotoItem]], selections: Dict[str, Selection],
                 save, max_display: Sequence[int]):
        self.pending = list(pending)
        self.detections = detections
        self.sibling_map = sibling_map
        self.selections = selections
        self.save = save
        self.max_w, self.max_h = max_display
        self.clicked: Optional[int] = None
        self.choices: List[int] = []    # indices into the detection list, in on-screen order 1..9
        self.scale = 1.0

    # -- drawing --
    def _taken(self, item: PhotoItem) -> Dict[int, str]:
        """Detection index -> sibling cow that already uses that box for this same photo."""
        taken = {}
        for other in self.sibling_map.get(item.key, []):
            sel = self.selections.get(other.key)
            if sel and sel.box_index is not None and sel.status in (AUTO, PICKED):
                taken[sel.box_index] = other.cow_id
        return taken

    def _render(self, item: PhotoItem, photo_bgr: np.ndarray, position: int, message: str) -> np.ndarray:
        boxes = self.detections[item.key]
        taken = self._taken(item)
        img = photo_bgr.copy()
        for n, idx in enumerate(self.choices, start=1):
            b = boxes[idx]
            color = TAKEN_COLOR if idx in taken else COLORS[(n - 1) % len(COLORS)]
            p1 = (int(b.x1 * self.scale), int(b.y1 * self.scale))
            p2 = (int(b.x2 * self.scale), int(b.y2 * self.scale))
            cv2.rectangle(img, p1, p2, color, 3)
            label = f"{n}" + (f" = {taken[idx]}" if idx in taken else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(img, (p1[0], p1[1]), (p1[0] + tw + 12, p1[1] + th + 14), color, -1)
            cv2.putText(img, label, (p1[0] + 6, p1[1] + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        header = np.zeros((HEADER_H, max(img.shape[1], 900), 3), dtype=np.uint8)
        others = [o.cow_id for o in self.sibling_map.get(item.key, [])]
        line1 = f"[{position + 1}/{len(self.pending)}]  Which box is  {item.cow_id}  ?   view: {item.view}"
        line2 = (f"file: {item.name}   detected cows: {len(boxes)}"
                 + (f"   same photo also filed under: {', '.join(others)}" if others else ""))
        line3 = message or "1-9/click: pick   d: draw box   s: skip   b: back   q: quit"
        cv2.putText(header, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(header, line2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(header, line3, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 255) if message else (200, 200, 200), 1)
        canvas = np.zeros((HEADER_H + img.shape[0], header.shape[1], 3), dtype=np.uint8)
        canvas[:HEADER_H] = header
        canvas[HEADER_H:, :img.shape[1]] = img
        return canvas

    def _load(self, item: PhotoItem) -> np.ndarray:
        rgb = np.asarray(load_image(item.path))
        h, w = rgb.shape[:2]
        self.scale = min(self.max_w / w, (self.max_h - HEADER_H) / h, 1.0)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_AREA)

    def _choices_for(self, item: PhotoItem) -> List[int]:
        """Up to 9 boxes (the largest ones), numbered left to right."""
        boxes = self.detections[item.key]
        largest = sorted(range(len(boxes)), key=lambda i: boxes[i].area, reverse=True)[:MAX_CHOICES]
        return sorted(largest, key=lambda i: (boxes[i].x1, boxes[i].y1))

    # -- input --
    def _on_mouse(self, event, x, y, flags, item_boxes) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or y < HEADER_H:
            return
        px, py = x / self.scale, (y - HEADER_H) / self.scale
        hits = [n for n, idx in enumerate(self.choices)
                if item_boxes[idx].x1 <= px <= item_boxes[idx].x2 and item_boxes[idx].y1 <= py <= item_boxes[idx].y2]
        if hits:  # the smallest box under the cursor (a cow in front of another)
            self.clicked = min(hits, key=lambda n: item_boxes[self.choices[n]].area)

    def _draw_box(self, canvas: np.ndarray, photo_shape) -> Optional[CowBox]:
        x, y, w, h = cv2.selectROI(WINDOW, canvas, showCrosshair=True, fromCenter=False)
        if w < 5 or h < 5:
            return None
        ph, pw = photo_shape[:2]
        x1, y1 = max(0, x), max(0, y - HEADER_H)
        x2, y2 = min(pw, x + w), min(ph, y - HEADER_H + h)
        if x2 <= x1 or y2 <= y1:
            return None
        s = self.scale
        return CowBox(x1 / s, y1 / s, x2 / s, y2 / s, confidence=1.0)

    def run(self) -> bool:
        """Returns True when every pending photo has a choice, False if the user quit early."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        i, message, confirm = 0, "", None
        loaded_key, photo = None, None
        while i < len(self.pending):
            item = self.pending[i]
            if item.key != loaded_key:
                photo, loaded_key = self._load(item), item.key
            self.choices = self._choices_for(item)
            boxes = self.detections[item.key]
            cv2.setMouseCallback(WINDOW, self._on_mouse, boxes)
            canvas = self._render(item, photo, i, message)
            cv2.imshow(WINDOW, canvas)
            message, step = "", 0
            while step == 0:
                key = cv2.waitKey(50) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return False
                choice = None
                if self.clicked is not None:
                    choice, self.clicked = self.clicked, None
                elif ord("1") <= key <= ord("9"):
                    choice = key - ord("1")
                    if choice >= len(self.choices):
                        message, step = f"There is no box {choice + 1} in this photo.", -99
                        continue
                if choice is not None:
                    idx = self.choices[choice]
                    taken = self._taken(item)
                    if idx in taken and confirm != (item.key, idx):
                        confirm = (item.key, idx)
                        message, step = (f"Box {choice + 1} is already {taken[idx]} for this same photo. "
                                         f"Press {choice + 1} again to use it anyway."), -99
                        continue
                    self.selections[item.key] = make_selection(item, PICKED, boxes[idx], idx)
                    step = 1
                elif key == ord("d"):
                    box = self._draw_box(canvas, photo.shape)
                    if box is None:
                        message, step = "Drawing cancelled.", -99
                        continue
                    self.selections[item.key] = make_selection(item, DRAWN, box)
                    step = 1
                elif key == ord("s"):
                    self.selections[item.key] = make_selection(item, SKIPPED)
                    step = 1
                elif key == ord("b"):
                    if i == 0:
                        message, step = "This is the first photo.", -99
                        continue
                    i -= 1
                    self.selections.pop(self.pending[i].key, None)
                    step = -1
                elif key in (ord("q"), 27):
                    return False
            if step == 1:
                confirm = None
                i += 1
            if step != -99:
                self.save()
        return True


# ---- main -----------------------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging()
    views = cfg["data"]["views"]
    extensions = cfg["data"].get("image_extensions", [".jpg", ".jpeg", ".png"])
    inputs = [resolve_cli_path(d) for d in args.inputs]
    output_dir = resolve_cli_path(args.output)
    selections_path = resolve_cli_path(args.selections)
    if output_dir in inputs:
        print("ERROR: --output must not be one of the --inputs folders.", file=sys.stderr)
        return 1

    items = collect_photos(inputs, views, extensions, PROJECT_ROOT)
    if not items:
        print("ERROR: no photos found in " + ", ".join(map(str, inputs)), file=sys.stderr)
        return 1
    sibling_map = siblings(items)
    selections = load_selections(selections_path)
    known = {i.key for i in items}
    print(f"{len(items)} photo(s) of {len({i.cow_id for i in items})} cows; "
          f"{len(sibling_map)} are copies of a photo filed under another cow; "
          f"{sum(k in selections for k in known)} already have a choice.")

    def save() -> None:
        save_selections(selections, selections_path)

    finished = True
    if not args.export_only:
        from cowid.detection.detector import CowCropper
        cropper = CowCropper(args.model, conf=args.conf, device=args.device)
        cache_path = resolve_path(cfg["paths"]["output_dir"]) / "crop_detections.json"
        cache = load_detection_cache(cache_path)
        detections = detect_all(items, cropper.detect, cache, save=lambda: save_detection_cache(cache, cache_path))

        auto = 0
        for item in items:
            boxes = detections[item.key]
            if item.key not in selections and not needs_review(boxes, item.key in sibling_map, args.review_all):
                selections[item.key] = make_selection(item, AUTO, boxes[0], 0)
                auto += 1
        save()
        pending = [i for i in items if i.key not in selections
                   or (args.review_all and selections[i.key].status == AUTO)]
        print(f"Picked automatically: {auto} new photo(s) with a single cow. Waiting for you: {len(pending)}.")
        if pending and not args.no_gui:
            if args.review_all:  # show auto picks again, but keep them until the user decides
                for item in pending:
                    if selections.get(item.key) and selections[item.key].status == AUTO:
                        selections.pop(item.key)
            print(__doc__.split("Keys in the window:")[1].split("Examples")[0])
            picker = Picker(pending, detections, sibling_map, selections, save, args.max_display)
            try:
                finished = picker.run()
            finally:
                cv2.destroyAllWindows()
                if args.review_all:  # photos the user did not reach keep their automatic pick
                    for item in pending:
                        boxes = detections[item.key]
                        if item.key not in selections and len(boxes) == 1 and item.key not in sibling_map:
                            selections[item.key] = make_selection(item, AUTO, boxes[0], 0)
                save()
        elif pending:
            finished = False

    counts = export_crops(items, selections, output_dir, args.padding, args.overwrite)
    print(f"\nCrops in {output_dir}: {counts['written']} written, {counts['up_to_date']} already up to date, "
          f"{counts['removed_skipped']} removed (skipped). {counts['not_selected']} photo(s) still need a choice.")
    print("\ncow        cropped  skipped  waiting")
    for cow, cropped, skipped, waiting in summarize(items, selections):
        print(f"{cow:<10}{cropped:>8}{skipped:>9}{waiting:>9}")
    print(f"\nChoices saved in {selections_path}")
    if not finished:
        print("Run the same command again to continue where you stopped.")
    else:
        print("All photos have a choice. Next: set paths.dataset_dir: data/processed in "
              "configs/training_config.yaml, then: python scripts/train_model.py --resplit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
