"""Optional cow detection/cropping with Ultralytics YOLO.

Hide-pattern embeddings work best on tight crops of a single cow. This module crops the largest
detected cow from an image, either for one-off comparisons or to mirror a whole ``Cow_XXX/<view>``
dataset of full frames into a cropped copy (e.g. ``data/processed``). Requires ``pip install ultralytics``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from PIL import Image

from cowid.data.dataset import DEFAULT_EXTENSIONS, DEFAULT_VIEWS, load_image, scan_dataset

logger = logging.getLogger(__name__)

COW_CLASS_NAMES = {"cow", "cows", "cattle", "bull", "calf", "heifer", "dairy cow", "beef cattle"}


@dataclass
class CowBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class CowCropper:
    """Detects cows (COCO class 'cow' or a custom cattle model) and crops them."""

    def __init__(self, model: str = "yolo11s.pt", conf: float = 0.35, device: str = "auto", padding: float = 0.05):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Cow detection needs the optional 'ultralytics' package: pip install ultralytics") from exc
        self.model = YOLO(model)
        self.conf = conf
        self.padding = padding
        cuda = torch.cuda.is_available() and device != "cpu"
        self.device = ("0" if device in ("auto", "cuda") else device.replace("cuda:", "")) if cuda else "cpu"
        names = self.model.names
        self.cow_class_ids = [i for i, n in names.items() if str(n).strip().lower() in COW_CLASS_NAMES]
        if not self.cow_class_ids:
            raise ValueError(f"Detector '{model}' has no cow/cattle class (classes: {list(names.values())[:20]})")

    def detect(self, image: Image.Image) -> List[CowBox]:
        result = self.model.predict(image, conf=self.conf, classes=self.cow_class_ids, device=self.device,
                                    verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            return []
        xyxy = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        return [CowBox(*box, confidence=c) for box, c in zip(xyxy, confs)]

    def crop_largest(self, image: Image.Image) -> Optional[Image.Image]:
        """Crop the largest detected cow (with relative padding), or ``None`` if no cow is found."""
        boxes = self.detect(image)
        if not boxes:
            return None
        box = max(boxes, key=lambda b: b.area)
        pad_x, pad_y = (box.x2 - box.x1) * self.padding, (box.y2 - box.y1) * self.padding
        w, h = image.size
        return image.crop((max(0, int(box.x1 - pad_x)), max(0, int(box.y1 - pad_y)),
                           min(w, int(box.x2 + pad_x)), min(h, int(box.y2 + pad_y))))


def crop_dataset(source_dir: Union[str, Path], target_dir: Union[str, Path], cropper: CowCropper,
                 views: Sequence[str] = DEFAULT_VIEWS, extensions: Sequence[str] = DEFAULT_EXTENSIONS,
                 overwrite: bool = False) -> Dict[str, int]:
    """Mirror ``source_dir`` (Cow_XXX/<view>/...) into ``target_dir`` with each image cropped to its largest cow.

    Images without a detected cow are skipped and reported; review the crops when frames contain
    several cows, because the largest one is not necessarily the labelled cow.
    """
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    counts = {"cropped": 0, "no_cow": 0, "skipped_existing": 0}
    for sample in scan_dataset(source_dir, views, extensions):
        out_path = target_dir / sample.image
        if out_path.exists() and not overwrite:
            counts["skipped_existing"] += 1
            continue
        crop = cropper.crop_largest(load_image(source_dir / sample.image))
        if crop is None:
            counts["no_cow"] += 1
            logger.warning("No cow detected in %s", sample.image)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)
        counts["cropped"] += 1
    return counts
