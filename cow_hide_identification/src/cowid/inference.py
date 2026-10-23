"""Embedding extraction and image comparison with a trained checkpoint."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from PIL import Image

from cowid.data.augmentations import build_eval_transform
from cowid.data.dataset import load_image
from cowid.reid.siamese_network import SiameseNetwork
from cowid.reid.similarity import cosine_similarity
from cowid.utils.checkpoint_utils import load_model_for_inference
from cowid.utils.device import get_device

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image]


class CowEmbedder:
    """Turns cow images into unit-length hide-pattern embeddings and compares them."""

    def __init__(self, model: SiameseNetwork, image_size: Sequence[int], device: torch.device,
                 cropper: Optional[Any] = None, checkpoint: Optional[Dict[str, Any]] = None):
        self.model = model.eval()
        self.device = device
        self.transform = build_eval_transform(image_size)
        self.cropper = cropper  # optional cowid.detection.detector.CowCropper
        self.checkpoint_info = {k: (checkpoint or {}).get(k) for k in ("epoch", "monitor", "best_metric", "saved_at")}

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: str = "auto", cropper: Optional[Any] = None
                        ) -> "CowEmbedder":
        torch_device = get_device(device)
        model, checkpoint = load_model_for_inference(path, torch_device)
        image_size = checkpoint.get("image_size") or [224, 224]
        return cls(model, image_size, torch_device, cropper, checkpoint)

    def _prepare(self, image: ImageInput) -> torch.Tensor:
        pil = image.convert("RGB") if isinstance(image, Image.Image) else load_image(image)
        if self.cropper is not None:
            crop = self.cropper.crop_largest(pil)
            if crop is None:
                logger.warning("No cow detected in %s; using the full image.",
                               image if not isinstance(image, Image.Image) else "image")
            else:
                pil = crop
        return self.transform(pil)

    @torch.no_grad()
    def embed(self, images: Sequence[ImageInput]) -> torch.Tensor:
        """``[N, embedding_dim]`` embeddings (on the CPU) for a list of image paths or PIL images."""
        batch = torch.stack([self._prepare(img) for img in images]).to(self.device)
        return self.model(batch).float().cpu()

    def compare(self, image_a: ImageInput, image_b: ImageInput) -> float:
        """Cosine similarity in [-1, 1]; higher means more likely the same cow."""
        emb = self.embed([image_a, image_b])
        return float(cosine_similarity(emb[0], emb[1]))
