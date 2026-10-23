"""Feature-extractor backbones.

A backbone builder takes ``pretrained: bool`` and returns ``(module, feature_dim)`` where the module
maps an image batch ``[B, 3, H, W]`` to features ``[B, feature_dim]``. Add new backbones with
:func:`register_backbone`; nothing else in the project needs to change.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch.nn as nn
from torchvision import models as tv_models

BackboneBuilder = Callable[[bool], Tuple[nn.Module, int]]

_REGISTRY: Dict[str, BackboneBuilder] = {}


def register_backbone(name: str, builder: BackboneBuilder) -> None:
    _REGISTRY[name.lower()] = builder


def available_backbones() -> List[str]:
    return sorted(_REGISTRY)


def _torchvision_resnet(name: str) -> BackboneBuilder:
    def build(pretrained: bool) -> Tuple[nn.Module, int]:
        model = tv_models.get_model(name, weights="DEFAULT" if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()  # keep the globally pooled features
        return model, feature_dim
    return build


def _torchvision_efficientnet(name: str) -> BackboneBuilder:
    def build(pretrained: bool) -> Tuple[nn.Module, int]:
        model = tv_models.get_model(name, weights="DEFAULT" if pretrained else None)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    return build


for _name in ("resnet18", "resnet34", "resnet50", "resnet101"):
    register_backbone(_name, _torchvision_resnet(_name))
for _name in ("efficientnet_b0", "efficientnet_b2"):
    register_backbone(_name, _torchvision_efficientnet(_name))


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Instantiate a registered backbone; ``pretrained`` loads ImageNet weights (downloaded once)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {', '.join(available_backbones())}")
    try:
        return _REGISTRY[key](pretrained)
    except (OSError, RuntimeError) as exc:
        if not pretrained:
            raise
        raise RuntimeError(
            f"Could not load ImageNet weights for '{name}': {exc}\nThe first run downloads them from "
            "download.pytorch.org into the torch hub cache. Check the internet connection, or set "
            "model.pretrained: false (not recommended for small datasets).") from exc
