"""Siamese embedding network for cow hide patterns."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cowid.reid.backbone import build_backbone
from cowid.reid.similarity import cosine_similarity


class EmbeddingHead(nn.Module):
    """Projects backbone features to an L2-normalised embedding."""

    def __init__(self, in_dim: int, embedding_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        layers: list = []
        last_dim = in_dim
        if hidden_dim:
            layers += [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), p=2, dim=1)


class SiameseNetwork(nn.Module):
    """Twin network whose branches share one backbone and head (weight sharing by construction).

    ``model(x)`` embeds a single batch; ``model(x1, x2)`` embeds both inputs of a pair with the same
    weights. Embeddings are unit length, so their dot product equals their cosine similarity.
    """

    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 128, hidden_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.config: Dict[str, Any] = {"backbone": backbone, "embedding_dim": int(embedding_dim),
                                       "hidden_dim": int(hidden_dim), "dropout": float(dropout),
                                       "pretrained": bool(pretrained)}
        self.backbone, feature_dim = build_backbone(backbone, pretrained)
        self.head = EmbeddingHead(feature_dim, embedding_dim, hidden_dim, dropout)
        self._backbone_frozen = False

    @classmethod
    def from_config(cls, model_cfg: Dict[str, Any], pretrained: Optional[bool] = None) -> "SiameseNetwork":
        """Build from a ``model:`` config section; ``pretrained`` overrides the config value."""
        return cls(backbone=model_cfg.get("backbone", "resnet50"),
                   embedding_dim=int(model_cfg.get("embedding_dim", 128)),
                   hidden_dim=int(model_cfg.get("hidden_dim", 512) or 0),
                   dropout=float(model_cfg.get("dropout", 0.2)),
                   pretrained=bool(model_cfg.get("pretrained", True)) if pretrained is None else pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config["embedding_dim"]

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` images -> ``[B, embedding_dim]`` unit-length embeddings."""
        return self.head(self.backbone(images))

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x2 is None:
            return self.embed(x1)
        return self.embed(x1), self.embed(x2)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of each image pair (row ``i`` of ``x1`` with row ``i`` of ``x2``)."""
        e1, e2 = self(x1, x2)
        return cosine_similarity(e1, e2)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the backbone (used to train only the head during the first epochs)."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self._backbone_frozen = not trainable
        if self.training:
            self.train()

    def train(self, mode: bool = True) -> "SiameseNetwork":
        super().train(mode)
        if mode and self._backbone_frozen:
            self.backbone.eval()  # keep pretrained BatchNorm statistics fixed while frozen
        return self
