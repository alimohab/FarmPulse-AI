"""Image transforms that preserve the identity-defining hide pattern."""
from __future__ import annotations

from typing import Sequence

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: Sequence[int]) -> T.Compose:
    """Training augmentations that simulate camera/lighting variation without altering the pattern.

    Deliberately excluded:
      * horizontal/vertical flips: a mirrored flank or back is a pattern that does not exist on the
        animal (a flipped left flank is not the right flank), so flips would teach false identities;
      * hue jitter and strong saturation changes: pattern colours (e.g. black/white/brown patches)
        must stay recognisable; only illumination is varied.
    Crops, rotation and perspective are kept mild so most of the visible pattern remains in frame.
    """
    h, w = int(image_size[0]), int(image_size[1])
    aspect = w / h
    return T.Compose([
        T.Resize((h, w)),
        T.RandomResizedCrop((h, w), scale=(0.8, 1.0), ratio=(aspect * 0.9, aspect * 1.1)),
        T.RandomApply([T.RandomRotation(degrees=8)], p=0.5),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.0),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Small occlusions (fences, feeders, other cows) covering a few percent of the image.
        T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
    ])


def build_eval_transform(image_size: Sequence[int]) -> T.Compose:
    """Deterministic preprocessing for validation, testing and inference."""
    h, w = int(image_size[0]), int(image_size[1])
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
