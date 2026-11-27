"""Image transforms. ImageNet statistics because the backbones are ImageNet-pretrained.

Stochastic augmentation is training-only. Validation and test images are resized and
normalized, so augmented copies of one image cannot cross a patient split. Geometric
operations use torchvision and therefore the torch RNG, which `torch.random.fork_rng`
can restore while training metrics are scored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AugmentationConfig:
    """Numeric augmentation recipe selected by `configs/data/augmentation/<name>.yaml`."""

    def __init__(
        self,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        rotation_degrees: float = 0.0,
        crop_scale: Sequence[float] | None = None,
        crop_ratio: Sequence[float] | None = None,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.rotation_degrees = float(rotation_degrees)
        self.crop_scale = None if crop_scale is None else (
            float(crop_scale[0]), float(crop_scale[1])
        )
        self.crop_ratio = None if crop_ratio is None else (
            float(crop_ratio[0]), float(crop_ratio[1])
        )
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        if self.crop_scale is None != self.crop_ratio is None:
            raise ValueError("crop_scale and crop_ratio must be set together")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        if not 0 <= self.hue <= 0.5:
            raise ValueError("hue jitter must be in [0, 0.5]")


def rotation_padding(image_size: int, degrees: float) -> int:
    """Reflect padding that keeps a center crop free of empty corners after rotation."""
    theta = math.radians(min(abs(degrees), 45.0))
    margin = (math.sin(theta) + math.cos(theta) - 1.0) / 2.0
    return math.ceil(image_size * margin)


def image_transform(
    image_size: int, training: bool, augmentation: AugmentationConfig | None = None
) -> T.Compose:
    """Deterministic resize and normalization, plus the training recipe when requested."""
    if not training or augmentation is None:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    ops: list = []
    if augmentation.crop_scale is not None and augmentation.crop_ratio is not None:
        ops.append(
            T.RandomResizedCrop(
                image_size,
                scale=augmentation.crop_scale,
                ratio=augmentation.crop_ratio,
            )
        )
    else:
        ops.append(T.Resize((image_size, image_size)))
    if augmentation.horizontal_flip > 0:
        ops.append(T.RandomHorizontalFlip(p=augmentation.horizontal_flip))
    if augmentation.vertical_flip > 0:
        ops.append(T.RandomVerticalFlip(p=augmentation.vertical_flip))
    if augmentation.rotation_degrees > 0:
        # Pad with reflected pixels before rotating, then crop back, so the lesion
        # border stays in frame and the corners are not filled with black.
        padding = rotation_padding(image_size, augmentation.rotation_degrees)
        ops.append(T.Pad(padding, padding_mode="reflect"))
        ops.append(
            T.RandomRotation(
                degrees=augmentation.rotation_degrees,
                interpolation=T.InterpolationMode.BILINEAR,
            )
        )
        ops.append(T.CenterCrop(image_size))
    if any(
        value > 0
        for value in (
            augmentation.brightness,
            augmentation.contrast,
            augmentation.saturation,
            augmentation.hue,
        )
    ):
        ops.append(
            T.ColorJitter(
                brightness=augmentation.brightness,
                contrast=augmentation.contrast,
                saturation=augmentation.saturation,
                hue=augmentation.hue,
            )
        )
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)
