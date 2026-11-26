"""Image transforms. ImageNet statistics because the backbones are ImageNet-pretrained."""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AUGMENTATION_PRESETS = ("none", "flip", "derm")


def image_transform(
    image_size: int, training: bool, augmentation: str = "flip"
) -> T.Compose:
    if augmentation not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"unknown augmentation preset {augmentation!r}; expected one of {AUGMENTATION_PRESETS}"
        )
    if not training or augmentation == "none":
        ops: list = [T.Resize((image_size, image_size))]
    elif augmentation == "flip":
        ops = [T.Resize((image_size, image_size)), T.RandomHorizontalFlip()]
    else:
        # Gentle crop/rotation/jitter suitable for centered clinical skin lesion photographs;
        # hue jitter stays small (0.02) to preserve diagnostic melanin/erythema color cues.
        ops = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)
