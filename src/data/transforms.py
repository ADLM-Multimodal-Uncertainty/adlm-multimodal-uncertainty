"""Image transforms. ImageNet statistics because the backbones are ImageNet-pretrained."""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_transform(image_size: int, training: bool) -> T.Compose:
    ops = [T.Resize((image_size, image_size))]
    if training:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)
