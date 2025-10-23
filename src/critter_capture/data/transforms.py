"""Image transformation utilities."""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T


def build_transforms(
    image_size: int, augment: bool = True
) -> tuple[Callable, Callable]:
    """Return training and evaluation transforms."""

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        train_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    eval_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ]
    )

    return train_transform, eval_transform


__all__ = ["build_transforms"]
