"""Detection-aware augmentation built on torchvision transforms v2.

Transforms operate jointly on ``(image, target)`` so bounding boxes are moved
along with the image on geometric ops. Augmentation is applied to the training
split only; validation/benchmark use the plain eval transform (no augmentation),
which keeps the shared eval set un-augmented as designed.

Core augmentations (on by default): brightness, contrast, saturation, hue
(HSV-style jitter) plus horizontal flip. Light geometry (scale/rotation) is
opt-in via non-zero parameters.

Important for the synthetic-ratio ablation: keep the augmentation config
constant across all sweep points, otherwise it confounds the accuracy curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import v2


@dataclass
class AugConfig:
    """Augmentation hyper-parameters (all zero -> identity except flip)."""
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.05
    hflip: float = 0.5
    scale: float = 0.0      # +/- fraction, e.g. 0.1 -> RandomAffine scale (0.9, 1.1)
    translate: float = 0.0  # fraction of image size, e.g. 0.05
    rotate: float = 0.0     # degrees, e.g. 5 -> RandomAffine degrees (-5, 5)

    def summary(self) -> str:
        return (
            f"brightness={self.brightness} contrast={self.contrast} "
            f"saturation={self.saturation} hue={self.hue} hflip={self.hflip} "
            f"scale={self.scale} translate={self.translate} rotate={self.rotate}"
        )


def build_transforms(train: bool, cfg: AugConfig | None = None) -> v2.Compose:
    """Build a v2 transform pipeline.

    train=False -> only PIL->float tensor conversion (no augmentation).
    train=True  -> augmentation per ``cfg`` (defaults if None).
    """
    tfs: list = [v2.ToImage()]

    if train:
        cfg = cfg or AugConfig()

        if any((cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)):
            tfs.append(
                v2.ColorJitter(
                    brightness=cfg.brightness or 0.0,
                    contrast=cfg.contrast or 0.0,
                    saturation=cfg.saturation or 0.0,
                    hue=cfg.hue or 0.0,
                )
            )

        if cfg.hflip > 0:
            tfs.append(v2.RandomHorizontalFlip(p=cfg.hflip))

        if cfg.scale > 0 or cfg.rotate > 0 or cfg.translate > 0:
            tfs.append(
                v2.RandomAffine(
                    degrees=(-cfg.rotate, cfg.rotate) if cfg.rotate > 0 else 0,
                    translate=(cfg.translate, cfg.translate) if cfg.translate > 0 else None,
                    scale=(1.0 - cfg.scale, 1.0 + cfg.scale) if cfg.scale > 0 else None,
                )
            )

    tfs.append(v2.ToDtype(torch.float32, scale=True))

    if train:
        # Drop boxes pushed out of frame / degenerate after geometric ops.
        tfs.append(v2.SanitizeBoundingBoxes())

    return v2.Compose(tfs)
