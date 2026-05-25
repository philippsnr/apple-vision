from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DepthEstimator(nn.Module):
    """ResNet50 encoder + U-Net decoder for metric depth estimation.

    Input:  RGB image  [B, 3, H, W],  ImageNet-normalised
    Output: depth map  [B, 1, H, W],  in metres (positive)

    Encoder feature sizes for a 800×1280 input:
        e0  64ch  400×640   (after conv1+bn+relu)
        e1 256ch  200×320   (layer1, after maxpool)
        e2 512ch  100×160   (layer2)
        e3 1024ch  50× 80   (layer3)
        e4 2048ch  25× 40   (layer4)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.dec4 = _DecoderBlock(2048, 1024, 512)
        self.dec3 = _DecoderBlock(512, 512, 256)
        self.dec2 = _DecoderBlock(256, 256, 128)
        self.dec1 = _DecoderBlock(128, 64, 64)
        self.dec0 = _DecoderBlock(64, 0, 32)

        self.head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d = self.dec4(e4, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.dec0(d)

        return self.head(d)


def si_log_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
    """Scale-invariant logarithmic loss (Eigen et al., 2014).

    lam=0.5 is the original formulation; lam=0 reduces to plain MSE in log space.
    Only pixels where target > 0 contribute (masks holes / sensor drop-outs).
    """
    eps = 1e-6
    mask = target > eps
    if mask.sum() == 0:
        return pred.sum() * 0.0

    log_diff = torch.log(pred[mask] + eps) - torch.log(target[mask] + eps)
    return log_diff.pow(2).mean() - lam * log_diff.mean().pow(2)
