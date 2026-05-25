from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBDDataset(Dataset):
    """Paired RGB + metric depth images from O3DE simulation.

    Directory structure:
        root/
          rgb/    rgb_<timestamp>.png      (uint8 RGB)
          depth/  depth_<timestamp>.png   (uint16, millimeters)
          camera/ camera_<timestamp>.json (ROS camera_info with K matrix)

    Returns (rgb, depth, K) where:
        rgb   – float32 tensor [3, H, W], ImageNet-normalised
        depth – float32 tensor [1, H, W], in metres
        K     – float32 tensor [3, 3], camera intrinsics
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        resize: tuple[int, int] | None = None,
    ) -> None:
        root = Path(root)
        depth_dir = root / "depth"
        camera_dir = root / "camera"
        all_rgb = sorted(
            p for p in (root / "rgb").iterdir()
            if p.suffix == ".png" and "Zone" not in p.name
            and (depth_dir / f"depth_{p.stem[len('rgb_'):]}.png").exists()
            and (camera_dir / f"camera_{p.stem[len('rgb_'):]}.json").exists()
        )
        if not all_rgb:
            raise FileNotFoundError(f"No complete RGB+depth+camera triplets found in {root}")

        rng = random.Random(seed)
        indices = list(range(len(all_rgb)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))

        selected = sorted(indices[:n_val] if split == "val" else indices[n_val:])

        self.rgb_paths = [all_rgb[i] for i in selected]
        self.depth_dir = depth_dir
        self.camera_dir = camera_dir
        self.resize = resize  # (width, height) in pixels

        self._normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        timestamp = rgb_path.stem[len("rgb_"):]

        depth_path = self.depth_dir / f"depth_{timestamp}.png"
        camera_path = self.camera_dir / f"camera_{timestamp}.json"

        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_img = Image.open(depth_path)

        if self.resize is not None:
            rgb_img = rgb_img.resize(self.resize, Image.BILINEAR)
            depth_img = depth_img.resize(self.resize, Image.NEAREST)

        rgb = self._normalize(T.ToTensor()(rgb_img))

        # uint16 millimetres → float32 metres
        depth_mm = np.array(depth_img, dtype=np.float32)
        depth = torch.from_numpy(depth_mm / 1000.0).unsqueeze(0)

        with open(camera_path) as f:
            cam = json.load(f)
        K = torch.tensor(cam["K"], dtype=torch.float32).reshape(3, 3)

        return rgb, depth, K


def collate_fn(batch):
    rgbs, depths, Ks = zip(*batch)
    return torch.stack(rgbs), torch.stack(depths), torch.stack(Ks)
