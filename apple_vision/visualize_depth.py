from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.rgbd import RGBDDataset
from .models.depth_estimator import DepthEstimator

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and convert to HWC uint8."""
    img = tensor.cpu() * _IMAGENET_STD + _IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def visualize(
    checkpoint: str | Path,
    dataset_root: str | Path,
    out_dir: str | Path,
    n: int = 4,
    split: str = "val",
    resize: tuple[int, int] | None = None,
    colormap: str = "plasma",
    max_depth: float = 10.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthEstimator(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    ds = RGBDDataset(dataset_root, split=split, resize=resize)
    indices = torch.linspace(0, len(ds) - 1, n).long().tolist()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        rgb, depth_gt, _ = ds[idx]

        with torch.no_grad():
            depth_pred = model(rgb.unsqueeze(0).to(device)).squeeze().cpu()

        depth_gt_np = depth_gt.squeeze().numpy()
        depth_pred_np = depth_pred.numpy()

        # Far pixels (sky, background) are irrelevant — mask them to NaN so they
        # render as grey and don't compress the colormap for the near range.
        far_mask = depth_gt_np > max_depth
        depth_gt_vis = depth_gt_np.copy().astype(np.float32)
        depth_pred_vis = depth_pred_np.copy().astype(np.float32)
        depth_gt_vis[far_mask] = np.nan
        depth_pred_vis[far_mask] = np.nan  # mask same region in prediction

        near = depth_gt_np[~far_mask]
        vmin = float(near.min()) if near.size else 0.0
        vmax = float(near.max()) if near.size else max_depth

        cmap = plt.get_cmap(colormap).copy()
        cmap.set_bad(color="lightgrey")  # NaN pixels → grey

        fig = plt.figure(figsize=(15, 4.5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

        ax_rgb   = fig.add_subplot(gs[0])
        ax_gt    = fig.add_subplot(gs[1])
        ax_pred  = fig.add_subplot(gs[2])
        ax_cbar  = fig.add_subplot(gs[3])

        ax_rgb.imshow(_to_rgb(rgb))
        ax_rgb.set_title("RGB", fontsize=12)
        ax_rgb.axis("off")

        im = ax_gt.imshow(depth_gt_vis, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_gt.set_title("Ground truth depth", fontsize=12)
        ax_gt.axis("off")

        ax_pred.imshow(depth_pred_vis, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_pred.set_title("Predicted depth", fontsize=12)
        ax_pred.axis("off")

        fig.colorbar(im, cax=ax_cbar, label="Depth (m)")

        out_path = out_dir / f"depth_comparison_{i:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def main_cli() -> None:
    p = argparse.ArgumentParser(description="Compare predicted vs ground-truth depth maps")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to depth estimator checkpoint")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="quickplots/depth")
    p.add_argument("--n", type=int, default=4, help="Number of samples to visualize")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--colormap", type=str, default="plasma")
    p.add_argument("--max-depth", type=float, default=10.0, help="Pixels beyond this depth are shown in grey")
    args = p.parse_args()

    visualize(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        n=args.n,
        split=args.split,
        resize=tuple(args.resize) if args.resize else None,
        colormap=args.colormap,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main_cli()
