from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.rgbd import RGBDDataset, collate_fn
from .models.depth_estimator import DepthEstimator


@torch.no_grad()
def evaluate(
    checkpoint: str | Path,
    dataset_root: str | Path,
    split: str = "val",
    resize: tuple[int, int] | None = None,
    max_depth: float = 10.0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthEstimator(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    ds = RGBDDataset(dataset_root, split=split, resize=resize)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    abs_errors = []   # metres
    rel_errors = []   # fraction

    for rgb, depth_gt, _ in tqdm(loader, desc="Evaluating", unit="img"):
        rgb = rgb.to(device)
        depth_pred = model(rgb).squeeze().cpu().numpy()
        depth_gt_np = depth_gt.squeeze().numpy()

        mask = (depth_gt_np > 0) & (depth_gt_np <= max_depth)
        if mask.sum() == 0:
            continue

        pred = depth_pred[mask]
        gt   = depth_gt_np[mask]

        abs_errors.append(np.abs(pred - gt))
        rel_errors.append(np.abs(pred - gt) / gt)

    all_abs = np.concatenate(abs_errors)
    all_rel = np.concatenate(rel_errors)

    results = {
        "MAE  (cm)":    float(all_abs.mean() * 100),
        "MedAE (cm)":   float(np.median(all_abs) * 100),
        "RMSE (cm)":    float(np.sqrt((all_abs**2).mean()) * 100),
        "Rel error (%)": float(all_rel.mean() * 100),
        "pixels evaluated": int(mask.sum()),
    }
    return results


def main_cli() -> None:
    p = argparse.ArgumentParser(description="Evaluate depth estimator accuracy in cm")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--max-depth", type=float, default=10.0)
    args = p.parse_args()

    results = evaluate(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        split=args.split,
        resize=tuple(args.resize) if args.resize else None,
        max_depth=args.max_depth,
    )

    print("\n--- Depth estimation accuracy ---")
    for k, v in results.items():
        print(f"  {k:<20} {v:.2f}" if isinstance(v, float) else f"  {k:<20} {v}")


if __name__ == "__main__":
    main_cli()
