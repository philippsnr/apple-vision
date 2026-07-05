from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms as T

from .data.minneapple import CocoAppleDataset
from .models.detector import create_model


def _draw_boxes(ax, boxes, color, label):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none",
        ))
        if i == 0:
            ax.plot([], [], color=color, linewidth=1.5, label=label)


def visualize(
    checkpoint: str | Path,
    dataset_root: str | Path,
    out_dir: str | Path,
    n: int = 4,
    split: str = "val",
    score_threshold: float = 0.5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    ds = CocoAppleDataset(
        dataset_root,
        f"annotations/instances_{split}.json",
        f"images/{split}",
    )

    indices = torch.linspace(0, len(ds) - 1, n).long().tolist()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = T.ToTensor()

    for i, idx in enumerate(indices):
        img_pil, target = ds[idx]
        img_np = np.array(img_pil)

        with torch.no_grad():
            output = model([to_tensor(img_pil).to(device)])[0]

        gt_boxes = target["boxes"].numpy()
        pred_boxes = output["boxes"].cpu().numpy()
        pred_scores = output["scores"].cpu().numpy()
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Sample {idx}", fontsize=11)

        ax_gt.imshow(img_np)
        _draw_boxes(ax_gt, gt_boxes, color="#00ff00", label=f"GT ({len(gt_boxes)})")
        ax_gt.set_title(f"Ground truth — {len(gt_boxes)} boxes", fontsize=10)
        ax_gt.axis("off")
        if len(gt_boxes):
            ax_gt.legend(loc="upper right", fontsize=8)

        ax_pred.imshow(img_np)
        _draw_boxes(ax_pred, pred_boxes, color="#ff4444", label=f"Pred ({len(pred_boxes)})")
        ax_pred.set_title(
            f"Predictions (score ≥ {score_threshold}) — {len(pred_boxes)} boxes", fontsize=10
        )
        ax_pred.axis("off")
        if len(pred_boxes):
            ax_pred.legend(loc="upper right", fontsize=8)

        out_path = out_dir / f"detection_comparison_{i:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def main_cli() -> None:
    p = argparse.ArgumentParser(description="Compare predicted vs ground-truth apple detections")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="quickplots/detections")
    p.add_argument("--n", type=int, default=4, help="Number of samples to visualize")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--score-threshold", type=float, default=0.5)
    args = p.parse_args()

    visualize(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        n=args.n,
        split=args.split,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main_cli()
