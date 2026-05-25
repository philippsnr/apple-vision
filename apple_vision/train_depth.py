from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.rgbd import RGBDDataset, collate_fn
from .models.depth_estimator import DepthEstimator, si_log_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a metric depth estimator on paired RGB+depth images")
    p.add_argument("--dataset-root", type=str, required=True, help="Root of the RGB+depth dataset")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="Train backbone from scratch")
    p.add_argument("--early-stop-patience", type=int, default=0)
    return p.parse_args()


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0
    start = time.time()
    for rgb, depth, _ in loader:
        rgb = rgb.to(device)
        depth = depth.to(device)

        pred = model(rgb)
        loss = si_log_loss(pred, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / max(1, len(loader))
    print(f"Epoch {epoch}: train loss={avg:.4f} ({time.time() - start:.1f}s)")
    return avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for rgb, depth, _ in loader:
        rgb = rgb.to(device)
        depth = depth.to(device)
        pred = model(rgb)
        total_loss += si_log_loss(pred, depth).item()
    avg = total_loss / max(1, len(loader))
    print(f"           val   loss={avg:.4f}")
    return avg


def main_cli():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path(args.dataset_root)
    train_ds = RGBDDataset(root, split="train", val_fraction=args.val_fraction)
    val_ds = RGBDDataset(root, split="val", val_fraction=args.val_fraction)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = DepthEstimator(pretrained=not args.no_pretrained).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "depth_estimator_best.pth"
    final_ckpt = out_dir / "depth_estimator.pth"

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"           → saved best checkpoint (val_loss={best_val:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main_cli()
