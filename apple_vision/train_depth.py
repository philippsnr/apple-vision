from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.rgbd import RGBDDataset, collate_fn
from .models.depth_estimator import DepthEstimator, si_log_loss
from .plot_metrics import plot_loss_curve, save_metrics_csv


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
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Resize images before training, e.g. --resize 640 400")
    p.add_argument("--max-depth", type=float, default=10.0,
                   help="Ignore pixels beyond this depth (metres) in the loss. 0 = disabled.")
    return p.parse_args()


def train_one_epoch(model, optimizer, loader, device, epoch, num_epochs, max_depth):
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False, unit="batch")
    for rgb, depth, _ in bar:
        rgb = rgb.to(device)
        depth = depth.to(device)

        pred = model(rgb)
        loss = si_log_loss(pred, depth, max_depth=max_depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    avg = total_loss / max(1, len(loader))
    return avg


@torch.no_grad()
def evaluate(model, loader, device, epoch, num_epochs, max_depth):
    model.eval()
    total_loss = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [val]  ", leave=False, unit="batch")
    for rgb, depth, _ in bar:
        rgb = rgb.to(device)
        depth = depth.to(device)
        pred = model(rgb)
        loss = si_log_loss(pred, depth, max_depth=max_depth)
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    avg = total_loss / max(1, len(loader))
    return avg


def main_cli():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path(args.dataset_root)
    resize = tuple(args.resize) if args.resize else None
    train_ds = RGBDDataset(root, split="train", val_fraction=args.val_fraction, resize=resize)
    val_ds = RGBDDataset(root, split="val", val_fraction=args.val_fraction, resize=resize)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples" +
          (f"  (resized to {resize[0]}×{resize[1]})" if resize else ""))

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
    history = []

    for epoch in range(1, args.epochs + 1):
        max_depth = args.max_depth if args.max_depth > 0 else None
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.epochs, max_depth)
        val_loss = evaluate(model, val_loader, device, epoch, args.epochs, max_depth)
        scheduler.step()

        marker = " *" if val_loss < best_val else ""
        print(f"Epoch {epoch:>3}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")

    csv_path = out_dir / "depth_estimator_metrics.csv"
    save_metrics_csv(csv_path, history)
    plot_loss_curve(csv_path, out_dir / "depth_estimator_loss.png", title="Depth Estimator — Training Loss")


if __name__ == "__main__":
    main_cli()
