from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from .data.minneapple import CocoAppleDataset, collate_fn
from .models.detector import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple apple detector on MinneApple (COCO-style)")
    p.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    p.add_argument("--train-ann", type=str, default="annotations/instances_train.json")
    p.add_argument("--train-images", type=str, default="images/train")
    p.add_argument("--val-ann", type=str, default="annotations/instances_val.json")
    p.add_argument("--val-images", type=str, default="images/val")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to a model checkpoint (.pth) to resume from")
    p.add_argument("--early-stop-patience", type=int, default=0, help="Early stopping patience in epochs (0 disables)")
    return p.parse_args()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    bar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False, unit="batch")
    for images, targets in bar:
        images = [T.ToTensor()(img).to(device) if not isinstance(img, torch.Tensor) else img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        bar.set_postfix(loss=f"{losses.item():.4f}")

    return total_loss / max(1, len(data_loader))


def evaluate(model, data_loader, device, epoch, num_epochs):
    """Compute average validation loss without updating model parameters.

    Note: torchvision detection models return losses only when model.training is True.
    To avoid affecting running BatchNorm statistics, we set the whole model to train()
    but put all BatchNorm layers into eval() and run under torch.no_grad().
    """
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    total_loss = 0.0
    bar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs} [val]  ", leave=False, unit="batch")
    with torch.no_grad():
        for images, targets in bar:
            images = [T.ToTensor()(img).to(device) if not isinstance(img, torch.Tensor) else img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            bar.set_postfix(loss=f"{losses.item():.4f}")

    if not was_training:
        model.eval()
    return total_loss / max(1, len(data_loader))


def main_cli():
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    root = Path(args.dataset_root)

    train_ds = CocoAppleDataset(root, args.train_ann, args.train_images)
    val_ds = CocoAppleDataset(root, args.val_ann, args.val_images)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    num_classes = 2  # background + apple
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    # Optionally resume from checkpoint (model weights only)
    if args.resume:
        resume_arg = str(args.resume)
        resume_path = Path(resume_arg.replace("\\", "/"))
        if not resume_path.exists():
            default_ckpt = Path("checkpoints/fasterrcnn_resnet50_fpn_apple.pth")
            if default_ckpt.exists():
                print(f"[train] Warning: resume checkpoint '{resume_arg}' not found. Using default '{default_ckpt.as_posix()}'")
                resume_path = default_ckpt
            else:
                import os
                hint = ""
                if os.name != "nt" and ("\\" in resume_arg):
                    hint = " Hint: On Linux/WSL shells, use forward slashes (e.g., checkpoints/fasterrcnn_resnet50_fpn_apple.pth) or quote the argument to keep backslashes."
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path.as_posix()} (provided: '{resume_arg}').{hint}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Resumed model weights from {resume_path.as_posix()}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "fasterrcnn_resnet50_fpn_apple_best.pth"

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.epochs)
        val_loss = evaluate(model, val_loader, device, epoch, args.epochs)

        marker = " *" if val_loss < best_val else ""
        print(f"Epoch {epoch:>3}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {bad_epochs} epochs).")
            break

    final_ckpt_path = out_dir / "fasterrcnn_resnet50_fpn_apple.pth"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved final checkpoint to {final_ckpt_path}")


if __name__ == "__main__":
    main_cli()
