from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(
    csv_path: Path,
    out_path: Path,
    title: str = "Training Loss",
) -> None:
    epochs, train_losses, val_losses = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    best_epoch = epochs[val_losses.index(min(val_losses))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train", linewidth=2)
    ax.plot(epochs, val_losses, label="Val", linewidth=2)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve to {out_path}")
