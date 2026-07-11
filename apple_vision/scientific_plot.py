"""Shared plotting helper for publication-style mean +/- std line charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_mean_std_line(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    out_basename: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    df = df[[x_column, y_column]].dropna()

    grouped = df.groupby(x_column)[y_column].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values(x_column)
    grouped["std"] = grouped["std"].fillna(0.0)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.linewidth": 1.0,
            "figure.dpi": 100,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        grouped[x_column],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        capsize=3,
        color="black",
        ecolor="gray",
        elinewidth=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()

    out_dir = Path(out_basename).parent
    if str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        out_path = f"{out_basename}.{ext}"
        fig.savefig(out_path, dpi=300)
        print(f"Saved {out_path}")

    plt.close(fig)
