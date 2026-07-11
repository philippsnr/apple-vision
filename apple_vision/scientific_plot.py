"""Shared plotting helpers for publication-style charts (mean +/- std line, raw scatter)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _style_axes(ax, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(True, alpha=0.3, linewidth=0.5)


def _save(fig, out_basename: str, formats: tuple[str, ...]) -> None:
    fig.tight_layout()

    out_dir = Path(out_basename).parent
    if str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        out_path = f"{out_basename}.{ext}"
        fig.savefig(out_path, dpi=300)
        print(f"Saved {out_path}")

    plt.close(fig)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.linewidth": 1.0,
        "figure.dpi": 100,
    }
)


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

    _style_axes(ax, title, x_label, y_label)
    _save(fig, out_basename, formats)


def plot_scatter(
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        df[x_column],
        df[y_column],
        s=35,
        color="black",
        alpha=0.6,
        edgecolor="none",
    )

    _style_axes(ax, title, x_label, y_label)
    _save(fig, out_basename, formats)
