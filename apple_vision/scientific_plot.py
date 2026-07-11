"""Shared plotting helpers for clean, modern charts (mean +/- std line, raw scatter)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"
ACCENT_SOFT = "#2a78d6"  # used at low alpha for area fill / scatter


def _style_axes(ax, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title, fontsize=13, fontweight=600, color=INK, pad=12)
    ax.set_xlabel(x_label, fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel(y_label, fontsize=10.5, color=INK_SECONDARY)

    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1.0)

    ax.tick_params(direction="out", colors=INK_SECONDARY, labelsize=9.5, length=4)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _save(fig, out_basename: str, formats: tuple[str, ...]) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()

    out_dir = Path(out_basename).parent
    if str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        out_path = f"{out_basename}.{ext}"
        fig.savefig(out_path, dpi=300, facecolor=SURFACE)
        print(f"Saved {out_path}")

    plt.close(fig)


plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
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

    ax.fill_between(
        grouped[x_column],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        color=ACCENT_SOFT,
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax.errorbar(
        grouped[x_column],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        markersize=8,
        markerfacecolor=ACCENT,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        linewidth=2.2,
        color=ACCENT,
        ecolor=BASELINE,
        elinewidth=1.2,
        capsize=0,
        zorder=3,
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
        s=70,
        color=ACCENT,
        alpha=0.55,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=3,
    )

    _style_axes(ax, title, x_label, y_label)
    _save(fig, out_basename, formats)
