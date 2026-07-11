"""Plot a scientific line chart from a local CSV file (mean +/- std for repeated X values)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apple_vision.scientific_plot import plot_mean_std_line

# --- Defaults (used when the corresponding CLI arg is omitted) ---------
CSV_PATH = "wandb_exports/wandb_export_aug1.csv"
X_COLUMN = "synthetic_n"
Y_COLUMN = "AP"

TITLE = "Average Precision vs. Number of Synthetic Images"
X_LABEL = "Synthetic Images (n)"
Y_LABEL = "AP"

OUTPUT_BASENAME = "plots/ap_over_synthetic_n"
OUTPUT_FORMATS = ("png",)  # add "pdf", "svg" if needed
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scientific line plot from a CSV (mean +/- std per x value)")
    p.add_argument("--csv", default=CSV_PATH, help="Path to input CSV")
    p.add_argument("--x-column", default=X_COLUMN, help="Column to use for the x-axis")
    p.add_argument("--y-column", default=Y_COLUMN, help="Column to use for the y-axis")
    p.add_argument("--title", default=TITLE, help="Plot title")
    p.add_argument("--x-label", default=X_LABEL, help="X-axis label")
    p.add_argument("--y-label", default=Y_LABEL, help="Y-axis label")
    p.add_argument("--out", default=OUTPUT_BASENAME, help="Output path without extension")
    p.add_argument("--formats", nargs="+", default=list(OUTPUT_FORMATS), help="Output formats, e.g. png pdf svg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    plot_mean_std_line(
        df,
        x_column=args.x_column,
        y_column=args.y_column,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        out_basename=args.out,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
