"""Fetch runs directly from the W&B API and plot a scientific line chart (mean +/- std).

Credentials are read from a local `.env` file (see `.env.example`):
    WANDB_API_KEY=...
    WANDB_PROJECT=apple-detector
    WANDB_ENTITY=            # optional, defaults to the API key's default entity

Filtering (per the project's sweep design, only one of aug_factor/synthetic_n
varies at a time while the other is held at its baseline):
    --x-column synthetic_n  -> keeps only runs with config.aug_factor == 1
    --x-column aug_factor   -> keeps only runs with config.synthetic_n == 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apple_vision.scientific_plot import plot_mean_std_line, plot_scatter

# --- Defaults (used when the corresponding CLI arg is omitted) ---------
X_COLUMN = "synthetic_n"
Y_COLUMN = "AP"

TITLE = "Average Precision vs. Number of Synthetic Images"
X_LABEL = "Synthetic Images (n)"
Y_LABEL = "AP"

OUTPUT_BASENAME = "plots/ap_over_synthetic_n"
OUTPUT_FORMATS = ("png",)  # add "pdf", "svg" if needed
STYLE = "line"  # "line" (mean +/- std) or "scatter" (raw per-run points)

PLOT_FN = {
    "line": plot_mean_std_line,
    "scatter": plot_scatter,
}

# Baseline value the *other* axis is pinned to for each x-column choice.
BASELINE_FOR_X_COLUMN = {
    "synthetic_n": ("aug_factor", 1),
    "aug_factor": ("synthetic_n", 0),
}
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch W&B runs and plot mean +/- std per x value")
    p.add_argument("--x-column", default=X_COLUMN, help="Run config key to use for the x-axis")
    p.add_argument("--y-column", default=Y_COLUMN, help="Run summary key to use for the y-axis")
    p.add_argument("--title", default=TITLE, help="Plot title")
    p.add_argument("--x-label", default=X_LABEL, help="X-axis label")
    p.add_argument("--y-label", default=Y_LABEL, help="Y-axis label")
    p.add_argument("--out", default=OUTPUT_BASENAME, help="Output path without extension")
    p.add_argument("--formats", nargs="+", default=list(OUTPUT_FORMATS), help="Output formats, e.g. png pdf svg")
    p.add_argument("--state", default="finished", help="Only include runs in this state (empty string = any)")
    p.add_argument("--style", choices=list(PLOT_FN), default=STYLE, help="Chart style")
    return p.parse_args()


def fetch_runs_df(project: str, entity: str | None) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    rows = []
    for run in api.runs(path):
        row = {"name": run.name, "state": run.state}
        row.update(run.config)
        row.update(run.summary._json_dict)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is not set. Copy .env.example to .env and fill in your key.")

    project = os.environ.get("WANDB_PROJECT", "apple-detector")
    entity = os.environ.get("WANDB_ENTITY") or None

    df = fetch_runs_df(project, entity)

    if args.state:
        df = df[df["state"] == args.state]

    if args.x_column in BASELINE_FOR_X_COLUMN:
        filter_col, filter_val = BASELINE_FOR_X_COLUMN[args.x_column]
        before = len(df)
        df = df[df[filter_col] == filter_val]
        print(f"Filtered {filter_col} == {filter_val}: {before} -> {len(df)} runs")

    PLOT_FN[args.style](
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
