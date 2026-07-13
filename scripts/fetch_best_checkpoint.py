"""Pull the best-AP detector checkpoint from W&B and place it in checkpoints/.

Iterates all runs in the W&B project, picks the one with the highest logged
`AP` in its summary, downloads its `detector-{run.name}` model artifact
(logged by the Kaggle training notebooks), and copies the contained
checkpoint to checkpoints/fasterrcnn_resnet50_fpn_apple_best.pth so that
apple_vision/api.py picks it up as the default model.

Credentials are read from a local `.env` file (see `.env.example`):
    WANDB_API_KEY=...
    WANDB_PROJECT=apple-detector
    WANDB_ENTITY=            # optional, defaults to the API key's default entity
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
TARGET_NAME = "fasterrcnn_resnet50_fpn_apple_best.pth"


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is not set. Copy .env.example to .env and fill in your key.")

    project = os.environ.get("WANDB_PROJECT", "apple-detector")
    entity = os.environ.get("WANDB_ENTITY") or None

    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    best_run = None
    best_ap = float("-inf")
    for run in api.runs(path):
        ap = run.summary.get("AP")
        if ap is None:
            continue
        if ap > best_ap:
            best_ap = ap
            best_run = run

    if best_run is None:
        raise SystemExit(f"No runs with a logged 'AP' summary found in project '{path}'.")

    print(f"Best run: {best_run.name} (id={best_run.id}, AP={best_ap:.4f})")

    artifact_name = f"detector-{best_run.name}:latest"
    try:
        artifact = api.artifact(f"{path}/{artifact_name}")
    except wandb.errors.CommError as e:
        raise SystemExit(
            f"Could not find artifact '{artifact_name}' in project '{path}'. "
            f"Was the checkpoint-logging cell run for this run? ({e})"
        )

    download_dir = artifact.download()
    pth_files = list(Path(download_dir).glob("*.pth"))
    if not pth_files:
        raise SystemExit(f"Artifact '{artifact_name}' does not contain a .pth file.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = CHECKPOINT_DIR / TARGET_NAME
    shutil.copy(pth_files[0], target)
    print(f"Saved {pth_files[0]} -> {target}")


if __name__ == "__main__":
    main()
