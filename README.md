# Apple Vision

Apple detection training pipeline using Faster R-CNN (torchvision) on COCO-style annotations. Supports three datasets: MinneApple, Apple Dataset Benchmark from Orchard Environment, and Apple MOTS.

## Paper

**AI-Based Detection of Apples for Autonomous Harvesting Systems**  
Philipp Staudinger, Nils Fleschhut — DHBW Ravensburg, 2026

[![Paper cover](paper/cover.png)](paper/studienarbeit.pdf)

## Requirements

- Python >= 3.10
- PyTorch >= 2.2, torchvision >= 0.17, pycocotools, pillow, numpy, tqdm
- Optional: CUDA-capable GPU (CPU training works but is much slower)

## Installation

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Datasets

### MinneApple

Raw data layout after download:
```
data/minneapple/detection/
  train/images/   train/masks/
  test/images/    test/masks/
```

Convert to COCO format:
```bash
uv run python scripts/prepare_minneapple.py --root data/minneapple/detection --val-ratio 0.15 --seed 42
```

Output: `data/minneapple/coco/`

---

### Apple Dataset Benchmark from Orchard Environment

Download from [Dataset Ninja](https://datasetninja.com/apple-dataset-benchmark-from-orchard-environment). After extracting, the raw folder should contain subfolders `ArtificialLight/`, `CropLoadEstimation/`, `HarvestingRobot2016/`, `HarvestingRobot2017/`.

Convert to COCO format:
```bash
uv run python scripts/prepare_orchard_benchmark.py --root data/orchard/raw --val-ratio 0.15 --seed 42
```

Options: `--test-ratio <float>` to create a test split, `--symlink` to symlink instead of copying images.

Output: `data/orchard/coco/`

---

### Apple MOTS

Download from [Dataset Ninja](https://datasetninja.com/apple-mots). After extracting, the raw folder should contain:
```
data/apple_mots/raw/
  train/images/<sequence>/*.png
  train/instances/<sequence>/*.png
  testing/images/<sequence>/*.png
  testing/instances/<sequence>/*.png
```

Convert instance masks to COCO bounding boxes:
```bash
# Use official testing set as validation
uv run python scripts/prepare_apple_mots.py \
  --root data/apple_mots/raw \
  --train-splits train \
  --val-splits testing \
  --seed 42

# Or: random val split from train, testing as a dedicated test set
uv run python scripts/prepare_apple_mots.py \
  --root data/apple_mots/raw \
  --train-splits train \
  --val-splits "" \
  --val-ratio 0.1 \
  --test-splits testing
```

Output: `data/apple_mots/coco/`

---

### Merging Datasets

Combine multiple COCO-format datasets into one. Image and annotation IDs are remapped to be globally unique; images are symlinked (or copied) into the output directory, prefixed with the source dataset name to avoid filename collisions.

```bash
uv run python scripts/merge_coco_datasets.py \
  data/minneapple/coco data/orchard/coco \
  --output data/merged/coco
```

Options: `--splits train val test` to control which splits are merged (default: all three), `--copy` to copy images instead of symlinking.

Output: `data/merged/coco/`

---

### Dataset Cleaning

If a dataset contains corrupted images, run the cleaning script to move them out and remove their entries from the annotation JSON:
```bash
uv run python scripts/clean_coco_dataset.py data/orchard/coco
```

The script scans `images/train`, `images/val`, and `images/test`, moves corrupted files to a `broken/` subfolder, and updates the corresponding annotation JSONs in-place (a `.bak` backup is created first).

## Training

```bash
uv run python -m apple_vision.train --dataset-root data/minneapple/coco
```

Key options (defaults shown):

| Flag | Default | Description |
|---|---|---|
| `--dataset-root` | *(required)* | Path to COCO-format dataset root |
| `--train-ann` | `annotations/instances_train.json` | Training annotation file |
| `--train-images` | `images/train` | Training images directory |
| `--val-ann` | `annotations/instances_val.json` | Validation annotation file |
| `--val-images` | `images/val` | Validation images directory |
| `--epochs` | `1` | Number of training epochs |
| `--batch-size` | `2` | Batch size |
| `--lr` | `0.005` | SGD learning rate |
| `--num-workers` | `2` | DataLoader workers |
| `--out-dir` | `checkpoints` | Directory for saved checkpoints |
| `--resume` | — | Path to a checkpoint to resume from |
| `--early-stop-patience` | `0` | Early stopping patience (0 = disabled) |

### Checkpoints

- `checkpoints/fasterrcnn_resnet50_fpn_apple_best.pth` — best val-loss checkpoint
- `checkpoints/fasterrcnn_resnet50_fpn_apple.pth` — final checkpoint after all epochs

## Evaluation (COCO mAP)

Evaluate a saved checkpoint against the validation set and report AP@[.5:.95], AP50, AP75, etc.:

```bash
uv run python -m apple_vision.evaluate_coco \
  --dataset-root data/minneapple/coco \
  --checkpoint checkpoints/fasterrcnn_resnet50_fpn_apple_best.pth
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--val-ann` | `annotations/instances_val.json` | Validation annotation file |
| `--val-images` | `images/val` | Validation images directory |
| `--results-json` | `coco_results.json` | Where to save raw detections |
| `--score-threshold` | `0.0` | Minimum score to keep a detection |
| `--device` | auto | `cuda` or `cpu` |

## Visualization

Quickly verify annotations by drawing ground-truth bounding boxes on a sample of images:

```bash
uv run python -m apple_vision.quickplot \
  --dataset-root data/minneapple/coco \
  --ann annotations/instances_val.json \
  --images images/val \
  --n 8 \
  --out-dir quickplots
```

Add `--show` to open the images after saving.

## Project Structure

```
apple_vision/
  train.py          # training loop
  evaluate_coco.py  # COCO mAP evaluation
  quickplot.py      # annotation visualizer
  models/
    detector.py     # Faster R-CNN factory
  data/
    minneapple.py   # COCO dataset class
scripts/
  prepare_minneapple.py          # MinneApple → COCO
  prepare_orchard_benchmark.py   # Orchard → COCO
  prepare_apple_mots.py          # Apple MOTS masks → COCO
  merge_coco_datasets.py         # merge multiple COCO datasets into one
  clean_coco_dataset.py          # remove corrupted images & fix annotations
```
