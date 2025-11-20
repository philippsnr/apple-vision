#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple Dataset Benchmark from Orchard Environment (bounding boxes) -> COCO format.

Expected source layout (after extracting the official archives or the DatasetNinja tarball):
root/
  ArtificialLight/
    *.png
    annotations.txt
  CropLoadEstimation/
  HarvestingRobot2016/
  HarvestingRobot2017/

The script concatenates all images, splits them into train/val[/test] splits and
creates COCO-style annotations compatible with apple_vision.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    img_path: Path
    rel_name: str
    boxes: Sequence[Sequence[float]]
    subset: str


def _read_annotation_file(path: Path) -> Dict[str, List[List[float]]]:
    """Parse one of the *.txt files shipped with the dataset."""
    mapping: Dict[str, List[List[float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = [token.strip() for token in raw.split(",") if token.strip() != ""]
            if not parts:
                continue
            img_name = Path(parts[0]).name
            coords = parts[1:]
            if not coords:
                # image w/o labels
                mapping.setdefault(img_name.lower(), [])
                continue
            if len(coords) % 4 != 0:
                # Some files contain a trailing, incomplete bbox (e.g. due to a dangling comma).
                # Be tolerant: drop the last 1-3 values and keep the rest.
                trimmed = len(coords) - (len(coords) % 4)
                if trimmed <= 0:
                    # Nothing usable on this line
                    print(f"[warn] No complete bbox found in {path} for image '{img_name}' -> skipping line: {line.strip()}")
                    mapping.setdefault(img_name.lower(), [])
                    continue
                print(
                    f"[warn] Incomplete bbox in {path} for image '{img_name}': expected groups of 4, got {len(coords)}; "
                    f"dropping last {len(coords) - trimmed} value(s)."
                )
                coords = coords[:trimmed]
            boxes: List[List[float]] = []
            for i in range(0, len(coords), 4):
                try:
                    x = float(coords[i])
                    y = float(coords[i + 1])
                    w = float(coords[i + 2])
                    h = float(coords[i + 3])
                except ValueError as exc:
                    raise ValueError(f"Failed to parse numbers in {path}: '{coords[i:i+4]}'") from exc
                boxes.append([x, y, w, h])
            mapping.setdefault(img_name.lower(), []).extend(boxes)
    return mapping


def _collect_samples(root: Path) -> List[Sample]:
    subsets = [p for p in root.iterdir() if p.is_dir()]
    if not subsets:
        # Allow datasets where images (and optional annotations.txt) are directly under root
        subsets = [root]

    samples: List[Sample] = []
    for subset_dir in sorted(subsets):
        ann_files = sorted(p for p in subset_dir.iterdir() if p.suffix.lower() == ".txt")
        mapping: Dict[str, List[List[float]]] = {}
        for ann_file in ann_files:
            parsed = _read_annotation_file(ann_file)
            for name_key, boxes in parsed.items():
                mapping.setdefault(name_key, []).extend(boxes)
        if not ann_files:
            print(f"[warn] No annotation file found in {subset_dir}, images will be treated as unlabeled.")
        images = sorted(
            p for p in subset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not images:
            print(f"[warn] Found no images inside {subset_dir}, skipping.")
            continue
        for img in images:
            rel_name = f"{subset_dir.name}_{img.name}"
            boxes = mapping.get(img.name.lower(), [])
            samples.append(Sample(img, rel_name, boxes, subset_dir.name))
    if not samples:
        raise RuntimeError(f"No images discovered under {root}.")
    return samples


def _split_samples(
    samples: Sequence[Sample], val_ratio: float, test_ratio: float, seed: int
) -> tuple[List[Sample], List[Sample], List[Sample]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Require 0 <= ratios and val_ratio + test_ratio < 1.")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    total = len(shuffled)

    def _ratio_to_count(n_total: int, ratio: float) -> int:
        if ratio <= 0 or n_total <= 1:
            return 0
        count = int(round(n_total * ratio))
        if count == 0 and ratio > 0:
            count = 1
        if count >= n_total:
            count = n_total - 1
        return count

    n_val = _ratio_to_count(total, val_ratio)
    n_test = _ratio_to_count(total, test_ratio)
    max_test = max(0, total - n_val - 1)
    if n_test > max_test:
        n_test = max_test

    val_split = shuffled[:n_val]
    test_split = shuffled[n_val : n_val + n_test]
    train_split = shuffled[n_val + n_test :]

    if len(train_split) == 0:
        raise RuntimeError("Split left no training samples. Reduce val/test ratios.")
    return train_split, val_split, test_split


def _sanitize_bbox(box, width: int, height: int):
    x, y, w, h = box
    x = max(0.0, float(x))
    y = max(0.0, float(y))
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    x2 = min(x + w, float(width))
    y2 = min(y + h, float(height))
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w == 0 or h == 0:
        return None
    return [x1, y1, w, h]


def _build_coco(split: Sequence[Sample], start_img_id: int = 0, start_ann_id: int = 1):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "apple"}],
    }
    img_id = start_img_id
    ann_id = start_ann_id

    for sample in tqdm(split, desc="build coco"):
        with Image.open(sample.img_path) as im:
            width, height = im.size

        coco["images"].append(
            {"id": img_id, "file_name": sample.rel_name, "width": int(width), "height": int(height)}
        )

        for box in sample.boxes:
            bbox = _sanitize_bbox(box, width, height)
            if not bbox:
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        img_id += 1
    return coco


def _stage_images(samples: Iterable[Sample], dest_dir: Path, symlink: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for sample in tqdm(samples, desc=f"stage -> {dest_dir.name}"):
        target = dest_dir / sample.rel_name
        if target.exists():
            continue
        if symlink:
            try:
                target.symlink_to(sample.img_path.resolve())
                continue
            except OSError:
                print(f"[warn] Symlink failed for {sample.img_path}, falling back to copy.")
        shutil.copy2(sample.img_path, target)


def main():
    ap = argparse.ArgumentParser(description="Convert Apple Dataset Benchmark from Orchard Environment to COCO format.")
    ap.add_argument("--root", type=str, default="data/orchard/raw", help="Dataset root. May contain scenario folders (ArtificialLight, CropLoadEstimation, ...), but all are optional. Images can also be directly under this root.")
    ap.add_argument("--out-root", type=str, default=None, help="Destination root for the COCO dataset (defaults to <root>/../coco).")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (0-1).")
    ap.add_argument("--test-ratio", type=float, default=0.0, help="Optional test split ratio (0-1).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before splitting.")
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source root not found: {root}")

    samples = _collect_samples(root)
    print(f"Discovered {len(samples)} images under {root}.")

    train_split, val_split, test_split = _split_samples(samples, args.val_ratio, args.test_ratio, args.seed)
    print(f"Split sizes: train={len(train_split)} val={len(val_split)} test={len(test_split)}")

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else root.parent / "coco"
    img_dir = out_root / "images"
    ann_dir = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    coco_train = _build_coco(train_split, start_img_id=0, start_ann_id=1)
    coco_val = _build_coco(val_split, start_img_id=100000, start_ann_id=1)
    _stage_images(train_split, img_dir / "train", args.symlink)
    _stage_images(val_split, img_dir / "val", args.symlink)
    with open(ann_dir / "instances_train.json", "w", encoding="utf-8") as f:
        json.dump(coco_train, f)
    with open(ann_dir / "instances_val.json", "w", encoding="utf-8") as f:
        json.dump(coco_val, f)

    if test_split:
        coco_test = _build_coco(test_split, start_img_id=200000, start_ann_id=1)
        _stage_images(test_split, img_dir / "test", args.symlink)
        with open(ann_dir / "instances_test.json", "w", encoding="utf-8") as f:
            json.dump(coco_test, f)

    print("\nDone ✅")
    print(f"COCO root: {out_root}")
    print(f"- {img_dir / 'train'}")
    print(f"- {img_dir / 'val'}")
    if test_split:
        print(f"- {img_dir / 'test'}")
    print(f"- {ann_dir / 'instances_train.json'}")
    print(f"- {ann_dir / 'instances_val.json'}")
    if test_split:
        print(f"- {ann_dir / 'instances_test.json'}")


if __name__ == "__main__":
    main()
