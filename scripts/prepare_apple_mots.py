#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple MOTS -> COCO (bounding boxes).

Expected layout after extracting APPLE_MOTS.zip (or the DatasetNinja tarball):
root/
  train/
    images/<scene>/*.png
    instances/<scene>/*.png
  testing/
    images/<scene>/*.png
    instances/<scene>/*.png

Each split contains per-scene folders; masks have the same relative path as the
images. This script converts the pixel-level instance masks into COCO-style
bounding boxes so they can be used with the existing detection pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    img_path: Path
    mask_path: Path | None
    rel_name: str
    split: str


def _parse_split_arg(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _collect_split_samples(root: Path, split_name: str) -> List[Sample]:
    split_dir = root / split_name
    images_dir = split_dir / "images"
    instances_dir = split_dir / "instances"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found for split '{split_name}' -> {images_dir}")

    has_masks = instances_dir.exists()
    if not has_masks:
        print(f"[warn] No masks found for split '{split_name}' at {instances_dir}. Images will be treated as unlabeled.")

    samples: List[Sample] = []
    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        rel = img_path.relative_to(images_dir)
        rel_flat = rel.as_posix().replace("/", "_")
        rel_name = f"{split_name}_{rel_flat}"
        mask_path = instances_dir / rel if has_masks else None
        if mask_path and not mask_path.exists():
            print(f"[warn] Missing mask for {img_path}, expected {mask_path}")
            mask_path = None
        samples.append(Sample(img_path=img_path, mask_path=mask_path, rel_name=rel_name, split=split_name))
    if not samples:
        raise RuntimeError(f"No images found under {images_dir}")
    return samples


def _boxes_from_mask(mask_path: Path | None):
    if not mask_path:
        return []
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        # If RGB (e.g., ignore overlays), use first channel
        mask = mask[:, :, 0]
    unique_vals = np.unique(mask)
    boxes = []
    for inst_id in unique_vals:
        if inst_id == 0:
            continue
        ys, xs = np.where(mask == inst_id)
        if ys.size == 0 or xs.size == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        w = int(max(1, x2 - x1))
        h = int(max(1, y2 - y1))
        if w == 0 or h == 0:
            continue
        boxes.append([int(x1), int(y1), w, h])
    return boxes


def _build_coco(samples: Sequence[Sample], start_img_id=0, start_ann_id=1):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "apple"}],
    }
    img_id = start_img_id
    ann_id = start_ann_id
    for sample in tqdm(samples, desc="build coco"):
        with Image.open(sample.img_path) as img:
            width, height = img.size
        coco["images"].append(
            {"id": img_id, "file_name": sample.rel_name, "width": int(width), "height": int(height)}
        )

        boxes = _boxes_from_mask(sample.mask_path)
        for bbox in boxes:
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
        dst = dest_dir / sample.rel_name
        if dst.exists():
            continue
        if symlink:
            try:
                dst.symlink_to(sample.img_path.resolve())
                continue
            except OSError:
                print(f"[warn] Symlink failed for {sample.img_path}, fallback to copy.")
        shutil.copy2(sample.img_path, dst)


def _split_for_validation(samples: List[Sample], val_ratio: float, seed: int):
    if not samples:
        return samples, []
    if not (0 < val_ratio < 1):
        raise ValueError("val-ratio must be between 0 and 1 when no validation split is provided.")
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(indices[:n_val])
    val_split = [samples[i] for i in val_idx]
    train_split = [samples[i] for i in indices[n_val:]]
    if not train_split:
        raise RuntimeError("Validation ratio removed all training samples; decrease --val-ratio.")
    return train_split, val_split


def _gather_samples(root: Path, split_names: List[str]) -> List[Sample]:
    collected: List[Sample] = []
    for split in split_names:
        collected.extend(_collect_split_samples(root, split))
    return collected


def main():
    ap = argparse.ArgumentParser(description="Convert Apple MOTS dataset to COCO bounding boxes.")
    ap.add_argument("--root", type=str, default="data/apple_mots/raw", help="Directory that contains 'train/', 'testing/' etc.")
    ap.add_argument("--out-root", type=str, default=None, help="Where to place the COCO dataset (defaults to <root>/../coco).")
    ap.add_argument("--train-splits", type=str, default="train", help="Comma-separated folder names used for training.")
    ap.add_argument("--val-splits", type=str, default="testing", help="Comma-separated folder names used for validation. Leave empty to sample from train via --val-ratio.")
    ap.add_argument("--test-splits", type=str, default="", help="Comma-separated folder names used for test export.")
    ap.add_argument("--val-ratio", type=float, default=0.0, help="Hold-out ratio from training splits if --val-splits is empty.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/val split when --val-splits is empty.")
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    train_names = _parse_split_arg(args.train_splits) or ["train"]
    val_names = _parse_split_arg(args.val_splits)
    test_names = _parse_split_arg(args.test_splits)

    train_samples = _gather_samples(root, train_names)
    val_samples = _gather_samples(root, val_names) if val_names else []
    if not val_samples:
        if args.val_ratio <= 0:
            raise RuntimeError(
                "No validation split discovered. Provide --val-splits or a positive --val-ratio to split part of the train data."
            )
        train_samples, extra_val = _split_for_validation(train_samples, args.val_ratio, args.seed)
        val_samples = extra_val

    test_samples = _gather_samples(root, test_names) if test_names else []

    print(f"Collected samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else root.parent / "coco"
    img_root = out_root / "images"
    ann_root = out_root / "annotations"
    ann_root.mkdir(parents=True, exist_ok=True)

    coco_train = _build_coco(train_samples, start_img_id=0, start_ann_id=1)
    coco_val = _build_coco(val_samples, start_img_id=100000, start_ann_id=1)

    _stage_images(train_samples, img_root / "train", args.symlink)
    _stage_images(val_samples, img_root / "val", args.symlink)

    with open(ann_root / "instances_train.json", "w", encoding="utf-8") as f:
        json.dump(coco_train, f)
    with open(ann_root / "instances_val.json", "w", encoding="utf-8") as f:
        json.dump(coco_val, f)

    if test_samples:
        coco_test = _build_coco(test_samples, start_img_id=200000, start_ann_id=1)
        _stage_images(test_samples, img_root / "test", args.symlink)
        with open(ann_root / "instances_test.json", "w", encoding="utf-8") as f:
            json.dump(coco_test, f)

    print("\nFertig ✅")
    print(f"COCO root: {out_root}")
    print(f"- {img_root / 'train'}")
    print(f"- {img_root / 'val'}")
    if test_samples:
        print(f"- {img_root / 'test'}")
    print(f"- {ann_root / 'instances_train.json'}")
    print(f"- {ann_root / 'instances_val.json'}")
    if test_samples:
        print(f"- {ann_root / 'instances_test.json'}")


if __name__ == "__main__":
    main()

