#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple Dataset Benchmark from Orchard Environment (Dataset Ninja / Supervisely) -> COCO.

Expected layout after extracting the Dataset Ninja archive:
  root/
    ds/
      img/   <subset>_<filename>.png
      ann/   <subset>_<filename>.png.json

Outputs:
  data/orchard/coco/
    images/{train,val}/
    annotations/{instances_train.json, instances_val.json}
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def bbox_from_rectangle(points):
    """Supervisely rectangle: exterior = [[x1,y1],[x2,y2]] -> COCO [x,y,w,h]."""
    if not points or len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    return [int(round(x1)), int(round(y1)), w, h]


def collect_pairs(root: Path):
    img_dir = root / "ds" / "img"
    ann_dir = root / "ds" / "ann"
    if not img_dir.exists():
        raise FileNotFoundError(f"img/ not found under: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"ann/ not found under: {ann_dir}")

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        ann_path = ann_dir / f"{img_path.name}.json"
        if ann_path.exists():
            pairs.append((img_path, ann_path))
    return pairs


def build_coco(pairs, start_img_id=0, start_ann_id=1):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "apple"}],
    }
    img_id = start_img_id
    ann_id = start_ann_id

    for img_path, ann_path in tqdm(pairs, desc="build COCO"):
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": int(w), "height": int(h)}
        )

        ann_data = json.load(open(ann_path))
        for obj in ann_data.get("objects", []):
            if obj.get("classTitle", "").lower() != "apple":
                continue
            points = obj.get("points", {}).get("exterior", [])
            bbox = bbox_from_rectangle(points)
            if bbox is None:
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": int(bbox[2] * bbox[3]),
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return coco


def stage_images(pairs, dest_dir: Path, symlink: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img_path, _ in tqdm(pairs, desc=f"stage -> {dest_dir.name}"):
        dst = dest_dir / img_path.name
        if dst.exists():
            continue
        if symlink:
            try:
                dst.symlink_to(img_path.resolve())
                continue
            except OSError:
                pass
        shutil.copy2(img_path, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/orchard/raw")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pairs = collect_pairs(root)
    print(f"Found {len(pairs)} images")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Split: train={len(train_pairs)} val={len(val_pairs)}")

    out_root = root.parent / "coco"
    out_ann = out_root / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)

    coco_train = build_coco(train_pairs, start_img_id=0, start_ann_id=1)
    coco_val = build_coco(val_pairs, start_img_id=100000, start_ann_id=1)

    stage_images(train_pairs, out_root / "images" / "train", args.symlink)
    stage_images(val_pairs, out_root / "images" / "val", args.symlink)

    with open(out_ann / "instances_train.json", "w") as f:
        json.dump(coco_train, f)
    with open(out_ann / "instances_val.json", "w") as f:
        json.dump(coco_val, f)

    print(f"\nDone. COCO root: {out_root}")
    print(f"  train: {len(coco_train['images'])} images, {len(coco_train['annotations'])} annotations")
    print(f"  val:   {len(coco_val['images'])} images, {len(coco_val['annotations'])} annotations")


if __name__ == "__main__":
    main()
