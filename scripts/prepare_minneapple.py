#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinneApple (detection) -> COCO (Bounding Boxes) + optional Val-Split aus train/.
Erzeugt Struktur:
data/minneapple/coco/
  images/{train,val}
  annotations/{instances_train.json, instances_val.json[, instances_test.json]}
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # Ensure strictly positive width/height (avoid zero-area boxes)
    w = int(max(1, x2 - x1))
    h = int(max(1, y2 - y1))
    return [int(x1), int(y1), w, h]


def collect_pairs(split_dir: Path):
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    pairs = []
    for ip in images:
        mp = mask_dir / ip.name
        if mp.exists():
            pairs.append((ip, mp))
    return pairs


def build_coco_for_pairs(pairs, start_img_id=0, start_ann_id=1):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "apple"}],
    }
    img_id = start_img_id
    ann_id = start_ann_id

    for img_path, mask_path in tqdm(pairs, desc="COCO build"):
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": int(w), "height": int(h)}
        )

        mask = np.array(Image.open(mask_path))
        # Maske ist instanzkodiert (0=Background, 1..N=Instanz)
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            m = (mask == inst_id).astype(np.uint8)
            bbox = bbox_from_mask(m)
            if bbox:
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": int(bbox[2] * bbox[3]),
                        "iscrowd": 0,
                    }
                )
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
            except Exception:
                # Fallback: copy, falls Symlink nicht erlaubt ist
                shutil.copy2(img_path, dst)
        else:
            shutil.copy2(img_path, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/minneapple/detection",
        help="Pfad zu 'detection' (enthält train/, test/).",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Anteil für Val aus train/, wenn kein val/ vorhanden ist.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true", help="Bilder lieber symlinken statt kopieren")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    train_dir = root / "train"
    val_dir = root / "val"  # evtl. nicht vorhanden
    test_dir = root / "test"

    out_root = root.parent / "coco"
    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"
    out_ann = out_root / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)

    # --- Train/Val bestimmen ---
    if val_dir.exists():
        train_pairs = collect_pairs(train_dir)
        val_pairs = collect_pairs(val_dir)
    else:
        # Val aus Train splitten
        all_pairs = collect_pairs(train_dir)
        random.Random(args.seed).shuffle(all_pairs)
        n_val = max(1, int(len(all_pairs) * args.val_ratio))
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]
        print(f"Kein 'val/' gefunden → Split aus 'train/': train={len(train_pairs)} val={len(val_pairs)}")

    # --- COCO JSONs bauen ---
    coco_train = build_coco_for_pairs(train_pairs, start_img_id=0, start_ann_id=1)
    coco_val = build_coco_for_pairs(val_pairs, start_img_id=100000, start_ann_id=1)

    # --- Bilder in Zielstruktur bereitstellen ---
    stage_images(train_pairs, out_img_train, symlink=args.symlink)
    stage_images(val_pairs, out_img_val, symlink=args.symlink)

    # --- JSON speichern ---
    with open(out_ann / "instances_train.json", "w") as f:
        json.dump(coco_train, f)
    with open(out_ann / "instances_val.json", "w") as f:
        json.dump(coco_val, f)

    # --- Optional: Test konvertieren (nur JSON, ohne Kopieren) ---
    if test_dir.exists():
        test_pairs = collect_pairs(test_dir)
        coco_test = build_coco_for_pairs(test_pairs, start_img_id=200000, start_ann_id=1)
        with open(out_ann / "instances_test.json", "w") as f:
            json.dump(coco_test, f)

    print("\nFertig ✅")
    print(f"COCO-Root: {out_root}")
    print(f"- {out_img_train}")
    print(f"- {out_img_val}")
    print(f"- {out_ann / 'instances_train.json'}")
    print(f"- {out_ann / 'instances_val.json'}")
    if (out_ann / "instances_test.json").exists():
        print(f"- {out_ann / 'instances_test.json'}")


if __name__ == "__main__":
    main()

'''
    Example usage:
        uv run python scripts/prepare_minneapple.py \
          --root data/minneapple/detection \
          --val-ratio 0.15 \
          --seed 42
'''
