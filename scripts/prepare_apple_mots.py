#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple MOTS (Supervisely) -> COCO (Bounding Boxes) + optional Val-Split aus train/.
Erzeugt Struktur:
data/apple_mots/coco/
  images/{train,val}
  annotations/{instances_train.json, instances_val.json[, instances_test.json]}

Erwartete Struktur unter --root (Standard: data/apple_mots/raw):

  root/
    train/
      img/   # RGB-Bilder
      ann/   # Supervisely-Annotationen (*.json)
    val/     # optional, sonst wird aus train/ gesplittet
      img/
      ann/
    test/    # optional
      img/
      ann/
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def bbox_from_polygon(points):
    """
    Supervisely-Polygon: points = [[x1, y1], [x2, y2], ...]
    -> COCO-BBox [x, y, w, h]
    """
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    return [int(round(x1)), int(round(y1)), w, h]


def bbox_from_bitmap(bitmap):
    """
    Supervisely-Bitmap:
      {
        "origin": [x, y],
        "size": {"width": w, "height": h},
        ...
      }
    -> COCO-BBox [x, y, w, h]
    """
    if not bitmap:
        return None
    origin = bitmap.get("origin")
    size = bitmap.get("size")
    if not origin or not size:
        return None
    x, y = origin
    w = max(1, int(size.get("width", 0)))
    h = max(1, int(size.get("height", 0)))
    return [int(round(x)), int(round(y)), w, h]


def collect_pairs_supervisely(split_dir: Path):
    """
    Sucht Bilder + passende Supervisely-Annotationen in:
      split_dir/img  und  split_dir/ann

    Erwartet:
      - Bildname:   frame_0001.png
      - Ann-Datei:  frame_0001.png.json  ODER  frame_0001.json
    """
    img_dir = split_dir / "img"
    ann_dir = split_dir / "ann"

    if not img_dir.exists():
        raise FileNotFoundError(f"img/ nicht gefunden unter: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"ann/ nicht gefunden unter: {ann_dir}")

    images = sorted(
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    pairs = []
    for ip in images:
        # Supervisely nimmt typischerweise den vollen Dateinamen + ".json"
        cand1 = ann_dir / f"{ip.name}.json"
        # Manche Exporte nehmen nur den Stem
        cand2 = ann_dir / f"{ip.stem}.json"

        if cand1.exists():
            ap = cand1
        elif cand2.exists():
            ap = cand2
        else:
            # Kein Annotation-File -> überspringen
            continue

        pairs.append((ip, ap))

    return pairs


def build_coco_from_supervisely(pairs, start_img_id=0, start_ann_id=1):
    """
    Baut ein COCO-Objekt aus (image_path, annotation_json_path)-Pairs.
    Es werden nur Objekte mit classTitle "apple" / "Apple" berücksichtigt.
    BBox-Berechnung:
      - bevorzugt Polygon (points.exterior)
      - fallback: bitmap.origin + bitmap.size
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "apple"}],
    }

    img_id = start_img_id
    ann_id = start_ann_id

    for img_path, ann_path in tqdm(pairs, desc="COCO build (Supervisely)"):
        # Bildgröße bestimmen (zur Sicherheit aus dem Bild selbst)
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": int(w), "height": int(h)}
        )

        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        objects = ann_data.get("objects", [])
        for obj in objects:
            class_title = obj.get("classTitle", "")
            if class_title.lower() != "apple":
                # Andere Klassen / "ignore regions" etc. überspringen
                continue

            bbox = None

            # 1) Polygon?
            points = (
                obj.get("points", {})
                .get("exterior", [])
            )
            if points:
                bbox = bbox_from_polygon(points)

            # 2) Fallback: Bitmap?
            if bbox is None:
                bitmap = obj.get("bitmap")
                if bitmap:
                    bbox = bbox_from_bitmap(bitmap)

            if bbox is None:
                # Konnte keine BBox bestimmen -> überspringen
                continue

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
    """
    Stellt Bilder in der Zielstruktur bereit (kopieren oder symlinken).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img_path, _ in tqdm(pairs, desc=f"stage -> {dest_dir.name}"):
        dst = dest_dir / img_path.name
        if dst.exists():
            continue
        if symlink:
            try:
                dst.symlink_to(img_path.resolve())
            except Exception:
                shutil.copy2(img_path, dst)
        else:
            shutil.copy2(img_path, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/apple_mots/raw",
        help="Path to Apple MOTS Supervisely root (contains subdirectories for each split).",
    )
    ap.add_argument(
        "--train-splits",
        type=str,
        default="train",
        help="Name of the subdirectory to use as the training split.",
    )
    ap.add_argument(
        "--val-splits",
        type=str,
        default="",
        help='Name of the subdirectory to use as the validation split. Pass "" to derive val from train using --val-ratio.',
    )
    ap.add_argument(
        "--test-splits",
        type=str,
        default="",
        help="Name of the subdirectory to use as the test split (optional).",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of train to use as val when --val-splits is empty.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    out_root = root.parent / "coco"
    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"
    out_ann = out_root / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)

    # --- Train/Val bestimmen ---
    train_dir = root / args.train_splits
    if args.val_splits:
        train_pairs = collect_pairs_supervisely(train_dir)
        val_pairs = collect_pairs_supervisely(root / args.val_splits)
    else:
        all_pairs = collect_pairs_supervisely(train_dir)
        random.Random(args.seed).shuffle(all_pairs)
        n_val = max(1, int(len(all_pairs) * args.val_ratio))
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]
        print(f"No val split given → splitting from train: train={len(train_pairs)} val={len(val_pairs)}")

    # --- COCO JSONs bauen ---
    coco_train = build_coco_from_supervisely(train_pairs, start_img_id=0, start_ann_id=1)
    coco_val = build_coco_from_supervisely(val_pairs, start_img_id=100000, start_ann_id=1)

    # --- Bilder in Zielstruktur bereitstellen ---
    stage_images(train_pairs, out_img_train, symlink=args.symlink)
    stage_images(val_pairs, out_img_val, symlink=args.symlink)

    # --- JSON speichern ---
    with open(out_ann / "instances_train.json", "w") as f:
        json.dump(coco_train, f)
    with open(out_ann / "instances_val.json", "w") as f:
        json.dump(coco_val, f)

    # --- Optional: Test ---
    if args.test_splits:
        test_pairs = collect_pairs_supervisely(root / args.test_splits)
        coco_test = build_coco_from_supervisely(test_pairs, start_img_id=200000, start_ann_id=1)
        out_img_test = out_root / "images" / "test"
        stage_images(test_pairs, out_img_test, symlink=args.symlink)
        with open(out_ann / "instances_test.json", "w") as f:
            json.dump(coco_test, f)

    print("\nDone")
    print(f"COCO root: {out_root}")
    print(f"- {out_img_train}")
    print(f"- {out_img_val}")
    print(f"- {out_ann / 'instances_train.json'}")
    print(f"- {out_ann / 'instances_val.json'}")
    if args.test_splits:
        print(f"- {out_ann / 'instances_test.json'}")


if __name__ == "__main__":
    main()

"""
Example usage:

    uv run python scripts/prepare_apple_mots.py \
      --root data/apple_mots/raw \
      --train-splits train \
      --val-splits testing \
      --seed 42

"""
