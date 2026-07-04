#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic apples (Supervisely polygon export) -> COCO (Bounding Boxes).

Erzeugt Struktur:
data/synthetic/coco/
  images/train[, images/val]
  annotations/{instances_train.json[, instances_val.json]}

Erwartete Struktur unter --root (Standard: data/synthetic/raw) ist ein
Supervisely-Projekt mit einem oder mehreren Dataset-Ordnern:

  root/
    meta.json                     # optional, wird nur informativ gelesen
    <dataset name>/
      img/   # RGB-Bilder
      ann/   # Supervisely-Annotationen (*.png.json mit Polygonen)

Nur Objekte mit classTitle "apple" (Polygon) werden übernommen. Bilder ohne
Objekte bleiben als Negativbeispiele im Datensatz erhalten (leere Targets).

Die synthetischen Daten sind ausschließlich für das Training gedacht und dürfen
niemals ins gemeinsame Benchmark-/Eval-Set gelangen. Deshalb landet per Default
alles in 'train' (--val-ratio 0.0).
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def bbox_from_polygon(points, img_w, img_h):
    """
    Supervisely-Polygon: points = [[x1, y1], [x2, y2], ...]
    -> COCO-BBox [x, y, w, h], auf die Bildgrenzen geclamped.
    Gibt None zurück, wenn keine gültige (positive) Box entsteht.
    """
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(img_w, max(xs))
    y2 = min(img_h, max(ys))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    if w < 1 or h < 1:
        return None
    return [int(round(x1)), int(round(y1)), w, h]


def find_dataset_dirs(root: Path):
    """
    Findet alle Supervisely-Dataset-Ordner unter root, d.h. Unterordner, die
    sowohl img/ als auch ann/ enthalten. Fällt auf root selbst zurück, falls
    root direkt img/+ann/ enthält.
    """
    if (root / "img").exists() and (root / "ann").exists():
        return [root]
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "img").exists() and (d / "ann").exists()
    )
    if not dirs:
        raise FileNotFoundError(
            f"Kein Supervisely-Dataset (img/+ann/) unter {root} gefunden."
        )
    return dirs


def collect_pairs_supervisely(dataset_dir: Path):
    """
    Sucht Bilder + passende Supervisely-Annotationen in:
      dataset_dir/img  und  dataset_dir/ann

    Ann-Datei heißt typischerweise <bildname>.json (voller Name inkl. Endung)
    oder <stem>.json.
    """
    img_dir = dataset_dir / "img"
    ann_dir = dataset_dir / "ann"

    images = sorted(
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    pairs = []
    missing = 0
    for ip in images:
        cand1 = ann_dir / f"{ip.name}.json"
        cand2 = ann_dir / f"{ip.stem}.json"
        if cand1.exists():
            ap = cand1
        elif cand2.exists():
            ap = cand2
        else:
            missing += 1
            continue
        pairs.append((ip, ap))

    if missing:
        print(f"  ! {missing} Bilder ohne Annotation übersprungen in {dataset_dir.name}")
    return pairs


def build_coco_from_supervisely(pairs, start_img_id=0, start_ann_id=1):
    """
    Baut ein COCO-Objekt aus (image_path, annotation_json_path)-Pairs.
    Nur classTitle == "apple" (Polygone). Leere Bilder bleiben als Negative.
    Gibt (coco, stats) zurück.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "apple"}],
    }

    img_id = start_img_id
    ann_id = start_ann_id
    n_empty = 0
    n_dropped = 0

    for img_path, ann_path in tqdm(pairs, desc="COCO build (synthetic)"):
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": int(w), "height": int(h)}
        )

        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        n_before = ann_id
        for obj in ann_data.get("objects", []):
            if obj.get("classTitle", "").lower() != "apple":
                continue
            points = obj.get("points", {}).get("exterior", [])
            bbox = bbox_from_polygon(points, w, h)
            if bbox is None:
                n_dropped += 1
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

        if ann_id == n_before:
            n_empty += 1
        img_id += 1

    stats = {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "empty_images": n_empty,
        "dropped_boxes": n_dropped,
    }
    return coco, stats


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
                shutil.copy2(img_path, dst)
        else:
            shutil.copy2(img_path, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/synthetic/raw",
        help="Pfad zum Supervisely-Projekt (enthält Dataset-Ordner mit img/+ann/).",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Anteil für einen internen Val-Split. Default 0.0 = alles in train "
        "(synthetische Daten sind nur fürs Training).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true", help="Bilder symlinken statt kopieren")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = root.parent / "coco"
    out_ann = out_root / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)

    # --- Alle Dataset-Ordner einsammeln ---
    dataset_dirs = find_dataset_dirs(root)
    all_pairs = []
    for d in dataset_dirs:
        pairs = collect_pairs_supervisely(d)
        print(f"Dataset '{d.name}': {len(pairs)} Bild/Ann-Paare")
        all_pairs.extend(pairs)
    print(f"Gesamt: {len(all_pairs)} Paare aus {len(dataset_dirs)} Dataset-Ordner(n)")

    # --- Train/Val bestimmen ---
    if args.val_ratio > 0.0:
        random.Random(args.seed).shuffle(all_pairs)
        n_val = max(1, int(len(all_pairs) * args.val_ratio))
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]
        print(f"Val-Split: train={len(train_pairs)} val={len(val_pairs)}")
    else:
        train_pairs = all_pairs
        val_pairs = []

    # --- COCO bauen ---
    coco_train, stats_train = build_coco_from_supervisely(train_pairs, start_img_id=0, start_ann_id=1)
    stage_images(train_pairs, out_root / "images" / "train", symlink=args.symlink)
    with open(out_ann / "instances_train.json", "w") as f:
        json.dump(coco_train, f)

    if val_pairs:
        coco_val, stats_val = build_coco_from_supervisely(val_pairs, start_img_id=1_000_000, start_ann_id=1)
        stage_images(val_pairs, out_root / "images" / "val", symlink=args.symlink)
        with open(out_ann / "instances_val.json", "w") as f:
            json.dump(coco_val, f)
    else:
        stats_val = None

    print("\nFertig")
    print(f"COCO-Root: {out_root}")
    print(f"  train: {stats_train['images']} Bilder, {stats_train['annotations']} Boxen, "
          f"{stats_train['empty_images']} leer (Negative), {stats_train['dropped_boxes']} Boxen verworfen")
    if stats_val:
        print(f"  val:   {stats_val['images']} Bilder, {stats_val['annotations']} Boxen, "
              f"{stats_val['empty_images']} leer, {stats_val['dropped_boxes']} Boxen verworfen")


if __name__ == "__main__":
    main()

"""
Example usage:

    uv run python scripts/prepare_synthetic.py \
      --root data/synthetic/raw \
      --seed 42
"""
