#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baut das gemeinsame Benchmark-/Eval-Set aus den realen Datensätzen.

Zieht deterministisch je Datensatz gleich viele annotierte Bilder (Default 50)
aus deren train/val-Pool und friert sie als eigenständiges COCO-Set ein. Dieses
Set wird zum fairen Vergleich ALLER Modelle auf gleicher Grundlage benutzt.

Wichtige Garantien:
  - real & un-augmentiert, mit Ground-Truth (nur MinneApple/AppleMOTS/Orchard)
  - balanciert nach *Bildern* (gleich viele Szenen je Domäne)
  - deterministisch reproduzierbar (Seed pro Datensatz)
  - eingefroren via Manifest (SHA-256 je Bild) -> Leakage-Sperre fürs Training

Ausgabe:
  data/benchmark/coco/
    images/test/<tag>_<origname>
    annotations/instances_test.json
  data/benchmark/benchmark_manifest.json

Das Manifest ist die Wahrheitsquelle: build des Trainings-Sets (Schritt 3) muss
jedes hier gelistete Bild ausschließen.

Beispiel:
    uv run python scripts/build_benchmark_set.py \
      data/minneapple/coco data/apple_mots/coco data/orchard/coco \
      --per-dataset 50 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _dataset_tag(root: Path) -> str:
    """Kurzname aus dem Dataset-Pfad (erste Komponente != 'coco').
    z.B. data/minneapple/coco -> 'minneapple'. Konsistent zu merge_coco_datasets.py.
    """
    for part in reversed(root.resolve().parts):
        if part and part.lower() != "coco":
            return part
    return root.name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_annotated_pool(ds_root: Path, source_splits: list[str]):
    """
    Sammelt aus den angegebenen Splits alle Bilder mit >=1 Annotation.
    Gibt eine sortierte Liste von Records zurück:
      {split, file_name, width, height, path, annotations: [ann, ...]}
    """
    pool = []
    for split in source_splits:
        ann_path = ds_root / "annotations" / f"instances_{split}.json"
        img_dir = ds_root / "images" / split
        if not ann_path.exists():
            continue
        with ann_path.open() as f:
            coco = json.load(f)

        anns_by_img = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_img[ann["image_id"]].append(ann)

        for img in coco.get("images", []):
            anns = anns_by_img.get(img["id"], [])
            if not anns:
                continue  # nur annotierte Bilder ins Benchmark
            src = img_dir / img["file_name"]
            if not src.exists():
                # Bild referenziert, aber Datei fehlt (z.B. entfernt) -> überspringen
                continue
            pool.append(
                {
                    "split": split,
                    "file_name": img["file_name"],
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "path": src,
                    "annotations": anns,
                }
            )
    # stabile Grundordnung, unabhängig von JSON-Reihenfolge
    pool.sort(key=lambda r: (r["split"], r["file_name"]))
    return pool


def main() -> None:
    p = argparse.ArgumentParser(
        description="Baut das gemeinsame Benchmark-/Eval-Set aus realen COCO-Datensätzen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        metavar="DATASET_ROOT",
        help="COCO-Roots der realen Datensätze (je annotations/ + images/).",
    )
    p.add_argument(
        "--per-dataset",
        type=int,
        default=50,
        help="Anzahl Bilder je Datensatz (Default 50, balanciert nach Bildern).",
    )
    p.add_argument(
        "--source-splits",
        nargs="+",
        default=["train", "val"],
        metavar="SPLIT",
        help="Aus welchen Splits gezogen wird (Default: train val). "
        "test wird bewusst ignoriert (z.B. MinneApple-Test hat keine GT).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/benchmark/coco"),
        help="Ausgabe-Root des Benchmark-COCO-Sets.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Pfad des Manifests (Default: <output>/../benchmark_manifest.json).",
    )
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Bilder symlinken statt kopieren (Default: kopieren -> self-contained).",
    )
    args = p.parse_args()

    dataset_roots = [Path(d).resolve() for d in args.datasets]
    for ds in dataset_roots:
        if not ds.is_dir():
            raise SystemExit(f"Fehler: Dataset-Root existiert nicht: {ds}")

    out_root = args.output.resolve()
    out_img_dir = out_root / "images" / "test"
    out_ann_dir = out_root / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest or (out_root.parent / "benchmark_manifest.json")).resolve()

    print(f"Benchmark -> {out_root}")
    print(f"Datasets: {[_dataset_tag(d) for d in dataset_roots]}")
    print(f"Pro Datensatz: {args.per_dataset}  Quelle: {args.source_splits}  Seed: {args.seed}\n")

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    merged_categories: list[dict] | None = None
    manifest_entries: list[dict] = []
    pending_copy: list[tuple[Path, Path]] = []
    per_dataset_summary: dict[str, dict] = {}

    next_img_id = 0
    next_ann_id = 1

    for ds_root in dataset_roots:
        tag = _dataset_tag(ds_root)

        # Kategorien vom ersten Datensatz übernehmen
        if merged_categories is None:
            for split in args.source_splits:
                ap = ds_root / "annotations" / f"instances_{split}.json"
                if ap.exists():
                    merged_categories = json.load(ap.open()).get("categories")
                    break

        pool = collect_annotated_pool(ds_root, args.source_splits)
        n = args.per_dataset
        if len(pool) < n:
            print(f"  ! {tag}: nur {len(pool)} annotierte Bilder verfügbar (< {n}) -> nehme alle")
            n = len(pool)

        # deterministische Auswahl, unabhängig von den anderen Datensätzen
        rng = random.Random(f"{tag}:{args.seed}")
        selected = rng.sample(pool, n)
        selected.sort(key=lambda r: (r["split"], r["file_name"]))

        n_boxes = 0
        for rec in selected:
            staged_name = f"{tag}_{rec['file_name']}"
            dst = out_img_dir / staged_name

            new_img_id = next_img_id
            next_img_id += 1
            merged_images.append(
                {
                    "id": new_img_id,
                    "file_name": staged_name,
                    "width": rec["width"],
                    "height": rec["height"],
                }
            )
            for ann in rec["annotations"]:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = new_img_id
                merged_annotations.append(new_ann)
                next_ann_id += 1
                n_boxes += 1

            pending_copy.append((rec["path"], dst))
            manifest_entries.append(
                {
                    "dataset": tag,
                    "source_split": rec["split"],
                    "orig_file_name": rec["file_name"],
                    "staged_file_name": staged_name,
                    "sha256": _sha256(rec["path"]),
                    "width": rec["width"],
                    "height": rec["height"],
                    "num_annotations": len(rec["annotations"]),
                }
            )

        per_dataset_summary[tag] = {
            "images": n,
            "boxes": n_boxes,
            "pool_size": len(pool),
        }
        print(f"  [{tag}] {n} Bilder / {n_boxes} Boxen (aus Pool {len(pool)})")

    # Bilder bereitstellen
    copied = 0
    for src, dst in pending_copy:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.symlink:
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
        copied += 1

    # COCO schreiben
    out_ann_path = out_ann_dir / "instances_test.json"
    with out_ann_path.open("w") as f:
        json.dump(
            {
                "images": merged_images,
                "annotations": merged_annotations,
                "categories": merged_categories or [{"id": 1, "name": "apple"}],
            },
            f,
        )

    # Manifest schreiben (Wahrheitsquelle für Leakage-Sperre)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "source_splits": args.source_splits,
        "datasets": {_dataset_tag(d): str(d) for d in dataset_roots},
        "summary": per_dataset_summary,
        "total_images": len(merged_images),
        "total_boxes": len(merged_annotations),
        "images": manifest_entries,
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    action = "symlinked" if args.symlink else "kopiert"
    print(
        f"\nFertig: {len(merged_images)} Bilder, {len(merged_annotations)} Boxen "
        f"({copied} {action})"
    )
    print(f"  COCO:     {out_ann_path}")
    print(f"  Images:   {out_img_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
