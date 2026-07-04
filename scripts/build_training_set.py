#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baut ein eindeutiges, konfigurierbares Trainings-Set aus mehreren COCO-Quellen.

Pro Quelle wird frei bestimmt, wie viele Bilder einfließen (`ROOT:ANZAHL` oder
`ROOT:all`). Die Auswahl ist deterministisch (Seed), schließt garantiert alle
Bilder des gemeinsamen Benchmark-/Eval-Sets aus (Leakage-Sperre über dessen
Manifest) und wird selbst als Manifest protokolliert -> jedes erzeugte Set ist
reproduzierbar und auditierbar.

Ausgabe:
  <output>/
    images/{train,val}/<tag>_<origname>
    annotations/{instances_train.json, instances_val.json}
  <output>/../training_manifest.json

Beispiel:
    uv run python scripts/build_training_set.py \
      --source data/minneapple/coco:400 \
      --source data/apple_mots/coco:all \
      --source data/orchard/coco:800 \
      --source data/synthetic/coco:500 \
      --benchmark-manifest data/benchmark/benchmark_manifest.json \
      --output data/train_v1/coco --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _dataset_tag(root: Path) -> str:
    """Kurzname aus dem Dataset-Pfad (erste Komponente != 'coco').
    Konsistent zu merge_coco_datasets.py und build_benchmark_set.py.
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


def parse_source(spec: str) -> tuple[Path, str]:
    """'data/minneapple/coco:400' -> (Path, '400'); ohne ':' -> (Path, 'all')."""
    root_str, sep, count = spec.rpartition(":")
    if not sep:
        return Path(spec), "all"
    # Vorsicht: leerer count ('root:') -> 'all'
    return Path(root_str), (count or "all")


def collect_eligible(ds_root: Path, source_splits: list[str], exclude_names: set[str]):
    """
    Sammelt aus den Quell-Splits alle Bilder MIT Datei auf Disk, die nicht im
    Benchmark sind (nach Dateiname ausgeschlossen). Records:
      {split, file_name, width, height, path, annotations}
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
            fn = img["file_name"]
            if fn in exclude_names:
                continue  # Benchmark-Bild -> Leakage-Sperre
            src = img_dir / fn
            if not src.exists():
                continue
            pool.append(
                {
                    "split": split,
                    "file_name": fn,
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "path": src,
                    "annotations": anns_by_img.get(img["id"], []),
                }
            )
    pool.sort(key=lambda r: (r["split"], r["file_name"]))
    return pool


def main() -> None:
    p = argparse.ArgumentParser(
        description="Baut ein konfigurierbares, benchmark-freies Trainings-Set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="ROOT:N",
        help="Quelle als 'coco_root:anzahl' oder 'coco_root:all'. Mehrfach angebbar.",
    )
    p.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("data/benchmark/benchmark_manifest.json"),
        help="Manifest des Benchmark-Sets (Leakage-Sperre). Leer/none zum Deaktivieren.",
    )
    p.add_argument("--output", "-o", type=Path, required=True, help="Output-COCO-Root.")
    p.add_argument(
        "--source-splits",
        nargs="+",
        default=["train", "val"],
        help="Aus welchen Quell-Splits gezogen wird (Default: train val).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Anteil je Quelle für den internen Val-Split (Early Stopping). 0 = kein Val.",
    )
    p.add_argument(
        "--val-exclude",
        nargs="*",
        default=[],
        metavar="TAG",
        help="Tags, deren Bilder komplett in train gehen (z.B. 'synthetic' für "
        "einen rein realen Val-Early-Stop-Signal).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--copy",
        action="store_true",
        help="Bilder kopieren statt symlinken (self-contained, aber groß).",
    )
    p.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Zusätzlich SHA-256 jedes gewählten Bildes gegen das Benchmark-Manifest prüfen.",
    )
    args = p.parse_args()

    sources = [parse_source(s) for s in args.source]
    for root, _ in sources:
        if not root.is_dir():
            raise SystemExit(f"Fehler: Quelle existiert nicht: {root}")

    # --- Benchmark-Manifest laden -> Ausschluss nach (tag -> Dateinamen) + Hashes ---
    exclude_by_tag: dict[str, set[str]] = defaultdict(set)
    exclude_hashes: set[str] = set()
    mpath = args.benchmark_manifest
    if mpath and str(mpath).lower() not in {"none", ""} and Path(mpath).exists():
        bm = json.load(Path(mpath).open())
        for e in bm.get("images", []):
            exclude_by_tag[e["dataset"]].add(e["orig_file_name"])
            exclude_hashes.add(e["sha256"])
        print(f"Leakage-Sperre: {sum(len(v) for v in exclude_by_tag.values())} "
              f"Benchmark-Bilder aus {len(exclude_by_tag)} Datensätzen geladen")
    else:
        print("WARNUNG: kein Benchmark-Manifest -> keine Leakage-Sperre aktiv!")

    out_root = args.output.resolve()
    out_ann_dir = out_root / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    # Akkumulatoren für die gemergten Splits
    merged = {"train": {"images": [], "annotations": []},
              "val": {"images": [], "annotations": []}}
    merged_categories: list[dict] | None = None
    pending_links: list[tuple[Path, Path, str]] = []  # (src, dst, split)
    next_img_id = 0
    next_ann_id = 1
    manifest_sources: list[dict] = []
    verify_records: list[tuple[str, Path]] = []  # (tag, path) für Hash-Check

    for root, count_spec in sources:
        tag = _dataset_tag(root)
        eligible = collect_eligible(root, args.source_splits, exclude_by_tag.get(tag, set()))

        if count_spec == "all":
            n = len(eligible)
        else:
            n = int(count_spec)
            if n > len(eligible):
                print(f"  ! {tag}: angefragt {n}, nur {len(eligible)} verfügbar -> nehme alle")
                n = len(eligible)

        # Geschachtelte Auswahl: einmal deterministisch shufflen, dann Präfix
        # nehmen. Dadurch ist jedes größere n ein echtes Superset des kleineren
        # -> saubere Ablation (z.B. Synthetik-Anteil hochfahren = dieselben
        # Bilder + zusätzliche, nicht neu gewürfelt).
        order = list(eligible)
        random.Random(f"train:{tag}:{args.seed}").shuffle(order)
        selected = order[:n]

        # Val-Split stratifiziert je Quelle
        if args.val_ratio > 0 and tag not in args.val_exclude:
            order = list(selected)
            random.Random(f"trainval:{tag}:{args.seed}").shuffle(order)
            n_val = int(round(n * args.val_ratio))
            val_set = set(id(r) for r in order[:n_val])
        else:
            n_val = 0
            val_set = set()

        if merged_categories is None:
            for split in args.source_splits:
                ap = root / "annotations" / f"instances_{split}.json"
                if ap.exists():
                    merged_categories = json.load(ap.open()).get("categories")
                    break

        included_names = []
        for rec in selected:
            split = "val" if id(rec) in val_set else "train"
            staged_name = f"{tag}_{rec['file_name']}"
            img_id = next_img_id
            next_img_id += 1
            merged[split]["images"].append(
                {"id": img_id, "file_name": staged_name,
                 "width": rec["width"], "height": rec["height"]}
            )
            for ann in rec["annotations"]:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = img_id
                merged[split]["annotations"].append(new_ann)
                next_ann_id += 1
            pending_links.append((rec["path"].resolve(), out_root / "images" / split / staged_name, split))
            included_names.append(rec["file_name"])
            if args.verify_hashes:
                verify_records.append((tag, rec["path"]))

        n_boxes = sum(len(r["annotations"]) for r in selected)
        manifest_sources.append({
            "tag": tag,
            "root": str(root.resolve()),
            "requested": count_spec,
            "included": n,
            "train": n - n_val,
            "val": n_val,
            "boxes": n_boxes,
            "pool_size": len(eligible),
            "file_names": sorted(included_names),
        })
        print(f"  [{tag}] {n} Bilder ({n - n_val} train / {n_val} val), "
              f"{n_boxes} Boxen (Pool {len(eligible)})")

    # --- Harte Leakage-Assertion (nach Dateiname) ---
    for src in manifest_sources:
        bad = set(src["file_names"]) & exclude_by_tag.get(src["tag"], set())
        if bad:
            raise SystemExit(f"LEAKAGE! {src['tag']}: {len(bad)} Benchmark-Bilder im Training: "
                             f"{sorted(bad)[:5]} ...")

    # --- Optionaler SHA-256-Cross-Check ---
    if args.verify_hashes and exclude_hashes:
        print(f"Hash-Verifikation von {len(verify_records)} Bildern ...")
        for tag, path in verify_records:
            if _sha256(path) in exclude_hashes:
                raise SystemExit(f"LEAKAGE (Hash)! {tag}: {path} ist im Benchmark-Set.")
        print("  Hash-Check ok: kein Trainingsbild im Benchmark.")

    # --- Bilder bereitstellen + COCO schreiben ---
    linked = 0
    for src, dst, _split in pending_links:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        linked += 1

    cats = merged_categories or [{"id": 1, "name": "apple"}]
    for split in ("train", "val"):
        data = merged[split]
        if not data["images"] and split == "val":
            continue
        with (out_ann_dir / f"instances_{split}.json").open("w") as f:
            json.dump({"images": data["images"], "annotations": data["annotations"],
                       "categories": cats}, f)

    # --- Trainings-Manifest ---
    manifest_path = (out_root.parent / "training_manifest.json").resolve()
    total_train = len(merged["train"]["images"])
    total_val = len(merged["val"]["images"])
    with manifest_path.open("w") as f:
        json.dump({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "output": str(out_root),
            "source_splits": args.source_splits,
            "val_ratio": args.val_ratio,
            "val_exclude": args.val_exclude,
            "benchmark_manifest": str(mpath) if mpath else None,
            "hash_verified": bool(args.verify_hashes and exclude_hashes),
            "totals": {"train": total_train, "val": total_val,
                       "images": total_train + total_val},
            "sources": manifest_sources,
        }, f, indent=2)

    action = "kopiert" if args.copy else "symlinkt"
    print(f"\nFertig: {total_train} train + {total_val} val Bilder ({linked} {action})")
    print(f"  COCO:     {out_root}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
