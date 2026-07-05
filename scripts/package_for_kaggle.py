#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packt die COCO-Quellen + Benchmark + Manifest in einen Upload-Ordner für ein
Kaggle-Dataset und schreibt die passende dataset-metadata.json.

Strategie: EINMAL hochladen, im Notebook pro Sweep-Punkt zusammenbauen. Der
Ordner enthält nur die kanonischen Quell-Sets und das eingefrorene Benchmark-
Set (keine raw/-Ordner, keine Archive, keine zusammengebauten Trainings-Sets).

Trick: Dateien werden per HARDLINK gespiegelt (gleiche Platte -> sofort, kein
zusätzlicher Speicher, und es sind echte Dateien -> Kaggle lädt sie sauber,
anders als Symlinks). Fällt auf Kopieren zurück, wenn Hardlinks nicht gehen.

Ziel-Layout (später unter /kaggle/input/<slug>/):
  minneapple/coco/...
  apple_mots/coco/...
  orchard/coco/...
  synthetic/coco/...
  benchmark/coco/...
  benchmark_manifest.json
  dataset-metadata.json

Upload danach:
  kaggle datasets create -p data/kaggle_upload --dir-mode zip      # erstmalig
  kaggle datasets version -p data/kaggle_upload -m "neue version"  # update

Beispiel:
    uv run python scripts/package_for_kaggle.py \
      --source data/minneapple/coco --source data/apple_mots/coco \
      --source data/orchard/coco --source data/synthetic/coco \
      --benchmark data/benchmark/coco \
      --manifest data/benchmark/benchmark_manifest.json \
      --dataset-id deinnutzername/apple-vision-data \
      --output data/kaggle_upload
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _dataset_tag(root: Path) -> str:
    """Kurzname aus dem Dataset-Pfad (erste Komponente != 'coco').
    Konsistent zu den anderen Skripten -> Tags bleiben unter /kaggle/input gleich.
    """
    for part in reversed(root.resolve().parts):
        if part and part.lower() != "coco":
            return part
    return root.name


def mirror_tree(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    """Spiegelt src_dir nach dst_dir per Hardlink (Fallback: Kopie).
    Gibt (n_linked, n_copied) zurück. Folgt vorhandenen Symlinks zur echten Datei.
    """
    linked = copied = 0
    for root, _dirs, files in os.walk(src_dir):
        rel = Path(root).relative_to(src_dir)
        out = dst_dir / rel
        out.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            real = s.resolve()  # falls die Quelle selbst ein Symlink ist
            d = out / name
            if d.exists():
                d.unlink()
            try:
                os.link(real, d)
                linked += 1
            except OSError:
                shutil.copy2(real, d)
                copied += 1
    return linked, copied


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Packt COCO-Quellen + Benchmark in einen Kaggle-Upload-Ordner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", action="append", default=[], metavar="COCO_ROOT",
                    help="Kanonisches COCO-Set (mehrfach). Wird unter <tag>/coco/ abgelegt.")
    ap.add_argument("--benchmark", type=Path, default=Path("data/benchmark/coco"),
                    help="Benchmark-COCO-Root -> benchmark/coco/.")
    ap.add_argument("--manifest", type=Path, default=Path("data/benchmark/benchmark_manifest.json"),
                    help="Benchmark-Manifest -> benchmark_manifest.json.")
    ap.add_argument("--output", "-o", type=Path, default=Path("data/kaggle_upload"),
                    help="Upload-Ordner (wird angelegt/überschrieben).")
    ap.add_argument("--dataset-id", required=True,
                    help="Kaggle-Dataset-ID im Format 'nutzername/slug'.")
    ap.add_argument("--title", default=None,
                    help="Anzeigename (Default: aus dem Slug abgeleitet).")
    args = ap.parse_args()

    if "/" not in args.dataset_id:
        raise SystemExit("Fehler: --dataset-id muss 'nutzername/slug' sein.")
    slug = args.dataset_id.split("/", 1)[1]
    title = args.title or slug.replace("-", " ").title()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    total_linked = total_copied = 0

    # Quell-Sets
    for s in args.source:
        root = Path(s).resolve()
        if not root.is_dir():
            raise SystemExit(f"Fehler: Quelle existiert nicht: {root}")
        tag = _dataset_tag(root)
        dst = out / tag / "coco"
        l, c = mirror_tree(root, dst)
        total_linked += l; total_copied += c
        print(f"  {tag}/coco: {_fmt(l)} gelinkt, {_fmt(c)} kopiert")

    # Benchmark
    if args.benchmark and Path(args.benchmark).is_dir():
        l, c = mirror_tree(Path(args.benchmark).resolve(), out / "benchmark" / "coco")
        total_linked += l; total_copied += c
        print(f"  benchmark/coco: {_fmt(l)} gelinkt, {_fmt(c)} kopiert")

    # Manifest
    if args.manifest and Path(args.manifest).exists():
        dst = out / "benchmark_manifest.json"
        if dst.exists():
            dst.unlink()
        try:
            os.link(Path(args.manifest).resolve(), dst)
        except OSError:
            shutil.copy2(Path(args.manifest).resolve(), dst)
        print("  benchmark_manifest.json")

    # dataset-metadata.json
    meta = {
        "title": title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with (out / "dataset-metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFertig -> {out}")
    print(f"  {_fmt(total_linked)} Dateien gelinkt, {_fmt(total_copied)} kopiert")
    print(f"  dataset-id: {args.dataset_id}")
    print("\nUpload:")
    print(f"  kaggle datasets create  -p {out} --dir-mode zip      # erstmalig")
    print(f"  kaggle datasets version -p {out} -m 'update'         # spätere Updates")


if __name__ == "__main__":
    main()
