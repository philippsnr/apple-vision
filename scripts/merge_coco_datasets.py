"""Merge multiple COCO-format datasets into one.

Reads annotations/instances_{split}.json and images/{split}/ from each
dataset root, remaps image and annotation IDs to be globally unique, symlinks
(or copies) images into the output directory with a per-dataset prefix to avoid
filename collisions, and writes a merged annotation file.

Usage:
    uv run python scripts/merge_coco_datasets.py \
        data/minneapple/coco data/orchard/coco \
        --output data/merged/coco

    # copy images instead of symlinking:
    uv run python scripts/merge_coco_datasets.py \
        data/minneapple/coco data/orchard/coco \
        --output data/merged/coco --copy

    # only merge specific splits:
    uv run python scripts/merge_coco_datasets.py \
        data/minneapple/coco data/orchard/coco \
        --output data/merged/coco --splits train val

# example usage: uv run python scripts/merge_coco_datasets.py data/minneapple/coco data/orchard/coco --output data/merged/coco
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _dataset_tag(root: Path) -> str:
    """Derive a short name from the dataset root path for prefixing image filenames.

    Walks up the resolved path to find the first component that isn't 'coco'.
    E.g. /data/minneapple/coco -> 'minneapple'.
    """
    for part in reversed(root.resolve().parts):
        if part and part.lower() != "coco":
            return part
    return root.name


def merge_split(
    dataset_roots: list[Path],
    output_root: Path,
    split: str,
    copy: bool,
) -> bool:
    """Merge one split across all datasets. Returns True if any dataset had this split."""
    out_ann_dir = output_root / "annotations"
    out_img_dir = output_root / "images" / split
    out_ann_path = out_ann_dir / f"instances_{split}.json"

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    merged_categories: list[dict] | None = None
    # Deferred image links: collected first, applied after dirs are created.
    pending_links: list[tuple[Path, Path]] = []

    next_img_id = 0
    next_ann_id = 1
    any_found = False

    for ds_root in dataset_roots:
        ann_path = ds_root / "annotations" / f"instances_{split}.json"
        img_dir = ds_root / "images" / split

        if not ann_path.exists():
            print(f"  [SKIP] {ds_root}: no annotation file for split '{split}'")
            continue

        any_found = True
        tag = _dataset_tag(ds_root)

        with ann_path.open() as f:
            coco = json.load(f)

        images: list[dict] = coco.get("images", [])
        annotations: list[dict] = coco.get("annotations", [])
        categories: list[dict] = coco.get("categories", [])

        if merged_categories is None and categories:
            merged_categories = categories

        # Remap image IDs and collect image-link tasks.
        id_map: dict[int, int] = {}
        for img in images:
            new_name = f"{tag}_{img['file_name']}"
            id_map[img["id"]] = next_img_id

            new_img = dict(img)
            new_img["id"] = next_img_id
            new_img["file_name"] = new_name
            merged_images.append(new_img)
            next_img_id += 1

            src = img_dir / img["file_name"]
            if src.exists():
                pending_links.append((src.resolve(), out_img_dir / new_name))

        # Remap annotation IDs and image_id references.
        for ann in annotations:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[ann["image_id"]]
            merged_annotations.append(new_ann)
            next_ann_id += 1

        print(f"  [{split}] {tag}: {len(images)} images, {len(annotations)} annotations")

    if not any_found:
        return False

    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    for src, dst in pending_links:
        if dst.exists():
            skipped += 1
            continue
        if copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        linked += 1

    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories or [],
    }
    with out_ann_path.open("w") as f:
        json.dump(merged, f)

    action = "copied" if copy else "symlinked"
    print(
        f"  [{split}] => {len(merged_images)} images, {len(merged_annotations)} annotations"
        f" | {linked} images {action}, {skipped} already present"
        f" | annotations -> {out_ann_path}"
    )
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge multiple COCO-format datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        metavar="DATASET_ROOT",
        help="Paths to COCO dataset roots (each must contain annotations/ and images/).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for the merged dataset.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        metavar="SPLIT",
        help="Which splits to merge (default: train val test).",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (slower, but output is self-contained).",
    )
    args = p.parse_args()

    dataset_roots = [Path(d).resolve() for d in args.datasets]
    output_root = Path(args.output).resolve()

    for ds in dataset_roots:
        if not ds.is_dir():
            print(f"Error: dataset root does not exist: {ds}")
            sys.exit(1)

    tags = [_dataset_tag(ds) for ds in dataset_roots]
    print(f"Merging {len(dataset_roots)} datasets -> {output_root}")
    print(f"Datasets: {tags}")
    print(f"Splits:   {args.splits}")
    print()

    for split in args.splits:
        print(f"=== Split: {split} ===")
        merge_split(dataset_roots, output_root, split, args.copy)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
