import os
import shutil
import sys
from pathlib import Path
import json

from PIL import Image, ImageFile, UnidentifiedImageError

# Do NOT allow truncated images to load silently.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def check_image(path: Path) -> bool:
    """
    Tries to fully load an image.
    Returns True if it is valid, False if corrupted or unreadable.
    """
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (OSError, UnidentifiedImageError) as e:
        print(f"[DEFECT] {path} -> {e}")
        return False


def move_to_broken(path: Path, broken_dir: Path) -> Path:
    """
    Moves a corrupted image into the 'broken' directory.
    If the filename already exists there, append a counter.
    """
    broken_dir.mkdir(parents=True, exist_ok=True)

    target = broken_dir / path.name

    # Avoid overwriting files in 'broken'
    if target.exists():
        stem = path.stem
        suffix = path.suffix
        counter = 1
        while True:
            candidate = broken_dir / f"{stem}_dup{counter}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            counter += 1

    shutil.move(str(path), str(target))
    print(f"[MOVED] {path} -> {target}")
    return target


def scan_and_move_corrupted(images_dir: Path) -> None:
    """
    Scans all images in images_dir, moves corrupted ones to images_dir/'broken'.
    """
    if not images_dir.is_dir():
        print(f"[SKIP] Images directory does not exist: {images_dir}")
        return

    broken_dir = images_dir / "broken"

    total = 0
    ok = 0
    bad = 0

    print(f"\n=== Checking images in: {images_dir} ===")
    print(f"Corrupted images will be moved to: {broken_dir}\n")

    for root, dirs, files in os.walk(images_dir):
        root_path = Path(root)

        # Skip the 'broken' directory itself
        if root_path == broken_dir:
            continue

        for name in files:
            path = root_path / name
            if path.suffix.lower() not in IMAGE_EXTS:
                continue

            total += 1

            if check_image(path):
                ok += 1
            else:
                bad += 1
                move_to_broken(path, broken_dir)

            if total % 100 == 0:
                print(f"Checked {total} images (OK: {ok}, DEFECT: {bad})")

    print("Image check finished.")
    print(f"Total scanned:    {total}")
    print(f"Valid images:     {ok}")
    print(f"Corrupted moved:  {bad}")


def clean_coco_annotations(images_dir: Path, ann_path: Path) -> None:
    """
    Cleans a COCO annotations file by removing entries for which the image
    file does not exist in images_dir (after corrupted images have been moved).

    Overwrites ann_path, but creates a .bak backup first.
    """
    if not ann_path.is_file():
        print(f"[SKIP] Annotation file does not exist: {ann_path}")
        return

    print(f"\n=== Cleaning annotations: {ann_path} ===")

    with ann_path.open("r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # Collect all existing image files in images_dir (top level only)
    existing_files = {
        p.name for p in images_dir.iterdir()
        if p.is_file()
    }
    print(f"Found {len(existing_files)} image files on disk in {images_dir}.")

    kept_images = [img for img in images if img.get("file_name") in existing_files]
    kept_image_ids = {img["id"] for img in kept_images}
    kept_annotations = [
        ann for ann in annotations if ann.get("image_id") in kept_image_ids
    ]

    removed_images = len(images) - len(kept_images)
    removed_annotations = len(annotations) - len(kept_annotations)

    print(f"Original images:      {len(images)}")
    print(f"Kept images:          {len(kept_images)}")
    print(f"Removed images:       {removed_images}")
    print(f"Original annotations: {len(annotations)}")
    print(f"Kept annotations:     {len(kept_annotations)}")
    print(f"Removed annotations:  {removed_annotations}")

    coco["images"] = kept_images
    coco["annotations"] = kept_annotations

    # Backup original
    backup_path = ann_path.with_suffix(ann_path.suffix + ".bak")
    shutil.copy2(ann_path, backup_path)
    print(f"Backup of original annotations written to: {backup_path}")

    # Overwrite with cleaned version
    with ann_path.open("w") as f:
        json.dump(coco, f)
    print(f"Cleaned annotations written to: {ann_path}")


def process_split(dataset_root: Path, split: str) -> None:
    """
    Process one split (train/val/test) if corresponding directories/files exist.
    """
    images_dir = dataset_root / "images" / split
    ann_path = dataset_root / "annotations" / f"instances_{split}.json"

    if not images_dir.is_dir() and not ann_path.is_file():
        # Nothing for this split
        return

    print(f"\n############################")
    print(f"Processing split: {split}")
    print("############################")

    if images_dir.is_dir():
        scan_and_move_corrupted(images_dir)
    else:
        print(f"[WARN] Images dir for split '{split}' does not exist: {images_dir}")

    if ann_path.is_file():
        clean_coco_annotations(images_dir, ann_path)
    else:
        print(f"[WARN] Annotation file for split '{split}' does not exist: {ann_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python clean_coco_dataset.py /path/to/coco_root")
        print("Example structure:")
        print("  coco_root/")
        print("    images/train, images/val, images/test (optional)")
        print("    annotations/instances_train.json, instances_val.json, instances_test.json")
        sys.exit(1)

    dataset_root = Path(sys.argv[1]).resolve()

    if not dataset_root.is_dir():
        print(f"Dataset root does not exist or is not a directory: {dataset_root}")
        sys.exit(1)

    print(f"COCO dataset root: {dataset_root}")

    for split in ["train", "val", "test"]:
        process_split(dataset_root, split)

    print("\nAll done.")


if __name__ == "__main__":
    main()

# example usage: uv run python scripts/clean_coco_dataset.py data/orchard/coco
