#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

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


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python test_images.py /path/to/images")
        sys.exit(1)

    image_root = Path(sys.argv[1]).resolve()

    if not image_root.is_dir():
        print(f"Directory does not exist or is not a folder: {image_root}")
        sys.exit(1)

    broken_dir = image_root / "broken"

    total = 0
    ok = 0
    bad = 0

    print(f"Scanning images in: {image_root}")
    print(f"Corrupted images will be moved to: {broken_dir}\n")

    for root, dirs, files in os.walk(image_root):
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

    print("\n-----")
    print("DONE.")
    print(f"Total scanned:    {total}")
    print(f"Valid images:     {ok}")
    print(f"Corrupted moved:  {bad}")


if __name__ == "__main__":
    main()

# Example usage:
# uv run python scripts/check_images.py data/orchard/coco/images/train
