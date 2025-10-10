from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw
from pycocotools.coco import COCO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick plot COCO ground-truth bounding boxes on sample images")
    p.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root (COCO layout)")
    p.add_argument("--ann", type=str, default="annotations/instances_val.json", help="Annotation JSON path (relative to root or absolute)")
    p.add_argument("--images", type=str, default="images/val", help="Images directory (relative to root or absolute)")
    p.add_argument("--n", type=int, default=8, help="Number of images to plot")
    p.add_argument("--out-dir", type=str, default="quickplots", help="Directory to save plotted images")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p.add_argument("--show", action="store_true", help="Open images after saving (may be slow)")
    return p.parse_args()


def _resolve(root: Path, maybe_rel: str | Path) -> Path:
    p = Path(str(maybe_rel).replace("\\", "/"))
    return p if p.is_absolute() else (root / p)


def _xywh_to_xyxy(box: Iterable[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def draw_boxes(img: Image.Image, boxes: List[Tuple[float, float, float, float]], outline=(255, 0, 0), width: int = 3) -> Image.Image:
    im = img.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
    return im


def main_cli():
    args = parse_args()

    root = Path(args.dataset_root)
    ann_path = _resolve(root, args.ann)
    img_dir = _resolve(root, args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(ann_path.as_posix())

    import random
    rng = random.Random(args.seed)
    img_ids = coco.getImgIds()
    if len(img_ids) == 0:
        raise RuntimeError("No images found in annotations.")
    sample_ids = rng.sample(img_ids, k=min(args.n, len(img_ids)))

    count = 0
    for img_id in sample_ids:
        info = coco.loadImgs([img_id])[0]
        file_name = info["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            print(f"[quickplot] Warning: image file missing: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        boxes = [
            _xywh_to_xyxy(ann["bbox"]) for ann in anns if "bbox" in ann and isinstance(ann["bbox"], (list, tuple))
        ]
        viz = draw_boxes(img, boxes)
        out_path = out_dir / f"{Path(file_name).stem}_gt.png"
        viz.save(out_path)
        print(f"Saved: {out_path}")
        count += 1
        if args.show:
            try:
                viz.show()
            except Exception:
                pass

    print(f"Done. Plotted {count} image(s) to {out_dir}.")


if __name__ == "__main__":
    main_cli()
