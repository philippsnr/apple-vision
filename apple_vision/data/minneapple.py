from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision.datasets import CocoDetection
except Exception:  # pragma: no cover - allow import without torchvision available at parse time
    CocoDetection = object  # type: ignore


def _to_xyxy(box):
    # COCO bbox is [x, y, width, height]
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _normalize_path(p) -> Path:
    """Normalize path-like input to be OS-agnostic.

    Converts any backslashes to forward slashes so that paths provided with
    Windows-style separators also work on POSIX (WSL/Linux/Mac). Forward
    slashes work on Windows too, so this is safe cross-platform.
    """
    return Path(str(p).replace("\\", "/"))


class CocoAppleDataset(CocoDetection if isinstance(CocoDetection, type) else Dataset):
    """
    Minimal COCO-compatible dataset wrapper for apple detection.

    Expected directory structure (COCO-style):
    dataset_root/
      annotations/
        instances_train.json
        instances_val.json
      images/
        train/
        val/

    Notes:
    - This wrapper uses only bounding boxes and labels (no masks) so it works
      with Faster R-CNN out of the box. If your annotations have only polygons,
      they still include 'bbox' fields in COCO.
    - All apple instances are mapped to label id 1 (background=0).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        ann_file: str | Path,
        img_dir: str | Path,
        transforms=None,
    ) -> None:
        # Normalize paths to support Windows-style input (e.g., backslashes) on POSIX
        raw_root = str(dataset_root)
        root = _normalize_path(dataset_root)

        # Attempt to recover from collapsed separators on POSIX shells (e.g., "dataminneapplecoco")
        # If no path separators are present and the root doesn't exist, try common default.
        if ("\\" not in raw_root) and ("/" not in raw_root) and not root.exists():
            default_candidate = Path("data/minneapple/coco")
            if default_candidate.exists():
                print(f"[minneapple] Warning: dataset_root '{raw_root}' had no separators. Assuming '{default_candidate.as_posix()}' as dataset root.")
                root = default_candidate

        ann_path = _normalize_path(root / ann_file)
        img_path = _normalize_path(root / img_dir)

        # Helpful hint for POSIX shells where unquoted backslashes get stripped
        import os
        hint = ""
        if not root.exists():
            if os.name != "nt" and ("\\" not in raw_root) and ("/" not in raw_root):
                hint = (" Hint: On Linux/WSL shells, use forward slashes (e.g., data/minneapple/coco) "
                        "or quote the argument to keep backslashes, e.g.: "
                        "--dataset-root 'data\\minneapple\\coco'")

        if not ann_path.exists():
            extra = f" (dataset_root provided: '{raw_root}', resolved: '{root.as_posix()}')."
            raise FileNotFoundError(f"Annotation file not found: {ann_path}{extra}{(' ' + hint) if hint else ''}")
        if not img_path.exists():
            extra = f" (dataset_root provided: '{raw_root}', resolved: '{root.as_posix()}')."
            raise FileNotFoundError(f"Image directory not found: {img_path}{extra}{(' ' + hint) if hint else ''}")

        super().__init__(img_path.as_posix(), ann_path.as_posix())
        self._transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        img, anns = super().__getitem__(idx)

        # Image size for clamping boxes
        img_w, img_h = img.size  # PIL gives (W, H)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            bbox = ann.get("bbox")
            if not bbox:
                continue

            # COCO bbox -> xyxy and sanitize
            x1, y1, x2, y2 = map(float, _to_xyxy(bbox))
            # Clamp to image bounds [0, W] / [0, H]
            x1 = max(0.0, min(x1, float(img_w)))
            y1 = max(0.0, min(y1, float(img_h)))
            x2 = max(0.0, min(x2, float(img_w)))
            y2 = max(0.0, min(y2, float(img_h)))
            # Skip degenerate or non-positive area boxes
            if not (x2 > x1 and y2 > y1):
                continue

            boxes.append([x1, y1, x2, y2])
            # Map all apple instances to class 1
            labels.append(1)
            # Recompute area after clamping to ensure consistency
            areas.append(float((x2 - x1) * (y2 - y1)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self._transforms is not None:
            # v2 path: transform image + boxes jointly (geometric augmentation
            # moves the boxes). Boxes are wrapped as a tv_tensor so v2 knows to
            # transform them; area/iscrowd are recomputed afterwards to stay
            # consistent with any boxes dropped by SanitizeBoundingBoxes.
            from torchvision import tv_tensors

            bbx = tv_tensors.BoundingBoxes(
                boxes_t, format="XYXY", canvas_size=(img_h, img_w)
            )
            img, out = self._transforms(img, {"boxes": bbx, "labels": labels_t})
            boxes_out = torch.as_tensor(out["boxes"], dtype=torch.float32)
            labels_out = out["labels"]
            wh = boxes_out[:, 2:] - boxes_out[:, :2] if boxes_out.numel() else boxes_out.new_zeros((0, 2))
            target = {
                "boxes": boxes_out,
                "labels": labels_out,
                "image_id": torch.tensor([idx]),
                "area": (wh[:, 0] * wh[:, 1]) if boxes_out.numel() else torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((labels_out.shape[0],), dtype=torch.int64),
            }
        else:
            # default: keep PIL image; tensor conversion happens in the train loop
            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([idx]),
                "area": torch.as_tensor(areas, dtype=torch.float32) if boxes else torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64) if boxes else torch.zeros((0,), dtype=torch.int64),
            }

        return img, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
