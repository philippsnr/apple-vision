from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T

from apple_vision import camera
from apple_vision.models.detector import create_model

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
_DETECTOR_GLOB = "fasterrcnn_*.pth"

app = FastAPI(title="Apple Vision API", description="Detect apples in RGB-D images")


class Apple(BaseModel):
    x: float
    y: float
    z: float
    width: float


class DetectResponse(BaseModel):
    model: str
    count: int
    apples: list[Apple]


class ModelsResponse(BaseModel):
    models: list[str]
    default: Optional[str]

_model_cache: dict[str, torch.nn.Module] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_checkpoints() -> list[str]:
    return sorted(p.name for p in CHECKPOINT_DIR.glob(_DETECTOR_GLOB))


def _default_checkpoint() -> Optional[str]:
    checkpoints = _list_checkpoints()
    if not checkpoints:
        return None
    for name in checkpoints:
        if "best" in name:
            return name
    return checkpoints[0]


def _get_model(checkpoint_name: str) -> torch.nn.Module:
    if checkpoint_name in _model_cache:
        return _model_cache[checkpoint_name]
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_name}")
    model = create_model(num_classes=2, pretrained=False)
    state = torch.load(ckpt_path, map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model_cache[checkpoint_name] = model
    return model


def _median_depth(depth_arr: np.ndarray, box: list[float], scale: float) -> Optional[float]:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = depth_arr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = depth_arr[y1:y2, x1:x2].astype(np.float32) * scale
    valid = crop[crop > 0]
    return float(np.median(valid)) if valid.size > 0 else None


@app.get("/models", response_model=ModelsResponse, summary="List available detector checkpoints")
def list_models():
    checkpoints = _list_checkpoints()
    return {"models": checkpoints, "default": _default_checkpoint()}


@app.post("/detect", response_model=DetectResponse, summary="Detect apples and return their 3D positions")
async def detect(
    rgb: UploadFile = File(..., description="RGB image (JPEG or PNG)"),
    depth: UploadFile = File(..., description="Depth image aligned with RGB (16-bit PNG; values in depth_scale units)"),
    score_threshold: float = Form(0.5, description="Minimum detection confidence [0, 1]"),
    depth_scale: float = Form(0.001, description="Multiplier to convert depth pixel values to metres (default: 0.001 for millimetres)"),
):
    checkpoint = _default_checkpoint()
    if checkpoint is None:
        raise HTTPException(status_code=503, detail="No detector checkpoints found in checkpoints/")

    detector = _get_model(checkpoint)

    rgb_bytes = await rgb.read()
    rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")

    depth_bytes = await depth.read()
    depth_arr = np.array(Image.open(io.BytesIO(depth_bytes)))

    intrinsics = camera.scaled_to(camera.REALSENSE_D435I_COLOR, rgb_img.width, rgb_img.height)

    img_tensor = T.ToTensor()(rgb_img).to(_device)
    with torch.no_grad():
        output = detector([img_tensor])[0]

    boxes = output.get("boxes", torch.empty((0, 4))).cpu().numpy()
    scores = output.get("scores", torch.empty((0,))).cpu().numpy()

    apples = []
    for box, score in zip(boxes, scores):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        z = _median_depth(depth_arr, box.tolist(), depth_scale)
        if z is None:
            continue
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        x, y, z = camera.backproject(u, v, z, intrinsics)
        width = camera.width_from_bbox(x2 - x1, z, intrinsics)
        apples.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "width": round(width, 3),
        })

    return {"model": checkpoint, "count": len(apples), "apples": apples}
