from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


REALSENSE_D435I_COLOR = CameraIntrinsics(
    fx=916.642944335938,
    fy=916.673278808594,
    cx=625.08154296875,
    cy=371.20751953125,
    width=1280,
    height=720,
)


def scaled_to(intr: CameraIntrinsics, width: int, height: int) -> CameraIntrinsics:
    """Scale intrinsics to a different image resolution, assuming the same sensor FOV."""
    if width == intr.width and height == intr.height:
        return intr
    sx = width / intr.width
    sy = height / intr.height
    return CameraIntrinsics(
        fx=intr.fx * sx,
        fy=intr.fy * sy,
        cx=intr.cx * sx,
        cy=intr.cy * sy,
        width=width,
        height=height,
    )


def backproject(u: float, v: float, z: float, intr: CameraIntrinsics) -> tuple[float, float, float]:
    """Back-project a pixel + depth into camera-frame 3D coordinates (metres)."""
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return x, y, z


def width_from_bbox(box_width_px: float, z: float, intr: CameraIntrinsics) -> float:
    """Real-world width (metres) of an object spanning box_width_px pixels at depth z."""
    return box_width_px * z / intr.fx
