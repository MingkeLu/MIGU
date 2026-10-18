"""Depth-assisted pointing-ray utilities used by the ROS interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0


INDEX_FINGER_INDICES = (5, 6, 7, 8)


def depth_at_pixel_robust(
    depth: np.ndarray,
    u: int,
    v: int,
    *,
    win: int = 5,
    percentile: float = 40.0,
    depth_scale: float = 1.0,
) -> float | None:
    """Return robust local depth in metres, or ``None`` when unavailable."""
    height, width = depth.shape[:2]
    u0, v0 = max(0, u - win // 2), max(0, v - win // 2)
    u1, v1 = min(width, u + win // 2 + 1), min(height, v + win // 2 + 1)
    patch = np.asarray(depth[v0:v1, u0:u1], dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if not valid.size:
        return None
    return float(np.percentile(valid, percentile) / depth_scale)


def unproject(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project one RGB-D pixel into the optical camera frame."""
    return np.asarray(
        [
            (u - intrinsics.cx) * z / intrinsics.fx,
            (v - intrinsics.cy) * z / intrinsics.fy,
            z,
        ],
        dtype=np.float32,
    )


def estimate_index_finger_ray(
    landmarks_px: Sequence[tuple[int, int]],
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    win: int = 5,
    percentile: float = 40.0,
    ensure_forward: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    """Fit a directed PCA ray through the four MediaPipe index-finger joints."""
    ranked_points: list[tuple[int, np.ndarray]] = []
    for index in INDEX_FINGER_INDICES:
        u, v = landmarks_px[index]
        z = depth_at_pixel_robust(
            depth, u, v, win=win, percentile=percentile,
            depth_scale=intrinsics.depth_scale,
        )
        if z is not None:
            ranked_points.append((index, unproject(u, v, z, intrinsics)))
    if len(ranked_points) < 2:
        return None

    points = np.stack([point for _, point in ranked_points])
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    direction = axes[0]
    direction /= np.linalg.norm(direction) + 1e-9
    ranks = np.asarray([index for index, _ in ranked_points], dtype=np.float32)
    projections = (points - center) @ direction
    slope, _ = np.linalg.lstsq(
        np.vstack([ranks, np.ones_like(ranks)]).T, projections, rcond=None
    )[0]
    if slope < 0:
        direction = -direction

    distal = [point for index, point in ranked_points if index in (7, 8)]
    origin = np.mean(np.stack(distal), axis=0) if distal else center
    if ensure_forward and np.dot(direction, origin) < 0:
        direction = -direction
    return origin.astype(np.float32), direction.astype(np.float32), list(points)
