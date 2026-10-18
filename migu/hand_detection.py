"""MediaPipe hand detection and 3-D pointing-ray estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0


class MediaPipeHandDetector:
    """Detect one hand with MediaPipe's legacy Hands API."""

    def __init__(self, *, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as error:
            raise ImportError(
                "MediaPipe hand detection requires mediapipe==0.10.21"
            ) from error
        if not hasattr(mp, "solutions"):
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "MIGU requires the MediaPipe legacy solutions API; installed "
                f"version {version!r} does not expose it. Install mediapipe==0.10.21."
            )
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, rgb: np.ndarray) -> dict:
        """Return 21 normalized and pixel landmarks, or an empty mapping."""
        height, width = rgb.shape[:2]
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return {}
        landmarks = result.multi_hand_landmarks[0].landmark
        return {
            "landmarks_norm": [(point.x, point.y, point.z) for point in landmarks],
            "landmarks_px": [
                (int(point.x * width), int(point.y * height)) for point in landmarks
            ],
            "image_size": (height, width),
        }


def depth_at_pixel_robust(depth: np.ndarray, u: int, v: int, *, win: int = 5,
                          percentile: float = 40.0,
                          depth_scale: float = 1.0) -> float | None:
    """Return a robust local depth in metres, or ``None`` when unavailable."""
    height, width = depth.shape[:2]
    u0, v0 = max(0, u - win // 2), max(0, v - win // 2)
    u1, v1 = min(width, u + win // 2 + 1), min(height, v + win // 2 + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if not valid.size:
        return None
    return float(np.percentile(valid, percentile) / depth_scale)


def unproject(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project a pixel and metric depth into camera coordinates."""
    return np.asarray([
        (u - intrinsics.cx) * z / intrinsics.fx,
        (v - intrinsics.cy) * z / intrinsics.fy,
        z,
    ], dtype=np.float32)


def estimate_finger_ray(
    landmarks_px: Sequence[tuple[int, int]],
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    landmark_indices: Sequence[int],
    fitting_joint_count: int = 4,
    filtering: str = "outlier",
    diagnostics: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    """Fit a PCA ray to consecutive depth samples along the selected finger."""
    selected = list(landmark_indices)[-fitting_joint_count:]
    samples: list[tuple[float, int, int]] = []
    for segment_index, (start_index, end_index) in enumerate(zip(selected[:-1], selected[1:])):
        start = np.asarray(landmarks_px[start_index], dtype=np.float64)
        end = np.asarray(landmarks_px[end_index], dtype=np.float64)
        steps = max(1, int(np.ceil(np.linalg.norm(end - start))))
        for step in range(steps + 1):
            if segment_index > 0 and step == 0:
                continue
            fraction = step / steps
            u, v = np.rint(start + fraction * (end - start)).astype(int)
            samples.append((segment_index + fraction, int(u), int(v)))

    ranked_points: list[tuple[float, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for rank, u, v in samples:
        if (u, v) in seen:
            continue
        seen.add((u, v))
        z = depth_at_pixel_robust(
            depth, u, v, depth_scale=intrinsics.depth_scale
        )
        if z is not None:
            ranked_points.append((rank, unproject(u, v, z, intrinsics)))
    if diagnostics is not None:
        diagnostics["sampled_points_count"] = len(ranked_points)
    if len(ranked_points) < 2:
        return None

    if filtering == "outlier":
        depths = np.asarray([point[2] for _, point in ranked_points])
        maximum_depth = float(depths.mean()) + 0.10
        ranked_points = [
            item for item, depth_value in zip(ranked_points, depths)
            if depth_value <= maximum_depth
        ]
    elif filtering != "none":
        raise ValueError("filtering must be 'none' or 'outlier'")
    if len(ranked_points) < 2:
        return None
    if diagnostics is not None:
        diagnostics["used_points_count"] = len(ranked_points)

    points = [point for _, point in ranked_points]
    matrix = np.stack(points)
    center = matrix.mean(axis=0)
    _, _, axes = np.linalg.svd(matrix - center, full_matrices=False)
    direction = axes[0]
    direction /= np.linalg.norm(direction) + 1e-9
    ranks = np.asarray([rank for rank, _ in ranked_points], dtype=np.float32)
    projections = np.asarray([
        np.dot(point - center, direction) for _, point in ranked_points
    ], dtype=np.float32)
    slope, _ = np.linalg.lstsq(
        np.vstack([ranks, np.ones_like(ranks)]).T, projections, rcond=None
    )[0]
    if slope < 0:
        direction = -direction
    # The fitted direction uses filtered samples, while the ray origin follows
    # the method definition: the mean of the two distal joint positions. Keeping
    # these roles separate prevents depth filtering from shifting the fingertip
    # origin toward the camera.
    tail = []
    for index in selected[-2:]:
        u, v = landmarks_px[index]
        z = depth_at_pixel_robust(
            depth, u, v, depth_scale=intrinsics.depth_scale
        )
        if z is not None:
            tail.append(unproject(u, v, z, intrinsics))
    origin = np.mean(np.stack(tail), axis=0) if tail else center
    return origin.astype(np.float32), direction.astype(np.float32), points
