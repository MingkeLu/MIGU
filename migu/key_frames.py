"""Generate pointing-ray JSON files from paired RGB and depth frames."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np

from .hand_detection import (
    CameraIntrinsics, MediaPipeHandDetector, depth_at_pixel_robust,
    estimate_finger_ray, unproject,
)


FINGERS = (
    ("index", (5, 6, 7, 8)),
    ("pinky", (17, 18, 19, 20)),
    ("thumb", (1, 2, 3, 4)),
)
JOINT_NAMES = ("MCP", "PIP", "DIP", "TIP")


def _load_intrinsics(path: Path) -> CameraIntrinsics:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"fx", "fy", "cx", "cy"}:
                values[key.strip()] = float(value)
    missing = {"fx", "fy", "cx", "cy"} - values.keys()
    if missing:
        raise ValueError(f"Camera intrinsics are missing: {', '.join(sorted(missing))}")
    return CameraIntrinsics(**values, depth_scale=1000.0)


def _select_finger(landmarks: Sequence[tuple[int, int]]) -> tuple[str, tuple[int, ...], dict]:
    wrist = np.asarray(landmarks[0], dtype=float)
    distances = {
        name: float(np.linalg.norm(np.asarray(landmarks[indices[-1]], dtype=float) - wrist))
        for name, indices in FINGERS
    }
    name, indices = max(FINGERS, key=lambda item: distances[item[0]])
    return name, indices, distances


def _build_ray(landmarks: Sequence[tuple[int, int]], depth: np.ndarray,
               intrinsics: CameraIntrinsics) -> dict | None:
    name, indices, distances = _select_finger(landmarks)
    diagnostics = {}
    fitted = estimate_finger_ray(
        landmarks, depth.astype(np.float32), intrinsics,
        landmark_indices=indices, fitting_joint_count=4,
        filtering="outlier", diagnostics=diagnostics,
    )
    if fitted is None:
        return None
    origin, direction, used_points = fitted
    finger_landmarks = []
    for index, joint_name in zip(indices, JOINT_NAMES):
        u, v = landmarks[index]
        z = depth_at_pixel_robust(
            depth.astype(np.float32), u, v, depth_scale=intrinsics.depth_scale
        )
        finger_landmarks.append({
            "landmark_index": index,
            "name": joint_name,
            "pixel": [int(u), int(v)],
            "depth": z,
            "point_3d": unproject(u, v, z, intrinsics).tolist() if z is not None else None,
        })
    return {
        "origin": origin.tolist(),
        "direction": direction.tolist(),
        "hand_landmarks_px": [[int(u), int(v)] for u, v in landmarks],
        "index_finger_landmarks": finger_landmarks,
        "selected_pointing_finger": name,
        "palm_root_to_fingertip_distances_px": distances,
        "exchange_enabled": True,
        "used_points_count": len(used_points),
        "sampled_points_count": diagnostics["sampled_points_count"],
        "fitting_mode": "consecutive",
        "filtering": "outlier",
        "fitting_points_3d": [point.tolist() for point in used_points],
        "timestamp_rgb": None,
        "timestamp_depth": None,
        "depth_match_dt": None,
        "hand_detector": "mediapipe",
    }


def generate_key_frames(rgb_dir: Path, depth_dir: Path, output_dir: Path,
                        intrinsics_path: Path, hand_config: dict) -> dict:
    """Generate all available pointing rays and return processing statistics."""
    intrinsics = _load_intrinsics(intrinsics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = {path.stem: path for path in depth_dir.glob("*.png")}
    stats = {
        "saved": 0, "missing_depth": 0, "no_hand_or_invalid_depth": 0,
        "unreadable": 0, "processing_time_seconds_by_frame": {},
    }
    detector = MediaPipeHandDetector(
        min_detection_confidence=float(hand_config["min_detection_confidence"]),
        min_tracking_confidence=float(hand_config["min_tracking_confidence"]),
    )
    try:
        for rgb_path in sorted(rgb_dir.glob("*.png")):
            started = time.perf_counter()
            frame_id = rgb_path.stem
            try:
                depth_path = depths.get(frame_id)
                if depth_path is None:
                    stats["missing_depth"] += 1
                    continue
                bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if bgr is None or depth is None or depth.ndim != 2:
                    stats["unreadable"] += 1
                    continue
                hand = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ray = _build_ray(hand["landmarks_px"], depth, intrinsics) if hand else None
                if ray is None:
                    stats["no_hand_or_invalid_depth"] += 1
                    continue
                (output_dir / f"{frame_id}.json").write_text(
                    json.dumps(ray, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                stats["saved"] += 1
            finally:
                stats["processing_time_seconds_by_frame"][frame_id] = (
                    time.perf_counter() - started
                )
    finally:
        detector.close()
    return stats
