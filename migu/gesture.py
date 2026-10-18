"""Gesture-posterior execution and hand-ray visualization helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .gesture_posterior import (
    compute_eye_position, load_camera_intrinsics, load_depth_meters,
    load_fitting_joint_coordinates, load_ray, normalize_vectors,
)


def run_gesture(depth_path: Path, ray_path: Path, intrinsics_path: Path,
                frame_dir: Path, mode: str):
    """Compute and save the configured pixel-wise gesture posterior."""
    depth = load_depth_meters(depth_path)
    intrinsics = load_camera_intrinsics(intrinsics_path)
    finger, direction = load_ray(ray_path)
    eye = compute_eye_position(finger, direction)
    if mode == "gesture_posterior":
        from .gesture_posterior import pixel_posterior_from_gesture
        result = pixel_posterior_from_gesture(
            depth, intrinsics, eye, finger, direction,
            sigma_u=0.009, alpha_depth=0.2,
            fitting_joint_coordinates=load_fitting_joint_coordinates(ray_path, 4),
            fitting_covariance_beta=1.5,
            device="auto",
        )
    elif mode == "vMF":
        from .gesture_posterior_vmf import pixel_posterior_from_gesture
        result = pixel_posterior_from_gesture(
            depth, intrinsics, finger, direction, kappa=20.0, device="auto"
        )
    elif mode == "distance_cone":
        from .gesture_posterior_distance_cone import pixel_posterior_from_gesture
        result = pixel_posterior_from_gesture(depth, intrinsics, finger, direction)
    elif mode in {"angle_error_gaussian", "angular_gaussian"}:
        from .gesture_posterior_angular_gaussian import pixel_posterior_from_gesture
        result = pixel_posterior_from_gesture(
            depth, intrinsics, finger, direction, angular_std=1.0
        )
    elif mode in {"dist_threshold", "distance_threshold"}:
        from .gesture_posterior_distance_threshold import pixel_posterior_from_gesture
        result = pixel_posterior_from_gesture(
            depth, intrinsics, finger, direction, distance_threshold=0.20
        )
    else:
        raise ValueError(f"Unsupported gesture mode: {mode}")

    path = frame_dir / f"{ray_path.stem}_posterior.npz"
    np.savez_compressed(
        path,
        probability=result["probability"],
        log_probability=result["log_probability"],
        valid_mask=result["valid_mask"],
        eye_position=eye,
        finger_position=finger,
        observed_direction=normalize_vectors(direction),
        fitting_covariance=result["fitting_covariance"],
        posterior=np.asarray(mode),
        kappa=np.asarray(20.0, dtype=np.float64),
    )
    return result["probability"], path


def load_hand_ray_overlay(ray_path: Path, intrinsics_path: Path) -> dict:
    """Load the fitted ray and project a visible line into image coordinates."""
    ray = json.loads(ray_path.read_text(encoding="utf-8"))
    intrinsics = load_camera_intrinsics(intrinsics_path)
    origin = np.asarray(ray["origin"], dtype=float)
    direction = normalize_vectors(np.asarray(ray["direction"], dtype=float))
    points = np.stack([origin, origin + direction * 1.5])
    valid = points[:, 2] > 1e-6
    pixels = np.empty((2, 2), dtype=float)
    pixels[:, 0] = intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx
    pixels[:, 1] = intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy
    return {
        "points": pixels,
        "valid": valid,
        "landmarks": ray.get("hand_landmarks_px", []),
    }


def draw_hand_ray_overlay(image: Image.Image, overlay: dict) -> Image.Image:
    """Draw detected hand landmarks and the projected pointing ray."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    radius = max(2, round(min(canvas.size) / 240))
    for u, v in overlay.get("landmarks", []):
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(0, 255, 255))
    points = np.asarray(overlay["points"])
    if np.all(overlay["valid"]) and np.all(np.isfinite(points)):
        draw.line(tuple(map(tuple, points)), fill=(255, 40, 40), width=max(3, radius))
    return canvas
