"""Uniform cylindrical gesture posterior around the pointing ray."""

from __future__ import annotations

import numpy as np

from .gesture_posterior import CameraIntrinsics, depth_to_points, normalize_vectors


def pixel_posterior_from_gesture(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    distance_threshold: float = 0.20,
    valid_mask: np.ndarray | None = None,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> dict[str, np.ndarray]:
    """Assign equal mass to pixels inside a forward 20 cm-radius cylinder."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H, W], got {depth.shape}")
    if (isinstance(distance_threshold, bool)
            or not isinstance(distance_threshold, (int, float))
            or not np.isfinite(distance_threshold) or distance_threshold <= 0):
        raise ValueError("distance_threshold must be finite and positive.")

    origin = np.asarray(ray_origin, dtype=np.float64)
    axis = np.asarray(ray_direction, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("ray_origin must be a finite vector with shape [3].")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-12:
        raise ValueError("ray_direction must be a finite non-zero vector with shape [3].")
    axis = normalize_vectors(axis)

    points = depth_to_points(depth, intrinsics)
    offsets = points - origin
    projected_distance = np.einsum("...i,i->...", offsets, axis)
    perpendicular = offsets - projected_distance[..., None] * axis
    line_distance = np.linalg.norm(perpendicular, axis=-1)
    current_valid_mask = (
        np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        & np.all(np.isfinite(points), axis=-1) & (projected_distance >= 0)
        & (line_distance <= float(distance_threshold))
    )
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != depth.shape:
            raise ValueError(f"valid_mask must have shape {depth.shape}, got {supplied_mask.shape}")
        current_valid_mask &= supplied_mask
    count = int(current_valid_mask.sum())
    if count == 0:
        raise ValueError("No valid depth pixels are inside the distance-threshold cylinder.")

    probability = np.zeros(depth.shape, dtype=np.float64)
    probability[current_valid_mask] = 1.0 / count
    log_probability = np.full(depth.shape, -np.inf, dtype=np.float64)
    log_probability[current_valid_mask] = -np.log(count)
    log_likelihood = np.full(depth.shape, -np.inf, dtype=np.float64)
    log_likelihood[current_valid_mask] = 0.0
    return {
        "probability": probability,
        "log_probability": log_probability,
        "log_likelihood": log_likelihood,
        "valid_mask": current_valid_mask,
        "points": points,
        "projected_distance": projected_distance,
        "line_distance": line_distance,
        "fitting_covariance": np.zeros((3, 3), dtype=np.float64),
    }
