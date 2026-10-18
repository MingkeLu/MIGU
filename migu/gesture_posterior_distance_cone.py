"""Pixel-wise spatial pointing likelihood using a distance-dependent cone."""

from __future__ import annotations

import numpy as np

from .gesture_posterior import CameraIntrinsics, depth_to_points, logsumexp, normalize_vectors


def pixel_posterior_from_gesture(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    sigma_0: float = 0.005,
    distance_rate: float = 0.0025,
    occlusion_factor: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> dict[str, np.ndarray]:
    """Evaluate the conic Gaussian pointing likelihood at every RGB-D pixel.

    All distances are in metres.  For a 3-D point ``x``, ``pd`` is its signed
    projection on the pointing ray and ``d`` is its perpendicular distance to
    the corresponding infinite line.  Only points with ``pd >= 0`` are valid.
    The variance is ``sigma_0**2 + distance_rate * pd``.

    ``occlusion_factor`` supplies theta for callers that have an occlusion
    model.  The current mask pipeline has no pairwise occlusion probabilities,
    so omitting it applies theta=1.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H, W], got {depth.shape}")
    if (isinstance(sigma_0, bool) or not isinstance(sigma_0, (int, float))
            or not np.isfinite(sigma_0) or sigma_0 <= 0):
        raise ValueError("sigma_0 must be finite and positive.")
    if (isinstance(distance_rate, bool) or not isinstance(distance_rate, (int, float))
            or not np.isfinite(distance_rate) or distance_rate < 0):
        raise ValueError("distance_rate must be finite and non-negative.")

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
    )
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != depth.shape:
            raise ValueError(f"valid_mask must have shape {depth.shape}, got {supplied_mask.shape}")
        current_valid_mask &= supplied_mask

    theta = np.ones(depth.shape, dtype=np.float64)
    if occlusion_factor is not None:
        theta = np.asarray(occlusion_factor, dtype=np.float64)
        if theta.shape != depth.shape:
            raise ValueError(
                f"occlusion_factor must have shape {depth.shape}, got {theta.shape}"
            )
        if not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > 1)):
            raise ValueError("occlusion_factor must contain finite values in [0, 1].")
        current_valid_mask &= theta > 0
    if not np.any(current_valid_mask):
        raise ValueError("No valid depth pixels are available for the distance-cone posterior.")

    variance = float(sigma_0) ** 2 + float(distance_rate) * projected_distance
    log_likelihood = np.full(depth.shape, -np.inf, dtype=np.float64)
    valid_variance = variance[current_valid_mask]
    log_likelihood[current_valid_mask] = (
        np.log(theta[current_valid_mask])
        - 0.5 * np.log(2.0 * np.pi * valid_variance)
        - line_distance[current_valid_mask] ** 2 / (2.0 * valid_variance)
    )
    log_normalizer = logsumexp(log_likelihood[current_valid_mask])
    log_probability = np.full(depth.shape, -np.inf, dtype=np.float64)
    log_probability[current_valid_mask] = log_likelihood[current_valid_mask] - log_normalizer
    probability = np.zeros(depth.shape, dtype=np.float64)
    probability[current_valid_mask] = np.exp(log_probability[current_valid_mask])

    return {
        "probability": probability,
        "log_probability": log_probability,
        "log_likelihood": log_likelihood,
        "valid_mask": current_valid_mask,
        "points": points,
        "projected_distance": projected_distance,
        "line_distance": line_distance,
        "variance": variance,
        "fitting_covariance": np.zeros((3, 3), dtype=np.float64),
    }
