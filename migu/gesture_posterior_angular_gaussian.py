"""Rotationally symmetric angular-Gaussian gesture posterior."""

from __future__ import annotations

import numpy as np

from .gesture_posterior import CameraIntrinsics, depth_to_points, logsumexp, normalize_vectors


def pixel_posterior_from_gesture(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    angular_std: float = 1.0,
    valid_mask: np.ndarray | None = None,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> dict[str, np.ndarray]:
    """Evaluate ``N(angle; 0, angular_std**2)`` around the pointing axis.

    The angular error is the unsigned angle in ``[0, pi]`` between the fitted
    pointing direction and the direction from the ray origin to each RGB-D
    point.  Consequently the likelihood is rotationally symmetric about the
    pointing axis.  The returned probabilities are normalized over valid
    image pixels.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H, W], got {depth.shape}")
    if (isinstance(angular_std, bool) or not isinstance(angular_std, (int, float))
            or not np.isfinite(angular_std) or angular_std <= 0):
        raise ValueError("angular_std must be finite and positive.")

    origin = np.asarray(ray_origin, dtype=np.float64)
    axis = np.asarray(ray_direction, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("ray_origin must be a finite vector with shape [3].")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-12:
        raise ValueError("ray_direction must be a finite non-zero vector with shape [3].")
    axis = normalize_vectors(axis)

    points = depth_to_points(depth, intrinsics)
    offsets = points - origin
    distances = np.linalg.norm(offsets, axis=-1)
    current_valid_mask = (
        np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        & np.all(np.isfinite(points), axis=-1) & (distances > 1e-12)
    )
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != depth.shape:
            raise ValueError(f"valid_mask must have shape {depth.shape}, got {supplied_mask.shape}")
        current_valid_mask &= supplied_mask
    if not np.any(current_valid_mask):
        raise ValueError("No valid depth pixels are available for the angular-Gaussian posterior.")

    directions = offsets / np.maximum(distances[..., None], 1e-12)
    cosine = np.clip(np.einsum("...i,i->...", directions, axis), -1.0, 1.0)
    angular_error = np.arccos(cosine)
    log_likelihood = np.full(depth.shape, -np.inf, dtype=np.float64)
    log_likelihood[current_valid_mask] = (
        -0.5 * (angular_error[current_valid_mask] / float(angular_std)) ** 2
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
        "angular_error": angular_error,
        "fitting_covariance": np.zeros((3, 3), dtype=np.float64),
    }
