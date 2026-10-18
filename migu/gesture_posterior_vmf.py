"""Pixel-wise gesture posterior based on a von Mises--Fisher distribution."""

from __future__ import annotations

import numpy as np

from .gesture_posterior import CameraIntrinsics, depth_to_points, logsumexp, normalize_vectors


def pixel_posterior_from_gesture(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    kappa: float = 20.0,
    valid_mask: np.ndarray | None = None,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Evaluate a vMF direction density from ``ray_origin`` at every RGB-D pixel.

    The fitted ray direction is the vMF mean axis.  Since all pixels share the
    same concentration and normalizing constant, discrete pixel normalization
    only requires the stable softmax of ``kappa * dot(axis, direction)``.
    ``device='auto'`` uses CUDA when available; the public result remains NumPy.
    """
    depth = np.asarray(depth, dtype=np.float64)
    device = str(device).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H, W], got {depth.shape}")
    if (not isinstance(kappa, (int, float)) or isinstance(kappa, bool)
            or not np.isfinite(kappa) or kappa < 0):
        raise ValueError("kappa must be finite and non-negative.")

    origin = np.asarray(ray_origin, dtype=np.float64)
    axis = np.asarray(ray_direction, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("ray_origin must be a finite vector with shape [3].")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-12:
        raise ValueError("ray_direction must be a finite non-zero vector with shape [3].")
    axis = normalize_vectors(axis)
    if valid_mask is not None and np.asarray(valid_mask).shape != depth.shape:
        raise ValueError(
            f"valid_mask must have shape {depth.shape}, got {np.asarray(valid_mask).shape}"
        )

    if device != "cpu":
        from .gesture_torch import resolve_device, vmf_posterior
        torch, resolved_device = resolve_device(device)
        if resolved_device is not None:
            return vmf_posterior(
                torch, resolved_device, depth, intrinsics, origin, axis, float(kappa),
                valid_mask, min_depth, max_depth,
            )

    points = depth_to_points(depth, intrinsics)
    offsets = points - origin
    distances = np.linalg.norm(offsets, axis=-1)
    current_valid_mask = (
        np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        & np.all(np.isfinite(points), axis=-1) & (distances > 1e-12)
    )
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != depth.shape:
            raise ValueError(f"valid_mask must have shape {depth.shape}, got {valid_mask.shape}")
        current_valid_mask &= valid_mask
    if not np.any(current_valid_mask):
        raise ValueError("No valid depth pixels are available for the vMF posterior.")

    candidate_directions = offsets / np.maximum(distances[..., None], 1e-12)
    log_likelihood = np.full(depth.shape, -np.inf, dtype=np.float64)
    log_likelihood[current_valid_mask] = float(kappa) * np.einsum(
        "...i,i->...", candidate_directions[current_valid_mask], axis
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
        "nominal_direction": candidate_directions,
        "fitting_covariance": np.zeros((3, 3), dtype=np.float64),
    }
