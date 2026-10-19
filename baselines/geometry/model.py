"""Pure geometry and likelihood functions for the pointing baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def mask_to_point_cloud(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
    mad_scale: float = 6.0,
) -> np.ndarray:
    """Back-project valid masked pixels, with robust per-object depth filtering."""
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError(f"Mask/depth shapes differ: {mask.shape} vs {depth.shape}")
    finite_depth = np.isfinite(depth)
    valid_range = np.zeros(depth.shape, dtype=bool)
    valid_range[finite_depth] = ((depth[finite_depth] > min_depth_m)
                                 & (depth[finite_depth] < max_depth_m))
    valid = mask & valid_range
    v, u = np.nonzero(valid)
    if not u.size:
        return np.empty((0, 3), dtype=np.float64)
    z = depth[v, u]
    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    if mad > np.finfo(float).eps and mad_scale > 0:
        # 1.4826 makes MAD comparable to standard deviation for Gaussian noise.
        keep = np.abs(z - median) <= mad_scale * 1.4826 * mad
        u, v, z = u[keep], v[keep], z[keep]
    x = (u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))


def angular_features(
    points: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> tuple[float, float, bool]:
    """Return centroid angle theta, closest-extent angle phi, and validity."""
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite(norm) or norm <= 0:
        raise ValueError("origin and direction must be finite 3-vectors; direction must be nonzero")
    direction = direction / norm
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        return np.nan, np.nan, False

    centroid_delta = points.mean(axis=0) - origin
    centroid_norm = float(np.linalg.norm(centroid_delta))
    centroid_forward = float(centroid_delta @ direction)
    if centroid_norm <= 0 or centroid_forward <= 0:
        return np.nan, np.nan, False
    theta = float(np.arccos(np.clip(centroid_forward / centroid_norm, -1.0, 1.0)))

    relative = points - origin
    projection = relative @ direction
    forward = projection > 0
    if not np.any(forward):
        return theta, np.nan, False
    rel_forward = relative[forward]
    proj_forward = projection[forward]
    perpendicular = rel_forward - proj_forward[:, None] * direction
    closest = rel_forward[int(np.argmin(np.linalg.norm(perpendicular, axis=1)))]
    closest_norm = float(np.linalg.norm(closest))
    if closest_norm <= 0:
        return theta, np.nan, False
    phi = float(np.arccos(np.clip((closest @ direction) / closest_norm, -1.0, 1.0)))
    return theta, phi, bool(np.isfinite(theta) and np.isfinite(phi))


def paper_log_likelihood(
    theta: np.ndarray,
    phi: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    *,
    feature_set: str = "theta_phi",
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute the paper's Eqs. (2)--(5) in the log domain.

    The source paper calls lambda a rate but sets each lambda to the scene-wise
    arithmetic mean of its feature (not the reciprocal mean). Its area density is
    lambda_A * exp(-lambda_A / A).
    """
    theta, phi, area, valid = map(np.asarray, (theta, phi, area, valid))
    if feature_set not in {"theta_phi", "theta_phi_area"}:
        raise ValueError("feature_set must be 'theta_phi' or 'theta_phi_area'")
    usable = valid.astype(bool) & np.isfinite(theta) & np.isfinite(phi) & (area > 0)
    log_likelihood = np.full(theta.shape, -np.inf, dtype=np.float64)
    if not np.any(usable):
        return log_likelihood, {"lambda_theta": np.nan, "lambda_phi": np.nan,
                                "lambda_area": np.nan}
    tiny = np.finfo(np.float64).tiny
    lambda_theta = max(float(np.mean(theta[usable])), tiny)
    lambda_phi = max(float(np.mean(phi[usable])), tiny)
    value = (np.log(lambda_theta) - lambda_theta * theta[usable]
             + np.log(lambda_phi) - lambda_phi * phi[usable])
    lambda_area = float(np.mean(area[usable]))
    if feature_set == "theta_phi_area":
        lambda_area = max(lambda_area, tiny)
        value += np.log(lambda_area) - lambda_area / area[usable]
    log_likelihood[usable] = value
    return log_likelihood, {"lambda_theta": lambda_theta, "lambda_phi": lambda_phi,
                            "lambda_area": lambda_area}


def normalize_log_likelihood(log_likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ordinary likelihoods and a stable softmax posterior."""
    values = np.asarray(log_likelihood, dtype=np.float64)
    finite = np.isfinite(values)
    probability = np.zeros_like(values)
    likelihood = np.zeros_like(values)
    if not np.any(finite):
        if values.size:
            probability.fill(1.0 / values.size)
            likelihood.fill(1.0)
        return likelihood, probability
    maximum = float(np.max(values[finite]))
    relative = np.zeros_like(values)
    relative[finite] = np.exp(values[finite] - maximum)
    probability = relative / relative.sum()
    # Preserve the literal paper likelihood in output. Posterior normalization
    # above does not depend on this potentially underflow-prone conversion.
    likelihood[finite] = np.exp(np.clip(values[finite], np.log(np.finfo(float).tiny),
                                        np.log(np.finfo(float).max)))
    return likelihood, probability
