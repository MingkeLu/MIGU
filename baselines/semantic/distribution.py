from __future__ import annotations

import numpy as np


def gaussian_kernel_sum(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    sigma_fraction: float = 0.035,
) -> dict:
    """Represent every sample by one Gaussian and add them with equal weight."""
    if not points:
        raise ValueError("At least one point is required")
    if sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be positive")

    sigma_x = width * sigma_fraction
    sigma_y = height * sigma_fraction
    covariance = [[sigma_x**2, 0.0], [0.0, sigma_y**2]]
    weight = 1.0 / len(points)
    kernels = [
        {
            "weight": weight,
            "mean": [float(x), float(y)],
            "covariance": covariance,
        }
        for x, y in points
    ]
    return {
        "type": "gaussian_kernel_sum",
        "combination": "weighted_sum",
        "coordinate_system": "grounding_frame_pixels_xy_top_left",
        "mean_units": "pixels",
        "covariance_units": "pixels_squared",
        "sigma_fraction": sigma_fraction,
        "sigma_pixels": [sigma_x, sigma_y],
        "n_kernels": len(kernels),
        "kernels": kernels,
    }


def density_map(distribution: dict, width: int, height: int) -> np.ndarray:
    """Evaluate the summed Gaussian kernels on the complete pixel grid."""
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    density = np.zeros((height, width), dtype=np.float32)
    sigma_x, sigma_y = distribution["sigma_pixels"]
    for kernel in distribution["kernels"]:
        mean_x, mean_y = kernel["mean"]
        exponent = -0.5 * (
            ((xs - mean_x) / sigma_x) ** 2 + ((ys - mean_y) / sigma_y) ** 2
        )
        density += float(kernel["weight"]) * np.exp(exponent)
    return density


def top_density_peaks(
    distribution: dict,
    width: int,
    height: int,
    count: int = 3,
) -> list[dict]:
    """Extract separated peaks from the summed density with non-max suppression."""
    density = density_map(distribution, width, height)
    working = density.copy()
    sigma_x, sigma_y = distribution["sigma_pixels"]
    suppress_x = max(1, round(1.5 * sigma_x))
    suppress_y = max(1, round(1.5 * sigma_y))
    peaks = []
    for rank in range(1, count + 1):
        flat_index = int(np.argmax(working))
        value = float(working.flat[flat_index])
        if value <= 0:
            break
        y, x = np.unravel_index(flat_index, working.shape)
        peaks.append({"x": float(x), "y": float(y), "density": value, "rank": rank})
        x0, x1 = max(0, x - suppress_x), min(width, x + suppress_x + 1)
        y0, y1 = max(0, y - suppress_y), min(height, y + suppress_y + 1)
        local_x = np.arange(x0, x1)[None, :]
        local_y = np.arange(y0, y1)[:, None]
        mask = ((local_x - x) / suppress_x) ** 2 + (
            (local_y - y) / suppress_y
        ) ** 2 <= 1
        region = working[y0:y1, x0:x1]
        region[mask] = 0
    return peaks


def sample_point_kernels(
    points: list[tuple[float, float]],
    count: int,
    width: int,
    height: int,
    rng: np.random.Generator,
    sigma_fraction: float,
) -> list[tuple[float, float]]:
    """Keep selected points and sample the rest evenly from their Gaussians."""
    if not points:
        raise ValueError("At least one point is required")
    if count <= 0:
        raise ValueError("count must be positive")
    if sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be positive")

    centers = np.asarray(points, dtype=float)
    centers = np.clip(centers, [0.0, 0.0], [width - 1.0, height - 1.0])

    # Elitism prevents an iterative refinement round from losing every point the
    # VLM just selected. The remaining proposal budget is divided evenly among
    # the individual point kernels, without fitting a mixture model.
    retained = centers[: min(len(centers), count)]
    remaining = count - len(retained)
    if remaining == 0:
        return [(float(x), float(y)) for x, y in retained]

    selected = np.resize(np.arange(len(centers)), remaining)
    rng.shuffle(selected)
    noise = rng.normal(
        loc=0.0,
        scale=np.asarray([width * sigma_fraction, height * sigma_fraction]),
        size=(remaining, 2),
    )
    sampled = centers[selected] + noise
    sampled = np.clip(sampled, [0.0, 0.0], [width - 1.0, height - 1.0])
    samples = np.concatenate([retained, sampled], axis=0)
    return [(float(x), float(y)) for x, y in samples]
