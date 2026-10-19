"""Minimal binary PLY helpers used by GIRAF visualization."""
from pathlib import Path
import numpy as np

def _back_project(
    depth_m: np.ndarray, intrinsics: tuple[float, float, float, float]
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    v, u = np.indices(depth_m.shape, dtype=np.float64)
    return np.stack(
        ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m), axis=-1
    )


def _ray_geometry(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Gesture ray direction must be non-zero and finite")
    direction /= norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ direction)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    side_1 = np.cross(direction, reference)
    side_1 /= np.linalg.norm(side_1)
    side_2 = np.cross(direction, side_1)

    t = np.linspace(0.0, length, max(80, int(length / 0.004)))
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    rings = (
        origin[None, None, :]
        + t[:, None, None] * direction[None, None, :]
        + radius
        * (np.cos(angles)[None, :, None] * side_1 + np.sin(angles)[None, :, None] * side_2)
    ).reshape(-1, 3)

    # A deterministic green sphere makes the ray origin easy to locate.
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    sphere_count = 300
    y = np.linspace(1.0, -1.0, sphere_count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = golden_angle * np.arange(sphere_count)
    sphere = origin + (radius * 2.5) * np.stack(
        (radial * np.cos(angle), y, radial * np.sin(angle)), axis=1
    )
    points = np.vstack((rings, sphere))
    colors = np.vstack(
        (
            np.tile(np.array([[255, 35, 35]], dtype=np.uint8), (rings.shape[0], 1)),
            np.tile(np.array([[30, 255, 60]], dtype=np.uint8), (sphere.shape[0], 1)),
        )
    )
    return points, colors


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    records = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    records["x"], records["y"], records["z"] = points.T
    records["r"], records["g"], records["b"] = colors.T
    with path.open("wb") as file:
        file.write(header)
        file.write(records.tobytes())
