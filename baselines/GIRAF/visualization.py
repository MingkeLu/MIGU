"""PLY point-cloud visualization for GIRAF results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .point_cloud import _back_project, _ray_geometry, _write_binary_ply
from .model import Intrinsics


_CENTER_COLORS = np.asarray([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [148, 103, 189],
    [140, 86, 75], [227, 119, 194], [188, 189, 34], [23, 190, 207],
    [255, 187, 120], [152, 223, 138], [197, 176, 213], [247, 182, 210],
], dtype=np.uint8)


def _sphere_geometry(center: np.ndarray, radius: float, color: np.ndarray,
                     count: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Create a dense colored sphere that remains visible in ordinary PLY viewers."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(1.0, -1.0, count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = golden_angle * np.arange(count)
    points = np.asarray(center, dtype=np.float64) + radius * np.stack(
        (radial * np.cos(angle), y, radial * np.sin(angle)), axis=1
    )
    return points, np.tile(np.asarray(color, dtype=np.uint8), (count, 1))


def export_giraf_point_cloud(
    rgb_path: str | Path,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    objects: Sequence[dict],
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    output_path: str | Path,
    *,
    depth_scale: float = 0.001,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
    stride: int = 2,
    ray_length_m: float = 1.0,
    ray_radius_m: float = 0.006,
    center_radius_m: float = 0.018,
) -> Path:
    """Export RGB points, center markers, and the GIRAF ray as binary PLY."""
    if stride < 1:
        raise ValueError("visualization stride must be at least 1")
    if min(depth_scale, ray_length_m, ray_radius_m, center_radius_m) <= 0:
        raise ValueError("visualization scales, lengths, and radii must be positive")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth_m = np.asarray(depth, dtype=np.float64) * float(depth_scale)
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(f"RGB/depth shapes differ: {rgb.shape[:2]} vs {depth_m.shape}")

    points_image = _back_project(
        depth_m, (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    )
    sampled_depth = depth_m[::stride, ::stride]
    valid = (np.isfinite(sampled_depth) & (sampled_depth > min_depth_m)
             & (sampled_depth < max_depth_m))
    point_parts = [points_image[::stride, ::stride][valid]]
    color_parts = [rgb[::stride, ::stride][valid]]

    ray_points, ray_colors = _ray_geometry(
        np.asarray(ray_origin, dtype=np.float64),
        np.asarray(ray_direction, dtype=np.float64),
        float(ray_length_m), float(ray_radius_m),
    )
    point_parts.append(ray_points)
    color_parts.append(ray_colors)

    center_labels = []
    for index, item in enumerate(objects):
        color = _CENTER_COLORS[index % len(_CENTER_COLORS)]
        center = item.get("center_3d")
        center_labels.append({
            "object_id": str(index + 1),
            "giraf_index": index,
            "name": str(item["name"]),
            "center_3d": None if center is None else np.asarray(center, dtype=float).tolist(),
            "color_rgb": color.tolist(),
        })
        if center is not None:
            center_points, center_colors = _sphere_geometry(
                np.asarray(center, dtype=np.float64), float(center_radius_m), color
            )
            point_parts.append(center_points)
            color_parts.append(center_colors)

    destination = Path(output_path).resolve()
    _write_binary_ply(destination, np.vstack(point_parts), np.vstack(color_parts))
    destination.with_name(f"{destination.stem}_centers.json").write_text(
        json.dumps(center_labels, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
