"""Deterministic GIRAF language filtering and 3-D pointing geometry."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


DEFAULT_SYNONYMS = {
    "cup": {"cup", "mug"},
    "container": {"container", "box", "bin", "bottle", "can", "jar"},
    "tool": {"tool", "implement", "utensil", "instrument"},
    "marker pen": {"marker pen", "marker", "pen"},
    "chalkboard eraser": {"chalkboard eraser", "board eraser", "eraser"},
}


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def mask_to_center_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    depth_scale: float = 1.0,
    min_points: int = 20,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> tuple[np.ndarray | None, int]:
    """Back-project a mask and return its robust median center in metres."""
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError(f"Mask/depth shapes differ: {mask.shape} vs {depth.shape}")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    if min_points < 1:
        raise ValueError("min_points must be positive")
    depth_m = depth * float(depth_scale)
    valid = (mask & np.isfinite(depth_m) & (depth_m > min_depth_m)
             & (depth_m < max_depth_m))
    rows, columns = np.nonzero(valid)
    if len(rows) < min_points:
        return None, int(len(rows))
    z = depth_m[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < min_points:
        return None, int(len(points))
    return np.median(points, axis=0), int(len(points))


def build_pointing_ray(
    pip_3d: np.ndarray, fingertip_3d: np.ndarray, *, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the GIRAF ray from PIP through TIP, starting at TIP."""
    pip = _vector(pip_3d, "pip_3d")
    tip = _vector(fingertip_3d, "fingertip_3d")
    direction = tip - pip
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < epsilon:
        raise ValueError(
            f"PIP and fingertip are too close to define a ray (distance={norm:.3g} m)"
        )
    return tip, direction / norm


def ray_to_depth_point(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    depth_scale: float = 1.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
    min_forward_m: float = 0.05,
) -> dict | None:
    """Return the observed scene point closest to the forward pointing ray.

    A depth image is a discrete sampling of the visible environment, so an exact
    analytic ray/surface intersection usually does not exist.  The nearest valid
    depth sample is therefore both the intersection estimate and the documented
    fallback when the ray passes between samples.
    """
    depth = np.asarray(depth, dtype=np.float64)
    origin = _vector(origin, "origin")
    direction = _vector(direction, "direction")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("direction must be nonzero")
    direction = direction / norm
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D array")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")

    depth_m = depth * float(depth_scale)
    valid = (np.isfinite(depth_m) & (depth_m > min_depth_m)
             & (depth_m < max_depth_m))
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return None
    z = depth_m[rows, columns]
    points = np.column_stack((
        (columns - intrinsics.cx) * z / intrinsics.fx,
        (rows - intrinsics.cy) * z / intrinsics.fy,
        z,
    ))
    relative = points - origin
    forward = relative @ direction
    keep = np.isfinite(forward) & (forward >= float(min_forward_m))
    if not np.any(keep):
        return None
    points, rows, columns, forward = (
        values[keep] for values in (points, rows, columns, forward)
    )
    perpendicular = relative[keep] - forward[:, None] * direction
    distances = np.linalg.norm(perpendicular, axis=1)
    index = int(np.argmin(distances))
    return {
        "point_3d": points[index],
        "pixel": [int(columns[index]), int(rows[index])],
        "ray_distance": float(distances[index]),
        "forward_distance": float(forward[index]),
    }


def _forms(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
    values = {text}
    for token in text.split():
        values.add(token)
        if token.endswith("ies") and len(token) > 3:
            values.add(token[:-3] + "y")
        elif token.endswith("es") and len(token) > 2:
            values.add(token[:-2])
        elif token.endswith("s") and len(token) > 1:
            values.add(token[:-1])
    return {value for value in values if value}


def lexical_semantic_candidates(
    instruction: str,
    object_names: Sequence[str],
    synonyms: dict[str, set[str]] | None = None,
) -> list[int]:
    """Case-insensitive exact/phrase matching with plurals and synonyms."""
    instruction_forms = _forms(instruction)
    instruction_text = " " + re.sub(r"[^a-z0-9 ]+", " ", instruction.lower()) + " "
    groups = synonyms or DEFAULT_SYNONYMS
    expanded_groups = [{form for item in values | {key} for form in _forms(item)}
                       for key, values in groups.items()]
    selected = []
    for index, name in enumerate(object_names):
        name_forms = _forms(str(name))
        related = set(name_forms)
        for group in expanded_groups:
            if name_forms & group:
                related |= group
        if any((" " + value + " ") in instruction_text for value in related) or related & instruction_forms:
            selected.append(index)
    return selected


def openai_semantic_candidates(
    instruction: str,
    object_names: Sequence[str],
    *,
    model: str = "gpt-5.4-mini",
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[int]:
    """Ask an OpenAI-compatible model which candidate categories fit the instruction."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI semantic filtering requires the 'openai' package") from error
    kwargs = {}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)
    prompt = (
        "Select every candidate object category that can be the semantic target of the "
        "instruction. Ignore pointing geometry. Match case-insensitively, account for simple "
        "singular/plural forms and ordinary synonyms (for example mug/cup). Return JSON only as "
        "{\"indices\": [zero-based integers]}.\n"
        f"Instruction: {instruction!r}\nCandidates: "
        + json.dumps(list(enumerate(map(str, object_names))), ensure_ascii=False)
    )
    response = client.responses.create(
        model=model,
        input=[{"role": "system", "content": "You perform semantic object-category filtering."},
               {"role": "user", "content": prompt}],
        text={"format": {"type": "json_schema", "name": "semantic_candidates",
                         "strict": True, "schema": {"type": "object",
                         "properties": {"indices": {"type": "array", "items": {"type": "integer"}}},
                         "required": ["indices"], "additionalProperties": False}}},
    )
    payload = json.loads(response.output_text)
    return sorted({int(i) for i in payload["indices"] if 0 <= int(i) < len(object_names)})


def get_semantic_candidates(
    instruction: str,
    object_names: Sequence[str],
    *,
    fallback: str = "all",
    provider: str = "openai",
    semantic_filter: Callable[[str, Sequence[str]], list[int]] | None = None,
    **openai_options: Any,
) -> list[int]:
    """Return semantic indices, applying the configured no-match fallback."""
    if fallback not in {"all", "none"}:
        raise ValueError("semantic fallback must be 'all' or 'none'")
    names = [str(name).lower() for name in object_names]
    if semantic_filter is not None:
        selected = semantic_filter(instruction.lower(), names)
    elif provider == "openai":
        selected = openai_semantic_candidates(instruction.lower(), names, **openai_options)
    elif provider == "lexical":
        selected = lexical_semantic_candidates(instruction.lower(), names)
    else:
        raise ValueError("semantic provider must be 'openai' or 'lexical'")
    selected = sorted({int(i) for i in selected if 0 <= int(i) < len(names)})
    return selected if selected or fallback == "none" else list(range(len(names)))


def select_target(
    instruction: str,
    objects: Sequence[dict],
    pip_3d: np.ndarray,
    fingertip_3d: np.ndarray,
    *,
    semantic_fallback: str = "all",
    front_fallback: str = "none",
    semantic_provider: str = "openai",
    semantic_filter: Callable[[str, Sequence[str]], list[int]] | None = None,
    semantic_options: dict[str, Any] | None = None,
    epsilon: float = 1e-6,
    ray_origin: np.ndarray | None = None,
    ray_direction: np.ndarray | None = None,
) -> dict:
    """Select the nearest semantic candidate to the forward pointing ray."""
    if front_fallback not in {"all", "none"}:
        raise ValueError("front fallback must be 'all' or 'none'")
    if (ray_origin is None) != (ray_direction is None):
        raise ValueError("ray_origin and ray_direction must be supplied together")
    if ray_origin is None:
        origin, direction = build_pointing_ray(pip_3d, fingertip_3d, epsilon=epsilon)
    else:
        origin = _vector(ray_origin, "ray_origin")
        direction = _vector(ray_direction, "ray_direction")
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < epsilon:
            raise ValueError(f"ray_direction is too small (norm={norm:.3g})")
        direction = direction / norm
    names = [str(item["name"]) for item in objects]
    semantic = get_semantic_candidates(
        instruction, names, fallback=semantic_fallback, provider=semantic_provider,
        semantic_filter=semantic_filter, **(semantic_options or {}),
    )
    scores = []
    for index in semantic:
        center = objects[index].get("center_3d")
        if center is None:
            continue
        point = _vector(center, f"objects[{index}].center_3d")
        relative = point - origin
        forward = float(relative @ direction)
        closest = origin + max(forward, 0.0) * direction
        scores.append({"index": index, "name": names[index],
                       "ray_distance": float(np.linalg.norm(point - closest)),
                       "forward_distance": forward, "is_in_front": bool(forward >= 0)})
    scores.sort(key=lambda item: item["ray_distance"])
    selectable = [item for item in scores if item["is_in_front"]]
    if not selectable and front_fallback == "all":
        selectable = scores
    winner = selectable[0] if selectable else None
    target_index = winner["index"] if winner else None
    return {
        "target_index": target_index,
        "target_name": names[target_index] if target_index is not None else None,
        "target_center_3d": (np.asarray(objects[target_index]["center_3d"], dtype=np.float64)
                             if target_index is not None else None),
        "ray_origin": origin, "ray_direction": direction,
        "candidate_scores": scores,
        "semantic_candidates_before": [{"index": i, "name": name} for i, name in enumerate(names)],
        "semantic_candidates_after": [{"index": i, "name": names[i]} for i in semantic],
    }
