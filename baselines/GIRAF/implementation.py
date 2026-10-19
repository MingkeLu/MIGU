"""Per-frame baseline inference, shared by cached and live runs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from batch_preparation.shared import segment_image
from .model import Intrinsics, mask_to_center_3d, ray_to_depth_point, select_target
from .visualization import export_giraf_point_cloud

def resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_intrinsics(path: Path) -> Intrinsics:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"fx", "fy", "cx", "cy"}:
                values[key.strip()] = float(value)
    missing = {"fx", "fy", "cx", "cy"} - values.keys()
    if missing:
        raise ValueError(f"Camera intrinsics missing: {', '.join(sorted(missing))}")
    return Intrinsics(**values)


def detect_objects(image_path: Path, config: dict) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    return segment_image(image_path, config)


def load_perception(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        names = data["box_names"].astype(str).tolist()
        scores = np.asarray(data["scores"])
        boxes = (np.asarray(data["bboxes"]).reshape(-1, 4)
                 if "bboxes" in data else np.full((len(names), 4), np.nan))
    return masks, names, scores, boxes


def load_finger_points(ray_path: Path, allow_precomputed_only: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    ray = json.loads(ray_path.read_text(encoding="utf-8"))
    if "origin" not in ray or "direction" not in ray:
        raise ValueError(f"{ray_path} does not contain a precomputed pointing ray")
    origin = np.asarray(ray["origin"], dtype=np.float64)
    direction = np.asarray(ray["direction"], dtype=np.float64)
    if (origin.shape != (3,) or direction.shape != (3,)
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) <= 0):
        raise ValueError(f"{ray_path} contains an invalid precomputed pointing ray")
    points = {str(item.get("name", "")).upper(): item.get("point_3d")
              for item in ray.get("index_finger_landmarks", [])}
    if points.get("PIP") is None or points.get("TIP") is None:
        if allow_precomputed_only:
            # select_target ignores these compatibility arguments when an
            # explicit ray is supplied. Keep batch processing independent of
            # landmark depth completeness.
            return origin, origin + direction, ray
        raise ValueError(f"{ray_path} does not contain valid PIP and TIP 3-D landmarks")
    return np.asarray(points["PIP"]), np.asarray(points["TIP"]), ray


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def ground_truth_index(point, masks: np.ndarray, scores: np.ndarray) -> int | None:
    if point is None:
        return None
    x, y = point.x, point.y
    if not (0 <= y < masks.shape[1] and 0 <= x < masks.shape[2]):
        return None
    matches = np.flatnonzero(masks[:, y, x]).tolist()
    return max(matches, key=lambda index: float(scores[index])) if matches else None


def process_frame(frame, instruction: dict, args, output: Path) -> dict:
    frame_started = time.perf_counter()
    masks, names, scores, boxes = (load_perception(args.batch_input / "perception" / f"{frame.frame_id}.npz")
                                    if args.batch else detect_objects(frame.rgb_path, args.segmentation))
    if not names:
        raise RuntimeError("segmentation returned no candidates")
    depth = np.asarray(Image.open(frame.depth_path), dtype=np.float64)
    pip, tip, source_ray = load_finger_points(frame.ray_path, allow_precomputed_only=True)
    objects, invalid_centers = [], []
    for index, (name, mask) in enumerate(zip(names, masks)):
        center, point_count = mask_to_center_3d(
            mask, depth, args.intrinsics, depth_scale=args.depth_scale,
            **args.depth_filter,
        )
        box = boxes[index] if index < len(boxes) and np.all(np.isfinite(boxes[index])) else None
        objects.append({"name": name, "mask": mask, "center_3d": center, "box": box,
                        "point_count": point_count})
        if center is None:
            invalid_centers.append({"index": index, "name": name, "point_count": point_count})
    result = select_target(
        instruction["instruction"], objects, pip, tip,
        semantic_fallback=args.semantic["fallback"],
        front_fallback=args.geometry["front_fallback"],
        semantic_provider=args.semantic["provider"],
        semantic_options={key: value for key, value in {
            "model": args.semantic.get("model"), "base_url": args.base_url,
            "api_key": args.api_key,
        }.items() if value is not None},
        epsilon=float(args.geometry["ray_epsilon_m"]),
        # Live and batch modes both consume the fitted ray. The PIP/TIP values
        # above remain compatibility arguments for the pure GIRAF API.
        ray_origin=source_ray["origin"],
        ray_direction=source_ray["direction"],
    )
    result["ray_environment_point"] = ray_to_depth_point(
        depth, args.intrinsics, result["ray_origin"], result["ray_direction"],
        depth_scale=args.depth_scale,
        min_depth_m=float(args.depth_filter["min_depth_m"]),
        max_depth_m=float(args.depth_filter["max_depth_m"]),
        min_forward_m=float(args.geometry.get("environment_min_forward_m", 0.05)),
    )
    gt_index = ground_truth_index(frame.ground_truth_point, masks, scores)
    result.update({
        "frame_id": frame.frame_id, "instruction": instruction["instruction"], "objects": [
            {"index": i, "name": item["name"], "center_3d": item["center_3d"],
             "box": item["box"], "point_count": item["point_count"]}
            for i, item in enumerate(objects)
        ],
        "invalid_centers": invalid_centers, "source_pointing_ray": source_ray,
        "ground_truth_index": gt_index,
        "correct": bool(gt_index is not None and result["target_index"] == gt_index),
        "processing_time_seconds": time.perf_counter() - frame_started,
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output / "perception.npz", masks=masks,
                        box_names=np.asarray(names), scores=scores, bboxes=boxes)
    if args.visualize:
        export_giraf_point_cloud(
            frame.rgb_path, depth, args.intrinsics, objects,
            result["ray_origin"], result["ray_direction"],
            output / "giraf_point_cloud.ply",
            depth_scale=args.depth_scale,
            min_depth_m=float(args.depth_filter["min_depth_m"]),
            max_depth_m=float(args.depth_filter["max_depth_m"]),
        )
    return jsonable(result)
