"""Per-frame baseline inference, shared by cached and live runs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from batch_preparation.shared import segment_image
from .model import (Intrinsics, angular_features, mask_to_point_cloud,
                    normalize_log_likelihood, paper_log_likelihood)

def json_numbers(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


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
        raise ValueError(f"camera intrinsics missing: {', '.join(sorted(missing))}")
    return Intrinsics(values["fx"], values["fy"], values["cx"], values["cy"])


def detect_objects(
    image: Image.Image, config: dict
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    return segment_image(image, config)


def ground_truth(point, masks: np.ndarray, scores: np.ndarray,
                 names: list[str]) -> dict | None:
    if point is None:
        return None
    x, y = point.x, point.y
    matches = np.flatnonzero(masks[:, y, x]).tolist() if 0 <= y < masks.shape[1] and 0 <= x < masks.shape[2] else []
    selected = max(matches, key=lambda i: float(scores[i])) if matches else None
    return {"point": {"x": x, "y": y}, "candidate_index": selected,
            "object_name": names[selected] if selected is not None else None,
            "status": "matched" if selected is not None else "no_mask"}


def load_perception(path: Path):
    with np.load(path, allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        names = data["box_names"].astype(str).tolist()
        scores = np.asarray(data["scores"])
        boxes = np.asarray(data["bboxes"], dtype=np.float32).reshape(-1, 4)
    return masks, names, scores, boxes


def process(frame, intr: Intrinsics, config: dict, segmentation: dict,
            output: Path, perception_root: Path | None = None) -> dict:
    frame_started = time.perf_counter()
    image = Image.open(frame.rgb_path).convert("RGB")
    raw_depth = np.asarray(Image.open(frame.depth_path))
    ray = json.loads(frame.ray_path.read_text(encoding="utf-8"))
    masks, names, scores, boxes = (
        load_perception(perception_root / f"{frame.frame_id}.npz")
        if perception_root is not None else detect_objects(image, segmentation)
    )
    if not names:
        raise RuntimeError("segmentation returned no candidates")

    origin, direction = np.asarray(ray["origin"]), np.asarray(ray["direction"])
    theta, phi, area, valid, counts = [], [], [], [], []
    for mask in masks:
        cloud = mask_to_point_cloud(mask, raw_depth.astype(np.float64) * 0.001,
                                    intr, **config["depth_filter"])
        th, ph, ok = angular_features(cloud, origin, direction)
        theta.append(th); phi.append(ph); area.append(int(np.count_nonzero(mask)))
        valid.append(ok); counts.append(len(cloud))
    arrays = [np.asarray(x) for x in (theta, phi, area, valid)]
    log_likelihood, parameters = paper_log_likelihood(*arrays, feature_set=config["feature_set"])
    likelihood, probability = normalize_log_likelihood(log_likelihood)
    predicted = int(np.argmax(probability))
    gt = ground_truth(frame.ground_truth_point, masks, scores, names)
    result = {
        "frame_id": frame.frame_id, "feature_set": config["feature_set"],
        "object_names": names, "theta": json_numbers(arrays[0]), "phi": json_numbers(arrays[1]),
        "area": arrays[2].astype(int).tolist(), "point_count": counts,
        "valid_candidate": arrays[3].astype(bool).tolist(), "likelihood": likelihood.tolist(),
        "probability": probability.tolist(), "predicted_index": predicted,
        "predicted_object": names[predicted],
        "parameters": {key: (float(value) if np.isfinite(value) else None)
                       for key, value in parameters.items()},
        "pointing_ray": ray, "ground_truth": gt,
        "correct": bool(gt and gt["candidate_index"] is not None and predicted == gt["candidate_index"]),
        "processing_time_seconds": time.perf_counter() - frame_started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "perception.npz", masks=masks, box_names=np.asarray(names),
        scores=scores, bboxes=boxes,
    )
    return result
