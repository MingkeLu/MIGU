"""Evaluate a MIGU run against prepared ground-truth masks."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def find_run(path: str | Path) -> Path:
    """Return *path* if it is a run, otherwise select its newest valid child."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MIGU output directory does not exist: {root}")
    if any(root.glob("*/*/result.json")):
        return root
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*/*/result.json"))]
    if not runs:
        raise ValueError(f"No MIGU run containing <group>/<frame>/result.json below {root}")
    timestamped = [p for p in runs if re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", p.name)]
    return max(timestamped or runs, key=lambda p: p.name)


def mask_coverage(mask: np.ndarray, ground_truth: np.ndarray) -> float:
    """Fraction of a predicted mask lying inside the ground-truth mask."""
    area = int(np.count_nonzero(mask))
    return float(np.count_nonzero(mask & ground_truth) / area) if area else 0.0


def _finite_mean(values: list[float]) -> float | None:
    valid = [value for value in values if math.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def _distance_summary(rows: list[dict[str, Any]],
                      prefix: str = "") -> dict[str, float | int | None]:
    distances_2d = [float(row["distance_2d_px"]) for row in rows
                    if row.get("distance_2d_px") is not None
                    and math.isfinite(float(row["distance_2d_px"]))]
    distances_3d = [float(row["distance_3d_m"]) for row in rows
                    if row.get("distance_3d_m") is not None
                    and math.isfinite(float(row["distance_3d_m"]))]
    return {
        f"{prefix}distance_2d_px": _finite_mean(distances_2d),
        f"{prefix}distance_2d_variance_px2": (
            float(np.var(distances_2d)) if distances_2d else None),
        f"{prefix}distance_2d_px_count": len(distances_2d),
        f"{prefix}distance_3d_m": _finite_mean(distances_3d),
        f"{prefix}distance_3d_variance_m2": (
            float(np.var(distances_3d)) if distances_3d else None),
        f"{prefix}distance_3d_m_count": len(distances_3d),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _normalise(values: list[float]) -> list[float]:
    cleaned = [max(0.0, value) if math.isfinite(value) else 0.0 for value in values]
    total = sum(cleaned)
    return ([value / total for value in cleaned] if total else
            ([1.0 / len(cleaned)] * len(cleaned) if cleaned else []))


def _restore_legacy_main_bboxes(data: dict[str, Any], bboxes: np.ndarray,
                                image_shape: tuple[int, int]) -> np.ndarray:
    """Recover pixel boxes for runs that saved normalized main boxes as integers."""
    if not bboxes.size or np.nanmax(np.abs(bboxes)) > 2:
        return bboxes
    reused_from = (data.get("vlm") or {}).get("reused_from")
    if not reused_from:
        return bboxes
    source = Path(str(reused_from)) / "perception.npz"
    if not source.is_file():
        return bboxes
    with np.load(source, allow_pickle=False) as archive:
        if "bboxes" not in archive:
            return bboxes
        original = np.asarray(archive["bboxes"], dtype=float).reshape(-1, 4)
    if original.shape != bboxes.shape or not np.isfinite(original).all():
        return bboxes
    if original.min() < -0.5 or original.max() > 1.5:
        return bboxes
    height, width = image_shape
    restored = original * np.asarray([width, height, width, height], dtype=float)
    restored[:, (0, 2)] = np.clip(restored[:, (0, 2)], 0, width - 1)
    restored[:, (1, 3)] = np.clip(restored[:, (1, 3)], 0, height - 1)
    return restored


def _intrinsics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = float(value)
    if not all(key in values for key in ("fx", "fy", "cx", "cy")):
        raise ValueError("invalid camera intrinsics")
    return values


def _point_3d(point: tuple[float, float], depth: np.ndarray,
              intrinsics: dict[str, float], depth_scale: float) -> tuple[float, float, float] | None:
    x = int(np.clip(round(point[0]), 0, depth.shape[1] - 1))
    y = int(np.clip(round(point[1]), 0, depth.shape[0] - 1))
    z = float(depth[y, x])
    if not math.isfinite(z) or z <= 0:
        crop = depth[max(0, y - 5):y + 6, max(0, x - 5):x + 6]
        valid = crop[np.isfinite(crop) & (crop > 0)]
        if not valid.size:
            return None
        z = float(np.median(valid))
    z *= depth_scale
    return ((point[0] - intrinsics["cx"]) * z / intrinsics["fx"],
            (point[1] - intrinsics["cy"]) * z / intrinsics["fy"], z)


def _localisation_metrics(data: dict[str, Any], frame_dir: Path, input_group: Path,
                          frame_id: str, masks: np.ndarray, confidences: list[float],
                          bboxes: np.ndarray, top_index: int, gesture: np.ndarray | None,
                          distance_point: str,
                          depth_scale: float) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "pred_x": None, "pred_y": None, "gt_x": None, "gt_y": None,
        "distance_2d_px": None, "distance_3d_m": None,
    }
    gt_data = data.get("ground_truth_point")
    if not isinstance(gt_data, dict) or gt_data.get("x") is None or gt_data.get("y") is None:
        return result
    gt_point = (float(gt_data["x"]), float(gt_data["y"]))
    pred_point: tuple[float, float] | None = None
    if distance_point == "bbox_center":
        if 0 <= top_index < len(bboxes):
            box = np.asarray(bboxes[top_index], dtype=float).reshape(-1)
            if len(box) == 4 and np.isfinite(box).all():
                pred_point = ((float(box[0]) + float(box[2])) / 2,
                              (float(box[1]) + float(box[3])) / 2)
    elif gesture is not None:
        # Reconstruct the numeric pixel-wise fusion heatmap saved by the pipeline
        # only as a visualization: max_i(mask_i * gesture * confidence_i).
        confidence_array = np.asarray(confidences, dtype=float)
        per_mask = masks * gesture[None, ...] * confidence_array[:, None, None]
        fusion = per_mask.max(axis=0) if len(masks) else np.zeros_like(gesture)
        valid = np.isfinite(fusion) & (fusion > 0)
        if valid.any():
            y, x = np.unravel_index(np.argmax(np.where(valid, fusion, -np.inf)), fusion.shape)
            pred_point = (float(x), float(y))
    top_mask = masks[top_index]
    if pred_point is None and top_mask.any():
        ys, xs = np.nonzero(top_mask)
        pred_point = ((float(xs.min()) + float(xs.max())) / 2,
                      (float(ys.min()) + float(ys.max())) / 2)
    result.update(gt_x=gt_point[0], gt_y=gt_point[1])
    if pred_point is None:
        return result
    result.update(pred_x=pred_point[0], pred_y=pred_point[1],
                  distance_2d_px=math.dist(pred_point, gt_point))
    depth_path = input_group / "key_depth" / f"{frame_id}.png"
    intrinsics_path = input_group / "camera_intrinsics.txt"
    if not depth_path.is_file():
        prepared = frame_dir.parent / "_prepared_input"
        depth_path = prepared / "key_depth" / f"{frame_id}.png"
        intrinsics_path = prepared / "camera_intrinsics.txt"
    try:
        depth = np.asarray(Image.open(depth_path), dtype=float)
        camera = _intrinsics(intrinsics_path)
        predicted_3d = _point_3d(pred_point, depth, camera, depth_scale)
        target_3d = _point_3d(gt_point, depth, camera, depth_scale)
        if predicted_3d is not None and target_3d is not None:
            result["distance_3d_m"] = math.dist(predicted_3d, target_3d)
    except (OSError, ValueError):
        pass
    return result


def _has_successful_hand_detection(frame_dir: Path) -> bool:
    """Whether a frame contains a readable fitted hand ray."""
    path = frame_dir / "hand_detection.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        origin = np.asarray(data["origin"], dtype=float)
        direction = np.asarray(data["direction"], dtype=float)
        return (origin.shape == (3,) and direction.shape == (3,)
                and np.isfinite(origin).all() and np.isfinite(direction).all()
                and float(np.linalg.norm(direction)) > 0)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _analyse_frame(result_path: Path, input_root: Path, threshold: float,
                   depth_scale: float, distance_point: str) -> dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    group = result_path.parent.parent.name
    frame_id = str(data.get("frame_id") or result_path.parent.name)
    gt_path = input_root / group / "ground_truth_masks" / f"{frame_id}.npz"
    if not gt_path.is_file():
        raise FileNotFoundError("ground-truth mask not found")
    segmentation_path = result_path.parent / "vlm_segmentation.npz"
    if not segmentation_path.is_file():
        raise FileNotFoundError("vlm_segmentation.npz not found")

    with np.load(gt_path, allow_pickle=False) as archive:
        gt = np.asarray(archive["mask"], dtype=bool)
    with np.load(segmentation_path, allow_pickle=False) as archive:
        masks = np.asarray(archive["masks"], dtype=bool)
        bboxes = (np.asarray(archive["bboxes"], dtype=float)
                  if "bboxes" in archive else np.empty((0, 4), dtype=float))
        archive_confidences = (np.asarray(archive["confidences"], dtype=float).reshape(-1)
                               if "confidences" in archive else np.array([], dtype=float))
    if masks.ndim != 3 or masks.shape[1:] != gt.shape:
        raise ValueError(f"mask shape mismatch: VLM {masks.shape}, GT {gt.shape}")
    if not len(masks):
        raise ValueError("VLM mask list is empty")
    bboxes = _restore_legacy_main_bboxes(data, bboxes, masks.shape[1:])

    entries = data.get("masks") or []
    probabilities = [_safe_float(item.get("probability")) for item in entries[:len(masks)]]
    if len(probabilities) < len(masks):
        probabilities.extend([0.0] * (len(masks) - len(probabilities)))
    top_index = int(np.argmax(probabilities))
    coverages = [mask_coverage(mask, gt) for mask in masks]
    matched = [index for index, value in enumerate(coverages) if value >= threshold]
    ranked = sorted(range(len(masks)), key=lambda index: probabilities[index], reverse=True)
    normalised_probabilities = _normalise(probabilities)
    target_probability = sum(normalised_probabilities[index] for index in matched)

    confidences = []
    for index in range(len(masks)):
        if index < len(entries) and entries[index].get("confidence") is not None:
            confidences.append(_safe_float(entries[index]["confidence"]))
        elif index < len(archive_confidences):
            confidences.append(_safe_float(archive_confidences[index]))
        else:
            confidences.append(0.0)

    row: dict[str, Any] = {
        "group": group,
        "frame_id": frame_id,
        "top_mask_index": top_index,
        "top_mask_probability": probabilities[top_index],
        "top_mask_coverage": coverages[top_index],
        "top_mask_high_overlap": coverages[top_index] >= threshold,
        "accuracy": float(any(index in matched for index in ranked[:1])),
        "top2_accuracy": float(any(index in matched for index in ranked[:2])),
        "top3_accuracy": float(any(index in matched for index in ranked[:3])),
        "nll": -math.log(max(target_probability, 1e-15)),
        "brier_score": ((target_probability - 1.0) ** 2 +
                        sum(normalised_probabilities[index] ** 2
                            for index in range(len(masks)) if index not in matched)),
        "vlm_has_high_overlap": bool(matched),
        "matched_mask_indices": ";".join(map(str, matched)),
        "matched_mask_count": len(matched),
        "matched_confidences": ";".join(f"{confidences[i]:.12g}" for i in matched),
        "best_matched_confidence": max((confidences[i] for i in matched), default=None),
        "gt_gesture_mass": None,
        "matched_gesture_mean": None,
        "other_gesture_mean": None,
        "gesture_mean_ratio": None,
        "matched_normalized_gesture_probability": None,
        "processing_time_seconds": None,
        "hand_detection_time_seconds": None,
        "segmentation_time_seconds": None,
    }

    timing = data.get("timing_seconds") or {}
    row["processing_time_seconds"] = (_safe_float(timing.get("total"), math.nan)
                                      if timing.get("total") is not None else None)
    row["hand_detection_time_seconds"] = (_safe_float(timing.get("hand_detection"), math.nan)
                                           if timing.get("hand_detection") is not None else None)
    row["segmentation_time_seconds"] = (_safe_float(timing.get("sam"), math.nan)
                                         if timing.get("sam") is not None else None)

    posterior_path = result_path.parent / "gesture_posterior.npz"
    gesture: np.ndarray | None = None
    if posterior_path.is_file():
        with np.load(posterior_path, allow_pickle=False) as archive:
            gesture = np.asarray(archive["probability"], dtype=float)
        if gesture.shape != gt.shape:
            raise ValueError(f"gesture shape mismatch: posterior {gesture.shape}, GT {gt.shape}")
        gesture = np.where(np.isfinite(gesture), gesture, 0.0)
        row["gt_gesture_mass"] = float(gesture[gt].sum())
        mask_means = [float(gesture[mask].mean()) if mask.any() else math.nan for mask in masks]
        matched_mean = _finite_mean([mask_means[i] for i in matched])
        other_mean = _finite_mean([mask_means[i] for i in range(len(masks)) if i not in matched])
        row["matched_gesture_mean"] = matched_mean
        row["other_gesture_mean"] = other_mean
        finite_mask_means = [value if math.isfinite(value) and value > 0 else 0.0
                             for value in mask_means]
        total_mask_mean = sum(finite_mask_means)
        if matched and total_mask_mean > 0:
            row["matched_normalized_gesture_probability"] = float(
                sum(finite_mask_means[index] for index in matched) / total_mask_mean
            )
        if matched_mean is not None and other_mean is not None:
            row["gesture_mean_ratio"] = (matched_mean / other_mean
                                         if other_mean > 0 else (math.inf if matched_mean > 0 else None))
    row.update(_localisation_metrics(data, result_path.parent, input_root / group,
                                     frame_id, masks, confidences, bboxes, top_index,
                                     gesture, distance_point, depth_scale))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def analyse_migu_run(output_dir: str | Path,
                     input_dir: str | Path = "captures",
                     save_dir: str | Path | None = None,
                     overlap_threshold: float = 0.8,
                     depth_scale: float = 0.001,
                     distance_point: str = "fusion_peak") -> dict[str, Any]:
    """Analyse one run and write per-frame data and a summary."""
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if distance_point not in {"fusion_peak", "bbox_center"}:
        raise ValueError("distance_point must be 'fusion_peak' or 'bbox_center'")
    run = find_run(output_dir)
    input_root = Path(input_dir).expanduser().resolve()
    destination = (Path(save_dir).expanduser().resolve() if save_dir else
                   run / "ground_truth_analysis")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    result_paths = sorted(run.glob("*/*/result.json"))
    successful_hand_frames = sum(_has_successful_hand_detection(path.parent)
                                 for path in result_paths)
    for result_path in result_paths:
        name = f"{result_path.parent.parent.name}/{result_path.parent.name}"
        try:
            rows.append(_analyse_frame(result_path, input_root, overlap_threshold,
                                       depth_scale, distance_point))
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            skipped.append(f"{name}: {exc}")

    top_hits = sum(bool(row["top_mask_high_overlap"]) for row in rows)
    vlm_hits = sum(bool(row["vlm_has_high_overlap"]) for row in rows)
    result_count = len(result_paths)
    matched_confidences = [float(row["best_matched_confidence"]) for row in rows
                           if row["best_matched_confidence"] is not None]
    gesture_masses = [float(row["gt_gesture_mass"]) for row in rows
                      if row["gt_gesture_mass"] is not None]
    ratios = [float(row["gesture_mean_ratio"]) for row in rows
              if row["gesture_mean_ratio"] is not None]
    finite_ratios = [value for value in ratios if math.isfinite(value)]
    valid_ids = {(row["group"], row["frame_id"]) for row in rows}
    valid_gt_hand_frames = sum(
        _has_successful_hand_detection(path.parent)
        and (path.parent.parent.name, path.parent.name) in valid_ids
        for path in result_paths
    )
    matched_probabilities = [float(row["matched_normalized_gesture_probability"])
                             for row in rows
                             if row["matched_normalized_gesture_probability"] is not None]
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows
                if row.get(key) is not None and math.isfinite(float(row[key]))]

    processing_times = values("processing_time_seconds")
    summary = {
        "run": str(run),
        "input_dir": str(input_root),
        "overlap_definition": "intersection(predicted_mask, ground_truth) / predicted_mask_area",
        "overlap_threshold": overlap_threshold,
        "depth_scale": depth_scale,
        "distance_point": distance_point,
        "result_frames": len(result_paths),
        "valid_frames": len(rows),
        "skipped_frames": len(skipped),
        "successful_hand_detection_frames": successful_hand_frames,
        "successful_hand_detection_rate": (successful_hand_frames / result_count
                                            if result_count else None),
        "valid_gt_successful_hand_detection_frames": valid_gt_hand_frames,
        "valid_gt_successful_hand_detection_rate": (valid_gt_hand_frames / result_count
                                                     if result_count else None),
        "top_mask_high_overlap_frames": top_hits,
        "top_mask_high_overlap_rate": top_hits / result_count if result_count else None,
        "accuracy": (sum(values("accuracy")) / result_count if result_count else None),
        "accuracy_count": result_count,
        "top2_accuracy": (sum(values("top2_accuracy")) / result_count if result_count else None),
        "top2_accuracy_count": result_count,
        "top3_accuracy": (sum(values("top3_accuracy")) / result_count if result_count else None),
        "top3_accuracy_count": result_count,
        "nll": _finite_mean(values("nll")),
        "nll_count": len(values("nll")),
        "brier_score": _finite_mean(values("brier_score")),
        "brier_score_count": len(values("brier_score")),
        "processing_time_mean_seconds": _finite_mean(processing_times),
        "processing_time_variance_seconds2": (float(np.var(processing_times))
                                                if processing_times else None),
        "processing_time_count": len(processing_times),
        "hand_detection_time_mean_seconds": _finite_mean(values("hand_detection_time_seconds")),
        "segmentation_time_mean_seconds": _finite_mean(values("segmentation_time_seconds")),
        "vlm_high_overlap_frames": vlm_hits,
        "vlm_high_overlap_rate": vlm_hits / result_count if result_count else None,
        "mean_best_matched_confidence": _finite_mean(matched_confidences),
        "gesture_valid_frames": len(gesture_masses),
        "mean_gt_gesture_mass": _finite_mean(gesture_masses),
        "gesture_ratio_frames": len(ratios),
        "finite_gesture_ratio_frames": len(finite_ratios),
        "infinite_gesture_ratio_frames": len(ratios) - len(finite_ratios),
        "mean_gesture_ratio": _finite_mean(finite_ratios),
        "median_gesture_ratio": (float(np.median(finite_ratios)) if finite_ratios else None),
        "matched_normalized_gesture_probability_frames": len(matched_probabilities),
        "mean_matched_normalized_gesture_probability": _finite_mean(matched_probabilities),
        "skip_reasons": skipped,
    }
    summary.update(_distance_summary(rows))
    for task in ("pick", "place"):
        task_rows = [row for row in rows if task in str(row.get("group", "")).lower()]
        summary.update(_distance_summary(task_rows, f"{task}_"))
    overlap_rows = [row for row in rows if bool(row.get("top_mask_high_overlap"))]
    non_overlap_rows = [row for row in rows if not bool(row.get("top_mask_high_overlap"))]
    summary.update(_distance_summary(overlap_rows, "top_mask_high_overlap_"))
    summary.update(_distance_summary(non_overlap_rows, "top_mask_not_high_overlap_"))
    _write_csv(destination / "frame_metrics.csv", rows)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MIGU run with prepared GT masks")
    parser.add_argument("output_dir", help="a run directory or a directory containing MIGU runs")
    parser.add_argument("--input-dir", default="captures")
    parser.add_argument("-o", "--save-dir")
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument(
        "--distance-point", choices=("fusion_peak", "bbox_center"), default="fusion_peak",
        help="prediction point used for 2-D/3-D distance (default: fusion_peak)",
    )
    args = parser.parse_args(argv)
    summary = analyse_migu_run(args.output_dir, args.input_dir, args.save_dir,
                               args.overlap_threshold, args.depth_scale,
                               args.distance_point)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
