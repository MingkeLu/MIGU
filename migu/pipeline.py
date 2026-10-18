#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from openai import OpenAI
from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from .datasets import open_dataset, pair_prepared_frames, select_capture_groups
from .gesture import draw_hand_ray_overlay, load_hand_ray_overlay, run_gesture
from .key_frames import generate_key_frames
from .vlm import parse_bbox_count, predict, validate_boxes
from .sam import masks_to_array, request_masks
from .visualization import (
    save_fusion_heatmap, save_gesture_heatmap, save_mask_distribution, save_vlm_masks,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed_top_level = {
        "input_dir", "subdirs", "output_dir", "hand_detection", "gesture",
        "vlm", "sam", "visualize",
    }
    unknown = set(config) - allowed_top_level
    if unknown:
        raise ValueError(f"Unknown configuration entries: {sorted(unknown)}")
    for key in ("input_dir", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty path string")
    subdirs = config.get("subdirs")
    if subdirs is not None and (
        not isinstance(subdirs, list)
        or not all(isinstance(value, str) and value.strip() for value in subdirs)
    ):
        raise ValueError("subdirs must be null or a list of non-empty strings")
    if not isinstance(config.get("visualize", True), bool):
        raise ValueError("visualize must be true or false")
    for section in ("hand_detection", "gesture", "vlm", "sam"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration mapping: {section}")

    expected_keys = {
        "hand_detection": {"min_detection_confidence", "min_tracking_confidence"},
        "gesture": {"mode"},
        "vlm": {"model", "bbox_count"},
        "sam": {"server_url", "timeout", "return_best"},
    }
    for section, allowed in expected_keys.items():
        extra = set(config[section]) - allowed
        if extra:
            raise ValueError(f"Unknown {section} entries: {sorted(extra)}")

    for key in ("min_detection_confidence", "min_tracking_confidence"):
        value = config["hand_detection"].get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"hand_detection.{key} must be a finite number in [0, 1]")
        config["hand_detection"][key] = float(value)

    mode = config["gesture"].get("mode", "gesture_posterior")
    if mode not in {
        "gesture_posterior", "vMF", "distance_cone", "angle_error_gaussian",
        "angular_gaussian", "dist_threshold", "distance_threshold",
    }:
        raise ValueError(
            "gesture.mode must be 'gesture_posterior', 'vMF', 'distance_cone', "
            "'angle_error_gaussian', or 'dist_threshold'"
        )
    config["gesture"]["mode"] = mode

    if not config["sam"].get("return_best", True):
        raise ValueError("MIGU requires sam.return_best: true (one mask per bbox)")
    if not isinstance(config["sam"].get("server_url"), str):
        raise ValueError("sam.server_url must be a string")
    timeout = config["sam"].get("timeout", 300)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("sam.timeout must be a positive finite number")
    try:
        parse_bbox_count(config["vlm"].get("bbox_count"))
    except ValueError as error:
        raise ValueError(
            "vlm.bbox_count must be a positive integer or a string like '<=5'"
        ) from error
    if not isinstance(config["vlm"].get("model"), str) or not config["vlm"]["model"].strip():
        raise ValueError("vlm.model must be a non-empty string")
    return config


def integrate_over_masks(masks: np.ndarray, probability: np.ndarray,
                         mean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    if masks.ndim != 3 or masks.shape[1:] != probability.shape:
        raise ValueError(f"Masks {masks.shape} and posterior {probability.shape} do not align")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0):
        raise ValueError("Gesture posterior must be finite and non-negative")
    masses = np.sum(masks * probability[None], axis=(1, 2))
    if mean:
        areas = masks.sum(axis=(1, 2))
        masses = np.divide(masses, areas, out=np.zeros_like(masses), where=areas > 0)
    total = float(masses.sum())
    return masses, masses / total if total > 0 else np.zeros_like(masses)


def fuse_pixel_wise(masks: np.ndarray, probability: np.ndarray,
                    confidences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fuse gesture and VLM confidence per pixel and return mask peaks + heatmap."""
    masks = np.asarray(masks, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if masks.ndim != 3 or masks.shape[1:] != probability.shape:
        raise ValueError(f"Masks {masks.shape} and posterior {probability.shape} do not align")
    if confidences.shape != (len(masks),):
        raise ValueError("There must be exactly one confidence per mask")
    if (not np.all(np.isfinite(probability)) or np.any(probability < 0)
            or not np.all(np.isfinite(confidences)) or np.any(confidences < 0)):
        raise ValueError("Gesture posterior and mask confidences must be finite and non-negative")
    per_mask = masks * probability[None] * confidences[:, None, None]
    scores = per_mask.max(axis=(1, 2)) if len(masks) else np.empty(0, dtype=np.float64)
    heatmap = per_mask.max(axis=0) if len(masks) else np.zeros_like(probability)
    return scores, heatmap


def run_vlm_sam(image_path: Path, instruction: str, config: dict,
                client: OpenAI) -> tuple[list[dict], np.ndarray, dict]:
    started = time.perf_counter()
    parsed, response_id, usage = predict(
        client, config["vlm"]["model"], image_path, instruction,
        config["vlm"]["bbox_count"],
    )
    bbox_seconds = time.perf_counter() - started
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    validate_boxes(parsed, image.width, image.height, config["vlm"]["bbox_count"])
    boxes = [{"label": item.label, "bbox": list(item.bbox),
              "confidence": float(item.confidence)} for item in parsed.boxes]
    sam_started = time.perf_counter()
    masks = masks_to_array(
        request_masks(config["sam"]["server_url"], image,
                      [item["bbox"] for item in boxes], config["sam"]["return_best"],
                      float(config["sam"].get("timeout", 300))), image.size
    )
    return boxes, masks, {"vlm_bbox_seconds": bbox_seconds,
                          "sam_seconds": time.perf_counter() - sam_started,
                          "response_id": response_id, "usage": usage}


def load_vlm_boxes(path: Path, image_size: tuple[int, int], bbox_count: object) -> list[dict]:
    """Load the JSON emitted by test_vlm and validate its reusable boxes."""
    if not path.is_file():
        raise FileNotFoundError(f"Reusable VLM result not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    raw_boxes = record.get("boxes") if isinstance(record, dict) else None
    if not isinstance(raw_boxes, list):
        raise ValueError(f"Reusable VLM result {path} has no boxes list")
    minimum, maximum = parse_bbox_count(bbox_count)
    if not minimum <= len(raw_boxes) <= maximum:
        expected = str(maximum) if minimum == maximum else f"1 to {maximum}"
        raise ValueError(f"Expected {expected} boxes in {path}, got {len(raw_boxes)}")
    width, height = image_size
    boxes = []
    for index, item in enumerate(raw_boxes, 1):
        if not isinstance(item, dict) or not isinstance(item.get("bbox"), list):
            raise ValueError(f"Invalid box {index} in reusable VLM result {path}")
        bbox = item["bbox"]
        if (len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, int)
                                  for value in bbox)):
            raise ValueError(f"Invalid bbox for box {index} in {path}: {bbox!r}")
        x_min, y_min, x_max, y_max = bbox
        if not (0 <= x_min < x_max < width and 0 <= y_min < y_max < height):
            raise ValueError(f"Box {index} in {path} is outside the {width}x{height} image: {bbox}")
        confidence = item.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not np.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError(f"Invalid confidence for box {index} in {path}: {confidence!r}")
        boxes.append({"label": str(item.get("label", "")), "bbox": bbox,
                      "confidence": float(confidence)})
    return boxes


def run_sam_for_reused_vlm(image_path: Path, reusable_path: Path,
                           config: dict) -> tuple[list[dict], np.ndarray, dict]:
    """Reuse test_vlm boxes while generating their SAM masks."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    boxes = load_vlm_boxes(reusable_path, image.size, config["vlm"]["bbox_count"])
    started = time.perf_counter()
    masks = masks_to_array(
        request_masks(config["sam"]["server_url"], image,
                      [item["bbox"] for item in boxes], config["sam"]["return_best"],
                      float(config["sam"].get("timeout", 300))), image.size
    )
    return boxes, masks, {
        "vlm_bbox_seconds": 0.0, "sam_seconds": time.perf_counter() - started,
        "response_id": None, "usage": None, "reused_from": str(reusable_path),
    }


def load_vlm_sam(path: Path, image_size: tuple[int, int]) -> tuple[list[dict], np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Reusable VLM result not found: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = {"masks", "bboxes", "labels", "confidences"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Reusable VLM result {path} is missing: {sorted(missing)}")
        masks = np.asarray(saved["masks"], dtype=bool)
        bboxes = np.asarray(saved["bboxes"])
        labels = np.asarray(saved["labels"])
        confidences = np.asarray(saved["confidences"])
    count = len(masks)
    if masks.ndim != 3 or masks.shape[1:] != (image_size[1], image_size[0]):
        raise ValueError(
            f"Reusable masks {masks.shape} from {path} do not match image size {image_size}"
        )
    if bboxes.shape != (count, 4) or len(labels) != count or len(confidences) != count:
        raise ValueError(f"Inconsistent reusable VLM arrays in {path}")
    boxes = [
        {"label": str(labels[index]), "bbox": [int(value) for value in bboxes[index]],
         "confidence": float(confidences[index])}
        for index in range(count)
    ]
    return boxes, masks, {
        "vlm_bbox_seconds": 0.0, "sam_seconds": 0.0,
        "response_id": None, "usage": None, "reused_from": str(path),
    }


def load_main_vlm_segmentation(
    perception_path: Path, image_size: tuple[int, int]
) -> tuple[list[dict], np.ndarray, dict]:
    """Load segmentation candidates and VLM probabilities emitted by ``main``."""
    result_path = perception_path.with_name("result.json")
    if not perception_path.is_file():
        raise FileNotFoundError(f"Reusable main segmentation not found: {perception_path}")
    if not result_path.is_file():
        raise FileNotFoundError(f"Reusable main VLM result not found: {result_path}")
    with np.load(perception_path, allow_pickle=False) as saved:
        required = {"masks", "bboxes", "box_names"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"Reusable main segmentation {perception_path} is missing: {sorted(missing)}"
            )
        masks = np.asarray(saved["masks"], dtype=bool)
        bboxes = np.asarray(saved["bboxes"])
        labels = np.asarray(saved["box_names"]).astype(str)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    objects = record.get("objects") if isinstance(record, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Reusable main VLM result {result_path} has no objects list")
    count = len(masks)
    if masks.ndim != 3 or masks.shape[1:] != (image_size[1], image_size[0]):
        raise ValueError(
            f"Reusable masks {masks.shape} from {perception_path} do not match image size "
            f"{image_size}"
        )
    if bboxes.shape != (count, 4) or labels.shape != (count,) or len(objects) != count:
        raise ValueError(f"Inconsistent reusable main arrays/objects in {perception_path.parent}")
    confidences = []
    for index, item in enumerate(objects, 1):
        if not isinstance(item, dict) or "semantic_probability" not in item:
            raise ValueError(
                f"Reusable main object {index} in {result_path} has no semantic_probability"
            )
        expected_id = str(index)
        if str(item.get("object_id")) != expected_id:
            raise ValueError(
                f"Reusable main object order mismatch in {result_path}: expected ID "
                f"{expected_id}, got {item.get('object_id')!r}"
            )
        confidence = item["semantic_probability"]
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not np.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError(
                f"Invalid semantic_probability for object {index} in {result_path}: "
                f"{confidence!r}"
            )
        confidences.append(float(confidence))
    pixel_bboxes = bboxes_to_pixels(bboxes, image_size)
    boxes = [
        {"label": str(labels[index]),
         "bbox": pixel_bboxes[index],
         "confidence": confidences[index]}
        for index in range(count)
    ]
    return boxes, masks, {
        "vlm_bbox_seconds": 0.0, "sam_seconds": 0.0,
        "response_id": None, "usage": None,
        "reused_from": str(perception_path.parent),
    }


def bboxes_to_pixels(bboxes: np.ndarray, image_size: tuple[int, int]) -> list[list[int]]:
    """Convert detector boxes to pixels, accepting normalized OWL-style boxes."""
    values = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(values).all():
        raise ValueError("Reusable bboxes contain non-finite coordinates")
    # Batch OWL detections are nominally in [0, 1], with small model overshoots.
    normalized = bool(values.size and values.min() >= -0.5 and values.max() <= 1.5)
    if normalized:
        width, height = image_size
        values = values * np.asarray([width, height, width, height], dtype=float)
    width, height = image_size
    values[:, (0, 2)] = np.clip(values[:, (0, 2)], 0, width - 1)
    values[:, (1, 3)] = np.clip(values[:, (1, 3)], 0, height - 1)
    return [[int(round(value)) for value in row] for row in values]


def load_semantic_grounding_segmentation(
    result_path: Path, image_size: tuple[int, int]
) -> tuple[list[dict], np.ndarray, dict]:
    """Reuse grid/PIVOT object probabilities and their batch perception masks."""
    if not result_path.is_file():
        raise FileNotFoundError(f"Reusable semantic-grounding result not found: {result_path}")
    record = json.loads(result_path.read_text(encoding="utf-8"))
    distribution = record.get("object_distribution")
    image_record = record.get("image")
    if not isinstance(distribution, list) or not isinstance(image_record, dict):
        raise ValueError(f"Invalid semantic-grounding result: {result_path}")
    source_image = Path(str(image_record.get("path", "")))
    frame_id = str(record.get("frame_id") or result_path.parents[1].name)
    perception_path = source_image.parent.parent / "perception" / f"{frame_id}.npz"
    if not perception_path.is_file():
        raise FileNotFoundError(
            f"Batch perception referenced by {result_path} not found: {perception_path}"
        )
    with np.load(perception_path, allow_pickle=False) as saved:
        required = {"masks", "bboxes", "box_names"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Reusable perception {perception_path} is missing: {sorted(missing)}")
        masks = np.asarray(saved["masks"], dtype=bool)
        bboxes = np.asarray(saved["bboxes"], dtype=float)
        labels = np.asarray(saved["box_names"]).astype(str)
    count = len(masks)
    if masks.ndim != 3 or masks.shape[1:] != (image_size[1], image_size[0]):
        raise ValueError(f"Reusable masks {masks.shape} do not match image size {image_size}")
    probabilities = np.zeros(count, dtype=float)
    seen = set()
    for item in distribution:
        if not isinstance(item, dict) or "object_index" not in item:
            raise ValueError(f"Invalid object_distribution entry in {result_path}")
        index = int(item["object_index"])
        probability = item.get("probability")
        if (index < 0 or index >= count or index in seen or isinstance(probability, bool)
                or not isinstance(probability, (int, float)) or not np.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError(f"Invalid object probability/index in {result_path}: {item!r}")
        probabilities[index] = float(probability)
        seen.add(index)
    if seen != set(range(count)):
        raise ValueError(f"object_distribution in {result_path} does not cover all {count} masks")
    pixel_bboxes = bboxes_to_pixels(bboxes, image_size)
    boxes = [
        {"label": str(labels[index]), "bbox": pixel_bboxes[index],
         "confidence": float(probabilities[index])}
        for index in range(count)
    ]
    return boxes, masks, {
        "vlm_bbox_seconds": 0.0, "sam_seconds": 0.0,
        "response_id": None, "usage": None, "reused_from": str(result_path),
    }


def reusable_vlm_sam_path(reuse_root: Path, group_name: str, frame_id: str) -> Path:
    """Return the artifact containing both the reused VLM boxes and SAM masks."""
    return reuse_root / group_name / frame_id / "vlm_segmentation.npz"


def reusable_vlm_path(reuse_root: Path, group_name: str, frame_id: str) -> Path:
    """Return a test_vlm JSON path, accepting either its run root or results dir."""
    direct = reuse_root / group_name / f"{frame_id}.json"
    nested = reuse_root / "results" / group_name / f"{frame_id}.json"
    return direct if direct.is_file() or not nested.is_file() else nested


def reusable_main_path(reuse_root: Path, group_name: str, frame_id: str) -> Path:
    return reuse_root / group_name / frame_id / "perception.npz"


def reusable_semantic_grounding_path(
    reuse_root: Path, group_name: str, frame_id: str, method: str
) -> Path:
    return reuse_root / group_name / frame_id / method / "result.json"


def require_reusable_results(reuse_root: Path, reuse_type: str,
                             group_name: str, frames) -> None:
    """Fail before processing if strict VLM or VLM/SAM reuse is incomplete."""
    missing = []
    for frame in frames:
        if reuse_type in {"grid", "pivot"}:
            path = reusable_semantic_grounding_path(
                reuse_root, group_name, frame.frame_id, reuse_type
            )
        else:
            path_for = {"vlm": reusable_vlm_path, "vlm+seg": reusable_vlm_sam_path,
                        "main": reusable_main_path}[reuse_type]
            path = path_for(reuse_root, group_name, frame.frame_id)
        required = [path, path.with_name("result.json")] if reuse_type == "main" else [path]
        missing.extend(candidate for candidate in required if not candidate.is_file())
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"vlm.reuse_result_dir enables strict {reuse_type!r} reuse; "
            "the following reusable files are missing:\n" + formatted
        )


def require_reusable_vlm_sam(reuse_root: Path, group_name: str, frames) -> None:
    """Backward-compatible wrapper for combined VLM/SAM reuse."""
    require_reusable_results(reuse_root, "vlm+seg", group_name, frames)


def process_frame(frame, instruction: str, intrinsics: Path, run_dir: Path,
                  config: dict, client: OpenAI | None, visualize: bool,
                  hand_seconds: float | None, reusable_vlm_path: Path | None = None,
                  reuse_type: str | None = None) -> dict:
    started = time.perf_counter()
    frame_dir = run_dir / frame.frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    has_gesture = (frame.depth_path is not None and frame.ray_path is not None
                   and frame.depth_path.is_file() and frame.ray_path.is_file())
    hand_path = frame_dir / "hand_detection.json" if has_gesture else None
    if hand_path is not None:
        shutil.copy2(frame.ray_path, hand_path)

    gesture_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        gesture_future = None
        if has_gesture:
            gesture_future = pool.submit(
                run_gesture, frame.depth_path, frame.ray_path, intrinsics, frame_dir,
                config["gesture"]["mode"],
            )
        if reuse_type is None:
            if client is None:
                raise RuntimeError("OpenAI client is required to generate a missing VLM result")
            vlm_future = pool.submit(run_vlm_sam, frame.rgb_path, instruction, config, client)
        elif reuse_type in {"vlm+seg", "main", "grid", "pivot"}:
            with Image.open(frame.rgb_path) as source:
                image_size = source.size
            loader = (load_main_vlm_segmentation if reuse_type == "main" else
                      load_semantic_grounding_segmentation
                      if reuse_type in {"grid", "pivot"} else load_vlm_sam)
            vlm_future = pool.submit(loader, reusable_vlm_path, image_size)
        else:
            vlm_future = pool.submit(
                run_sam_for_reused_vlm, frame.rgb_path, reusable_vlm_path, config
            )
        if gesture_future is not None:
            gesture_pixel, generated_posterior_path = gesture_future.result()
            gesture_seconds = time.perf_counter() - gesture_started
        else:
            gesture_pixel = generated_posterior_path = None
            gesture_seconds = None
        boxes, masks, vlm_trace = vlm_future.result()

    posterior_path = None
    if generated_posterior_path is not None:
        posterior_path = frame_dir / "gesture_posterior.npz"
        generated_posterior_path.replace(posterior_path)
    mask_path = frame_dir / "vlm_segmentation.npz"
    np.savez_compressed(
        mask_path, masks=masks, bboxes=np.asarray([box["bbox"] for box in boxes], np.int32),
        labels=np.asarray([box["label"] for box in boxes], dtype=str),
        confidences=np.asarray([box["confidence"] for box in boxes], np.float32),
        source_image=np.asarray(str(frame.rgb_path)), image_size=np.asarray(masks.shape[-2:][::-1]),
    )
    masses = distribution = fusion_heatmap = None
    if gesture_pixel is not None:
        confidences = np.asarray([box["confidence"] for box in boxes], dtype=np.float64)
        masses, fusion_heatmap = fuse_pixel_wise(masks, gesture_pixel, confidences)
        total = float(masses.sum())
        distribution = masses / total if total > 0 else np.zeros_like(masses)
    gt = None if frame.ground_truth_point is None else {
        "x": frame.ground_truth_point.x, "y": frame.ground_truth_point.y,
        "source": str(frame.ground_truth_point.source),
    }
    post_hand_seconds = time.perf_counter() - started
    result = {
        "frame_id": frame.frame_id, "instruction": instruction,
        "gesture_available": has_gesture,
        "ground_truth_point": gt,
        "paths": {"rgb": str(frame.rgb_path),
                  "depth": str(frame.depth_path) if frame.depth_path else None,
                  "hand_detection": str(hand_path) if hand_path else None,
                  "gesture_posterior": str(posterior_path) if posterior_path else None,
                  "vlm_segmentation": str(mask_path),
                  "fusion_heatmap": str(frame_dir / "fusion_heatmap.png")
                  if fusion_heatmap is not None else None},
        "fusion": "pixel_wise",
        "masks": [
            {"rank": index + 1, **box,
             "gesture_mass": float(masses[index]) if masses is not None else None,
             "probability": float(distribution[index]) if distribution is not None else None,
             "contains_ground_truth": bool(gt and masks[index, gt["y"], gt["x"]])
             if gt and 0 <= gt["y"] < masks.shape[1] and 0 <= gt["x"] < masks.shape[2] else False}
            for index, box in enumerate(boxes)
        ],
        "timing_seconds": {
            "hand_detection": hand_seconds, "gesture_posterior": gesture_seconds,
            "vlm_bbox": vlm_trace["vlm_bbox_seconds"], "sam": vlm_trace["sam_seconds"],
            "owlv2": vlm_trace.get("owlv2_seconds"),
            "parallel_grounding_and_segmentation": post_hand_seconds,
            # Hand detection is performed once in the group preparation pass. Adding
            # its measured frame time yields a comparable full per-frame latency.
            "total": post_hand_seconds + (hand_seconds or 0.0),
        },
        "vlm": {"detector": "vlm",
                "response_id": vlm_trace["response_id"], "usage": vlm_trace["usage"],
                "descriptions": vlm_trace.get("descriptions"),
                "reused_from": vlm_trace.get("reused_from")},
    }
    (frame_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # This is a required pixel-wise fusion artifact, independent of the optional
    # diagnostic visualizations below.
    if fusion_heatmap is not None:
        with Image.open(frame.rgb_path) as source:
            fusion_image = source.convert("RGB")
        save_fusion_heatmap(
            fusion_image, boxes, fusion_heatmap, frame.ground_truth_point,
            frame_dir / "fusion_heatmap.png",
        )
    if visualize:
        with Image.open(frame.rgb_path) as source:
            image = source.convert("RGB")
        gesture_image = image
        if has_gesture:
            gesture_image = draw_hand_ray_overlay(
                image, load_hand_ray_overlay(frame.ray_path, intrinsics)
            )
        if gesture_pixel is not None:
            save_gesture_heatmap(gesture_image, gesture_pixel, frame.ground_truth_point,
                                 frame_dir / "gesture_heatmap.png")
        save_vlm_masks(image, boxes, masks, frame.ground_truth_point,
                       frame_dir / "vlm_masks.png")
        if distribution is not None:
            save_mask_distribution(image, masks, distribution, frame.ground_truth_point,
                                   frame_dir / "mask_distribution.png")
    top_mask = (int(np.argmax(distribution)) + 1
                if distribution is not None and len(distribution) else "none")
    print(f"[{frame.frame_id}] gesture={'yes' if has_gesture else 'no'}, top mask={top_mask}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIGU multimodal grounding")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--subdirs", nargs="*")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--analyse", action="store_true",
                        help="analyse accuracy after all selected datasets finish")
    parser.add_argument("--analysis-input-dir", type=Path,
                        help="dataset root containing <group>/ground_truth_masks")
    parser.add_argument("--reuse-result-type", choices=("vlm", "vlm+seg", "main", "grid", "pivot"),
                        default="vlm+seg", help=argparse.SUPPRESS)
    parser.add_argument("--reuse-result-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    input_dir = (args.input_dir.resolve() if args.input_dir else _resolve(config["input_dir"]))
    output_dir = (args.output_dir.resolve() if args.output_dir else _resolve(config["output_dir"]))
    visualize = config.get("visualize", True) if args.visualize is None else args.visualize
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = output_dir / stamp
    reuse_root = args.reuse_result_dir.resolve() if args.reuse_result_dir else None
    reuse_type = args.reuse_result_type if reuse_root else None
    if reuse_root is not None and not reuse_root.is_dir():
        raise NotADirectoryError(f"vlm.reuse_result_dir is not a directory: {reuse_root}")
    client = None
    summaries = []
    groups = select_capture_groups(input_dir,
        args.subdirs if args.subdirs is not None else config.get("subdirs"))
    for group_name, group_dir in groups:
        run_dir = run_root / group_name
        prepared = open_dataset(group_dir).prepare(run_dir / "_prepared_input")
        hand_stats = generate_key_frames(
            prepared.rgb_dir, prepared.depth_dir, prepared.ray_dir,
            prepared.intrinsics_path, config["hand_detection"],
        )
        # VLM/SAM is deliberately run for every RGB frame. Frames without a
        # usable hand ray still produce reusable segmentation artifacts.
        frames = pair_prepared_frames(prepared, require_gesture=False)
        if reuse_root is not None:
            require_reusable_results(reuse_root, reuse_type, group_name, frames)
        if reuse_root is None and client is None:
            client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        results = []
        for frame in frames:
            instruction = prepared.instruction_by_frame.get(frame.frame_id)
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"No instruction for frame {frame.frame_id}")
            results.append(process_frame(
                frame, instruction.strip(), prepared.intrinsics_path, run_dir,
                config, client, visualize,
                hand_stats["processing_time_seconds_by_frame"].get(frame.frame_id),
                None if reuse_root is None else (
                    reusable_vlm_path(reuse_root, group_name, frame.frame_id)
                    if reuse_type == "vlm" else
                    reusable_main_path(reuse_root, group_name, frame.frame_id)
                    if reuse_type == "main" else
                    reusable_semantic_grounding_path(
                        reuse_root, group_name, frame.frame_id, reuse_type
                    ) if reuse_type in {"grid", "pivot"} else
                    reusable_vlm_sam_path(reuse_root, group_name, frame.frame_id)
                ), reuse_type,
            ))
        summary = {"group": group_name, "hand_detection": hand_stats, "frames": results}
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary.json").write_text(
        json.dumps({"created_at": stamp, "groups": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.analyse:
        from .result_analysis import analyse_migu_run
        analysis_input = (args.analysis_input_dir.resolve()
                          if args.analysis_input_dir else input_dir)
        analysis = analyse_migu_run(run_root, analysis_input)
        print(f"accuracy: {analysis['accuracy']}")
    print(f"output: {run_root}")


if __name__ == "__main__":
    main()
