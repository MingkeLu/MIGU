"""Online grounding API backed by the same configuration and models as MIGU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
from openai import OpenAI
from PIL import Image
import yaml

from migu.gesture_posterior import (
    CameraIntrinsics,
    compute_eye_position,
    depth_to_points,
    pixel_posterior_from_gesture,
)
from migu.pipeline import fuse_pixel_wise
from migu.sam import masks_to_array, request_masks
from migu.vlm import predict, validate_boxes


PUBLIC_CONFIG_KEYS = {
    "input_dir", "subdirs", "output_dir", "hand_detection", "gesture",
    "vlm", "sam", "visualize",
}


def load_config(path: Path) -> dict:
    """Load the public MIGU configuration and reject accidental private options."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = set(config) - PUBLIC_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unsupported public configuration entries: {sorted(unknown)}")
    if set(config["hand_detection"]) != {
        "min_detection_confidence", "min_tracking_confidence",
    }:
        raise ValueError("hand_detection must contain only MediaPipe confidence settings")
    if set(config["gesture"]) != {"mode"}:
        raise ValueError("gesture must contain only mode")
    if set(config["vlm"]) != {"model", "bbox_count"}:
        raise ValueError("vlm must contain only model and bbox_count")
    if not config["sam"].get("return_best", True):
        raise ValueError("sam.return_best must be true")
    return config


def _load_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            depth = np.asarray(archive["depth"], dtype=np.float64)
    else:
        with Image.open(path) as source:
            depth = np.asarray(source, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Depth input must have shape [height, width]")
    return depth * scale


def _fitting_points(payload: dict) -> np.ndarray:
    keypoints = payload["grounding"]["hand_keypoints"]
    try:
        result = np.asarray([keypoints[str(index)] for index in (5, 6, 7, 8)], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The grounding payload requires 3-D joints 5, 6, 7, and 8") from error
    if result.shape != (4, 3) or not np.all(np.isfinite(result)):
        raise ValueError("Index-finger joints must be finite camera-frame XYZ points")
    return result


class GroundingTool:
    """Ground one language instruction using an online pointing payload."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY before starting GroundingTool")
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key,
        )

    def __call__(self, instruction: str, payload: dict) -> dict:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        grounding = payload.get("grounding")
        if not isinstance(grounding, dict):
            raise ValueError("The ray query has no grounding payload")
        rgb_path = Path(grounding["rgb"]).resolve()
        depth_path = Path(grounding["depth"]).resolve()
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError("The grounding RGB-D snapshot is unavailable")
        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        depth = _load_depth(depth_path, float(grounding["depth_scale"]))
        if depth.shape != (image.height, image.width):
            raise ValueError("RGB and depth resolutions differ")

        intrinsics_value = grounding["intrinsics"]
        intrinsics = CameraIntrinsics(**{
            key: float(intrinsics_value[key]) for key in ("fx", "fy", "cx", "cy")
        })
        origin = np.asarray(payload["origin"], dtype=float)
        direction = np.asarray(payload["direction"], dtype=float)
        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("Ray origin and direction must have shape [3]")
        direction /= np.linalg.norm(direction) + 1e-12
        mode = self.config["gesture"]["mode"]
        if mode != "gesture_posterior":
            raise ValueError("The online GroundingTool currently requires gesture.mode=gesture_posterior")

        started = time.perf_counter()
        posterior_result = pixel_posterior_from_gesture(
            depth=depth,
            intrinsics=intrinsics,
            eye_position=compute_eye_position(origin, direction),
            finger_position=origin,
            observed_direction=direction,
            sigma_u=0.009,
            alpha_depth=0.2,
            fitting_joint_coordinates=_fitting_points(payload),
            fitting_covariance_beta=1.5,
            device="auto",
        )
        gesture_seconds = time.perf_counter() - started

        vlm_started = time.perf_counter()
        parsed, response_id, usage = predict(
            self.client, self.config["vlm"]["model"], rgb_path,
            instruction.strip(), self.config["vlm"]["bbox_count"],
        )
        validate_boxes(parsed, image.width, image.height, self.config["vlm"]["bbox_count"])
        boxes = [
            {"label": item.label, "bbox": list(item.bbox),
             "confidence": float(item.confidence)}
            for item in parsed.boxes
        ]
        vlm_seconds = time.perf_counter() - vlm_started

        sam_started = time.perf_counter()
        server_url = os.environ.get("MIGU_SAM_SERVER_URL", self.config["sam"]["server_url"])
        masks = masks_to_array(
            request_masks(
                server_url, image, [item["bbox"] for item in boxes],
                True, float(self.config["sam"].get("timeout", 300)),
            ),
            image.size,
        )
        sam_seconds = time.perf_counter() - sam_started
        scores, _ = fuse_pixel_wise(
            masks, posterior_result["probability"],
            np.asarray([item["confidence"] for item in boxes], dtype=float),
        )
        total = float(scores.sum())
        probabilities = scores / total if total > 0 else np.zeros_like(scores)
        if not len(probabilities) or not np.any(np.isfinite(probabilities)):
            raise ValueError("Grounding produced no valid candidate probability")
        selected = int(np.nanargmax(probabilities))
        valid = masks[selected] & np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            raise ValueError("The selected SAM mask contains no valid depth")
        target = np.median(depth_to_points(depth, intrinsics)[valid], axis=0)
        target_direction = target - origin
        target_direction /= np.linalg.norm(target_direction) + 1e-12
        candidates = [
            {**item, "probability": float(probabilities[index])}
            for index, item in enumerate(boxes)
        ]
        return {
            "status": "ok",
            "capture_id": payload.get("capture_id"),
            "instruction": instruction.strip(),
            "selected_index": selected,
            "selected_candidate": candidates[selected],
            "candidates": candidates,
            "observed_ray": {"origin": origin.tolist(), "direction": direction.tolist()},
            "target_center": target.tolist(),
            "target_direction": target_direction.tolist(),
            "timing_seconds": {
                "gesture_posterior": gesture_seconds,
                "vlm_bbox": vlm_seconds,
                "sam": sam_seconds,
            },
            "vlm": {"response_id": response_id, "usage": usage},
        }


def save_result(result: dict, output_dir: Path) -> Path:
    """Persist one JSON result using its capture identifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_id = str(result.get("capture_id") or int(time.time() * 1000))
    path = output_dir / f"{capture_id}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
