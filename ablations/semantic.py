"""Adapt baseline semantic sources to MIGU's boxes, masks, and confidence API."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from baselines.runner import run_som, write_json
from batch_preparation.shared import load_perception, segment_image, segmentation_config_for_group
from migu.datasets import DatasetFrame
from migu.pipeline import bboxes_to_pixels, load_vlm_sam
from migu.sam import masks_to_array, request_masks

BASELINE_NAMES = {"semantic-som": "VLM-SoM", "semantic-grid": "VLM-Grid",
                  "semantic-pivot": "VLM-PIVOT"}


def find_snapshot(root: Path, group: str, frame_id: str, variant: str) -> Path:
    """Accept current baseline outputs, original nested outputs, and MIGU outputs."""
    frame = root / group / frame_id
    if variant in {"without-geometry", "ray-distance", "ray-angle", "distance-threshold"}:
        path = frame / "vlm_segmentation.npz"
    else:
        path = frame / "result.json"
        method = "grid" if variant == "semantic-grid" else "pivot"
        if not path.is_file() and variant in {"semantic-grid", "semantic-pivot"}:
            path = frame / method / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reusable {variant} result: {path}")
    return path


def object_scores(record: dict, variant: str, count: int) -> np.ndarray:
    """Read baseline semantic scores in the segmentation's original object order."""
    if variant == "semantic-som":
        objects = record.get("objects", [])
        if len(objects) != count:
            raise ValueError("SoM object count does not match the segmentation masks")
        scores = np.asarray([x["semantic_probability"] for x in objects], dtype=float)
    else:
        scores = np.zeros(count, dtype=float)
        seen = set()
        for item in record.get("object_distribution", []):
            index = item["object_index"]
            if not isinstance(index, int) or isinstance(index, bool) or index in seen or not 0 <= index < count:
                raise ValueError("Invalid or duplicate semantic object index")
            scores[index] = item["probability"]
            seen.add(index)
        if seen != set(range(count)):
            raise ValueError("Semantic probabilities do not cover every segmentation mask")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Semantic probabilities must be finite values in [0, 1]")
    return scores


def candidate_boxes(perception, image_size, confidence):
    masks, names, _, bboxes = perception
    masks = np.asarray(masks, dtype=bool)
    width, height = image_size
    if masks.ndim != 3 or masks.shape[1:] != (height, width) or len(masks) != len(names):
        raise ValueError("Candidate masks and labels do not match the source image")
    if not len(masks):
        raise ValueError("Segmentation returned no candidates")
    pixels = bboxes_to_pixels(np.asarray(bboxes), image_size)
    if len(pixels) != len(masks) or len(confidence) != len(masks):
        raise ValueError("Candidate box/mask/confidence counts differ")
    return [{"label": str(name), "bbox": box, "confidence": float(score)}
            for name, box, score in zip(names, pixels, confidence)]


def make_runner(variant, group, group_run, baseline_config, args, cache_root):
    """Build a per-group semantic source without mutating global MIGU functions."""
    snapshot_root = args.reuse_result_dir
    grounder = None
    if variant in {"semantic-grid", "semantic-pivot"} and snapshot_root is None:
        from baselines.semantic.grounder import Grounder, GroundingConfig
        options = baseline_config["methods"][BASELINE_NAMES[variant]]
        accepted = {field.name for field in fields(GroundingConfig)}
        grounder = Grounder(GroundingConfig(**{k: v for k, v in options.items() if k in accepted}),
                            model=baseline_config["vlm"]["model"])

    def run(image_path, instruction, config, client):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        fid = image_path.stem
        trace = {"vlm_bbox_seconds": 0.0, "sam_seconds": 0.0,
                 "response_id": None, "usage": None, "detector": variant}
        if snapshot_root is not None and variant in {
            "without-geometry", "ray-distance", "ray-angle", "distance-threshold"
        }:
            snapshot = find_snapshot(snapshot_root, group, fid, variant)
            return load_vlm_sam(snapshot, image.size)
        if variant == "semantic-owlv2":
            from .owlv2 import run_vlm_owlv2
            boxes, owl_trace = run_vlm_owlv2(
                image_path, instruction, config, client,
                server_url=args.owlv2_server_url or baseline_config["segmentation"]["owl_server_url"],
                threshold=args.owlv2_bbox_conf_threshold, confidence=args.owlv2_confidence)
            started = time.perf_counter()
            masks = masks_to_array(request_masks(
                config["sam"]["server_url"], image, [x["bbox"] for x in boxes], True,
                float(config["sam"]["timeout"])), image.size)
            return boxes, masks, {**trace, **owl_trace, "sam_seconds": time.perf_counter() - started}

        artifact_dir = group_run / fid / "semantic_source"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if snapshot_root is not None:
            snapshot = find_snapshot(snapshot_root, group, fid, variant)
            record = json.loads(snapshot.read_text())
            possible = [snapshot.with_name("perception.npz"),
                        snapshot.parent.parent / "perception.npz",
                        cache_root / group / "perception" / f"{fid}.npz"]
            perception_path = next((p for p in possible if p.is_file()), None)
            if perception_path is None:
                raise FileNotFoundError(
                    f"Masks for {snapshot} are missing; supply the matching original --prepared-dir")
            perception = load_perception(perception_path)
            trace["reused_from"] = str(snapshot)
        else:
            started = time.perf_counter()
            if args.batch:
                perception = load_perception(cache_root / group / "perception" / f"{fid}.npz")
            else:
                perception = segment_image(image, segmentation_config_for_group(baseline_config, group))
                trace["sam_seconds"] = time.perf_counter() - started
            record = None
        masks, names, scores, bboxes = perception
        np.savez_compressed(artifact_dir / "perception.npz", masks=masks,
                            box_names=np.asarray(names), scores=scores, bboxes=bboxes,
                            image_size=np.asarray(image.size))
        if variant == "without-semantics":
            confidence = np.ones(len(masks), dtype=float)
        else:
            if record is None:
                started = time.perf_counter()
                if variant == "semantic-som":
                    frame = DatasetFrame(fid, image_path, None, None)
                    record = run_som(frame, instruction, perception, baseline_config, artifact_dir)
                else:
                    method = "grid" if variant == "semantic-grid" else "pivot"
                    record = grounder.ground(image_path, instruction, None, method, artifact_dir,
                                             batch_objects=(masks, names, scores))
                trace["vlm_bbox_seconds"] = time.perf_counter() - started
            confidence = object_scores(record, variant, len(masks))
            write_json(artifact_dir / "result.json", record)
        return candidate_boxes(perception, image.size, confidence), np.asarray(masks, bool), trace

    return run
