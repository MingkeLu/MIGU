"""Shared data preparation for live and cached baseline inference."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from migu.datasets import open_dataset, select_capture_groups
from migu.key_frames import generate_key_frames
from migu.sam import masks_to_array, request_masks
from .perception import encode_image, post_json, validate_detection_response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "baselines" / "config.yaml"


def select_groups(source: Path, names=None):
    """Accept both grouped captures and grouped normalized RGB-D datasets."""
    available = sorted(p for p in source.iterdir() if p.is_dir() and
                       ((p / "key_rgb").is_dir() and (p / "key_depth").is_dir()
                        or any(p.glob("*/metadata.json"))))
    if available:
        by_name = {p.name: p for p in available}
        missing = [name for name in (names or []) if name not in by_name]
        if missing:
            raise ValueError(f"Unknown input groups: {', '.join(missing)}")
        return [(name, by_name[name]) for name in names] if names else list(by_name.items())
    return select_capture_groups(source, names)


def segmentation_config_for_group(config: dict, group_name: str) -> dict:
    """Resolve a shared vocabulary or a per-group vocabulary."""
    segmentation = deepcopy(config["segmentation"])
    vocabulary = segmentation.get("vocabulary")
    if isinstance(vocabulary, dict):
        vocabulary = vocabulary.get(group_name)
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError(f"No non-empty segmentation vocabulary for {group_name!r}")
    segmentation["vocabulary"] = vocabulary
    return segmentation


def segment_image(image_or_path: Image.Image | Path, config: dict):
    """Detect with OWLv2, then segment its boxes with the SAM service."""
    image = (Image.open(image_or_path).convert("RGB")
             if isinstance(image_or_path, Path) else image_or_path.convert("RGB"))
    response = post_json(
        config["owl_server_url"], "owl_detect",
        {"image": encode_image(image), "text_queries": config["vocabulary"],
         "bbox_conf_threshold": config["bbox_conf_threshold"]},
        config["timeout"],
    )
    validate_detection_response(response)
    normalized_boxes = np.asarray(response["bboxes"], dtype=np.float32).reshape(-1, 4)
    width, height = image.size
    pixel_boxes = (
        normalized_boxes * np.asarray([width, height, width, height])
    ).astype(int).tolist()
    masks = request_masks(
        config["sam_server_url"], image, pixel_boxes, True, config["timeout"]
    ) if len(pixel_boxes) else []
    return (
        masks_to_array(masks, image.size),
        list(map(str, response["box_names"])),
        np.asarray(response["scores"], dtype=np.float32),
        normalized_boxes,
    )


def generate_live_rays(prepared, config: dict, project_root: Path = PROJECT_ROOT):
    """Use exactly MIGU's MediaPipe settings and fixed ray-fitting defaults."""
    return generate_key_frames(prepared.rgb_dir, prepared.depth_dir, prepared.ray_dir,
                               prepared.intrinsics_path, config["hand_detection"])


def load_perception(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return (np.asarray(data["masks"]), data["box_names"].astype(str).tolist(),
                np.asarray(data["scores"]), np.asarray(data["bboxes"]).reshape(-1, 4))


def save_perception(image_path: Path, output_path: Path, segmentation: dict):
    image = Image.open(image_path).convert("RGB")
    masks, names, scores, boxes = segment_image(image, segmentation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, masks=masks, box_names=np.asarray(names),
                        scores=scores, bboxes=boxes, image_size=np.asarray(image.size),
                        source_image=np.asarray(str(image_path)))


def prepare_group(source: Path, cache: Path, config: dict, group: str, *,
                  overwrite: bool = False, segmentation_only: bool = False,
                  visualize: bool = False):
    """Normalize RGB-D, cache hand rays/masks, and record preparation timings."""
    from .prepare import save_ground_truth_mask, visualize_frame
    cache = cache.resolve()
    source = source.resolve()
    if source == cache or cache in source.parents or source in cache.parents:
        raise ValueError("Preparation source and cache must not contain each other")
    cache.mkdir(parents=True, exist_ok=True)
    previous_path = cache / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    previous_times = previous.get("frame_timings", {})
    # Normalize legacy datasets into the cache too, without mutating source rays.
    original = open_dataset(source).prepare(cache)
    for name, directory in (("key_rgb", original.rgb_dir), ("key_depth", original.depth_dir)):
        target = cache / name
        target.mkdir(exist_ok=True)
        if target.resolve() != directory.resolve():
            for item in directory.glob("*.png"):
                if overwrite or not (target / item.name).exists():
                    shutil.copy2(item, target / item.name)
    intrinsics = cache / "camera_intrinsics.txt"
    if original.intrinsics_path.resolve() != intrinsics.resolve():
        shutil.copy2(original.intrinsics_path, intrinsics)
    (cache / "instructions.json").write_text(json.dumps(original.instruction_by_frame, indent=2))
    (cache / "ground_truth.json").write_text(json.dumps(
        {fid: {"x": p.x, "y": p.y} for fid, p in original.ground_truth_by_frame.items()}, indent=2))
    prepared = open_dataset(cache).prepare(cache)
    expected = {p.stem for p in prepared.rgb_dir.glob("*.png")}
    # Overwrite invalidates generated artifacts, including obsolete rays.
    if overwrite:
        for folder, suffix in (("key_frames", "*.json"), ("perception", "*.npz"),
                               ("ground_truth_masks", "*.npz"), ("visualization", "*.png")):
            for path in (cache / folder).glob(suffix):
                path.unlink()
    ray_ids = {p.stem for p in prepared.ray_dir.glob("*.json")}
    hand = {"processing_time_seconds_by_frame": {}}
    if not segmentation_only and (overwrite or not expected.issubset(previous_times) or not ray_ids):
        hand = generate_live_rays(prepared, config)
    segmentation = segmentation_config_for_group(config, group)
    failures, timings, ground_truth_masks = [], {}, {}
    for rgb in sorted(prepared.rgb_dir.glob("*.png")):
        fid = rgb.stem
        path = cache / "perception" / f"{fid}.npz"
        segmentation_time = previous_times.get(fid, {}).get("segmentation_seconds")
        if overwrite or not path.is_file():
            started = time.perf_counter()
            try:
                save_perception(rgb, path, segmentation)
            except Exception as error:
                failures.append({"frame_id": fid, "error": str(error)})
            segmentation_time = time.perf_counter() - started
        if path.is_file() and fid in original.ground_truth_by_frame:
            ground_truth_masks[fid] = save_ground_truth_mask(
                path, cache / "ground_truth_masks" / f"{fid}.npz",
                original.ground_truth_by_frame[fid])
        if visualize and path.is_file():
            visualize_frame(rgb, path, prepared.ray_dir / f"{fid}.json",
                            cache / "visualization" / rgb.name, segmentation)
        hand_time = hand["processing_time_seconds_by_frame"].get(
            fid, previous_times.get(fid, {}).get("hand_detection_seconds"))
        timings[fid] = {"hand_detection_seconds": hand_time,
                        "segmentation_seconds": segmentation_time}
    manifest = {"group": group, "source_input": str(source), "frame_timings": timings,
                "ground_truth_masks": ground_truth_masks, "failures": failures,
                "hand_detection": hand}
    previous_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return prepared, manifest
