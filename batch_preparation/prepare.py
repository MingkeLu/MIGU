"""Prepare shared RGB-D inputs, MediaPipe rays, and OWL+SAM masks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import yaml
from migu.datasets import select_capture_groups
from .shared import PROJECT_ROOT, DEFAULT_CONFIG, prepare_group, select_groups
from .perception import overlay_results

HAND_CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17))

def save_ground_truth_mask(perception_path: Path, output_path: Path, point) -> dict:
    """Save the largest mask containing a ground-truth RGB point."""
    # Do not leave a previously matched mask behind if recomputation no longer matches.
    output_path.unlink(missing_ok=True)
    result = {"point": {"x": point.x, "y": point.y}, "path": None}
    with np.load(perception_path, allow_pickle=False) as bundle:
        masks = np.asarray(bundle["masks"], dtype=bool)
        scores = np.asarray(bundle["scores"], dtype=float)
        height, width = masks.shape[-2:]
        if not (0 <= point.x < width and 0 <= point.y < height):
            return {**result, "status": "out_of_bounds", "matching_indices": []}
        matching = np.flatnonzero(masks[:, point.y, point.x]).tolist()
        if not matching:
            return {**result, "status": "no_mask", "matching_indices": []}
        areas = masks.reshape(len(masks), -1).sum(axis=1)
        selected = max(matching, key=lambda index: int(areas[index]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            mask=masks[selected],
            mask_index=np.asarray(selected, dtype=np.int64),
            box_name=np.asarray(bundle["box_names"][selected]),
            score=np.asarray(scores[selected], dtype=np.float32),
            area=np.asarray(areas[selected], dtype=np.int64),
            bbox=np.asarray(bundle["bboxes"][selected], dtype=np.float32),
            ground_truth_point=np.asarray([point.x, point.y], dtype=np.int64),
            source_perception=np.asarray(str(perception_path)),
        )
    return {
        **result, "status": "matched", "path": str(output_path),
        "mask_index": selected, "matching_indices": matching,
        "area": int(areas[selected]),
    }


def draw_hand_keypoints(image: Image.Image, ray_path: Path) -> Image.Image:
    """Overlay cached MediaPipe hand landmarks, when available."""
    if not ray_path.is_file():
        return image
    ray = json.loads(ray_path.read_text(encoding="utf-8"))
    landmarks = ray.get("hand_landmarks_px")
    if not isinstance(landmarks, list):
        landmarks = [item.get("pixel") for item in ray.get("index_finger_landmarks", [])]
    points = [tuple(map(float, point[:2])) for point in landmarks
              if isinstance(point, (list, tuple)) and len(point) >= 2]
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result, "RGBA")
    connections = HAND_CONNECTIONS if len(points) == 21 else tuple(
        (index, index + 1) for index in range(len(points) - 1)
    )
    for start, end in connections:
        draw.line((points[start], points[end]), fill=(0, 255, 80, 235), width=4)
    for x, y in points:
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=(30, 255, 100, 245), outline=(0, 0, 0, 230), width=2)
    return result


def visualize_frame(image_path: Path, perception_path: Path, ray_path: Path,
                    output_path: Path, config: dict) -> None:
    """Draw segmentation and hand keypoints together on one RGB image."""
    with np.load(perception_path, allow_pickle=False) as bundle:
        masks = bundle["masks"]
        bboxes = bundle["bboxes"]
        box_names = bundle["box_names"]
    image = Image.open(image_path).convert("RGB")
    combined = overlay_results(
        image, bboxes, masks, None, box_names=box_names,
        box_label_style=config.get("box_label_style", "number"),
    )
    combined = draw_hand_keypoints(combined, ray_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--subdirs", nargs="*")
    parser.add_argument("--mode", choices=("all", "segmentation"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = args.input_dir or PROJECT_ROOT / config["input_dir"]
    root = args.prepared_dir or PROJECT_ROOT / "batch_preparation" / source.name / "input"
    failed = False
    for group, path in select_groups(source, args.subdirs if args.subdirs is not None else config.get("subdirs")):
        _, manifest = prepare_group(path, root / group, config, group,
                                    overwrite=args.overwrite,
                                    segmentation_only=args.mode == "segmentation",
                                    visualize=args.visualize)
        print(f"{group}: {len(manifest['frame_timings'])} frames; {len(manifest['failures'])} failures")
        failed |= bool(manifest["failures"])
    print(f"Prepared inputs: {root}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
