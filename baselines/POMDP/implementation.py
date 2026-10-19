"""Per-frame baseline inference, shared by cached and live runs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from batch_preparation.shared import segment_image
from .model import (CandidateObject, EmbeddingLanguageSimilarityModel, POMDPConfig,
                    OpenAIEmbeddingBackend, OpenAISoMLanguageModel,
                    hashing_embedding_backend, infer)
from .visualization import save_visualizations

def resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_intrinsics(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"fx", "fy", "cx", "cy"}:
                values[key.strip()] = float(value)
    missing = {"fx", "fy", "cx", "cy"} - values.keys()
    if missing:
        raise ValueError(f"camera intrinsics missing: {', '.join(sorted(missing))}")
    return values


def detect_objects(image_path: Path, config: dict):
    return segment_image(image_path, config)


def load_perception(path: Path):
    with np.load(path, allow_pickle=False) as data:
        names = data["box_names"].astype(str).tolist()
        boxes = (np.asarray(data["bboxes"]).reshape(-1, 4)
                 if "bboxes" in data else np.full((len(names), 4), np.nan))
        return (np.asarray(data["masks"]), names,
                np.asarray(data["scores"]), boxes)


def load_pointing_ray(path: Path) -> tuple[np.ndarray, np.ndarray, str | None, dict]:
    ray = json.loads(path.read_text(encoding="utf-8"))
    origin = np.asarray(ray.get("origin"), dtype=np.float64)
    direction = np.asarray(ray.get("direction"), dtype=np.float64)
    if (origin.shape != (3,) or direction.shape != (3,)
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) == 0):
        raise ValueError(f"{path} does not contain a valid cached pointing ray")
    return origin, direction, ray.get("pointing_arm"), ray


def mask_centroid(mask: np.ndarray) -> list[float] | None:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    return [float(columns.mean()), float(rows.mean())] if len(rows) else None


def position_for(center: list[float] | None, width: int) -> str | None:
    if center is None:
        return None
    fraction = center[0] / width
    return "left" if fraction < 1 / 3 else "right" if fraction > 2 / 3 else "center"


def make_som_image(image: Image.Image, candidates: list[CandidateObject]) -> Image.Image:
    """Overlay segmentation candidates with translucent masks, boxes, and IDs."""
    annotated = image.convert("RGBA")
    font_size = max(18, round(min(image.size) * 0.035))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()
    colors = ((255, 48, 48), (0, 190, 255), (255, 210, 0), (50, 220, 90),
              (220, 70, 255), (255, 130, 20))
    line_width = max(3, round(min(image.size) / 240))
    for index, candidate in enumerate(candidates):
        color = colors[index % len(colors)]
        mask = np.asarray(candidate.mask, dtype=bool) if candidate.mask is not None else None
        if mask is not None and mask.shape == (image.height, image.width):
            overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
            overlay[mask] = (*color, 45)
            annotated = Image.alpha_composite(annotated, Image.fromarray(overlay, "RGBA"))
        box = candidate.bbox
        if box is None and mask is not None and np.any(mask):
            rows, columns = np.nonzero(mask)
            box = [columns.min(), rows.min(), columns.max(), rows.max()]
        if box is None:
            continue
        draw = ImageDraw.Draw(annotated, "RGBA")
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=line_width)
        label = f"ID {candidate.id}"
        label_y = min(float(image.height - 1), max(float(font_size + 3), y1))
        label_box = draw.textbbox((x1, label_y), label, font=font, anchor="ls", stroke_width=2)
        draw.rectangle((label_box[0] - 3, label_box[1] - 2,
                        label_box[2] + 3, label_box[3] + 2), fill=(0, 0, 0, 225))
        draw.text((x1, label_y), label, font=font, anchor="ls", fill=(*color, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return annotated.convert("RGB")


def ground_truth_index(point, masks: np.ndarray, scores: np.ndarray) -> int | None:
    if point is None:
        return None
    if not (0 <= point.y < masks.shape[1] and 0 <= point.x < masks.shape[2]):
        return None
    matches = np.flatnonzero(masks[:, point.y, point.x]).tolist()
    return max(matches, key=lambda i: float(scores[i])) if matches else None


def process_frame(frame, utterance: str, args, output: Path) -> dict:
    frame_started = time.perf_counter()
    masks, names, scores, boxes = (load_perception(args.batch_input / "perception" / f"{frame.frame_id}.npz")
                                    if args.batch else detect_objects(frame.rgb_path, args.segmentation))
    if not names:
        raise RuntimeError("segmentation returned no candidates")
    image = Image.open(frame.rgb_path).convert("RGB")
    width, height = image.size
    candidates = []
    for index, (name, mask) in enumerate(zip(names, masks)):
        center = mask_centroid(mask)
        box = boxes[index].tolist() if index < len(boxes) and np.all(np.isfinite(boxes[index])) else None
        candidates.append(CandidateObject(
            id=str(index), label=name,
            attributes={"color": None, "category": name,
                        "position": position_for(center, width), "other": []},
            center_2d=center, bbox=box, mask=mask,
        ))
    if frame.ray_path is not None and frame.ray_path.is_file():
        ray_origin, ray_direction, arm, source_ray = load_pointing_ray(frame.ray_path)
    else:
        ray_origin, ray_direction, arm, source_ray = None, None, None, None
    language_model = args.language_model
    if args.language_provider == "som":
        som_image = make_som_image(image, candidates)
        output.mkdir(parents=True, exist_ok=True)
        som_image.save(output / "som_prompt.jpg", quality=92, optimize=True)
        language_model = OpenAISoMLanguageModel(
            args.language_model_name, som_image, base_url=args.language_base_url,
            api_key=args.language_api_key,
        )
    result = infer(
        candidates, utterance, image_size=(width, height), intrinsics=args.intrinsics,
        language_model=language_model, config=args.model_config, pointing_arm=arm,
        pointing_origin=ray_origin, pointing_direction=ray_direction,
    )
    predicted = int(result["top1_id"])
    gt_index = ground_truth_index(frame.ground_truth_point, masks, scores)
    result.update({
        "frame_id": frame.frame_id, "utterance": utterance,
        "candidate_objects": [{"id": item.id, "label": item.label,
                               "attributes": item.attributes, "center_2d": item.center_2d,
                               "bbox": item.bbox} for item in candidates],
        "predicted_index": predicted, "predicted_object": names[predicted],
        "source_pointing_ray": source_ray, "ground_truth_index": gt_index,
        "correct": bool(gt_index is not None and predicted == gt_index),
        "processing_time_seconds": time.perf_counter() - frame_started,
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "perception.npz", masks=masks,
                        box_names=np.asarray(names), scores=scores, bboxes=boxes)
    if args.visualize:
        save_visualizations(frame.rgb_path, masks, names, result, output)
    return result
