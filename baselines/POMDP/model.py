"""POMDP baseline using the original LEGS language--gesture fusion model."""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hashlib
import io
import json
import re
from typing import Any, Callable, List, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image


EmbeddingBackend = Callable[[str, List[str]], np.ndarray]


@dataclass
class CandidateObject:
    id: str
    label: str
    attributes: dict = field(default_factory=dict)
    center_2d: Sequence[float] | None = None
    bbox: Sequence[float] | None = None
    mask: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class POMDPConfig:
    epsilon_l_minus: float = 0.05
    epsilon_l_plus: float = 0.95
    w_g: float = 1.0
    w_l: float = 1.0
    sigma_g_min_deg: float = 5.0
    sigma_g_max_deg: float = 45.0
    sigma_g_scale: float = 2.0
    sigma_g_ray_deg: float = 15.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not 0 < self.epsilon_l_minus < self.epsilon_l_plus < 1:
            raise ValueError("expected 0 < epsilon_l_minus < epsilon_l_plus < 1")
        if self.w_g < 0 or self.w_l < 0:
            raise ValueError("fusion weights must be non-negative")
        if not 0 < self.sigma_g_min_deg <= self.sigma_g_max_deg:
            raise ValueError("sigma_g bounds must be positive and ordered")
        if self.sigma_g_scale < 0 or self.sigma_g_ray_deg <= 0 or not 0 < self.eps < 1:
            raise ValueError("sigma_g_scale must be non-negative and eps must be in (0,1)")


class LanguageSimilarityModel(Protocol):
    def score(self, utterance: str, candidate: CandidateObject,
              all_candidates: list[CandidateObject]) -> float: ...


def candidate_description(candidate: CandidateObject) -> str:
    attrs = candidate.attributes or {}
    parts = [str(attrs.get("color") or "").strip(),
             str(attrs.get("category") or candidate.label).strip()]
    other = attrs.get("other") or []
    parts.extend(str(value).strip() for value in other if str(value).strip())
    text = " ".join(value for value in parts if value) or str(candidate.label)
    position = str(attrs.get("position") or "").strip()
    return f"a {text}" + (f" on the {position} side" if position else "")


class EmbeddingLanguageSimilarityModel:
    """Cosine similarity model backed by an injected batch embedding callable."""

    def __init__(self, embed: EmbeddingBackend):
        self.embed = embed

    def score_all(self, utterance: str, candidates: list[CandidateObject]) -> np.ndarray:
        descriptions = [candidate_description(item) for item in candidates]
        vectors = np.asarray(self.embed(utterance, descriptions), dtype=np.float64)
        if vectors.shape[0] != len(descriptions) + 1 or vectors.ndim != 2:
            raise ValueError("embedding backend must return shape [1 + candidates, dimension]")
        query, targets = vectors[0], vectors[1:]
        denominator = np.linalg.norm(targets, axis=1) * np.linalg.norm(query)
        cosine = np.divide(targets @ query, denominator, out=np.zeros(len(targets)),
                           where=denominator > 0)
        return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

    def score(self, utterance: str, candidate: CandidateObject,
              all_candidates: list[CandidateObject]) -> float:
        index = next((i for i, item in enumerate(all_candidates) if item is candidate), None)
        if index is None:
            raise ValueError("candidate is not present in all_candidates")
        return float(self.score_all(utterance, all_candidates)[index])


class OpenAIEmbeddingBackend:
    """Batch embedding callable using the official OpenAI Python SDK."""

    def __init__(self, model: str = "text-embedding-3-small", *,
                 base_url: str | None = None, api_key: str | None = None,
                 dimensions: int | None = None, client: Any = None):
        if not model.strip():
            raise ValueError("OpenAI embedding model must be non-empty")
        if dimensions is not None and dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.model = model.strip()
        self.dimensions = dimensions
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "OpenAI embeddings require the 'openai' package; "
                    "install it or select the hashing provider"
                ) from error
            options = {}
            if base_url:
                options["base_url"] = base_url
            if api_key:
                options["api_key"] = api_key
            client = OpenAI(**options)
        self.client = client

    def __call__(self, utterance: str, descriptions: list[str]) -> np.ndarray:
        inputs = [utterance, *descriptions]
        if any(not isinstance(value, str) or not value.strip() for value in inputs):
            raise ValueError("OpenAI embedding inputs must be non-empty strings")
        request = {"model": self.model, "input": inputs, "encoding_format": "float"}
        if self.dimensions is not None:
            request["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(inputs):
            raise RuntimeError("OpenAI returned an unexpected number of embeddings")
        return np.asarray([item.embedding for item in ordered], dtype=np.float64)


class OpenAISoMLanguageModel:
    """Score candidate semantics from a Set-of-Marks image with a vision model."""

    def __init__(self, model: str, image: Image.Image, *,
                 base_url: str | None = None, api_key: str | None = None,
                 client: Any = None):
        if not model.strip():
            raise ValueError("SoM vision-language model must be non-empty")
        self.model = model.strip()
        self.image = image.convert("RGB")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("SoM language grounding requires the 'openai' package") from error
            options = {}
            if base_url:
                options["base_url"] = base_url
            if api_key:
                options["api_key"] = api_key
            client = OpenAI(**options)
        self.client = client

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def score_all(self, utterance: str, candidates: list[CandidateObject]) -> np.ndarray:
        identifiers = [item.id for item in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("SoM candidate IDs must be unique")
        catalog = [{"id": item.id, "category": item.label,
                    "description": candidate_description(item)} for item in candidates]
        prompt = (
            "Ground the target described by the instruction in this Set-of-Marks image. "
            "Every candidate is visibly marked with its ID. Use the instruction, candidate "
            "appearance, category, attributes, and spatial/object relations. Ignore hands, "
            "pointing direction, gaze, and gesture because those are modeled separately. "
            "Return every catalog ID exactly once with an independent semantic match score "
            "kappa in [0,1]; scores do not need to sum to one.\n"
            f"Instruction: {utterance!r}\nCandidate catalog: "
            + json.dumps(catalog, ensure_ascii=False)
        )
        schema = {
            "type": "object",
            "properties": {"scores": {"type": "array", "items": {
                "type": "object",
                "properties": {"id": {"type": "string"},
                               "kappa": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["id", "kappa"], "additionalProperties": False}}},
            "required": ["scores"], "additionalProperties": False,
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content":
                 "You estimate language-conditioned object identity likelihoods for a robot."},
                {"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": self._data_url(self.image),
                     "detail": "high"},
                ]},
            ],
            text={"format": {"type": "json_schema", "name": "som_language_scores",
                              "strict": True, "schema": schema}},
        )
        score_by_id = {}
        for item in json.loads(response.output_text).get("scores", []):
            identifier = str(item["id"])
            if identifier in score_by_id:
                raise RuntimeError(f"SoM model returned duplicate candidate ID {identifier!r}")
            score_by_id[identifier] = float(item["kappa"])
        missing = set(identifiers) - score_by_id.keys()
        unknown = score_by_id.keys() - set(identifiers)
        if missing or unknown:
            raise RuntimeError(
                f"SoM model returned invalid candidate IDs; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return np.asarray([score_by_id[item] for item in identifiers], dtype=np.float64)


def hashing_embedding_backend(utterance: str, descriptions: list[str], dimensions: int = 256) -> np.ndarray:
    """Small deterministic offline backend; callers can replace it with real embeddings."""
    rows = []
    for text in [utterance, *descriptions]:
        vector = np.zeros(dimensions, dtype=np.float64)
        for token in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        rows.append(vector)
    return np.stack(rows)


def _normalize_points(points: np.ndarray, image_size: tuple[int, int] | None,
                      coordinate_space: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not np.all(np.isfinite(result)):
        raise ValueError("2-D points must be a finite array of shape [K,2]")
    space = coordinate_space
    if space == "auto":
        space = "normalized" if np.all((result >= 0) & (result <= 1)) else "pixel"
    if space == "pixel":
        if image_size is None or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("pixel coordinates require positive image_size=(width,height)")
        result = result / np.asarray(image_size, dtype=np.float64)
    elif space != "normalized":
        raise ValueError("coordinate_space must be auto, pixel, or normalized")
    return result


def project_finger_joints(finger_joints: np.ndarray, image_size: tuple[int, int] | None,
                          coordinate_space: str = "auto",
                          intrinsics: Mapping[str, float] | None = None) -> np.ndarray:
    joints = np.asarray(finger_joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] not in (2, 3):
        raise ValueError("finger_joints must have shape [K,2] or [K,3], K >= 2")
    if joints.shape[1] == 3:
        if intrinsics is None or image_size is None:
            raise ValueError("3-D finger joints require camera intrinsics and image_size")
        z = joints[:, 2]
        if not np.all(np.isfinite(joints)) or np.any(z <= 0):
            raise ValueError("3-D finger joints must be finite and in front of the camera")
        pixels = np.column_stack((
            float(intrinsics["fx"]) * joints[:, 0] / z + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * joints[:, 1] / z + float(intrinsics["cy"]),
        ))
        return _normalize_points(pixels, image_size, "pixel")
    return _normalize_points(joints, image_size, coordinate_space)


def _candidate_center(candidate: CandidateObject, image_size: tuple[int, int] | None,
                      coordinate_space: str) -> np.ndarray:
    if candidate.mask is not None:
        rows, columns = np.nonzero(np.asarray(candidate.mask, dtype=bool))
        if len(rows):
            return _normalize_points(np.asarray([[columns.mean(), rows.mean()]]), image_size, "pixel")[0]
    if candidate.center_2d is not None:
        return _normalize_points(np.asarray([candidate.center_2d]), image_size, coordinate_space)[0]
    if candidate.bbox is not None:
        box = np.asarray(candidate.bbox, dtype=np.float64)
        if box.shape != (4,):
            raise ValueError(f"candidate {candidate.id!r} bbox must contain four numbers")
        return _normalize_points(np.asarray([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]),
                                 image_size, coordinate_space)[0]
    raise ValueError(f"candidate {candidate.id!r} has no mask, center_2d, or bbox")


def _softmax(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values.copy()
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _distribution(likelihood: np.ndarray) -> np.ndarray:
    return _softmax(np.log(np.clip(likelihood, np.finfo(float).tiny, None)))


def infer(candidates: Sequence[CandidateObject], utterance: str | None = None,
          finger_joints: np.ndarray | None = None, *, image_size: tuple[int, int] | None = None,
          coordinate_space: str = "auto", intrinsics: Mapping[str, float] | None = None,
          language_model: LanguageSimilarityModel | None = None,
          config: POMDPConfig | None = None, pointing_arm: str | None = None,
          pointing_origin: np.ndarray | None = None,
          pointing_direction: np.ndarray | None = None) -> dict:
    """Return normalized gesture, language, and fused candidate distributions."""
    items = list(candidates)
    if not items:
        raise ValueError("at least one candidate is required")
    cfg = config or POMDPConfig()
    warnings: list[str] = []
    centers = np.stack([_candidate_center(item, image_size, coordinate_space) for item in items])

    cached_ray_active = pointing_origin is not None or pointing_direction is not None
    gesture_active = cached_ray_active or finger_joints is not None
    angles = np.full(len(items), np.nan)
    gesture_likelihood = np.ones(len(items))
    origin = direction = None
    sigma_deg = None
    valid_cues: list[str] = []
    if gesture_active:
        try:
            if cached_ray_active:
                ray_origin = np.asarray(pointing_origin, dtype=np.float64)
                ray_direction = np.asarray(pointing_direction, dtype=np.float64)
                if ray_origin.shape != (3,) or ray_direction.shape != (3,):
                    raise ValueError("cached pointing origin and direction must have shape [3]")
                if not np.all(np.isfinite(ray_origin)) or not np.all(np.isfinite(ray_direction)):
                    raise ValueError("cached pointing ray must be finite")
                ray_norm = float(np.linalg.norm(ray_direction))
                if ray_norm <= cfg.eps:
                    raise ValueError("cached pointing direction has zero length")
                ray_direction = ray_direction / ray_norm
                projected = project_finger_joints(
                    np.stack((ray_origin, ray_origin + ray_direction)),
                    image_size, "pixel", intrinsics,
                )
                central = projected[1] - projected[0]
                norm = float(np.linalg.norm(central))
                if norm <= cfg.eps:
                    raise ValueError("cached pointing ray has a degenerate image projection")
                direction = central / norm
                origin = projected[0]
                sigma_deg = float(np.clip(cfg.sigma_g_ray_deg,
                                          cfg.sigma_g_min_deg, cfg.sigma_g_max_deg))
                valid_cues = ["cached_3d_pointing_ray"]
            else:
                points = project_finger_joints(finger_joints, image_size, coordinate_space, intrinsics)
                central = points[-1] - points[0]
                norm = float(np.linalg.norm(central))
                if norm <= cfg.eps:
                    raise ValueError("first and last finger joints are coincident")
                direction = central / norm
                origin = points[-1]
                segments = np.diff(points, axis=0)
                norms = np.linalg.norm(segments, axis=1)
                usable = norms > cfg.eps
                if not np.any(usable):
                    raise ValueError("finger joints contain no valid segments")
                unit = segments[usable] / norms[usable, None]
                spread = np.arccos(np.clip(unit @ direction, -1.0, 1.0))
                sigma_deg = float(np.clip(
                    np.degrees(np.sqrt(np.mean(spread ** 2))) * cfg.sigma_g_scale,
                    cfg.sigma_g_min_deg, cfg.sigma_g_max_deg,
                ))
                valid_cues = ["finger_joint_direction", "finger_direction_spread"]
            delta = centers - origin
            delta_norm = np.linalg.norm(delta, axis=1)
            cosine = np.divide(delta @ direction, delta_norm, out=np.full(len(items), -1.0),
                               where=delta_norm > cfg.eps)
            angles = np.arccos(np.clip(cosine, -1.0, 1.0))
            sigma = np.radians(sigma_deg)
            gesture_likelihood = np.exp(-(angles ** 2) / (2.0 * sigma ** 2))
        except (ValueError, KeyError) as error:
            gesture_active = False
            warnings.append(f"gesture unavailable: {error}")

    language_active = bool(utterance and utterance.strip())
    similarity = np.full(len(items), np.nan)
    language_likelihood = np.ones(len(items))
    language_uninformative = False
    if language_active:
        model = language_model or EmbeddingLanguageSimilarityModel(hashing_embedding_backend)
        if hasattr(model, "score_all"):
            similarity = np.asarray(model.score_all(utterance.strip(), items), dtype=np.float64)
        else:
            similarity = np.asarray([model.score(utterance.strip(), item, items) for item in items])
        if similarity.shape != (len(items),) or not np.all(np.isfinite(similarity)):
            raise ValueError("language model must return one finite score per candidate")
        similarity = np.clip(similarity, 0.0, 1.0)
        language_likelihood = (cfg.epsilon_l_minus
                               + (cfg.epsilon_l_plus - cfg.epsilon_l_minus) * similarity)
        language_uninformative = bool(np.ptp(similarity) <= cfg.eps)
        if language_uninformative:
            warnings.append("language similarities are identical across candidates")

    gesture_probability = (_distribution(gesture_likelihood) if gesture_active
                           else np.full(len(items), 1.0 / len(items)))
    language_probability = (_distribution(language_likelihood) if language_active
                            else np.full(len(items), 1.0 / len(items)))
    log_scores = np.zeros(len(items))
    active = []
    if gesture_active:
        log_scores += cfg.w_g * np.log(np.clip(gesture_likelihood, cfg.eps, 1.0))
        active.append("gesture")
    if language_active:
        log_scores += cfg.w_l * np.log(np.clip(language_likelihood, cfg.eps, 1.0))
        active.append("language")
    fused_probability = _softmax(log_scores)
    records = [{
        "id": item.id,
        "gesture_angle_deg": float(np.degrees(angles[i])) if np.isfinite(angles[i]) else None,
        "gesture_likelihood": float(gesture_likelihood[i]) if gesture_active else None,
        "gesture_probability": float(gesture_probability[i]),
        "language_similarity": float(similarity[i]) if language_active else None,
        "language_likelihood": float(language_likelihood[i]) if language_active else None,
        "language_probability": float(language_probability[i]),
        "fused_log_score": float(log_scores[i]),
        "fused_probability": float(fused_probability[i]),
    } for i, item in enumerate(items)]
    records.sort(key=lambda value: value["fused_probability"], reverse=True)
    return {
        "ranked_candidates": records,
        "top1_id": records[0]["id"],
        "gesture_distribution": {item.id: float(gesture_probability[i]) for i, item in enumerate(items)},
        "language_distribution": {item.id: float(language_probability[i]) for i, item in enumerate(items)},
        "fused_distribution": {item.id: float(fused_probability[i]) for i, item in enumerate(items)},
        "gesture": {
            "origin": origin.tolist() if origin is not None else None,
            "central_direction": direction.tolist() if direction is not None else None,
            "sigma_g_deg": sigma_deg,
            "valid_cues": valid_cues,
            "pointing_arm": pointing_arm,
        },
        "diagnostics": {
            "active_modalities": active,
            "language_uninformative": language_uninformative,
            "warnings": warnings,
        },
    }
