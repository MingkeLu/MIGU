from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Literal, Protocol

import numpy as np

from .distribution import gaussian_kernel_sum, sample_point_kernels, top_density_peaks
from .schemas import (
    DEFAULT_MODEL,
    DirectModifiedPrediction,
    DirectPrediction,
    GridPrediction,
    ModelReply,
    PivotPrediction,
    SoMPrediction,
)
from .vision import (
    cell_center,
    draw_candidates,
    draw_grid,
    draw_heatmap_overlay,
    draw_object_probabilities,
    load_image,
    reshape_for_grounding,
    valid_cell_ids,
)
from .vlm import OpenAIVLM

Method = Literal["direct", "direct_modified", "grid", "pivot", "som"]


class VLM(Protocol):
    def direct(self, image, instruction: str, query: str | None, trial: int) -> ModelReply: ...

    def direct_modified(
        self, image, instruction: str, query: str, trial: int
    ) -> ModelReply: ...

    def grid(
        self,
        clean_image,
        grid_image,
        instruction: str,
        query: str | None,
        rows: int,
        cols: int,
        trial: int,
    ) -> ModelReply: ...

    def pivot(
        self,
        clean_image,
        proposal_image,
        instruction: str,
        query: str | None,
        candidate_count: int,
        iteration: int,
    ) -> ModelReply: ...

    def som(
        self,
        annotated_image,
        instruction: str,
        query: str | None,
        trial: int,
    ) -> ModelReply: ...


@dataclass(frozen=True)
class GroundingConfig:
    model: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key: str | None = None
    generations: int = 3
    grid_rows: int = 10
    grid_cols: int = 10
    pivot_candidates: int = 10
    pivot_iterations: int = 3
    grounding_width: int = 1280
    grounding_height: int = 720
    gaussian_sigma_fraction: float = 0.035
    seed: int = 7

    def __post_init__(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid dimensions must be positive")
        if self.pivot_candidates < 3 or self.pivot_iterations < 1:
            raise ValueError("PIVOT requires at least 3 candidates and 1 iteration")
        if (self.grounding_width, self.grounding_height) != (1280, 720):
            raise ValueError("The canonical grounding frame must be exactly 1280x720")
        if self.gaussian_sigma_fraction <= 0:
            raise ValueError("gaussian_sigma_fraction must be positive")


def _point_dict(x: float, y: float, **extra) -> dict:
    return {"x": float(x), "y": float(y), **extra}


def _validate_unique(values: list, label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"The model returned duplicate {label}: {values}")


class Grounder:
    """Runs a strategy in the canonical 1280x720 grounding frame."""

    def __init__(
        self,
        config: GroundingConfig | None = None,
        vlm: VLM | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        base = config or GroundingConfig()
        self.config = GroundingConfig(
            model=model if model is not None else base.model,
            base_url=base_url if base_url is not None else base.base_url,
            api_key=api_key if api_key is not None else base.api_key,
            generations=base.generations,
            grid_rows=base.grid_rows,
            grid_cols=base.grid_cols,
            pivot_candidates=base.pivot_candidates,
            pivot_iterations=base.pivot_iterations,
            grounding_width=base.grounding_width,
            grounding_height=base.grounding_height,
            gaussian_sigma_fraction=base.gaussian_sigma_fraction,
            seed=base.seed,
        )
        self.vlm = vlm or OpenAIVLM(
            self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def ground(
        self,
        image_path: str | Path,
        instruction: str,
        query: str | None,
        method: Method,
        output_dir: str | Path,
        *,
        batch_objects: tuple[np.ndarray, list[str], np.ndarray] | None = None,
    ) -> dict:
        if method in {"grid", "pivot"} and batch_objects is None:
            raise ValueError("Grid and PIVOT require segmentation masks for point voting")
        frame_started = time.perf_counter()
        image_path = Path(image_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        source_image = load_image(image_path)
        grounding_image = reshape_for_grounding(
            source_image, self.config.grounding_width, self.config.grounding_height
        )

        if method == "direct":
            payload = self._direct(grounding_image, instruction, query, batch_objects is not None)
        elif method == "direct_modified":
            payload = self._direct_modified(grounding_image, instruction, query)
        elif method == "grid":
            payload = self._grid(grounding_image, instruction, query, output_dir, batch_objects is not None)
        elif method == "pivot":
            payload = self._pivot(grounding_image, instruction, query, output_dir, batch_objects is not None)
        elif method == "som":
            payload = self._som(grounding_image, instruction, query, output_dir)
        else:
            raise ValueError(f"Unknown method: {method}")

        artifacts = dict(payload.get("artifacts", {}))
        if batch_objects is not None and method in {"direct", "grid", "pivot"}:
            masks, names, scores = batch_objects
            payload.pop("distribution", None)
            payload.update(
                self._points_to_objects(
                    payload["samples"], masks, names, scores, grounding_image.size
                )
            )
        elif method != "som":
            samples = [(item["x"], item["y"]) for item in payload["samples"]]
            heatmap_path = output_dir / "grounding_heatmap.jpg"
            draw_heatmap_overlay(
                grounding_image,
                samples,
                payload["distribution"],
                payload["top_points"],
            ).save(heatmap_path, quality=92, optimize=True)
            artifacts["heatmap"] = str(heatmap_path)
        result = {
            "method": method,
            "model": self.config.model,
            "instruction": instruction,
            "query": query,
            "image": {
                "path": str(image_path),
                "source_width": source_image.width,
                "source_height": source_image.height,
                "width": grounding_image.width,
                "height": grounding_image.height,
                "coordinate_system": "(x,y), origin top-left, 1280x720 grounding-frame pixels",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_seconds": time.perf_counter() - frame_started,
            "config": asdict(self.config),
            **payload,
            "artifacts": {
                **artifacts,
            },
        }
        result_path = output_dir / "result.json"
        result["artifacts"]["result"] = str(result_path)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    @staticmethod
    def _points_to_objects(
        samples: list[dict],
        masks: np.ndarray,
        names: list[str],
        scores: np.ndarray,
        grounding_size: tuple[int, int],
    ) -> dict:
        """Convert point votes to an object posterior without spatial smoothing."""
        masks = np.asarray(masks, dtype=bool)
        scores = np.asarray(scores, dtype=float)
        if masks.ndim != 3 or masks.shape[0] != len(names):
            raise ValueError("perception masks must have shape (objects, height, width)")
        if len(scores) != len(names):
            raise ValueError("perception scores and object names must have equal length")
        ground_width, ground_height = grounding_size
        mask_height, mask_width = masks.shape[1:]
        counts = np.zeros(len(names), dtype=int)
        weights = np.zeros(len(names), dtype=float)
        point_hits = []
        for point in samples:
            x = min(mask_width - 1, max(0, int(float(point["x"]) * mask_width / ground_width)))
            y = min(mask_height - 1, max(0, int(float(point["y"]) * mask_height / ground_height)))
            matches = np.flatnonzero(masks[:, y, x]).tolist()
            weight = float(point.get("confidence", 1.0))
            for index in matches:
                counts[index] += 1
                weights[index] += weight
            point_hits.append({**point, "mask_x": x, "mask_y": y,
                               "matching_object_indices": matches})
        total = float(weights.sum())
        distribution = [
            {"object_index": index, "object_name": str(name),
             "selection_count": int(counts[index]),
             "confidence_weight": float(weights[index]),
             "probability": float(weights[index] / total) if total else 0.0,
             "perception_score": float(scores[index])}
            for index, name in enumerate(names)
        ]
        distribution.sort(key=lambda item: (-item["probability"],
                                            -item["selection_count"],
                                            item["object_index"]))
        return {"samples": point_hits, "top_objects": distribution[:5],
                "object_distribution": distribution}

    def _som(
        self,
        image,
        instruction: str,
        query: str | None,
        output_dir: Path,
    ) -> dict:
        prompt_path = output_dir / "som_prompt.jpg"
        image.save(prompt_path, quality=92, optimize=True)
        runs = []
        selections: list[dict] = []
        for trial in range(1, self.config.generations + 1):
            reply = self.vlm.som(image, instruction, query, trial)
            prediction = SoMPrediction.model_validate(reply.parsed)
            object_ids = [item.object_id for item in prediction.objects]
            _validate_unique(object_ids, "SoM object IDs")

            selected = []
            for rank, candidate in enumerate(prediction.objects, start=1):
                item = {
                    "object_id": candidate.object_id,
                    "rank": rank,
                    "confidence": candidate.confidence,
                }
                selected.append(item)
                selections.append({"trial": trial, **item})
            runs.append(
                {
                    "trial": trial,
                    "response_id": reply.response_id,
                    "usage": reply.usage,
                    "target_description": prediction.target_description,
                    "objects": selected,
                }
            )

        counts: dict[str, int] = {}
        confidence_sums: dict[str, float] = {}
        for item in selections:
            object_id = item["object_id"]
            counts[object_id] = counts.get(object_id, 0) + 1
            confidence_sums[object_id] = confidence_sums.get(object_id, 0.0) + item[
                "confidence"
            ]
        total_weight = sum(confidence_sums.values())
        object_distribution = [
            {
                "object_id": object_id,
                "selection_count": counts[object_id],
                "confidence_weight": confidence_sums[object_id],
                "probability": (
                    confidence_sums[object_id] / total_weight if total_weight else 0.0
                ),
            }
            for object_id in sorted(
                confidence_sums,
                key=lambda item: (-confidence_sums[item], -counts[item], item),
            )
        ]
        probability_path = output_dir / "som_probabilities.jpg"
        draw_object_probabilities(image, object_distribution).save(
            probability_path, quality=92, optimize=True
        )
        return {
            "top_objects": object_distribution[:5],
            "object_distribution": object_distribution,
            "selections": selections,
            "trace": {"runs": runs},
            "artifacts": {
                "visual_prompt": str(prompt_path),
                "probability_overlay": str(probability_path),
            },
        }

    def _direct(self, image, instruction: str, query: str | None, batch: bool = False) -> dict:
        runs = []
        samples: list[dict] = []
        for trial in range(1, self.config.generations + 1):
            reply = self.vlm.direct(image, instruction, query, trial)
            prediction = DirectPrediction.model_validate(reply.parsed)
            points = []
            for rank, point in enumerate(prediction.points, start=1):
                x = min(max(point.x, 0), image.width - 1)
                y = min(max(point.y, 0), image.height - 1)
                converted = _point_dict(x, y, rank=rank, confidence=point.confidence)
                points.append(converted)
                samples.append(converted)
            runs.append(
                {
                    "trial": trial,
                    "response_id": reply.response_id,
                    "usage": reply.usage,
                    "target_description": prediction.target_description,
                    "points": points,
                }
            )
        distribution = None if batch else gaussian_kernel_sum(
            [(point["x"], point["y"]) for point in samples],
            image.width,
            image.height,
            self.config.gaussian_sigma_fraction,
        )
        return {
            "top_points": samples[:3] if batch else top_density_peaks(distribution, image.width, image.height),
            "samples": samples,
            "distribution": distribution,
            "trace": {"runs": runs},
        }

    def _direct_modified(self, image, instruction: str, query: str) -> dict:
        runs = []
        samples: list[dict] = []
        for trial in range(1, self.config.generations + 1):
            reply = self.vlm.direct_modified(image, instruction, query, trial)
            prediction = DirectModifiedPrediction.model_validate(reply.parsed)
            points = []
            for rank, point in enumerate(prediction.points, start=1):
                x = min(max(point.x, 0), image.width - 1)
                y = min(max(point.y, 0), image.height - 1)
                converted = _point_dict(x, y, rank=rank, confidence=point.confidence)
                points.append(converted)
                samples.append(converted)
            runs.append(
                {
                    "trial": trial,
                    "response_id": reply.response_id,
                    "usage": reply.usage,
                    "target_description": prediction.target_description,
                    "points": points,
                }
            )
        distribution = gaussian_kernel_sum(
            [(point["x"], point["y"]) for point in samples],
            image.width,
            image.height,
            self.config.gaussian_sigma_fraction,
        )
        return {
            "top_points": top_density_peaks(distribution, image.width, image.height),
            "samples": samples,
            "distribution": distribution,
            "trace": {"runs": runs},
        }

    def _grid(
        self,
        image,
        instruction: str,
        query: str | None,
        output_dir: Path,
        batch: bool = False,
    ) -> dict:
        annotated = draw_grid(image, self.config.grid_rows, self.config.grid_cols)
        grid_path = output_dir / "grid_prompt.jpg"
        annotated.save(grid_path, quality=92, optimize=True)
        allowed = valid_cell_ids(self.config.grid_rows, self.config.grid_cols)
        runs = []
        samples: list[dict] = []
        for trial in range(1, self.config.generations + 1):
            reply = self.vlm.grid(
                image,
                annotated,
                instruction,
                query,
                self.config.grid_rows,
                self.config.grid_cols,
                trial,
            )
            prediction = GridPrediction.model_validate(reply.parsed)
            normalized_ids = [identifier.upper() for identifier in prediction.cell_ids]
            _validate_unique(normalized_ids, "grid cells")
            invalid = [
                identifier for identifier in normalized_ids if identifier not in allowed
            ]
            if invalid:
                raise ValueError(f"The model returned invalid grid cells: {invalid}")
            points = []
            for rank, identifier in enumerate(normalized_ids, start=1):
                x, y = cell_center(
                    identifier,
                    self.config.grid_rows,
                    self.config.grid_cols,
                    image.width,
                    image.height,
                )
                converted = _point_dict(x, y, rank=rank, cell_id=identifier)
                points.append(converted)
                samples.append(converted)
            runs.append(
                {
                    "trial": trial,
                    "response_id": reply.response_id,
                    "usage": reply.usage,
                    "target_description": prediction.target_description,
                    "cells": points,
                }
            )
        distribution = None if batch else gaussian_kernel_sum(
            [(point["x"], point["y"]) for point in samples],
            image.width,
            image.height,
            self.config.gaussian_sigma_fraction,
        )
        return {
            "top_points": samples[:3] if batch else top_density_peaks(distribution, image.width, image.height),
            "samples": samples,
            "distribution": distribution,
            "trace": {"runs": runs},
            "artifacts": {"visual_prompt": str(grid_path)},
        }

    def _initial_pivot_points(
        self, width: int, height: int, rng: np.random.Generator
    ) -> list[tuple[float, float]]:
        count = self.config.pivot_candidates
        cols = int(np.ceil(np.sqrt(count * width / height)))
        rows = int(np.ceil(count / cols))
        cells = [(row, col) for row in range(rows) for col in range(cols)]
        rng.shuffle(cells)
        points = []
        for row, col in cells[:count]:
            x = (col + rng.uniform(0.2, 0.8)) * width / cols
            y = (row + rng.uniform(0.2, 0.8)) * height / rows
            points.append((float(x), float(y)))
        return points

    def _pivot(
        self,
        image,
        instruction: str,
        query: str | None,
        output_dir: Path,
        batch: bool = False,
    ) -> dict:
        rng = np.random.default_rng(self.config.seed)
        proposals = self._initial_pivot_points(image.width, image.height, rng)
        rounds = []
        final_selected: list[tuple[float, float]] = []
        selected_history: list[dict] = []

        for iteration in range(1, self.config.pivot_iterations + 1):
            annotated = draw_candidates(image, proposals)
            prompt_path = output_dir / f"pivot_iteration_{iteration}.jpg"
            annotated.save(prompt_path, quality=92, optimize=True)
            reply = self.vlm.pivot(
                image,
                annotated,
                instruction,
                query,
                len(proposals),
                iteration,
            )
            prediction = PivotPrediction.model_validate(reply.parsed)
            candidate_ids = prediction.candidate_ids
            _validate_unique(candidate_ids, "PIVOT candidate IDs")
            invalid = [
                index for index in candidate_ids if not 0 <= index < len(proposals)
            ]
            if invalid:
                raise ValueError(
                    f"The model returned invalid PIVOT candidate IDs: {invalid}"
                )
            final_selected = [proposals[index] for index in candidate_ids]
            selected_history.extend(
                _point_dict(
                    x, y, iteration=iteration, rank=rank, candidate_id=candidate_id
                )
                for rank, (candidate_id, (x, y)) in enumerate(
                    zip(candidate_ids, final_selected), start=1
                )
            )
            rounds.append(
                {
                    "iteration": iteration,
                    "response_id": reply.response_id,
                    "usage": reply.usage,
                    "target_description": prediction.target_description,
                    "proposals_display_pixels": [
                        _point_dict(x, y, candidate_id=index)
                        for index, (x, y) in enumerate(proposals)
                    ],
                    "selected_candidate_ids": candidate_ids,
                    "visual_prompt": str(prompt_path),
                }
            )
            if iteration < self.config.pivot_iterations:
                # The isotropic term provides the exploration needed to move from
                # coarse proposals to precise locations, and halves each round.
                exploration = 0.12 * (0.5 ** (iteration - 1))
                proposals = sample_point_kernels(
                    final_selected,
                    self.config.pivot_candidates,
                    image.width,
                    image.height,
                    rng,
                    exploration,
                )

        top_points = [
            _point_dict(x, y, rank=index)
            for index, (x, y) in enumerate(final_selected, start=1)
        ]
        samples = selected_history
        distribution = None if batch else gaussian_kernel_sum(
            [(point["x"], point["y"]) for point in samples],
            image.width,
            image.height,
            self.config.gaussian_sigma_fraction,
        )
        return {
            "top_points": top_points,
            "samples": samples,
            "distribution": distribution,
            "trace": {"rounds": rounds},
            "artifacts": {"visual_prompts": [item["visual_prompt"] for item in rounds]},
        }
