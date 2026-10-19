"""VLM semantic-query generation followed by OWLv2 detection.

This module is intentionally separate from ``vlm_bbox``: the original VLM
bounding-box prompt and response schema are part of the baseline and must not be
changed by the OWLv2 ablation.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, conlist, create_model

from migu.vlm import _image_data_url as image_data_url, parse_bbox_count


SYSTEM_PROMPT = """You convert robot instructions and an RGB scene into short
visual search phrases for an open-vocabulary object detector. Interpret the full
instruction and Robot State to identify the intended referent's semantic role,
category, attributes, and affordances. A referent may be an object or a supporting
surface/destination.

Return only concise, concrete descriptions of visible object or surface types.
Use ordinary detector-friendly noun phrases such as "red mug", "wooden table",
or "floor". Include only visually recognizable attributes that help distinguish
the referent. Do not mention pointing, hands, gaze, image positions, relationships,
actions, intended use, free space, or bounding boxes. Do not describe a compact
region within a surface: name the visible supporting surface itself. Do not return
the pointing hand unless it is explicitly requested."""


class SearchDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="A short detector-friendly noun phrase")


def description_model(count: object) -> type[BaseModel]:
    """Build a query schema with the same maximum as the configured box count."""
    _, maximum = parse_bbox_count(count)
    descriptions_type = conlist(SearchDescription, min_length=1, max_length=maximum)
    return create_model(
        f"UpTo{maximum}DetectorQueries",
        __config__=ConfigDict(extra="forbid"),
        descriptions=(descriptions_type, ...),
    )


def generate_descriptions(client: OpenAI, model: str, image_path: Path,
                          instruction: str, count: object):
    """Ask the VLM for semantic detector queries, never for coordinates."""
    _, maximum = parse_bbox_count(count)
    prompt = f"""Return up to {maximum} distinct short visual search phrases,
ranked from most to least semantically plausible for the requested referent.
Use fewer phrases when alternatives are not genuinely plausible. Each phrase must
be a simple noun phrase an open-vocabulary detector can recognize.

Instruction: {instruction!r}"""
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_url(image_path),
                 "detail": "high"},
            ]},
        ],
        text_format=description_model(count),
        reasoning={"effort": "low"},
        max_output_tokens=1024,
    )
    if response.output_parsed is None:
        raise RuntimeError(
            f"Response {response.id} has no parsed output "
            f"(status={response.status}, incomplete={response.incomplete_details})"
        )
    descriptions = [item.text.strip() for item in response.output_parsed.descriptions]
    descriptions = list(dict.fromkeys(item for item in descriptions if item))
    if not descriptions:
        raise RuntimeError("VLM returned no usable OWLv2 search descriptions")
    usage = response.usage.model_dump(mode="json") if response.usage else None
    return descriptions, response.id, usage


def _encoded_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _post(server_url: str, endpoint: str, payload: dict, timeout: float) -> dict:
    response = requests.post(
        f"{server_url.rstrip('/')}/{endpoint}", json=payload, timeout=timeout
    )
    if not response.ok:
        raise RuntimeError(
            f"{endpoint} returned HTTP {response.status_code}: {response.text}"
        )
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"{endpoint} returned a non-object response")
    return result


def detect(image: Image.Image, descriptions: list[str], server_url: str,
           threshold: float, timeout: float, max_boxes: int,
           confidence: float) -> list[dict]:
    """Run OWLv2 and convert its normalized boxes to pipeline pixel boxes."""
    result = _post(server_url, "owl_detect", {
        "image": _encoded_png(image),
        "text_queries": descriptions,
        "bbox_conf_threshold": threshold,
    }, timeout)
    bboxes = result.get("bboxes")
    scores = result.get("scores")
    names = result.get("box_names")
    if not all(isinstance(items, list) for items in (bboxes, scores, names)):
        raise ValueError("owl_detect response must contain bboxes, scores, and box_names lists")
    if not (len(bboxes) == len(scores) == len(names)):
        raise ValueError("owl_detect response fields have inconsistent lengths")

    width, height = image.size
    detections = []
    for bbox, score, name in zip(bboxes, scores, names):
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(isinstance(value, (int, float)) for value in bbox)
                or not isinstance(score, (int, float))):
            raise ValueError("owl_detect returned an invalid detection")
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if not all(np.isfinite([x1, y1, x2, y2, float(score)])):
            raise ValueError("owl_detect returned a non-finite detection")
        pixel_box = [
            max(0, min(width - 1, int(round(x1 * width)))),
            max(0, min(height - 1, int(round(y1 * height)))),
            max(0, min(width - 1, int(round(x2 * width)))),
            max(0, min(height - 1, int(round(y2 * height)))),
        ]
        if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
            continue
        detections.append((float(score), {
            "label": str(name), "bbox": pixel_box, "confidence": confidence,
        }))
    detections.sort(key=lambda item: item[0], reverse=True)
    boxes = [item[1] for item in detections[:max_boxes]]
    if not boxes:
        raise RuntimeError(
            "OWLv2 returned no valid boxes; lower --owlv2-bbox-conf-threshold "
            "or revise the instruction/query prompt"
        )
    return boxes


def run_vlm_owlv2(image_path: Path, instruction: str, config: dict,
                   client: OpenAI, *, server_url: str, threshold: float = 0.15,
                   confidence: float = 1.0):
    """Generate detector queries and return OWLv2 boxes plus trace metadata."""
    started = time.perf_counter()
    descriptions, response_id, usage = generate_descriptions(
        client, config["vlm"]["model"], image_path, instruction,
        config["vlm"]["bbox_count"],
    )
    vlm_seconds = time.perf_counter() - started
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    detector_started = time.perf_counter()
    _, maximum = parse_bbox_count(config["vlm"]["bbox_count"])
    boxes = detect(
        image, descriptions, server_url,
        threshold,
        float(config["sam"].get("timeout", 300)), maximum,
        confidence,
    )
    return boxes, {
        "vlm_bbox_seconds": vlm_seconds,
        "owlv2_seconds": time.perf_counter() - detector_started,
        "response_id": response_id,
        "usage": usage,
        "descriptions": descriptions,
    }
