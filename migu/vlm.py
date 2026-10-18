"""Structured VLM bounding-box prediction for MIGU."""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, conlist, create_model


SYSTEM_PROMPT = """You visually ground robot instructions in RGB images.
Interpret the complete instruction and identify the referent's semantic role,
category, attributes, and affordances. Treat Robot State as important semantic
evidence for references, held objects, and action requirements. Judge destinations
and supporting surfaces by their physical and functional compatibility with any
held object.

Use pointing hands, finger direction, gaze, relative layout, support relationships,
and scene geometry only to form a broad plausible candidate set. Determine confidence
primarily from semantic compatibility. Do not create large confidence differences
from uncertain 2-D gesture alignment.

For an object referent, tightly bound the visible object. For a destination, bound a
compact usable region with suitable clearance. Do not return the pointing hand unless
explicitly requested. Bounding boxes use pixel coordinates
[x_min, y_min, x_max, y_max], with (0, 0) at the top-left. Return only the requested
structured result."""


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(description="Short description of the object or region")
    x_min: int = Field(description="Left edge in pixels")
    y_min: int = Field(description="Top edge in pixels")
    x_max: int = Field(description="Right edge in pixels")
    y_max: int = Field(description="Bottom edge in pixels")
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max


def parse_bbox_count(value: object) -> tuple[int, int]:
    """Return the allowed minimum and maximum number of boxes."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value, value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*<=\s*([1-9]\d*)\s*", value)
        if match:
            return 1, int(match.group(1))
    raise ValueError("bbox_count must be a positive integer or a string like '<=5'")


def _prediction_model(count: object) -> type[BaseModel]:
    minimum, maximum = parse_bbox_count(count)
    boxes_type = conlist(BoundingBox, min_length=minimum, max_length=maximum)
    count_name = str(count).replace("<=", "UpTo").strip()
    return create_model(
        f"Top{count_name}BoundingBoxes", __config__=ConfigDict(extra="forbid"),
        boxes=(boxes_type, ...),
    )


def _image_data_url(path: Path) -> str:
    buffer = io.BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict(client: OpenAI, model: str, image_path: Path, instruction: str,
            bbox_count: object) -> tuple[BaseModel, str, dict | None]:
    """Ask the configured VLM for ranked, confidence-scored bounding boxes."""
    with Image.open(image_path) as image:
        width, height = image.size
    minimum, maximum = parse_bbox_count(bbox_count)
    selection = (f"exactly {maximum}" if minimum == maximum else
                 f"up to {maximum} (at least one)")
    prompt = f"""Return {selection} distinct object-or-region hypotheses for the
requested referent, ranked from most to least likely. Infer the intended role,
category, attributes, and affordances from the complete instruction and Robot State.
Use spatial evidence only to form a broad plausible set and confidence primarily
from semantic evidence. Keep confidence modest under ambiguity.

Instruction: {instruction!r}
Image size: width={width}, height={height} pixels.

Each box must tightly cover a plausible visible object or compact usable destination,
with 0 <= x_min < x_max <= {width - 1} and
0 <= y_min < y_max <= {height - 1}. Confidence is in [0, 1] and values do not need
to sum to one."""
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _image_data_url(image_path),
                 "detail": "high"},
            ]},
        ],
        text_format=_prediction_model(bbox_count),
        reasoning={"effort": "low"},
        max_output_tokens=4096,
    )
    if response.output_parsed is None:
        raise RuntimeError(
            f"Response {response.id} has no parsed output "
            f"(status={response.status}, incomplete={response.incomplete_details})"
        )
    usage = response.usage.model_dump(mode="json") if response.usage else None
    return response.output_parsed, response.id, usage


def validate_boxes(result: BaseModel, width: int, height: int, count: object) -> None:
    """Validate count and image bounds for a structured VLM response."""
    minimum, maximum = parse_bbox_count(count)
    if not minimum <= len(result.boxes) <= maximum:
        expected = str(maximum) if minimum == maximum else f"1 to {maximum}"
        raise ValueError(f"Expected {expected} boxes, got {len(result.boxes)}")
    for index, item in enumerate(result.boxes, 1):
        if not (0 <= item.x_min < item.x_max < width
                and 0 <= item.y_min < item.y_max < height):
            raise ValueError(
                f"Box {index} is outside the {width}x{height} image: {item.bbox}"
            )
