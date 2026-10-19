"""The original main pipeline's SoM branch without gesture inference or fusion."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from PIL import Image
def parse_selection_count(value) -> tuple[int, int | None]:
    """Parse n, <=n, or ~ into an inclusive (minimum, maximum) range."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value, value
    if not isinstance(value, str):
        raise ValueError("invalid selection count")
    normalized = value.strip()
    if normalized == "~":
        return 1, None
    if normalized.isdigit() and int(normalized) > 0:
        count = int(normalized)
        return count, count
    if normalized.startswith("<=") and normalized[2:].strip().isdigit():
        maximum = int(normalized[2:].strip())
        if maximum > 0:
            return 1, maximum
    raise ValueError("invalid selection count")


def build_object_ids(box_names: list[str], label_style: str) -> list[str]:
    """Return IDs exactly matching the labels drawn by overlay_results."""
    if label_style == "number":
        return [str(index) for index in range(1, len(box_names) + 1)]
    if label_style != "name-number":
        raise ValueError(f"Unknown box label style: {label_style}")
    counts = defaultdict(int)
    identifiers = []
    for raw_name in box_names:
        name = str(raw_name)
        counts[name] += 1
        identifiers.append(f"{name}#{counts[name]}")
    return identifiers


def run_semantic(
    annotated: Image.Image,
    instruction: dict[str, str],
    object_ids: list[str],
    object_names: list[str],
    args,
):
    from .schemas import DEFAULT_MODEL
    from .vlm import OpenAIVLM

    vlm = OpenAIVLM(
        args.model or DEFAULT_MODEL,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    object_index_by_id = {identifier: index for index, identifier in enumerate(object_ids)}
    if args.send_categories_to_vlm and len(object_names) != len(object_ids):
        raise ValueError(
            f"Object ID/name count mismatch: {len(object_ids)} IDs and "
            f"{len(object_names)} names"
        )
    object_catalog = (
        list(zip(object_ids, map(str, object_names)))
        if args.send_categories_to_vlm
        else None
    )
    confidence = np.zeros(len(object_ids), dtype=np.float64)
    traces = []
    minimum_selections, maximum_selections = args.selection_range
    if minimum_selections > len(object_ids):
        raise ValueError(
            f"VLM must select {minimum_selections} objects per generation, but only "
            f"{len(object_ids)} marked objects are available"
        )
    for trial in range(1, args.generations + 1):
        reply = vlm.som(
            annotated,
            instruction["instruction"],
            instruction.get("query"),
            trial,
            min_selections=minimum_selections,
            max_selections=maximum_selections,
            object_catalog=object_catalog,
        )
        # The reply has already been validated against the configured dynamic schema.
        prediction = reply.parsed
        seen = set()
        selected = []
        for candidate in prediction.objects:
            identifier = candidate.object_id.strip()
            object_index = object_index_by_id.get(identifier)
            if object_index is None or object_index in seen:
                raise ValueError(f"SoM returned invalid/duplicate object ID {candidate.object_id!r}")
            seen.add(object_index)
            confidence[object_index] += candidate.confidence
            selected.append({"object_id": identifier, "confidence": candidate.confidence})
        traces.append({"trial": trial, "target_description": prediction.target_description,
                       "objects": selected, "response_id": reply.response_id, "usage": reply.usage})
    total = confidence.sum()
    probability = confidence / total if total else confidence
    return probability, traces
