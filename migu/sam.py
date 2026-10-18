"""SAM HTTP client used by the MIGU pipeline."""
from __future__ import annotations

import base64
import io

import numpy as np
import requests
from PIL import Image


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_mask(mask: np.ndarray) -> str:
    """Encode a Boolean mask as a compressed, non-pickle NumPy archive."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, mask=np.asarray(mask, dtype=bool))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_mask(value: str) -> np.ndarray:
    """Decode a mask emitted by the self-hosted MIGU SAM service."""
    try:
        payload = base64.b64decode(value, validate=True)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return np.asarray(archive["mask"], dtype=bool)
    except (ValueError, KeyError, OSError) as error:
        raise ValueError("Invalid SAM mask payload") from error


def request_masks(server_url: str, image: Image.Image, boxes: list[list[int]],
                  return_best: bool, timeout: float) -> list[np.ndarray]:
    """Request one or more SAM masks for every bounding box."""
    response = requests.post(
        f"{server_url.rstrip('/')}/sam_mask_by_bbox",
        json={"image": _encode_image(image), "bboxes": boxes,
              "return_best": return_best},
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"sam_mask_by_bbox returned HTTP {response.status_code}: {response.text}"
        )
    items = response.json().get("result")
    if not isinstance(items, list) or len(items) != len(boxes):
        raise ValueError(f"Expected {len(boxes)} SAM results, received {len(items or [])}")
    masks = []
    for item in items:
        if not isinstance(item, dict) or item.get("encoding") != "npz-base64":
            raise ValueError("SAM response uses an unsupported mask encoding")
        masks.append(decode_mask(item.get("segmentation", "")))
    return masks


def masks_to_array(masks: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    """Validate and stack masks returned by the SAM service."""
    width, height = image_size
    normalized = []
    for index, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[-2:] != (height, width):
            raise ValueError(
                f"Mask {index + 1} has shape {mask.shape[-2:]}, expected {(height, width)}"
            )
        normalized.append(mask.reshape(-1, height, width))
    if not normalized:
        return np.empty((0, height, width), dtype=bool)
    if len({mask.shape[0] for mask in normalized}) != 1:
        raise ValueError("Boxes returned different numbers of SAM mask candidates")
    stacked = np.stack(normalized)
    return stacked[:, 0] if stacked.shape[1] == 1 else stacked
