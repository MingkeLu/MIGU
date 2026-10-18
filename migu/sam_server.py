"""Self-hosted HTTP service for bounding-box-prompted SAM segmentation."""
from __future__ import annotations

import argparse
import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from .sam import encode_mask


def _decode_image(value: object) -> Image.Image:
    if not isinstance(value, str):
        raise ValueError("image must be a base64 string")
    try:
        data = base64.b64decode(value, validate=True)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (ValueError, OSError) as error:
        raise ValueError("image is not a valid base64-encoded image") from error


def _validate_boxes(value: object, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("bboxes must be a non-empty list")
    result = []
    for index, box in enumerate(value, 1):
        if (not isinstance(box, list) or len(box) != 4
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       for item in box)):
            raise ValueError(f"bbox {index} must contain four numeric coordinates")
        x_min, y_min, x_max, y_max = map(float, box)
        if not (0 <= x_min < x_max < width and 0 <= y_min < y_max < height):
            raise ValueError(f"bbox {index} is outside the {width}x{height} image")
        result.append([round(x_min), round(y_min), round(x_max), round(y_max)])
    return result


class Segmenter:
    """Own the processor and model used by the Flask request handler."""

    def __init__(self, model_id: str, requested_device: str):
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install the SAM server dependencies with: pip install -e '.[sam-server]'"
            ) from error
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self._torch = torch
        self.device = torch.device(requested_device)
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, boxes: list[list[int]],
                return_best: bool) -> list[np.ndarray]:
        """Return one mask group per input box."""
        inputs = self.processor(
            images=image, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(multimask_output=not return_best, **inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        values = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        if return_best:
            values = values[:, 0]
        return [np.asarray(mask, dtype=bool) for mask in values]


def create_app(segmenter: Any):
    """Create a Flask application around an initialized segmenter."""
    try:
        from flask import Flask, jsonify, request
    except ImportError as error:
        raise RuntimeError(
            "Install the SAM server dependencies with: pip install -e '.[sam-server]'"
        ) from error

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "device": str(segmenter.device)})

    @app.post("/sam_mask_by_bbox")
    def sam_mask_by_bbox():
        try:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            image = _decode_image(payload.get("image"))
            boxes = _validate_boxes(payload.get("bboxes"), image.width, image.height)
            return_best = payload.get("return_best", True)
            if not isinstance(return_best, bool):
                raise ValueError("return_best must be a Boolean")
            masks = segmenter.predict(image, boxes, return_best)
            if len(masks) != len(boxes):
                raise RuntimeError(
                    f"Model returned {len(masks)} mask groups for {len(boxes)} boxes"
                )
            return jsonify({
                "result": [
                    {"segmentation": encode_mask(mask), "encoding": "npz-base64"}
                    for mask in masks
                ],
                "type": "mask by bounding box",
            })
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-hosted MIGU SAM service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--model", default="facebook/sam-vit-huge")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    segmenter = Segmenter(args.model, args.device)
    create_app(segmenter).run(
        host=args.host, port=args.port, debug=False, use_reloader=False
    )


if __name__ == "__main__":
    main()
