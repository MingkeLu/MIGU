from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = ((255, 45, 45), (0, 190, 255), (255, 200, 0),
          (80, 220, 100), (210, 80, 255))


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size
        )
    except OSError:
        return ImageFont.load_default()


def draw_ground_truth(image: Image.Image, point) -> Image.Image:
    canvas = image.convert("RGB").copy()
    if point is None:
        return canvas
    draw = ImageDraw.Draw(canvas)
    x, y = int(point.x), int(point.y)
    width = max(3, round(min(canvas.size) / 300))
    radius = max(10, round(min(canvas.size) / 70))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill="white", outline=(255, 0, 255), width=width)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 0, 255), width=width)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 0, 255), width=width)
    draw.text((x + radius + 5, y - radius), "GT", fill=(255, 0, 255),
              font=_font(max(18, round(min(canvas.size) / 45))),
              stroke_width=2, stroke_fill="white")
    return canvas


def save_gesture_heatmap(image: Image.Image, probability: np.ndarray, point, path: Path) -> None:
    values = np.asarray(probability, dtype=np.float32)
    peak = float(values.max()) if values.size else 0.0
    scaled = np.uint8(np.clip(values / peak if peak else values, 0, 1) * 255)
    heat = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = Image.blend(image.convert("RGB"), Image.fromarray(heat), 0.48)
    draw_ground_truth(overlay, point).save(path)


def save_fusion_heatmap(image: Image.Image, boxes: list[dict], probability: np.ndarray,
                        point, path: Path) -> None:
    """Overlay positive fused probabilities and annotate VLM boxes and ground truth."""
    values = np.asarray(probability, dtype=np.float32)
    if values.shape != (image.height, image.width):
        raise ValueError(
            f"Fusion heatmap {values.shape} does not match image size {image.size}"
        )
    peak = float(values.max()) if values.size else 0.0
    scaled = np.uint8(np.clip(values / peak if peak else values, 0, 1) * 255)
    heat = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = source.copy()
    selected = values > 0
    # Keep the original image outside the selected masks, while rendering the
    # positive-probability area as a standalone heatmap (without image texture).
    result[selected] = heat[selected]
    canvas = Image.fromarray(result)
    draw = ImageDraw.Draw(canvas)
    font = _font(max(18, round(min(canvas.size) / 45)))
    width = max(3, round(min(canvas.size) / 300))
    for index, item in enumerate(boxes):
        color = COLORS[index % len(COLORS)]
        bbox = tuple(item["bbox"])
        draw.rectangle(bbox, outline=color, width=width)
        label = f"{index + 1}: {float(item['confidence']):.3f}"
        draw.text((bbox[0] + 3, max(0, bbox[1] - 24)), label, fill=color,
                  font=font, stroke_width=2, stroke_fill="black")
    draw_ground_truth(canvas, point).save(path)


def _mask_canvas(image: Image.Image, masks: np.ndarray) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for index, mask in enumerate(masks):
        selected = np.asarray(mask, dtype=bool)
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        canvas[selected] = canvas[selected] * 0.5 + color * 0.5
    return Image.fromarray(canvas.astype(np.uint8))


def save_vlm_masks(image: Image.Image, boxes: list[dict], masks: np.ndarray,
                   point, path: Path) -> None:
    result = _mask_canvas(image, masks)
    draw = ImageDraw.Draw(result)
    font = _font(max(18, round(min(result.size) / 45)))
    width = max(3, round(min(result.size) / 300))
    for index, item in enumerate(boxes):
        color = COLORS[index % len(COLORS)]
        bbox = tuple(item["bbox"])
        draw.rectangle(bbox, outline=color, width=width)
        # Deliberately omit the VLM attribute/name; only rank and confidence are shown.
        label = f"{index + 1}: {float(item['confidence']):.3f}"
        draw.text((bbox[0] + 3, max(0, bbox[1] - 24)), label, fill=color,
                  font=font, stroke_width=2, stroke_fill="black")
    draw_ground_truth(result, point).save(path)


def save_mask_distribution(image: Image.Image, masks: np.ndarray,
                           probabilities: np.ndarray, point, path: Path) -> None:
    result = _mask_canvas(image, masks)
    draw = ImageDraw.Draw(result, "RGBA")
    font = _font(max(18, round(min(result.size) / 45)))
    order = np.argsort(-np.asarray(probabilities))
    panel_height = 42 + 25 * len(order)
    draw.rounded_rectangle((8, 8, 330, min(result.height - 8, panel_height)),
                           radius=8, fill=(0, 0, 0, 210))
    draw.text((18, 16), "Mask-level gesture distribution", fill="white", font=font)
    for row, index in enumerate(order):
        y = 44 + row * 25
        if y + 20 >= result.height:
            break
        draw.text((18, y), f"Mask {index + 1}: {float(probabilities[index]):.2%}",
                  fill=COLORS[index % len(COLORS)], font=font)
    draw_ground_truth(result, point).save(path)
