"""Client for the existing combined OWL detection and SAM segmentation service."""
from __future__ import annotations
import base64
import io
from collections import defaultdict
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def masks_to_array(masks, image_size):
    """Convert per-object SAM results to one boolean array.

    The default (``return_best=True``) shape is ``[N, H, W]``.  When
    ``--all-masks`` is used, the shape is normally ``[N, K, H, W]``.
    """
    width, height = image_size
    normalized = []
    for index, mask_group in enumerate(masks):
        mask_group = np.asarray(mask_group, dtype=bool)
        if mask_group.shape[-2:] != (height, width):
            raise ValueError(
                f"Mask {index} has spatial shape {mask_group.shape[-2:]}; "
                f"expected {(height, width)}"
            )
        if mask_group.ndim == 2:
            mask_group = mask_group[None, ...]
        else:
            mask_group = mask_group.reshape(-1, height, width)
        normalized.append(mask_group)

    if not normalized:
        return np.empty((0, height, width), dtype=bool)
    candidate_counts = {item.shape[0] for item in normalized}
    if len(candidate_counts) != 1:
        raise ValueError("Objects have different numbers of SAM mask candidates")

    stacked = np.stack(normalized, axis=0)
    return stacked[:, 0] if stacked.shape[1] == 1 else stacked


def post_json(server_url, endpoint, payload, timeout):
    response = requests.post(
        f"{server_url.rstrip('/')}/{endpoint}",
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"{endpoint} returned HTTP {response.status_code}: {response.text}"
        )
    return response.json()


def overlay_results(
    image,
    bboxes,
    masks,
    output_path,
    box_names=None,
    box_label_style="number",
    draw_masks=True,
):
    canvas = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    colors = np.array([
        [255, 80, 80],
        [80, 200, 120],
        [80, 140, 255],
        [245, 190, 60],
        [190, 90, 230],
    ], dtype=np.float32)

    if draw_masks:
        for index, mask_group in enumerate(masks):
            mask_group = np.asarray(mask_group)
            if mask_group.ndim == 2:
                mask_group = mask_group[None, ...]
            else:
                mask_group = mask_group.reshape(-1, *mask_group.shape[-2:])

            color = colors[index % len(colors)]
            for mask in mask_group:
                selected = mask.astype(bool)
                canvas[selected] = canvas[selected] * 0.5 + color * 0.5

    result = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    width, height = result.size
    try:
        number_font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=20)
    except OSError:
        number_font = ImageFont.load_default()

    name_counts = defaultdict(int)
    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        color = tuple(int(value) for value in colors[index % len(colors)])
        draw.rectangle(box, outline=color, width=3)

        if box_label_style == "name-number":
            name = str(box_names[index])
            name_counts[name] += 1
            label = f"{name}#{name_counts[name]}"
        else:
            label = str(index + 1)

        if hasattr(draw, "textbbox"):
            text_box = draw.textbbox((0, 0), label, font=number_font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
        else:  # Pillow < 8.0
            text_width, text_height = draw.textsize(label, font=number_font)
        if box_label_style == "name-number":
            padding_x, padding_y = 6, 4
            left = min(max(box[0], 0), max(width - text_width - 2 * padding_x, 0))
            top = min(max(box[1], 0), max(height - text_height - 2 * padding_y, 0))
            draw.rectangle(
                (
                    left,
                    top,
                    left + text_width + 2 * padding_x,
                    top + text_height + 2 * padding_y,
                ),
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
            draw.text(
                (left + padding_x, top + padding_y),
                label,
                fill=(255, 255, 255),
                font=number_font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            continue

        # Put a matching colored number badge at the box's top-left corner.
        badge_radius = max(
            14,
            (text_width + 10) // 2,
            (text_height + 10) // 2,
        )
        badge_x = min(max(box[0], badge_radius), width - badge_radius)
        badge_y = min(max(box[1], badge_radius), height - badge_radius)
        draw.ellipse(
            (
                badge_x - badge_radius,
                badge_y - badge_radius,
                badge_x + badge_radius,
                badge_y + badge_radius,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        draw.text(
            (badge_x - text_width / 2, badge_y - text_height / 2),
            label,
            fill=(255, 255, 255),
            font=number_font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    if output_path is not None:
        result.save(output_path)
    return result


def validate_detection_response(response, include_masks=False):
    expected_length = len(response["bboxes"])
    fields = ["scores", "box_names"]
    if include_masks:
        fields.append("masks")

    for field in fields:
        actual_length = len(response[field])
        if actual_length != expected_length:
            raise AssertionError(
                f"Expected {expected_length} {field}, received {actual_length}"
            )
