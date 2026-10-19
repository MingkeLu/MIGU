from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .distribution import density_map


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def reshape_for_grounding(
    image: Image.Image, width: int = 1280, height: int = 720
) -> Image.Image:
    """Reshape a source observation into the canonical grounding frame."""
    return image.resize((width, height), Image.Resampling.LANCZOS)


def to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def cell_id(row: int, col: int) -> str:
    return f"R{row + 1:02d}C{col + 1:02d}"


def valid_cell_ids(rows: int, cols: int) -> set[str]:
    return {cell_id(row, col) for row in range(rows) for col in range(cols)}


def cell_center(
    identifier: str, rows: int, cols: int, width: int, height: int
) -> tuple[float, float]:
    if len(identifier) != 6 or identifier[0] != "R" or identifier[3] != "C":
        raise ValueError(f"Invalid grid cell ID: {identifier}")
    row, col = int(identifier[1:3]) - 1, int(identifier[4:6]) - 1
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"Grid cell is outside the {rows}x{cols} grid: {identifier}")
    return ((col + 0.5) * width / cols, (row + 0.5) * height / rows)


def draw_grid(image: Image.Image, rows: int, cols: int) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    width, height = annotated.size
    cell_width, cell_height = width / cols, height / rows
    line_width = max(3, round(min(width, height) / 300))
    font_size = max(24, round(min(cell_width * 0.24, cell_height * 0.42)))
    font = _font(font_size)
    padding_x = max(5, round(font_size * 0.22))
    padding_y = max(3, round(font_size * 0.12))

    for row in range(rows + 1):
        y = round(row * height / rows)
        draw.line((0, y, width, y), fill=(0, 255, 80, 210), width=line_width)
    for col in range(cols + 1):
        x = round(col * width / cols)
        draw.line((x, 0, x, height), fill=(0, 255, 80, 210), width=line_width)

    # Large axis labels keep every RxxCxx cell readable without placing 100
    # opaque labels over potential manipulation targets.
    for row in range(rows):
        label = f"R{row + 1:02d}"
        row_position = 0.76 if row == 0 else row + 0.5
        x, y = 5, row_position * height / rows
        box = draw.textbbox((x, y), label, font=font, anchor="lm", stroke_width=2)
        draw.rectangle(
            (
                box[0] - padding_x,
                box[1] - padding_y,
                box[2] + padding_x,
                box[3] + padding_y,
            ),
            fill=(0, 0, 0, 225),
        )
        draw.text(
            (x, y),
            label,
            font=font,
            anchor="lm",
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    for col in range(cols):
        label = f"C{col + 1:02d}"
        x, y = (col + 0.5) * width / cols, 4
        box = draw.textbbox((x, y), label, font=font, anchor="mt", stroke_width=2)
        draw.rectangle(
            (
                box[0] - padding_x,
                box[1] - padding_y,
                box[2] + padding_x,
                box[3] + padding_y,
            ),
            fill=(0, 0, 0, 225),
        )
        draw.text(
            (x, y),
            label,
            font=font,
            anchor="mt",
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    return annotated.convert("RGB")


def draw_candidates(
    image: Image.Image, points: list[tuple[float, float]]
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    width, height = annotated.size
    radius = max(20, round(min(width, height) * 0.032))
    font = _font(max(26, round(radius * 1.25)))
    colors = ((255, 48, 48, 255), (0, 210, 255, 255), (255, 220, 0, 255))
    for index, (x, y) in enumerate(points):
        color = colors[index % len(colors)]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(0, 0, 0, 220),
            outline=color,
            width=max(5, radius // 4),
        )
        label = str(index)
        draw.text(
            (x, y),
            label,
            font=font,
            anchor="mm",
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255),
        )
    return annotated.convert("RGB")


def draw_object_probabilities(
    image: Image.Image, probabilities: list[dict]
) -> Image.Image:
    """Add a compact probability legend without obscuring the existing SoM marks."""
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font_size = max(20, round(min(image.size) * 0.032))
    title_font = _font(round(font_size * 1.08))
    font = _font(font_size)
    line_height = round(font_size * 1.45)
    padding = round(font_size * 0.55)
    shown = probabilities[:10]
    panel_width = max(250, round(image.width * 0.25))
    panel_height = padding * 2 + line_height * (len(shown) + 1)
    draw.rounded_rectangle(
        (8, 8, 8 + panel_width, 8 + panel_height),
        radius=10,
        fill=(0, 0, 0, 205),
        outline=(255, 255, 255, 220),
        width=2,
    )
    draw.text(
        (8 + padding, 8 + padding),
        "Object probability",
        font=title_font,
        fill=(255, 255, 255, 255),
    )
    for index, item in enumerate(shown, start=1):
        text = f"ID {item['object_id']}: {item['probability']:.1%}"
        draw.text(
            (8 + padding, 8 + padding + index * line_height),
            text,
            font=font,
            fill=(255, 225, 70, 255) if index == 1 else (255, 255, 255, 255),
        )
    return canvas.convert("RGB")


def draw_heatmap_overlay(
    image: Image.Image,
    samples: list[tuple[float, float]],
    distribution: dict,
    top_points: list[dict],
) -> Image.Image:
    density = density_map(distribution, image.width, image.height)
    peak = float(density.max())
    normalized = density / peak if peak > 0 else density

    # Blue -> cyan -> yellow -> red heat colors with density-controlled alpha.
    red = np.clip(2.2 * normalized - 0.35, 0, 1)
    green = np.clip(1.8 - np.abs(3.6 * normalized - 1.8), 0, 1)
    blue = np.clip(1.35 - 1.8 * normalized, 0, 1)
    alpha = np.where(normalized > 0.03, np.clip(normalized**0.65 * 190, 0, 190), 0)
    heat = np.dstack((red * 255, green * 255, blue * 255, alpha)).astype(np.uint8)

    canvas = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(heat, "RGBA"))
    draw = ImageDraw.Draw(canvas, "RGBA")
    radius = max(4, round(min(canvas.size) * 0.006))
    for x, y in samples:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255, 210),
            outline=(0, 0, 0, 210),
            width=2,
        )

    font = _font(max(16, radius * 2))
    for index, point in enumerate(top_points, start=1):
        x, y = point["x"], point["y"]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 55, 55, 235),
        )
        draw.text(
            (x + radius + 2, y - radius),
            f"P{index}",
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    return canvas.convert("RGB")
