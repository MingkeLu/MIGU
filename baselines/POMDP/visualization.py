"""Per-frame visualizations for LEGS modality distributions."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_visualizations(image_path: Path, masks: np.ndarray, labels: Sequence[str],
                        result: dict, output_dir: Path) -> None:
    """Save a gesture mask heatmap and a language-probability bar chart."""
    # Import lazily so normal inference does not require matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask_array = np.asarray(masks, dtype=bool)
    if mask_array.ndim != 3 or mask_array.shape[1:] != image.shape[:2]:
        raise ValueError("visualization masks must have shape [N, image_height, image_width]")

    candidate_ids = [str(index) for index in range(len(labels))]
    gesture = result["gesture_distribution"]
    language = result["language_distribution"]
    gesture_values = np.asarray([gesture[item_id] for item_id in candidate_ids], dtype=float)
    language_values = np.asarray([language[item_id] for item_id in candidate_ids], dtype=float)
    names = [f"{index}: {label}" for index, label in enumerate(labels)]

    heatmap = np.full(image.shape[:2], np.nan, dtype=float)
    for mask, probability in zip(mask_array, gesture_values):
        current = np.nan_to_num(heatmap[mask], nan=-np.inf)
        heatmap[mask] = np.maximum(current, probability)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.imshow(image)
    overlay = axis.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.55)
    for index, mask in enumerate(mask_array):
        rows, columns = np.nonzero(mask)
        if len(rows):
            axis.text(columns.mean(), rows.mean(),
                      f"{names[index]}\n{gesture_values[index]:.3f}",
                      ha="center", va="center", fontsize=8, color="white",
                      bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
    status = ("active" if "gesture" in result["diagnostics"]["active_modalities"]
              else "unavailable; uniform fallback")
    axis.set_title(f"Gesture probability heatmap ({status})")
    axis.axis("off")
    figure.colorbar(overlay, ax=axis, label="Gesture probability", fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "gesture_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    height = max(3.5, 0.55 * len(names) + 1.5)
    figure, axis = plt.subplots(figsize=(9, height))
    positions = np.arange(len(names))
    bars = axis.barh(positions, language_values, color="#4C78A8")
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Probability")
    status = ("active" if "language" in result["diagnostics"]["active_modalities"]
              else "unavailable; uniform fallback")
    axis.set_title(f"Language distribution ({status})")
    axis.grid(axis="x", alpha=0.25)
    for bar, probability in zip(bars, language_values):
        axis.text(min(probability + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                  f"{probability:.3f}", va="center", ha="left", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "language_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
