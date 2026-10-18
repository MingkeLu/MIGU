"""Dataset adapters for the multimodal grounding pipeline.

Every adapter exposes the same canonical workspace layout.  Supporting a future
capture format therefore only requires adding another adapter here; none of the
perception/grounding code needs to know how the source dataset is organized.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundTruthPoint:
    x: int
    y: int
    source: Path


@dataclass(frozen=True)
class DatasetFrame:
    frame_id: str
    rgb_path: Path
    depth_path: Path | None
    ray_path: Path | None
    ground_truth_point: GroundTruthPoint | None = None


@dataclass(frozen=True)
class PreparedDataset:
    rgb_dir: Path
    depth_dir: Path
    ray_dir: Path
    intrinsics_path: Path
    ground_truth_by_frame: dict[str, GroundTruthPoint]
    instruction_by_frame: dict[str, str]


class DatasetAdapter(ABC):
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    @abstractmethod
    def matches(cls, root: Path) -> bool:
        pass

    @abstractmethod
    def prepare(self, workspace: Path) -> PreparedDataset:
        pass


class LegacyInputAdapter(DatasetAdapter):
    """Original Input/key_rgb + key_depth + key_frames layout."""

    @classmethod
    def matches(cls, root: Path) -> bool:
        return (root / "key_rgb").is_dir() and (root / "key_depth").is_dir()

    def prepare(self, workspace: Path) -> PreparedDataset:
        del workspace
        ground_truth = {}
        instructions = {}
        ground_truth_path = self.root / "ground_truth.json"
        if ground_truth_path.is_file():
            records = json.loads(ground_truth_path.read_text(encoding="utf-8"))
            for frame_id, point in records.items():
                ground_truth[str(frame_id)] = GroundTruthPoint(
                    int(point["x"]), int(point["y"]), ground_truth_path
                )
        instructions_path = self.root / "instructions.json"
        if instructions_path.is_file():
            value = json.loads(instructions_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                instructions = {str(key): str(sentence) for key, sentence in value.items()
                                if isinstance(sentence, str) and sentence.strip()}
        return PreparedDataset(
            rgb_dir=self.root / "key_rgb",
            depth_dir=self.root / "key_depth",
            ray_dir=self.root / "key_frames",
            intrinsics_path=self.root / "camera_intrinsics.txt",
            ground_truth_by_frame=ground_truth,
            instruction_by_frame=instructions,
        )


class CaptureFoldersAdapter(DatasetAdapter):
    """One capture per folder, with rgb/depth/metadata/point files."""

    @classmethod
    def matches(cls, root: Path) -> bool:
        captures = root / "captures"
        return ((captures.is_dir() and any(captures.glob("*/metadata.json")))
                or any(root.glob("*/metadata.json")))

    def _captures_root(self) -> Path:
        nested = self.root / "captures"
        return nested if nested.is_dir() and any(nested.glob("*/metadata.json")) else self.root

    @staticmethod
    def _intrinsics(metadata: dict, source: Path) -> tuple[float, float, float, float]:
        value = metadata.get("rgb_intrinsics")
        if not isinstance(value, dict):
            raise ValueError(f"Missing rgb_intrinsics mapping in {source}")
        try:
            return tuple(float(value[key]) for key in ("fx", "fy", "ppx", "ppy"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid fx/fy/ppx/ppy in {source}") from error

    def prepare(self, workspace: Path) -> PreparedDataset:
        rgb_dir, depth_dir, ray_dir = (
            workspace / "key_rgb", workspace / "key_depth", workspace / "key_frames"
        )
        for directory in (rgb_dir, depth_dir, ray_dir):
            directory.mkdir(parents=True, exist_ok=True)

        common_intrinsics = None
        ground_truth = {}
        instructions = {}
        captures_root = self._captures_root()
        for capture_dir in sorted(captures_root.iterdir()):
            if not capture_dir.is_dir():
                continue
            metadata_path = capture_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame_id = str(metadata.get("id") or capture_dir.name)
            intrinsics = self._intrinsics(metadata, metadata_path)
            if common_intrinsics is None:
                common_intrinsics = intrinsics
            elif not np.allclose(common_intrinsics, intrinsics, rtol=0, atol=1e-5):
                raise ValueError(
                    "CaptureFoldersAdapter currently requires identical RGB intrinsics "
                    f"for all frames; {metadata_path} differs from the first frame"
                )

            rgb_source = capture_dir / str(metadata.get("rgb_file", "rgb.png"))
            depth_source = capture_dir / str(metadata.get("depth_file", "depth.npz"))
            if not rgb_source.is_file() or not depth_source.is_file():
                raise FileNotFoundError(f"Missing RGB or depth file in {capture_dir}")
            shutil.copy2(rgb_source, rgb_dir / f"{frame_id}.png")
            with np.load(depth_source, allow_pickle=False) as bundle:
                key = str(metadata.get("depth_array_key", "depth"))
                depth = np.asarray(bundle[key])
                scale = float(metadata.get("depth_scale_meters", bundle.get("depth_scale_meters", 0.001)))
            depth_mm = np.rint(depth.astype(np.float64) * scale * 1000.0)
            depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            Image.fromarray(depth_mm).save(depth_dir / f"{frame_id}.png")

            point_path = capture_dir / "point.json"
            if point_path.is_file():
                point = json.loads(point_path.read_text(encoding="utf-8"))
                if "x" in point and "y" in point:
                    ground_truth[frame_id] = GroundTruthPoint(
                        x=int(point["x"]), y=int(point["y"]), source=point_path
                    )
                sentence = point.get("sentence")
                if isinstance(sentence, str) and sentence.strip():
                    instructions[frame_id] = sentence.strip()

        if common_intrinsics is None:
            raise ValueError(f"No valid captures found below {captures_root}")
        fx, fy, cx, cy = common_intrinsics
        intrinsics_path = workspace / "camera_intrinsics.txt"
        intrinsics_path.write_text(
            f"fx: {fx}\nfy: {fy}\ncx: {cx}\ncy: {cy}\n", encoding="utf-8"
        )
        return PreparedDataset(rgb_dir, depth_dir, ray_dir, intrinsics_path,
                               ground_truth, instructions)


ADAPTERS = (LegacyInputAdapter, CaptureFoldersAdapter)


def open_dataset(root: Path) -> DatasetAdapter:
    for adapter_type in ADAPTERS:
        if adapter_type.matches(root):
            return adapter_type(root)
    names = ", ".join(adapter.__name__ for adapter in ADAPTERS)
    raise ValueError(f"Unsupported dataset layout at {root}; known adapters: {names}")


def select_capture_groups(input_dir: Path, names: list[str] | None = None) -> list[tuple[str, Path]]:
    """Resolve new ``captures/<group>/<frame>`` inputs, retaining old layouts."""
    input_dir = input_dir.resolve()
    if names is not None and (
        not isinstance(names, list)
        or not all(isinstance(name, str) and name.strip() for name in names)
    ):
        raise ValueError("subdirs must be null or a list of non-empty directory names")
    available = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and any(path.glob("*/metadata.json"))
    ) if input_dir.is_dir() else []
    if available:
        by_name = {path.name: path for path in available}
        if names:
            missing = [name for name in names if name not in by_name]
            if missing:
                raise ValueError(
                    f"Unknown capture subdirectories: {', '.join(missing)}; available: "
                    f"{', '.join(by_name)}"
                )
            return [(name, by_name[name]) for name in names]
        return [(path.name, path) for path in available]
    if names:
        raise ValueError("--subdirs can only be used with a captures directory containing groups")
    return [(input_dir.name, input_dir)]


def pair_prepared_frames(
    dataset: PreparedDataset, *, require_gesture: bool = True,
    require_complete_index_finger: bool = False,
    require_valid_pointing_ray: bool = False,
) -> list[DatasetFrame]:
    depth = {path.stem: path for path in dataset.depth_dir.glob("*.png")}
    rays = {path.stem: path for path in dataset.ray_dir.glob("*.json")}
    return [
        DatasetFrame(rgb.stem, rgb, depth.get(rgb.stem), rays.get(rgb.stem),
                     dataset.ground_truth_by_frame.get(rgb.stem))
        for rgb in sorted(dataset.rgb_dir.glob("*.png"))
        if (not require_gesture or (rgb.stem in depth and rgb.stem in rays))
        and (not require_complete_index_finger
             or has_complete_distal_index_finger(rays.get(rgb.stem)))
        and (not require_valid_pointing_ray
             or has_valid_pointing_ray(rays.get(rgb.stem)))
    ]


def has_complete_distal_index_finger(ray_path: Path | None) -> bool:
    """Return whether PIP, DIP, and TIP all have finite 3-D coordinates."""
    if ray_path is None or not ray_path.is_file():
        return False
    try:
        ray = json.loads(ray_path.read_text(encoding="utf-8"))
        points = {
            int(item["landmark_index"]): np.asarray(item["point_3d"], dtype=np.float64)
            for item in ray.get("index_finger_landmarks", [])
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return all(
        index in points and points[index].shape == (3,) and np.all(np.isfinite(points[index]))
        for index in (6, 7, 8)
    )


def has_valid_pointing_ray(ray_path: Path | None) -> bool:
    """Return whether a cached ray has finite origin and non-zero direction."""
    if ray_path is None or not ray_path.is_file():
        return False
    try:
        ray = json.loads(ray_path.read_text(encoding="utf-8"))
        origin = np.asarray(ray["origin"], dtype=np.float64)
        direction = np.asarray(ray["direction"], dtype=np.float64)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return bool(
        origin.shape == (3,)
        and direction.shape == (3,)
        and np.all(np.isfinite(origin))
        and np.all(np.isfinite(direction))
        and np.linalg.norm(direction) > 0
    )
