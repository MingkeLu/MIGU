"""Exercise live/cached inference without external VLM or segmentation services."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import yaml

from baselines.runner import METHODS, main
from baselines.semantic.schemas import GridPrediction, PivotPrediction, ModelReply
from batch_preparation.shared import DEFAULT_CONFIG


class FakeVLM:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args):
        return ModelReply(GridPrediction(target_description="target", cell_ids=["R01C01", "R01C02", "R02C01"]), "fake", None)

    def pivot(self, *args):
        return ModelReply(PivotPrediction(target_description="target", candidate_ids=[0, 1, 2]), "fake", None)


class FakeLanguage:
    def __init__(self, *args, **kwargs):
        pass

    def score_all(self, utterance, candidates):
        return np.array([0.8, 0.2])


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "captures" / "room_pick"
    for folder in ("key_rgb", "key_depth", "key_frames"):
        (source / folder).mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(source / "key_rgb/frame.png")
    Image.fromarray(np.full((8, 8), 1000, np.uint16)).save(source / "key_depth/frame.png")
    (source / "camera_intrinsics.txt").write_text("fx: 10\nfy: 10\ncx: 4\ncy: 4\n")
    (source / "instructions.json").write_text(json.dumps({"frame": "pick up this"}))
    (source / "ground_truth.json").write_text(json.dumps({"frame": {"x": 1, "y": 1}}))
    ray = {"origin": [0, 0, 0.1], "direction": [0, 0, 1]}
    (source / "key_frames/frame.json").write_text(json.dumps(ray))
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, :, :4] = True
    masks[1, :, 4:] = True
    perception = (masks, ["cup", "box"], np.array([0.9, 0.8]),
                  np.array([[0, 0, 0.5, 1], [0.5, 0, 1, 1]]))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["segmentation"]["vocabulary"] = ["cup", "box"]
    config["visualize"] = False
    config["methods"]["VLM-SoM"]["selections_per_generation"] = 2
    config["methods"]["VLM-Grid"]["grid_rows"] = 2
    config["methods"]["VLM-Grid"]["grid_cols"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cache = tmp_path / "prepared" / "room_pick"

    def hand(prepared, config):
        prepared.ray_dir.mkdir(exist_ok=True)
        (prepared.ray_dir / "frame.json").write_text(json.dumps(ray))
        return {"saved": 1, "processing_time_seconds_by_frame": {"frame": 0.1}}

    return SimpleNamespace(source=source, config=config_path, cache=cache,
                           perception=perception, hand=hand, root=tmp_path)


@pytest.mark.parametrize("method", METHODS)
def test_live_and_batch_use_identical_masks_and_algorithms(dataset, method):
    from batch_preparation.shared import prepare_group
    config = yaml.safe_load(dataset.config.read_text())
    with patch("batch_preparation.shared.segment_image", return_value=dataset.perception), \
         patch("batch_preparation.shared.generate_live_rays", side_effect=dataset.hand), \
         patch("baselines.semantic.grounder.OpenAIVLM", FakeVLM), \
         patch("baselines.POMDP.implementation.OpenAISoMLanguageModel", FakeLanguage), \
         patch("baselines.GIRAF.model.get_semantic_candidates", return_value=[0, 1]), \
         patch("baselines.semantic.som.run_semantic", return_value=(np.array([0.8, 0.2]), [])):
        prepare_group(dataset.source, dataset.cache, config, "room_pick")
        outputs = []
        for batch in (False, True):
            output = dataset.root / ("batch" if batch else "live") / method
            argv = ["baseline", method, "--config", str(dataset.config),
                    "--input-dir", str(dataset.source.parent),
                    "--prepared-dir", str(dataset.cache.parent),
                    "--output-dir", str(output), "--no-visualize", "--analyze"]
            if batch:
                argv.append("--batch")
            with patch("sys.argv", argv):
                main()
            run = next(output.iterdir())
            result = json.loads((run / "room_pick/frame/result.json").read_text())
            report = json.loads((run / "analysis/metrics_summary.json").read_text())
            assert report["runs"][0]["frames"] == 1
            assert report["runs"][0]["method"] == method
            outputs.append(result)
        keys = {"geometry": ["probability"], "POMDP": ["fused_distribution"],
                "GIRAF": ["target_index"], "VLM-Grid": ["object_distribution"],
                "VLM-PIVOT": ["object_distribution"], "VLM-SoM": ["objects"]}[method]
        for key in keys:
            assert outputs[0][key] == outputs[1][key]
        if method in {"VLM-Grid", "VLM-PIVOT"}:
            assert "distribution" not in outputs[0]
            assert "distribution" not in outputs[1]


@pytest.mark.parametrize("method", ["VLM-Grid", "VLM-PIVOT"])
def test_single_image_votes_into_masks(dataset, method):
    with patch("batch_preparation.shared.segment_image", return_value=dataset.perception), \
         patch("baselines.semantic.grounder.OpenAIVLM", FakeVLM):
        output = dataset.root / "single" / method
        with patch("sys.argv", ["baseline", method, str(dataset.source / "key_rgb/frame.png"),
                                "--config", str(dataset.config), "--instruction", "pick up this",
                                "--output-dir", str(output), "--no-visualize"]):
            main()
        result = json.loads(next(output.glob("*/single/frame/result.json")).read_text())
        assert len(result["object_distribution"]) == 2
        assert "distribution" not in result
        assert not list(output.rglob("grounding_heatmap.jpg"))


@pytest.mark.parametrize("method", ["POMDP", "GIRAF"])
def test_visualization_has_no_legacy_repository_dependencies(dataset, method):
    with patch("batch_preparation.shared.segment_image", return_value=dataset.perception), \
         patch("batch_preparation.shared.generate_live_rays", side_effect=dataset.hand), \
         patch("baselines.POMDP.implementation.OpenAISoMLanguageModel", FakeLanguage), \
         patch("baselines.GIRAF.model.get_semantic_candidates", return_value=[0, 1]):
        output = dataset.root / "visualization" / method
        with patch("sys.argv", ["baseline", method, "--config", str(dataset.config),
                                "--input-dir", str(dataset.source.parent),
                                "--output-dir", str(output), "--visualize"]):
            main()
        artifact = "gesture_heatmap.png" if method == "POMDP" else "giraf_point_cloud.ply"
        assert list(output.rglob(artifact))
