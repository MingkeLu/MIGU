"""Verify the nine public ablation workflows without external service calls."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from ablations.runner import GEOMETRY_MODES, VARIANTS, main
from ablations.semantic import object_scores, make_runner
from batch_preparation.shared import PROJECT_ROOT, prepare_group
from test_baseline_workflows import dataset, FakeVLM


def gesture(depth, ray, intrinsics, output, mode):
    probability = np.full((8, 8), 0.1)
    probability[:, :4] = 0.5
    probability[:, 4:] = 0.9
    probability /= probability.sum()
    path = output / f"{ray.stem}_posterior.npz"
    np.savez_compressed(path, probability=probability, posterior=mode)
    return probability, path


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batch", [False, True])
def test_all_ablation_commands(dataset, variant, batch):
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    config["visualize"] = False
    config_path = dataset.root / "migu.yaml"
    config_path.write_text(yaml.safe_dump(config))
    masks, names, scores, boxes = dataset.perception
    predictions = [{"label": "cup", "bbox": [0, 0, 3, 7], "confidence": 0.9},
                   {"label": "box", "bbox": [4, 0, 7, 7], "confidence": 0.1}]
    trace = {"vlm_bbox_seconds": 0.1, "sam_seconds": 0.2,
             "response_id": "fake", "usage": None}
    som = {"frame_id": "frame", "objects": [
        {"object_id": "1", "semantic_probability": 0.8},
        {"object_id": "2", "semantic_probability": 0.2}]}

    def hand(rgb, depth, output, intrinsics, hand_config):
        output.mkdir(exist_ok=True)
        (output / "frame.json").write_text(json.dumps({"origin": [0, 0, 0.1], "direction": [0, 0, 1]}))
        return {"saved": 1, "processing_time_seconds_by_frame": {"frame": 0.1}}

    with patch("batch_preparation.shared.segment_image", return_value=dataset.perception), \
         patch("batch_preparation.shared.generate_live_rays", side_effect=dataset.hand):
        prepare_group(dataset.source, dataset.cache, yaml.safe_load(dataset.config.read_text()), "room_pick")
    output = dataset.root / "output" / variant
    argv = ["ablation", variant, "--config", str(config_path),
            "--baseline-config", str(dataset.config), "--input-dir", str(dataset.source.parent),
            "--output-dir", str(output), "--prepared-dir", str(dataset.cache.parent),
            "--analysis-input-dir", str(dataset.cache.parent), "--no-visualize", "--analyze"]
    if batch:
        argv.append("--batch")
    with patch("sys.argv", argv), \
         patch("ablations.runner.OpenAI", return_value=SimpleNamespace()) as client, \
         patch("ablations.runner.generate_key_frames", side_effect=hand) as hand_call, \
         patch("migu.pipeline.run_gesture", side_effect=gesture) as gesture_call, \
         patch("migu.pipeline.run_vlm_sam", return_value=(predictions, masks, trace)) as regular_semantic, \
         patch("ablations.semantic.segment_image", return_value=dataset.perception), \
         patch("ablations.semantic.run_som", return_value=som), \
         patch("baselines.semantic.grounder.OpenAIVLM", FakeVLM), \
         patch("ablations.owlv2.run_vlm_owlv2", return_value=(predictions, trace)), \
         patch("ablations.semantic.request_masks", return_value=list(masks)):
        main()
        if variant == "without-geometry":
            hand_call.assert_not_called()
            gesture_call.assert_not_called()
        if variant == "without-semantics":
            client.assert_not_called()
            regular_semantic.assert_not_called()
        if variant in GEOMETRY_MODES:
            assert gesture_call.call_args.args[-1] == GEOMETRY_MODES[variant]
    result_path = next(output.glob("*/room_pick/frame/result.json"))
    result = json.loads(result_path.read_text())
    assert result["ablation"] == variant
    assert len(result["masks"]) == 2
    assert sum(x["probability"] for x in result["masks"]) == pytest.approx(1)
    if variant == "without-semantics":
        assert [x["confidence"] for x in result["masks"]] == [1, 1]
        assert result["masks"][1]["probability"] > result["masks"][0]["probability"]
    if variant == "without-geometry":
        assert [x["probability"] for x in result["masks"]] == pytest.approx([0.9, 0.1])
        assert not result["gesture_available"]
        assert result["paths"]["gesture_posterior"] is None
        assert all(x["gesture_mass"] is None for x in result["masks"])
    if variant in GEOMETRY_MODES:
        assert [x["probability"] for x in result["masks"]] == pytest.approx([5/6, 1/6])
    report = json.loads((result_path.parents[2] / "ground_truth_analysis/summary.json").read_text())
    assert report["result_frames"] == 1
    assert report["valid_frames"] == 1


@pytest.mark.parametrize("variant", ["semantic-grid", "semantic-pivot"])
def test_replacement_scores_are_indexed_not_rank_order(variant):
    record = {"object_distribution": [{"object_index": 1, "probability": 0.8},
                                       {"object_index": 0, "probability": 0.2}]}
    np.testing.assert_array_equal(object_scores(record, variant, 2), [0.2, 0.8])
    record["object_distribution"][0]["object_index"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        object_scores(record, variant, 2)


@pytest.mark.parametrize("variant", ["semantic-som", "semantic-grid", "semantic-pivot"])
def test_reuse_replacement_results_with_local_masks(dataset, variant):
    root = dataset.root / "snapshot"
    frame = root / "room_pick/frame"
    frame.mkdir(parents=True)
    masks, names, scores, boxes = dataset.perception
    np.savez_compressed(frame / "perception.npz", masks=masks, box_names=names,
                        scores=scores, bboxes=boxes)
    if variant == "semantic-som":
        record = {"objects": [{"object_id": "1", "semantic_probability": 0.2},
                              {"object_id": "2", "semantic_probability": 0.8}]}
    else:
        record = {"object_distribution": [{"object_index": 1, "probability": 0.8},
                                           {"object_index": 0, "probability": 0.2}]}
    (frame / "result.json").write_text(json.dumps(record))
    args = SimpleNamespace(reuse_result_dir=root, batch=False)
    runner = make_runner(variant, "room_pick", dataset.root / "run/room_pick",
                         yaml.safe_load(dataset.config.read_text()), args, dataset.root / "missing_cache")
    with patch("ablations.semantic.segment_image", side_effect=AssertionError("No segmentation requests")), \
         patch("ablations.semantic.run_som", side_effect=AssertionError("No VLM requests")):
        candidates, actual, trace = runner(dataset.source / "key_rgb/frame.png", "pick up this", {}, None)
    assert [x["confidence"] for x in candidates] == pytest.approx([0.2, 0.8])
    np.testing.assert_array_equal(actual, masks)
    assert trace["reused_from"] == str(frame / "result.json")
