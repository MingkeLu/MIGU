import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baselines.metrics import calculate_target_object_top_ratio
from baselines.metrics import (
    _add_preparation_times,
    _aggregate,
    _geometric_rows,
    _giraf_metric_point,
    evaluate_batch_outputs,
    evaluate_main_outputs,
    evaluate_touch_line_np_outputs,
)


def _write_result(path, target_id, scores):
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "ground_truth": {"object_id": target_id},
                "objects": [
                    {"object_id": object_id, "fused_probability": score}
                    for object_id, score in scores.items()
                ],
            }
        ),
        encoding="utf-8",
    )


class TargetObjectTopRatioTests(unittest.TestCase):
    def test_touch_line_np_evaluation_reports_all_applicable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "output" / "touch_line_np" / "run" / "room_pick"
            input_group = root / "input" / "room_pick"
            run.mkdir(parents=True)
            input_group.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({"frames": [{
                "frame_id": "frame_1",
                "box_xyxy": [0, 0, 20, 20],
                "processing_time_seconds": 2.0,
            }]}), encoding="utf-8")
            (input_group / "ground_truth.json").write_text(
                json.dumps({"frame_1": {"x": 10, "y": 10}}), encoding="utf-8"
            )

            report = evaluate_touch_line_np_outputs(
                root / "output", root / "input", root / "metrics"
            )

            summary = report["runs"][0]
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["top2_accuracy"], 1.0)
            self.assertEqual(summary["top3_accuracy"], 1.0)
            self.assertEqual(summary["distance_2d_px"], 0.0)
            self.assertEqual(summary["pick_distance_2d_px"], 0.0)
            self.assertEqual(summary["distance_2d_variance_px2"], 0.0)
            self.assertEqual(summary["processing_time_mean_seconds"], 2.0)
            self.assertIsNone(summary["nll"])
            self.assertIsNone(summary["brier_score"])

    def test_touch_line_np_accepts_standalone_outputs_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "baseline" / "touch_line_np" / "outputs"
            group = outputs / "run" / "room_pick"
            input_group = root / "input" / "room_pick"
            group.mkdir(parents=True)
            input_group.mkdir(parents=True)
            (group / "summary.json").write_text(json.dumps({"frames": [{
                "frame_id": "frame_1", "box_xyxy": [0, 0, 20, 20]
            }]}), encoding="utf-8")
            (input_group / "ground_truth.json").write_text(
                json.dumps({"frame_1": {"x": 10, "y": 10}}), encoding="utf-8"
            )

            report = evaluate_batch_outputs(
                outputs, root / "input", root / "metrics", method="touch_line_np"
            )

            self.assertEqual(report["runs"][0]["method"], "touch_line_np")
            self.assertEqual(report["runs"][0]["accuracy"], 1.0)

    def test_main_evaluation_uses_probability_for_each_pipeline_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output" / "main" / "run" / "group"
            input_group = root / "input" / "group"
            input_group.mkdir(parents=True)
            for index, (mode, key) in enumerate((
                ("fusion", "fused_probability"),
                ("semantic", "semantic_probability"),
                ("gesture", "gesture_probability"),
            )):
                frame_id = f"frame_{index}"
                frame = output / frame_id
                frame.mkdir(parents=True)
                (frame / "result.json").write_text(json.dumps({
                    "frame_id": frame_id,
                    "mode": mode,
                    "ground_truth": {
                        "point": {"x": 1, "y": 1},
                        "object_id": "2",
                    },
                    "objects": [
                        {"object_id": "1", key: 0.1},
                        {"object_id": "2", key: 0.9},
                    ],
                }), encoding="utf-8")
                np.savez_compressed(
                    frame / "perception.npz",
                    bboxes=np.array([[0, 0, 1, 1], [0, 0, 2, 2]]),
                )

            report = evaluate_main_outputs(
                root / "output", root / "input", root / "metrics"
            )

            summary = report["runs"][0]
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertAlmostEqual(summary["nll"], -np.log(0.9))
            self.assertAlmostEqual(summary["brier_score"], 0.02)

    def test_main_localization_point_selects_bbox_center_or_posterior_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "output" / "main" / "run" / "group" / "frame_1"
            input_group = root / "input" / "group"
            frame.mkdir(parents=True)
            (input_group / "key_depth").mkdir(parents=True)
            (frame / "result.json").write_text(json.dumps({
                "frame_id": "frame_1", "mode": "gesture",
                "ground_truth": {"point": {"x": 3, "y": 2}, "object_id": "bottle#1"},
                "objects": [{"object_id": "bottle#1", "gesture_probability": 1.0}],
            }), encoding="utf-8")
            mask = np.ones((1, 4, 4), dtype=bool)
            np.savez_compressed(frame / "perception.npz", masks=mask,
                                bboxes=np.array([[0, 0, 4, 4]]))
            posterior = np.zeros((4, 4), dtype=float)
            posterior[2, 3] = 1.0
            np.savez_compressed(frame / "frame_1_posterior.npz", probability=posterior)
            Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(
                input_group / "key_depth" / "frame_1.png"
            )
            (input_group / "camera_intrinsics.txt").write_text(
                "fx: 1\nfy: 1\ncx: 0\ncy: 0\n", encoding="utf-8"
            )

            peak = evaluate_main_outputs(
                root / "output", root / "input", root / "peak",
                localization_point="posterior_peak",
            )["runs"][0]
            center = evaluate_main_outputs(
                root / "output", root / "input", root / "center",
                localization_point="bbox_center",
            )["runs"][0]

            self.assertEqual(peak["distance_2d_px"], 0.0)
            self.assertEqual(peak["distance_3d_m"], 0.0)
            self.assertAlmostEqual(center["distance_2d_px"], 1.0)
            self.assertAlmostEqual(center["distance_3d_m"], 1.0)

    def test_metrics_can_select_one_method_case_insensitively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            input_root = root / "input"
            save = root / "metrics"
            (output / "GIRAF" / "20260101_000000_000000").mkdir(parents=True)
            (output / "LEGS" / "20260102_000000_000000").mkdir(parents=True)
            input_root.mkdir()

            report = evaluate_batch_outputs(
                output, input_root, save, method="giraf"
            )

            self.assertEqual(report["runs"], [])
            self.assertEqual(report["warnings"], [])

    def test_metrics_rejects_unknown_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (output / "GIRAF").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "available methods: GIRAF"):
                evaluate_batch_outputs(output, root / "input", root / "metrics", method="LEGS")

    def test_giraf_missing_target_is_counted_as_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output" / "GIRAF" / "run" / "group"
            input_group = root / "input" / "group"
            output.mkdir(parents=True)
            input_group.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"frames": [{
                "frame_id": "frame_1", "ground_truth_index": 0,
                "target_index": None, "processing_time_seconds": 1.0,
            }]}), encoding="utf-8")
            (input_group / "ground_truth.json").write_text(
                json.dumps({"frame_1": {"x": 1, "y": 2}}), encoding="utf-8"
            )

            report = evaluate_batch_outputs(
                root / "output", root / "input", root / "metrics", method="GIRAF"
            )

            self.assertEqual(report["runs"][0]["frames"], 1)
            self.assertEqual(report["runs"][0]["accuracy"], 0.0)
            self.assertEqual(report["runs"][0]["distance_2d_px_count"], 0)

    def test_calculates_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result(root / "frame_1" / "result.json", "1", {"1": 0.8, "2": 0.2})
            _write_result(root / "frame_2" / "result.json", "1", {"1": 0.3, "2": 0.7})

            self.assertEqual(calculate_target_object_top_ratio(root), 0.5)

    def test_ignores_frames_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result(root / "valid" / "result.json", "1", {"1": 1.0})
            (root / "invalid").mkdir()
            (root / "invalid" / "result.json").write_text(
                json.dumps({"ground_truth": None, "objects": []}), encoding="utf-8"
            )

            self.assertEqual(calculate_target_object_top_ratio(root), 1.0)

    def test_giraf_point_is_used_only_inside_selected_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            perception = Path(directory) / "perception.npz"
            masks = np.zeros((2, 3, 4), dtype=bool)
            masks[1, 2, 3] = True
            np.savez_compressed(perception, masks=masks)
            frame = {"ray_environment_point": {
                "pixel": [3, 2], "point_3d": [0.1, 0.2, 1.0],
            }}
            self.assertEqual(
                _giraf_metric_point(frame, perception, 1, []),
                ((3.0, 2.0), (0.1, 0.2, 1.0)),
            )
            self.assertEqual(_giraf_metric_point(frame, perception, 0, []), (None, None))

    def test_raises_when_no_frame_is_evaluable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No evaluable"):
                calculate_target_object_top_ratio(directory)

    def test_processing_time_mean_and_population_variance(self):
        rows = [
            {"method": "main", "run": "run", "processing_time_seconds": 1.0},
            {"method": "main", "run": "run", "processing_time_seconds": 3.0},
        ]
        summary = _aggregate(rows)[0]
        self.assertEqual(summary["processing_time_mean_seconds"], 2.0)
        self.assertEqual(summary["processing_time_variance_seconds2"], 1.0)
        self.assertEqual(summary["processing_time_count"], 2)

    def test_distance_population_variances(self):
        rows = [
            {"method": "main", "run": "run", "group": "pantry_pick",
             "distance_2d_px": 1.0,
             "distance_3d_m": 0.1},
            {"method": "main", "run": "run", "group": "office_pick",
             "distance_2d_px": 3.0,
             "distance_3d_m": 0.3},
            {"method": "main", "run": "run", "group": "office_place",
             "distance_2d_px": 10.0, "distance_3d_m": 1.0},
        ]
        summary = _aggregate(rows)[0]
        self.assertEqual(summary["pick_distance_2d_px"], 2.0)
        self.assertEqual(summary["pick_distance_2d_variance_px2"], 1.0)
        self.assertAlmostEqual(summary["pick_distance_3d_variance_m2"], 0.01)
        self.assertEqual(summary["pick_distance_2d_px_count"], 2)
        self.assertEqual(summary["place_distance_2d_px"], 10.0)
        self.assertEqual(summary["place_distance_2d_variance_px2"], 0.0)
        self.assertEqual(summary["place_distance_3d_m_count"], 1)

    def test_geometric_distance_scales_normalized_input_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "output" / "geometric_based" / "run"
            frame = run / "room" / "frame1"
            perception = root / "input" / "room" / "perception"
            frame.mkdir(parents=True)
            perception.mkdir(parents=True)
            (frame / "result.json").write_text(json.dumps({
                "frame_id": "frame1",
                "probability": [1.0],
                "ground_truth": {
                    "candidate_index": 0,
                    "point": {"x": 100, "y": 50},
                },
            }), encoding="utf-8")
            np.savez_compressed(
                perception / "frame1.npz",
                bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
                image_size=np.array([200, 100]),
            )

            rows = _geometric_rows("geometric_based", run, root / "input", 0.001, [])

            self.assertEqual(rows[0]["pred_x"], 100.0)
            self.assertEqual(rows[0]["pred_y"], 50.0)
            self.assertEqual(rows[0]["distance_2d_px"], 0.0)

    def test_batch_preparation_times_are_added_per_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            group = input_root / "room"
            group.mkdir()
            (group / "manifest.json").write_text(json.dumps({
                "frame_timings": {"frame_1": {
                    "hand_detection_seconds": 0.25,
                    "segmentation_seconds": 0.75,
                }}
            }), encoding="utf-8")
            rows = [{"group": "room", "frame_id": "frame_1",
                     "processing_time_seconds": 2.0}]
            warnings = []
            _add_preparation_times(rows, input_root, warnings)
            self.assertEqual(rows[0]["pipeline_processing_time_seconds"], 2.0)
            self.assertEqual(rows[0]["preparation_time_seconds"], 1.0)
            self.assertEqual(rows[0]["processing_time_seconds"], 3.0)
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
