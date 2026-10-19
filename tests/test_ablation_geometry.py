"""Regression tests retained from the original main_new geometric ablations."""
import numpy as np
import pytest
from PIL import Image
from migu.gesture_posterior import CameraIntrinsics
from migu.gesture_posterior_distance_cone import pixel_posterior_from_gesture as distance_cone_posterior
from migu.gesture_posterior_angular_gaussian import pixel_posterior_from_gesture as angular_gaussian_posterior
from migu.gesture_posterior_distance_threshold import pixel_posterior_from_gesture as distance_threshold_posterior
from ablations.owlv2 import detect

def test_owlv2_detection_uses_equal_confidence_and_caps_boxes(monkeypatch):
    response = {
        "bboxes": [[0.1, 0.1, 0.4, 0.5], [0.5, 0.2, 0.9, 0.8]],
        "scores": [0.4, 0.9],
        "box_names": ["cup", "table"],
    }
    monkeypatch.setattr("ablations.owlv2._post", lambda *args: response)
    boxes = detect(
        Image.new("RGB", (100, 50)), ["cup", "table"], "http://server",
        0.15, 10, max_boxes=2, confidence=0.6,
    )
    assert boxes == [
        {"label": "table", "bbox": [50, 10, 90, 40], "confidence": 0.6},
        {"label": "cup", "bbox": [10, 5, 40, 25], "confidence": 0.6},
    ]


def test_distance_cone_prefers_ray_and_expands_with_distance():
    depth = np.asarray([[1.0, 1.0, 2.0, 2.0]], dtype=np.float64)
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    result = distance_cone_posterior(
        depth, intrinsics, np.zeros(3), np.asarray([0.0, 0.0, 1.0]),
    )
    assert result["line_distance"][0, 0] == 0
    assert result["variance"][0, 2] > result["variance"][0, 1]
    assert result["probability"][0, 0] > result["probability"][0, 1]
    np.testing.assert_allclose(result["probability"].sum(), 1.0)


def test_distance_cone_rejects_points_behind_hand():
    with pytest.raises(ValueError, match="No valid depth pixels"):
        distance_cone_posterior(
            np.ones((1, 2)), CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
            np.asarray([0.0, 0.0, 2.0]), np.asarray([0.0, 0.0, 1.0]),
            valid_mask=np.asarray([[True, False]]),
        )


def test_angular_gaussian_uses_rotationally_symmetric_angular_error():
    depth = np.ones((3, 3), dtype=np.float64)
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    result = angular_gaussian_posterior(
        depth, intrinsics, np.zeros(3), np.asarray([0.0, 0.0, 1.0]),
    )
    assert result["probability"][1, 1] > result["probability"][1, 0]
    np.testing.assert_allclose(result["probability"][1, 0], result["probability"][0, 1])
    np.testing.assert_allclose(result["probability"].sum(), 1.0)


def test_distance_threshold_is_uniform_inside_forward_cylinder():
    depth = np.ones((1, 3), dtype=np.float64)
    intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=0.0, cy=0.0)
    result = distance_threshold_posterior(
        depth, intrinsics, np.zeros(3), np.asarray([0.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(result["valid_mask"], [[True, True, True]])
    np.testing.assert_allclose(result["probability"], [[1 / 3, 1 / 3, 1 / 3]])
