"""Protocol checks for the external perception-services integration."""
import base64
import pickle
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from batch_preparation.shared import segment_image
from migu.sam import request_masks


def test_request_masks_decodes_perception_services_pickle_response():
    expected = np.array([[True, False], [False, True]])
    response = Mock(ok=True)
    response.json.return_value = {
        "result": [{
            "segmentation": base64.b64encode(pickle.dumps(expected)).decode("ascii")
        }]
    }
    image = Image.new("RGB", (2, 2))

    with patch("migu.sam.requests.post", return_value=response) as post:
        masks = request_masks("http://127.0.0.1:4001/", image, [[0, 0, 1, 1]], True, 10)

    np.testing.assert_array_equal(masks[0], expected)
    assert post.call_args.args[0] == "http://127.0.0.1:4001/sam_mask_by_bbox"
    assert post.call_args.kwargs["json"]["bboxes"] == [[0, 0, 1, 1]]


def test_baseline_segmentation_calls_separate_owl_and_sam_services():
    image = Image.new("RGB", (200, 100))
    detection = {
        "bboxes": [[0.1, 0.2, 0.5, 0.8]],
        "scores": [0.75],
        "box_names": ["cup"],
    }
    mask = np.ones((100, 200), dtype=bool)
    config = {
        "owl_server_url": "http://127.0.0.1:4000",
        "sam_server_url": "http://127.0.0.1:4001",
        "vocabulary": ["cup"],
        "bbox_conf_threshold": 0.15,
        "timeout": 30,
    }

    with patch("batch_preparation.shared.post_json", return_value=detection) as owl, \
         patch("batch_preparation.shared.request_masks", return_value=[mask]) as sam:
        masks, names, scores, boxes = segment_image(image, config)

    assert owl.call_args.args[:2] == ("http://127.0.0.1:4000", "owl_detect")
    assert sam.call_args.args[0] == "http://127.0.0.1:4001"
    assert sam.call_args.args[2] == [[20, 20, 100, 80]]
    np.testing.assert_array_equal(masks, mask[None])
    assert names == ["cup"]
    np.testing.assert_allclose(scores, [0.75])
    np.testing.assert_allclose(boxes, detection["bboxes"])
