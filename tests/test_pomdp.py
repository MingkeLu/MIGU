from __future__ import annotations

import unittest

import numpy as np

from baselines.POMDP.model import (CandidateObject, EmbeddingLanguageSimilarityModel, POMDPConfig,
                  OpenAIEmbeddingBackend, OpenAISoMLanguageModel, infer)
from PIL import Image


class FixedLanguageModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def score_all(self, utterance, candidates):
        return self.scores


class LEGSTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            CandidateObject("left", "cup", center_2d=[0.2, 0.5]),
            CandidateObject("right", "box", center_2d=[0.8, 0.5]),
        ]

    def test_fusion_prefers_jointly_supported_candidate_and_sums_to_one(self):
        result = infer(
            self.candidates, "the box", np.array([[0.4, 0.5], [0.5, 0.5]]),
            coordinate_space="normalized", language_model=FixedLanguageModel([0.0, 1.0]),
        )
        self.assertEqual(result["top1_id"], "right")
        self.assertEqual(result["diagnostics"]["active_modalities"], ["gesture", "language"])
        self.assertAlmostEqual(sum(result["fused_distribution"].values()), 1.0)
        self.assertAlmostEqual(sum(result["gesture_distribution"].values()), 1.0)
        self.assertAlmostEqual(sum(result["language_distribution"].values()), 1.0)

    def test_missing_modalities_are_not_zero_likelihoods(self):
        language_only = infer(
            self.candidates, "box", language_model=FixedLanguageModel([0.0, 1.0]),
            coordinate_space="normalized",
        )
        self.assertEqual(language_only["diagnostics"]["active_modalities"], ["language"])
        self.assertEqual(language_only["top1_id"], "right")
        both_missing = infer(self.candidates, coordinate_space="normalized")
        self.assertEqual(both_missing["diagnostics"]["active_modalities"], [])
        self.assertEqual(list(both_missing["fused_distribution"].values()), [0.5, 0.5])
        self.assertTrue(all(row["gesture_likelihood"] is None for row in both_missing["ranked_candidates"]))

    def test_pixel_and_normalized_coordinates_are_equivalent(self):
        normalized = infer(self.candidates, finger_joints=np.array([[0.4, .5], [.5, .5]]),
                           coordinate_space="normalized")
        pixels = [CandidateObject("left", "cup", center_2d=[20, 50]),
                  CandidateObject("right", "box", center_2d=[80, 50])]
        pixel = infer(pixels, finger_joints=np.array([[40, 50], [50, 50]]),
                      image_size=(100, 100), coordinate_space="pixel")
        np.testing.assert_allclose(list(normalized["fused_distribution"].values()),
                                   list(pixel["fused_distribution"].values()))

    def test_3d_projection_and_spread_control_sigma(self):
        joints = np.array([[-0.2, 0, 1], [-0.1, .05, 1], [0, 0, 1]])
        result = infer(
            self.candidates, finger_joints=joints, image_size=(100, 100),
            intrinsics={"fx": 100, "fy": 100, "cx": 50, "cy": 50},
            config=POMDPConfig(sigma_g_min_deg=1, sigma_g_max_deg=80, sigma_g_scale=2),
        )
        self.assertGreater(result["gesture"]["sigma_g_deg"], 1)
        self.assertLessEqual(result["gesture"]["sigma_g_deg"], 80)

    def test_cached_3d_ray_is_projected_and_used_directly(self):
        result = infer(
            self.candidates,
            pointing_origin=np.array([0.0, 0.0, 1.0]),
            pointing_direction=np.array([1.0, 0.0, 1.0]),
            image_size=(100, 100),
            intrinsics={"fx": 100, "fy": 100, "cx": 50, "cy": 50},
            config=POMDPConfig(sigma_g_ray_deg=12.0),
        )
        self.assertEqual(result["gesture"]["valid_cues"], ["cached_3d_pointing_ray"])
        self.assertEqual(result["gesture"]["sigma_g_deg"], 12.0)
        self.assertEqual(result["top1_id"], "right")

    def test_mask_centroid_precedes_explicit_center(self):
        mask = np.zeros((10, 10), bool)
        mask[4:6, 8:10] = True
        candidate = CandidateObject("mask", "cup", center_2d=[0, 0], mask=mask)
        result = infer([candidate], finger_joints=np.array([[2, 5], [5, 5]]),
                       image_size=(10, 10), coordinate_space="pixel")
        self.assertEqual(result["top1_id"], "mask")
        self.assertLess(result["ranked_candidates"][0]["gesture_angle_deg"], 15)

    def test_embedding_backend_maps_cosine_to_unit_interval(self):
        def backend(utterance, descriptions):
            return np.array([[1, 0], [1, 0], [-1, 0]], dtype=float)

        model = EmbeddingLanguageSimilarityModel(backend)
        np.testing.assert_allclose(model.score_all("cup", self.candidates), [1, 0])

    def test_openai_backend_batches_inputs_and_preserves_response_order(self):
        class Item:
            def __init__(self, index, embedding):
                self.index, self.embedding = index, embedding

        class Embeddings:
            def __init__(self):
                self.request = None

            def create(self, **request):
                self.request = request
                return type("Response", (), {"data": [Item(1, [0, 1]), Item(0, [1, 0])]})()

        client = type("Client", (), {"embeddings": Embeddings()})()
        backend = OpenAIEmbeddingBackend("text-embedding-3-small", dimensions=2, client=client)
        vectors = backend("cup", ["a cup"])
        np.testing.assert_array_equal(vectors, [[1, 0], [0, 1]])
        self.assertEqual(client.embeddings.request["input"], ["cup", "a cup"])
        self.assertEqual(client.embeddings.request["dimensions"], 2)

    def test_som_language_model_returns_scores_in_candidate_order(self):
        class Responses:
            def __init__(self):
                self.request = None

            def create(self, **request):
                self.request = request
                return type("Response", (), {
                    "output_text": '{"scores":[{"id":"right","kappa":0.9},'
                                   '{"id":"left","kappa":0.2}]}'
                })()

        responses = Responses()
        client = type("Client", (), {"responses": responses})()
        model = OpenAISoMLanguageModel(
            "vision-model", Image.new("RGB", (32, 24)), client=client
        )
        np.testing.assert_allclose(model.score_all("the box", self.candidates), [0.2, 0.9])
        self.assertEqual(responses.request["model"], "vision-model")
        image_part = responses.request["input"][1]["content"][1]
        self.assertTrue(image_part["image_url"].startswith("data:image/jpeg;base64,"))

    def test_som_language_model_rejects_missing_candidates(self):
        class Responses:
            def create(self, **request):
                return type("Response", (), {
                    "output_text": '{"scores":[{"id":"left","kappa":0.2}]}'
                })()

        client = type("Client", (), {"responses": Responses()})()
        model = OpenAISoMLanguageModel(
            "vision-model", Image.new("RGB", (32, 24)), client=client
        )
        with self.assertRaisesRegex(RuntimeError, "missing=.*right"):
            model.score_all("the box", self.candidates)


if __name__ == "__main__":
    unittest.main()
