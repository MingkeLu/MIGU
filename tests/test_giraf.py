from __future__ import annotations

import unittest

import numpy as np

from baselines.GIRAF.model import (Intrinsics, build_pointing_ray, get_semantic_candidates,
                   mask_to_center_3d, ray_to_depth_point, select_target)


class GirafTests(unittest.TestCase):
    def test_mask_center_is_pointwise_median_and_uses_scale(self):
        depth_mm = np.array([[1000, 1000], [1000, 0]], dtype=float)
        center, count = mask_to_center_3d(
            np.ones((2, 2), bool), depth_mm, Intrinsics(1, 1, 0, 0),
            depth_scale=0.001, min_points=3,
        )
        np.testing.assert_allclose(center, [0, 0, 1])
        self.assertEqual(count, 3)

    def test_mask_center_rejects_too_few_points(self):
        center, count = mask_to_center_3d(
            np.ones((1, 2), bool), np.array([[1000, 0]]), Intrinsics(1, 1, 0, 0),
            depth_scale=0.001, min_points=2,
        )
        self.assertIsNone(center)
        self.assertEqual(count, 1)

    def test_ray_is_pip_to_tip_and_starts_at_tip(self):
        origin, direction = build_pointing_ray(np.array([0, 0, 0]), np.array([1, 0, 0]))
        np.testing.assert_allclose(origin, [1, 0, 0])
        np.testing.assert_allclose(direction, [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "too close"):
            build_pointing_ray(np.zeros(3), np.zeros(3))

    def test_ray_to_depth_point_returns_nearest_forward_sample_and_pixel(self):
        hit = ray_to_depth_point(
            np.array([[1000, 1000], [1000, 1000]], dtype=float),
            Intrinsics(1000, 1000, 0, 0),
            np.zeros(3), np.array([0, 0, 1]), depth_scale=0.001,
        )
        self.assertEqual(hit["pixel"], [0, 0])
        np.testing.assert_allclose(hit["point_3d"], [0, 0, 1])
        self.assertAlmostEqual(hit["ray_distance"], 0.0)

    def test_ray_to_depth_point_ignores_samples_too_close_to_fingertip(self):
        hit = ray_to_depth_point(
            np.array([[20]], dtype=float), Intrinsics(1, 1, 0, 0),
            np.zeros(3), np.array([0, 0, 1]), depth_scale=1.0,
            min_depth_m=0.0, max_depth_m=100.0, min_forward_m=21.0,
        )
        self.assertIsNone(hit)

    def test_semantics_plural_synonym_and_fallback(self):
        self.assertEqual(
            get_semantic_candidates("pick up the mugs", ["cup", "box"], provider="lexical"),
            [0],
        )
        self.assertEqual(
            get_semantic_candidates("pick that", ["cup", "box"], provider="lexical", fallback="all"),
            [0, 1],
        )
        self.assertEqual(
            get_semantic_candidates("pick that", ["cup", "box"], provider="lexical", fallback="none"),
            [],
        )

    def test_point_to_ray_and_front_filter(self):
        objects = [
            {"name": "cup", "center_3d": np.array([3.0, 0.2, 0.0])},
            {"name": "cup", "center_3d": np.array([0.99, 0.01, 0.0])},
        ]
        result = select_target(
            "pick the cup", objects, np.array([0, 0, 0]), np.array([1, 0, 0]),
            semantic_provider="lexical",
        )
        self.assertEqual(result["target_index"], 0)
        self.assertAlmostEqual(result["candidate_scores"][0]["ray_distance"], np.sqrt(0.0002))
        self.assertFalse(result["candidate_scores"][0]["is_in_front"])
        self.assertAlmostEqual(result["candidate_scores"][0]["forward_distance"], -0.01)

    def test_select_target_can_reuse_precomputed_ray(self):
        objects = [
            {"name": "cup", "center_3d": np.array([0.0, 3.0, 0.0])},
            {"name": "cup", "center_3d": np.array([3.0, 0.0, 0.0])},
        ]
        result = select_target(
            "pick the cup", objects,
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            semantic_provider="lexical",
            ray_origin=np.array([0.0, 0.0, 0.0]),
            ray_direction=np.array([0.0, 2.0, 0.0]),
        )
        self.assertEqual(result["target_index"], 0)
        np.testing.assert_allclose(result["ray_origin"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["ray_direction"], [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
