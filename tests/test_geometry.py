import unittest

import numpy as np

from baselines.geometry.model import (Intrinsics, angular_features, mask_to_point_cloud,
                       normalize_log_likelihood, paper_log_likelihood)


class GeometricBaselineTests(unittest.TestCase):
    def test_back_projection_and_invalid_depth(self):
        depth = np.array([[1.0, 0.0], [np.nan, 2.0]])
        points = mask_to_point_cloud(np.ones((2, 2), bool), depth,
                                     Intrinsics(1, 1, 0, 0), mad_scale=0)
        np.testing.assert_allclose(points, [[0, 0, 1], [2, 2, 2]])

    def test_angles_and_behind_rejection(self):
        theta, phi, valid = angular_features(
            np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
            np.zeros(3), np.array([1.0, 0.0, 0.0]),
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(theta, np.arctan2(0.5, 1.5))
        self.assertAlmostEqual(phi, 0.0)
        self.assertFalse(angular_features(np.array([[-1.0, 0, 0]]), np.zeros(3),
                                          np.array([1.0, 0, 0]))[2])

    def test_paper_equations_and_normalization(self):
        theta = np.array([0.1, 0.3])
        phi = np.array([0.2, 0.4])
        area = np.array([10.0, 20.0])
        log_l, params = paper_log_likelihood(theta, phi, area, np.ones(2, bool),
                                             feature_set="theta_phi_area")
        self.assertAlmostEqual(params["lambda_theta"], 0.2)
        self.assertAlmostEqual(params["lambda_phi"], 0.3)
        self.assertAlmostEqual(params["lambda_area"], 15.0)
        expected = (np.log(0.2) - 0.2 * theta + np.log(0.3) - 0.3 * phi
                    + np.log(15.0) - 15.0 / area)
        np.testing.assert_allclose(log_l, expected)
        _, probability = normalize_log_likelihood(log_l)
        self.assertAlmostEqual(float(probability.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
