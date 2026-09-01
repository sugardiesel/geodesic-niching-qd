import unittest

import numpy as np

from algorithms.trajectory_autoencoder import trajectory_to_sequence


class TrajectoryAutoencoderTests(unittest.TestCase):
    def test_trajectory_to_sequence_resamples_requested_features(self) -> None:
        trajectory = np.asarray(
            [
                [0.0, 0.0, 0.1, 0.2, 1.0, 0.0, 0.0],
                [1.0, 0.5, 0.3, 0.4, 0.8, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        sequence = trajectory_to_sequence(
            trajectory, sequence_length=4, feature_names=["x", "y", "energy"]
        )
        self.assertEqual(sequence.shape, (4, 3))
        self.assertAlmostEqual(float(sequence[0, 0]), 0.0)
        self.assertAlmostEqual(float(sequence[-1, 0]), 1.0)
        self.assertAlmostEqual(float(sequence[0, 2]), 1.0)
        self.assertAlmostEqual(float(sequence[-1, 2]), 0.8)


if __name__ == "__main__":
    unittest.main()
