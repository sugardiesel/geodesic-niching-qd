import unittest

import numpy as np

from algorithms.map_elites import GridArchive
from algorithms.policies import MLPPolicyGenome
from envs.forage_maze import RolloutResult


class Phase2MapElitesTests(unittest.TestCase):
    def test_archive_replaces_only_when_fitter(self) -> None:
        archive = GridArchive(
            bins=(4, 4),
            bd_min=np.asarray([-1.0, -1.0]),
            bd_max=np.asarray([1.0, 1.0]),
            genome_size=3,
        )
        genome = MLPPolicyGenome(weights=np.asarray([0.1, 0.2, 0.3]), obs_dim=1, hidden_dim=0)
        low = rollout(1.0)
        high = rollout(2.0)
        inserted, replaced, cell = archive.add(genome, low)
        self.assertTrue(inserted)
        self.assertFalse(replaced)
        self.assertEqual(cell, (2, 2))
        inserted, replaced, _ = archive.add(genome, low)
        self.assertFalse(inserted)
        self.assertFalse(replaced)
        inserted, replaced, _ = archive.add(genome, high)
        self.assertTrue(inserted)
        self.assertTrue(replaced)
        self.assertEqual(archive.occupied_count, 1)
        self.assertAlmostEqual(archive.metrics()["raw_qd_score"], 2.0)
        self.assertAlmostEqual(archive.metrics()["qd_score"], 7.0)


def rollout(fitness: float) -> RolloutResult:
    return RolloutResult(
        name="test",
        fitness=fitness,
        steps=10,
        food_collected=1,
        hazard_contacts=0,
        wall_collisions=0,
        final_position=np.asarray([0.0, 0.0]),
        trajectory=np.zeros((0, 7)),
        termination="test",
    )


if __name__ == "__main__":
    unittest.main()
