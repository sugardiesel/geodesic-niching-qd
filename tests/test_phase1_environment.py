import unittest

import numpy as np

from algorithms.policies import StayStillPolicy
from envs.forage_maze import ForageMaze2D
from envs.maze_distance import GridShortestPath


class Phase1EnvironmentTests(unittest.TestCase):
    def test_reset_step_and_rollout(self) -> None:
        env = ForageMaze2D.from_config_path("configs/phase1_horseshoe.yaml")
        obs = env.reset(seed=123)
        self.assertEqual(obs.shape, (env.observation_dim,))
        rollout = env.rollout(StayStillPolicy(), seed=123, name="stay")
        self.assertGreater(rollout.steps, 0)
        self.assertEqual(rollout.food_collected, 0)

    def test_horseshoe_landmark_has_large_detour(self) -> None:
        env = ForageMaze2D.from_config_path("configs/phase1_horseshoe.yaml")
        shortest = GridShortestPath(env, grid_size=100, connectivity=8)
        pair = env.map_cfg["diagnostic_pairs"][0]
        a = np.asarray(pair["a"], dtype=np.float64)
        b = np.asarray(pair["b"], dtype=np.float64)
        euclidean = float(np.linalg.norm(a - b))
        geodesic = shortest.distance(a, b)
        self.assertGreater(geodesic / euclidean, 5.0)

    def test_food_waits_full_configured_steps_before_respawning(self) -> None:
        env = ForageMaze2D.from_config_path("configs/phase1_horseshoe.yaml")
        env.reset(seed=123)
        food = next(item for item in env.foods if item.respawn_steps > 0)
        assert env.state is not None
        env.state.position = food.position.copy()

        env._food_reward()
        self.assertFalse(food.active)
        self.assertEqual(food.respawn_timer, food.respawn_steps + 1)

        env._update_food_respawns()
        for _ in range(food.respawn_steps - 1):
            env._update_food_respawns()
            self.assertFalse(food.active)
        env._update_food_respawns()
        self.assertTrue(food.active)

    def test_grid_shortest_path_rejects_diagonal_corner_cutting(self) -> None:
        env = ForageMaze2D.from_config_path("configs/phase1_horseshoe.yaml")
        shortest = GridShortestPath(env, grid_size=5, connectivity=8)
        shortest.free[:] = True
        shortest.free[1, 2] = False
        shortest.free[2, 1] = False

        neighbors = {neighbor for neighbor, _ in shortest._neighbors((1, 1))}
        self.assertNotIn((2, 2), neighbors)


if __name__ == "__main__":
    unittest.main()
