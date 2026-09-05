import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from algorithms.policies import StayStillPolicy
from envs.forage_maze import ForageMaze2D, load_env_config
from scripts.correct_phase1_distance_diagnostic import HistoricalPhase1ReplayMaze
from scripts.run_phase5_sweep import CONDITIONS, main, make_run_config, run_condition_seed


class BeforeSearch(Exception):
    pass


class HistoricalRespawnTests(unittest.TestCase):
    def test_default_remains_corrected(self):
        env = ForageMaze2D.from_config_path("configs/phase1_horseshoe.yaml")
        self.assertEqual(env.food_respawn_timing, "corrected")

    def test_historical_timer_preserves_immediate_decrement(self):
        env = ForageMaze2D.from_config_path(
            "configs/phase1_horseshoe.yaml", food_respawn_timing="historical"
        )
        env.reset(seed=123)
        food = env.foods[0]
        food.respawn_steps = 3
        env.state.position = food.position.copy()
        env._food_reward()
        self.assertEqual(food.respawn_timer, 3)
        env._update_food_respawns()
        self.assertEqual(food.respawn_timer, 2)
        env._update_food_respawns()
        self.assertFalse(food.active)
        env._update_food_respawns()
        self.assertTrue(food.active)

    def test_historical_mode_matches_original_reference_trajectory(self):
        config = load_env_config("configs/phase1_horseshoe.yaml")
        config["agent"]["start_position"] = config["map"]["foods"][0]["position"]
        config["agent"]["start_jitter"] = 0.0
        expected = HistoricalPhase1ReplayMaze(config).rollout(StayStillPolicy(), seed=123)
        actual = ForageMaze2D(config, food_respawn_timing="historical").rollout(
            StayStillPolicy(), seed=123
        )
        self.assertGreater(expected.food_collected, 1)
        np.testing.assert_array_equal(actual.trajectory, expected.trajectory)
        self.assertEqual(actual.fitness, expected.fitness)
        self.assertEqual(actual.food_collected, expected.food_collected)

    def test_unknown_timing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown food-respawn timing"):
            ForageMaze2D.from_config_path(
                "configs/phase1_horseshoe.yaml", food_respawn_timing="typo"
            )

    def test_sweep_config_propagates_explicit_timing_for_every_condition(self):
        for condition, info in CONDITIONS.items():
            with self.subTest(condition=condition):
                base = make_run_config(condition, info, 1001, Path("unused"), 20000)
                historical = make_run_config(
                    condition,
                    info,
                    1001,
                    Path("unused"),
                    20000,
                    protocol={"food_respawn_timing": "historical"},
                )
                self.assertNotIn("food_respawn_timing", base["experiment"])
                self.assertEqual(historical["experiment"].pop("food_respawn_timing"), "historical")
                self.assertEqual(historical, base)

    def test_runner_honors_config_and_cli_without_executing_search(self):
        runners = (
            ("scripts.run_phase2_map_elites", "run_map_elites"),
            ("scripts.run_phase3_aurora_euclidean", "run_phase3"),
            ("scripts.run_phase4_geodesic_niching", "run_phase4"),
        )
        for (module_name, search_name), info in zip(runners, CONDITIONS.values(), strict=True):
            module = importlib.import_module(module_name)
            with tempfile.TemporaryDirectory() as temp:
                config = yaml.safe_load(Path(info["base_config"]).read_text())
                config["experiment"]["output_dir"] = temp
                config["experiment"]["food_respawn_timing"] = "historical"
                path = Path(temp) / "config.yaml"
                path.write_text(yaml.safe_dump(config))
                for override, expected in (
                    ([], "historical"),
                    (["--food-respawn-timing", "corrected"], "corrected"),
                ):
                    with self.subTest(module=module_name, timing=expected):
                        argv = [module_name, "--config", str(path), *override]
                        with (
                            patch.object(sys, "argv", argv),
                            patch.object(module, search_name, side_effect=BeforeSearch) as search,
                        ):
                            with self.assertRaises(BeforeSearch):
                                module.main()
                            self.assertEqual(search.call_args.args[0].food_respawn_timing, expected)

    def test_resume_rejects_unknown_or_different_timing_without_starting_process(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            config_path = path / "config.yaml"
            summary_path = path / "summary.json"
            config_path.write_text(
                yaml.safe_dump({"experiment": {"food_respawn_timing": "historical"}})
            )
            spec = {
                "summary_path": summary_path,
                "config_path": config_path,
                "condition": "baseline",
                "seed": 1001,
            }
            for recorded in (None, "corrected", "historical"):
                summary_path.write_text(json.dumps({"food_respawn_timing": recorded}))
                with patch("scripts.run_phase5_sweep.subprocess.Popen") as process:
                    if recorded == "historical":
                        run_condition_seed(spec, resume=True)
                    else:
                        with self.assertRaisesRegex(ValueError, "Cannot resume"):
                            run_condition_seed(spec, resume=True)
                    process.assert_not_called()

    def test_aggregation_rejects_timing_override_before_touching_results(self):
        argv = [
            "run_phase5_sweep.py",
            "--aggregate-only",
            "--food-respawn-timing",
            "historical",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
            patch("scripts.run_phase5_sweep.Path.read_text") as read,
        ):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("not --aggregate-only results", stderr.getvalue())
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
