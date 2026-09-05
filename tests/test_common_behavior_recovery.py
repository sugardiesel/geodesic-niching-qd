import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_common_behavior_space import (
    read_csv,
    recover_elite_evaluation_rows,
    recovery_context,
)


def evaluation(index, code, efficiency=0.2):
    return {
        "evaluation": str(index),
        "fitness": "12",
        "steps": "280",
        "food_collected": "4",
        "wall_collisions": "0",
        "hazard_contacts": "0",
        "latent_0": str(code[0]),
        "latent_1": str(code[1]),
        "path_length": "5",
        "displacement": str(efficiency * 5),
        "loop_score": "3",
    }


def elite():
    return {
        **evaluation(0, (2, 3)),
        "cell_x": "9",
        "cell_y": "9",
        "descriptor_x": "2",
        "descriptor_y": "3",
    }


class CommonBehaviorRecoveryTests(unittest.TestCase):
    def test_saved_contribution_seed_1002_selects_evaluation_8102(self):
        root = Path(__file__).resolve().parents[1]
        condition = "contribution_geodesic_niching"
        seed_dir = root / "results" / "phase5" / condition / "seed_1002"
        rows = read_csv(seed_dir / "evaluations.csv")
        archive = [
            row
            for row in read_csv(seed_dir / "archive_cells.csv")
            if (row["cell_x"], row["cell_y"]) == ("9", "9")
        ]
        matched, _ = recover_elite_evaluation_rows(
            archive, rows, **recovery_context(seed_dir, rows, condition)
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][1]["evaluation"], "8102")
        self.assertEqual(matched[0][2]["descriptor_distance"], 0.0)

    def test_saved_baseline_seed_1001_uses_verified_historical_code(self):
        root = Path(__file__).resolve().parents[1]
        condition = "baseline_b_learned_bd_euclidean"
        seed_dir = root / "results" / "phase5" / condition / "seed_1001"
        rows = read_csv(seed_dir / "evaluations.csv")
        archive = [
            row
            for row in read_csv(seed_dir / "archive_cells.csv")
            if (row["cell_x"], row["cell_y"]) == ("3", "16")
        ]
        matched, _ = recover_elite_evaluation_rows(
            archive, rows, **recovery_context(seed_dir, rows, condition)
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][1]["evaluation"], "4944")
        self.assertLess(matched[0][2]["descriptor_distance"], 1e-4)

    def test_final_encoder_codes_override_misleading_online_codes(self):
        rows = [evaluation(8102, (20, 30)), evaluation(14908, (2, 3), efficiency=0.8)]
        matched, _ = recover_elite_evaluation_rows(
            [elite()], rows, final_latents={8102: np.array([2, 3]), 14908: np.array([8, 9])}
        )
        self.assertEqual(matched[0][1]["evaluation"], "8102")
        self.assertEqual(matched[0][2]["descriptor_distance"], 0.0)

    def test_missing_final_space_codes_fail_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "requires final-encoder codes"):
            recover_elite_evaluation_rows([elite()], [evaluation(1, (2, 3)), evaluation(2, (8, 9))])

    def test_retained_elite_cannot_be_a_later_noninserted_candidate(self):
        matched, _ = recover_elite_evaluation_rows(
            [elite()],
            [evaluation(5597, (20, 30)), evaluation(12983, (2, 3))],
            retrain_evaluation=10000,
            final_insertions={},
        )
        self.assertEqual(matched[0][1]["evaluation"], "5597")

    def test_recorded_final_insertion_takes_precedence(self):
        matched, _ = recover_elite_evaluation_rows(
            [elite()],
            [evaluation(5597, (2, 3)), evaluation(12983, (20, 30))],
            retrain_evaluation=10000,
            final_insertions={(9, 9): 12983},
        )
        self.assertEqual(matched[0][1]["evaluation"], "12983")

    def test_nonmatching_final_codes_fail_instead_of_selecting_nearest(self):
        with self.assertRaisesRegex(ValueError, "no matching final-space descriptor"):
            recover_elite_evaluation_rows(
                [elite()],
                [evaluation(1, (2, 3)), evaluation(2, (8, 9))],
                final_latents={1: np.array([7, 8]), 2: np.array([9, 10])},
            )
