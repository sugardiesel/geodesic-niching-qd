"""Recompute Phase 1.5 distances on the exact historical rollout pair set.

The original Phase 1.5 results predate two independent July fixes: food-respawn timing and
diagonal corner cutting in the grid shortest-path solver. This script replays the historical
respawn behavior, verifies the resulting 4,186 Euclidean pair distances against the saved CSV,
and then changes only the shortest-path solver. It therefore isolates the distance correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase1_plots import save_distance_scatter, save_region_distance_scatter
from envs.forage_maze import ForageMaze2D, load_env_config
from envs.maze_distance import GridShortestPath, pairwise_euclidean_and_geodesic
from scripts.run_phase1_validation import (
    classify_pairs_by_horseshoe_region,
    compute_landmark_stats,
    run_diagnostic_rollouts,
    subset_correlation,
    write_pairwise_csv,
)


class HistoricalPhase1ReplayMaze(ForageMaze2D):
    """Replay only the pre-audit food-respawn timing used by the saved pair set."""

    def _food_reward(self) -> float:
        if self.state is None:
            raise RuntimeError("Call reset() before reward computation.")
        reward = 0.0
        for food in self.foods:
            if not food.active:
                continue
            distance = float(np.linalg.norm(self.state.position - food.position))
            if distance <= self.agent_radius + food.radius:
                food.active = False
                food.respawn_timer = food.respawn_steps
                self.food_collected += 1
                reward += food.reward * float(self.fitness_cfg["food_reward_scale"])
                self.state.energy = min(
                    float(self.agent_cfg["max_energy"]),
                    self.state.energy + food.energy_gain,
                )
        return reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_horseshoe.yaml")
    parser.add_argument(
        "--legacy-pairs",
        default="results/phase1_environment/phase1_euclidean_vs_geodesic_pairs.csv",
    )
    parser.add_argument(
        "--legacy-summary",
        default="results/phase1_environment/phase1_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1_distance_diagnostic_corrected",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = HistoricalPhase1ReplayMaze(load_env_config(args.config))
    base_seed = int(env.config["experiment"]["base_seed"])
    random_policy_count = int(env.config["validation"]["random_policy_count"])

    rollouts = run_diagnostic_rollouts(env, base_seed, random_policy_count)
    final_points = np.asarray([rollout.final_position for rollout in rollouts])
    shortest = GridShortestPath(
        env,
        grid_size=int(env.config["shortest_path"]["grid_size"]),
        connectivity=int(env.config["shortest_path"]["connectivity"]),
    )
    euclidean, geodesic = pairwise_euclidean_and_geodesic(final_points, shortest)
    finite = np.isfinite(euclidean) & np.isfinite(geodesic) & (euclidean > 1e-9)

    legacy_euclidean = _read_float_column(Path(args.legacy_pairs), "euclidean_final_xy")
    legacy_geodesic = _read_float_column(
        Path(args.legacy_pairs), "maze_geodesic_shortest_path"
    )
    legacy_region_mask = _read_bool_column(
        Path(args.legacy_pairs), "shortest_path_touches_horseshoe_region"
    )
    replay_euclidean = euclidean[finite]
    if len(legacy_euclidean) != len(replay_euclidean):
        raise RuntimeError("Historical replay produced a different number of finite pairs.")
    formatted_match = all(
        f"{current:.8f}" == f"{legacy:.8f}"
        for current, legacy in zip(replay_euclidean, legacy_euclidean, strict=True)
    )
    if not formatted_match:
        max_difference = float(np.max(np.abs(replay_euclidean - legacy_euclidean)))
        raise RuntimeError(
            "Historical replay does not reproduce the saved pair set "
            f"(maximum Euclidean difference {max_difference:.12g})."
        )

    correlation = float(np.corrcoef(replay_euclidean, geodesic[finite])[0, 1])
    ratios = geodesic[finite] / replay_euclidean
    region_mask_all = classify_pairs_by_horseshoe_region(final_points, shortest, env)
    region_mask = region_mask_all[finite]
    geodesic_change = np.abs(geodesic[finite] - legacy_geodesic)
    changed_geodesic_count = sum(
        f"{current:.8f}" != f"{legacy:.8f}"
        for current, legacy in zip(geodesic[finite], legacy_geodesic, strict=True)
    )
    horseshoe_r = subset_correlation(replay_euclidean, geodesic[finite], region_mask)
    open_r = subset_correlation(replay_euclidean, geodesic[finite], ~region_mask)
    landmark_stats, _ = compute_landmark_stats(env, shortest)

    scatter_files = save_distance_scatter(
        euclidean,
        geodesic,
        output_dir / "phase1_euclidean_vs_geodesic_scatter",
        "Final-position distances: Euclidean vs maze geodesic",
        correlation,
    )
    report_scatter_files = save_region_distance_scatter(
        euclidean,
        geodesic,
        region_mask_all,
        output_dir / "phase1_report_euclidean_vs_geodesic_by_region",
        "Maze distance separates horseshoe and open-field pairs",
        correlation,
        horseshoe_r,
        open_r,
    )
    pair_csv = output_dir / "phase1_euclidean_vs_geodesic_pairs.csv"
    write_pairwise_csv(pair_csv, replay_euclidean, geodesic[finite], region_mask)

    legacy_summary = json.loads(Path(args.legacy_summary).read_text(encoding="utf-8"))
    legacy_diagnostic = legacy_summary["phase1_5_distance_diagnostic"]
    summary = {
        "config": args.config,
        "map": env.map_cfg["name"],
        "base_seed": base_seed,
        "correction_scope": (
            "Exact historical Phase 1.5 rollout pair set; only diagonal corner cutting in "
            "GridShortestPath is corrected."
        ),
        "pair_set_validation": {
            "legacy_pair_source": args.legacy_pairs,
            "pair_count": len(replay_euclidean),
            "euclidean_values_match_legacy_csv_at_8_decimals": formatted_match,
            "max_absolute_euclidean_difference_from_saved_csv": float(
                np.max(np.abs(replay_euclidean - legacy_euclidean))
            ),
            "changed_geodesic_pair_count_at_8_decimals": changed_geodesic_count,
            "mean_absolute_geodesic_change": float(np.mean(geodesic_change)),
            "max_absolute_geodesic_change": float(np.max(geodesic_change)),
            "region_labels_match_legacy_csv": bool(np.array_equal(region_mask, legacy_region_mask)),
        },
        "phase1_5_distance_diagnostic": {
            "rollouts": len(rollouts),
            "pair_count": len(replay_euclidean),
            "pearson_correlation": correlation,
            "horseshoe_region_pair_count": int(np.count_nonzero(region_mask)),
            "open_field_pair_count": int(np.count_nonzero(~region_mask)),
            "horseshoe_region_pearson_correlation": horseshoe_r,
            "open_field_pearson_correlation": open_r,
            "median_geodesic_to_euclidean_ratio": float(np.median(ratios)),
            "p95_geodesic_to_euclidean_ratio": float(np.percentile(ratios, 95)),
            "max_geodesic_to_euclidean_ratio": float(np.max(ratios)),
            "detour_pair_count_ratio_gt_3": int(np.count_nonzero(ratios > 3.0)),
            "detour_pair_count_ratio_gt_5": int(np.count_nonzero(ratios > 5.0)),
        },
        "report_authoritative_values": {
            "use_these_values_wherever_phase1_5_correlations_are_quoted_in_results": True,
            "report_text": {
                "overall_pearson_r": f"{correlation:.3f}",
                "horseshoe_region_pearson_r": f"{horseshoe_r:.3f}",
                "open_field_pearson_r": f"{open_r:.3f}",
            },
            "overall_pearson_r_rounded_3dp": round(correlation, 3),
            "horseshoe_region_pearson_r_rounded_3dp": round(horseshoe_r, 3),
            "open_field_pearson_r_rounded_3dp": round(open_r, 3),
            "superseded_overall_pearson_r_rounded_3dp": round(
                float(legacy_diagnostic["pearson_correlation"]), 3
            ),
            "superseded_horseshoe_region_pearson_r_rounded_3dp": round(
                float(legacy_diagnostic["horseshoe_region_pearson_correlation"]), 3
            ),
            "superseded_open_field_pearson_r_rounded_3dp": round(
                float(legacy_diagnostic["open_field_pearson_correlation"]), 3
            ),
            "note": (
                "Use 0.813 overall, 0.470 horseshoe, and 0.999 open-field in the report. "
                "These supersede 0.814/0.472/0.999 from the pre-corner-fix output."
            ),
        },
        "legacy_phase1_5_distance_diagnostic": legacy_diagnostic,
        "landmark_diagnostic_pairs": landmark_stats,
        "output_files": [*scatter_files, *report_scatter_files, str(pair_csv)],
    }
    summary_path = output_dir / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _read_float_column(path: Path, column: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([float(row[column]) for row in rows], dtype=np.float64)


def _read_bool_column(path: Path, column: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([row[column].strip().lower() == "true" for row in rows], dtype=bool)


if __name__ == "__main__":
    main()
