"""Run Phase 1 environment validation and the Phase 1.5 distance diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.policies import (
    RandomActionPolicy,
    RandomMLPPolicy,
    SeekNearestFoodPolicy,
    StayStillPolicy,
    WaypointPolicy,
)
from analysis.phase1_plots import (
    save_distance_scatter,
    save_region_distance_scatter,
    save_rollout_plot,
)
from envs.forage_maze import ForageMaze2D, RolloutResult
from envs.maze_distance import GridShortestPath, pairwise_euclidean_and_geodesic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_horseshoe.yaml")
    parser.add_argument("--output-dir", default="results/phase1_environment")
    args = parser.parse_args()

    env = ForageMaze2D.from_config_path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(env.config["experiment"]["base_seed"])
    random_policy_count = int(env.config["validation"]["random_policy_count"])
    speed_evaluations = int(env.config["validation"]["speed_benchmark_evaluations"])

    demo_rollouts = run_demo_rollouts(env, base_seed)
    shortest = GridShortestPath(
        env,
        grid_size=int(env.config["shortest_path"]["grid_size"]),
        connectivity=int(env.config["shortest_path"]["connectivity"]),
    )
    landmark_stats, landmark_paths = compute_landmark_stats(env, shortest)
    rollout_plot_files = save_rollout_plot(
        env,
        demo_rollouts,
        output_dir / "phase1_rollout_demo",
        "ForageMaze2D Phase 1 rollouts",
        extra_paths=landmark_paths,
    )

    diagnostic_rollouts = run_diagnostic_rollouts(env, base_seed, random_policy_count)
    final_points = np.asarray([rollout.final_position for rollout in diagnostic_rollouts])
    euclidean, geodesic = pairwise_euclidean_and_geodesic(final_points, shortest)
    finite = np.isfinite(euclidean) & np.isfinite(geodesic) & (euclidean > 1e-9)
    correlation = float(np.corrcoef(euclidean[finite], geodesic[finite])[0, 1])
    ratios = geodesic[finite] / euclidean[finite]
    scatter_plot_files = save_distance_scatter(
        euclidean,
        geodesic,
        output_dir / "phase1_euclidean_vs_geodesic_scatter",
        "Final-position distances: Euclidean vs maze geodesic",
        correlation,
    )
    region_mask_all = classify_pairs_by_horseshoe_region(final_points, shortest, env)
    region_mask = region_mask_all[finite]
    horseshoe_r = subset_correlation(euclidean[finite], geodesic[finite], region_mask)
    open_r = subset_correlation(euclidean[finite], geodesic[finite], ~region_mask)
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
    write_pairwise_csv(
        output_dir / "phase1_euclidean_vs_geodesic_pairs.csv",
        euclidean[finite],
        geodesic[finite],
        region_mask,
    )

    speed = run_speed_benchmark(env, base_seed + 500_000, speed_evaluations)
    degenerate = check_degenerate_policies(demo_rollouts)

    summary = {
        "config": args.config,
        "map": env.map_cfg["name"],
        "base_seed": base_seed,
        "seed_policy": env.config["experiment"]["seed_policy"],
        "observation_dim": env.observation_dim,
        "random_mlp_hidden_dim": 16,
        "random_mlp_parameter_count": RandomMLPPolicy.from_rng(
            env.observation_dim, 16, np.random.default_rng(base_seed)
        ).parameter_count,
        "demo_rollouts": [rollout_summary(rollout) for rollout in demo_rollouts],
        "degenerate_policy_check": degenerate,
        "speed_benchmark": speed,
        "phase1_5_distance_diagnostic": {
            "rollouts": len(diagnostic_rollouts),
            "pair_count": int(np.count_nonzero(finite)),
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
        "landmark_diagnostic_pairs": landmark_stats,
        "output_files": [
            *rollout_plot_files,
            *scatter_plot_files,
            *report_scatter_files,
            str(output_dir / "phase1_euclidean_vs_geodesic_pairs.csv"),
            str(output_dir / "phase1_summary.json"),
        ],
    }
    (output_dir / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print_summary(summary)


def run_demo_rollouts(env: ForageMaze2D, base_seed: int) -> list[RolloutResult]:
    policies = [
        ("stay_still", StayStillPolicy()),
        ("random_action", RandomActionPolicy(base_seed + 1)),
        ("seek_nearest_food", SeekNearestFoodPolicy()),
        (
            "safe_corner",
            WaypointPolicy([(-0.88, -0.86)]),
        ),
        (
            "inside_horseshoe_route",
            WaypointPolicy(
                [(-0.90, -0.42), (-0.90, 0.64), (-0.10, 0.78), (-0.10, 0.38), (0.00, -0.49)]
            ),
        ),
        (
            "right_field_route",
            WaypointPolicy(
                [(-0.90, -0.42), (-0.90, 0.66), (0.34, 0.78), (0.86, 0.32), (0.78, -0.50)]
            ),
        ),
    ]
    return [
        env.rollout(policy, seed=base_seed + idx * 1009, name=name)
        for idx, (name, policy) in enumerate(policies)
    ]


def run_diagnostic_rollouts(
    env: ForageMaze2D,
    base_seed: int,
    random_policy_count: int,
) -> list[RolloutResult]:
    rollouts = run_demo_rollouts(env, base_seed + 90_000)
    targeted = [
        ("diag_left_of_u_wall", WaypointPolicy([(-0.90, -0.42), (-0.34, -0.56)])),
        (
            "diag_inside_left_wall",
            WaypointPolicy([(-0.90, -0.42), (-0.90, 0.64), (-0.12, 0.78), (-0.12, -0.56)]),
        ),
        ("diag_below_u_bottom", WaypointPolicy([(-0.42, -0.78), (0.00, -0.78)])),
        (
            "diag_inside_bottom",
            WaypointPolicy([(-0.90, -0.42), (-0.90, 0.64), (-0.04, 0.78), (-0.04, -0.56)]),
        ),
        (
            "diag_right_outer_wall",
            WaypointPolicy([(-0.90, -0.42), (-0.90, 0.66), (0.40, 0.78), (0.32, -0.54)]),
        ),
        (
            "diag_inside_right_wall",
            WaypointPolicy([(-0.90, -0.42), (-0.90, 0.64), (0.12, 0.78), (0.12, -0.56)]),
        ),
    ]
    for idx, (name, policy) in enumerate(targeted):
        rollouts.append(env.rollout(policy, seed=base_seed + 91_000 + idx * 1009, name=name))
    rng = np.random.default_rng(base_seed + 123)
    for idx in range(random_policy_count):
        policy = RandomMLPPolicy.from_rng(env.observation_dim, 16, rng)
        seed = base_seed + idx * 1009 + 77
        rollouts.append(env.rollout(policy, seed=seed, name=f"random_mlp_{idx:03d}", record=False))
    return rollouts


def compute_landmark_stats(
    env: ForageMaze2D,
    shortest: GridShortestPath,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    stats: list[dict[str, float | str]] = []
    paths: dict[str, np.ndarray] = {}
    for pair in env.map_cfg.get("diagnostic_pairs", []):
        a = np.asarray(pair["a"], dtype=np.float64)
        b = np.asarray(pair["b"], dtype=np.float64)
        euclidean = float(np.linalg.norm(a - b))
        geodesic = shortest.distance(a, b)
        ratio = geodesic / max(euclidean, 1e-9)
        name = str(pair["name"])
        stats.append(
            {
                "name": name,
                "euclidean": euclidean,
                "geodesic": geodesic,
                "geodesic_to_euclidean_ratio": ratio,
            }
        )
        paths[name] = shortest.shortest_path(a, b)
    return stats, paths


def classify_pairs_by_horseshoe_region(
    points: np.ndarray,
    shortest: GridShortestPath,
    env: ForageMaze2D,
) -> np.ndarray:
    bounds = env.config.get("diagnostic", {}).get("horseshoe_region")
    if bounds is None:
        return np.zeros(points.shape[0] * (points.shape[0] - 1) // 2, dtype=bool)
    region = (
        float(bounds["x_min"]),
        float(bounds["y_min"]),
        float(bounds["x_max"]),
        float(bounds["y_max"]),
    )
    mask: list[bool] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            path = shortest.shortest_path(points[i], points[j])
            mask.append(path_touches_region(path, region))
    return np.asarray(mask, dtype=bool)


def path_touches_region(path: np.ndarray, region: tuple[float, float, float, float]) -> bool:
    if len(path) == 0:
        return False
    x_min, y_min, x_max, y_max = region
    inside_x = (path[:, 0] >= x_min) & (path[:, 0] <= x_max)
    inside_y = (path[:, 1] >= y_min) & (path[:, 1] <= y_max)
    return bool(np.any(inside_x & inside_y))


def subset_correlation(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x_subset = x[mask]
    y_subset = y[mask]
    if float(np.std(x_subset)) <= 1e-12 or float(np.std(y_subset)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_subset, y_subset)[0, 1])


def run_speed_benchmark(env: ForageMaze2D, seed: int, evaluations: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    fitnesses = []
    foods = []
    start = time.perf_counter()
    for idx in range(evaluations):
        policy = RandomMLPPolicy.from_rng(env.observation_dim, 16, rng)
        rollout = env.rollout(policy, seed=seed + idx * 1009, name="speed_random_mlp", record=False)
        fitnesses.append(rollout.fitness)
        foods.append(rollout.food_collected)
    elapsed = time.perf_counter() - start
    return {
        "evaluations": evaluations,
        "elapsed_seconds": elapsed,
        "evaluations_per_second": evaluations / elapsed,
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "mean_food_collected": float(np.mean(foods)),
    }


def check_degenerate_policies(rollouts: list[RolloutResult]) -> dict[str, float | str | bool]:
    by_name = {rollout.name: rollout for rollout in rollouts}
    stay = by_name["stay_still"].fitness
    corner = by_name["safe_corner"].fitness
    active_names = ["seek_nearest_food", "inside_horseshoe_route", "right_field_route"]
    best_active = max(by_name[name].fitness for name in active_names)
    best_degenerate = max(stay, corner)
    passed = best_active > best_degenerate + 2.0
    return {
        "status": "pass" if passed else "needs_adjustment",
        "passed": passed,
        "stay_still_fitness": stay,
        "safe_corner_fitness": corner,
        "best_active_fitness": best_active,
        "margin_best_active_minus_degenerate": best_active - best_degenerate,
    }


def write_pairwise_csv(
    path: Path,
    euclidean: np.ndarray,
    geodesic: np.ndarray,
    horseshoe_region: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "euclidean_final_xy",
                "maze_geodesic_shortest_path",
                "ratio",
                "shortest_path_touches_horseshoe_region",
            ]
        )
        for euc, geo, is_horseshoe in zip(euclidean, geodesic, horseshoe_region, strict=False):
            writer.writerow(
                [
                    f"{float(euc):.8f}",
                    f"{float(geo):.8f}",
                    f"{float(geo / euc):.8f}",
                    str(bool(is_horseshoe)).lower(),
                ]
            )


def rollout_summary(rollout: RolloutResult) -> dict[str, float | int | str | list[float]]:
    return {
        "name": rollout.name,
        "fitness": rollout.fitness,
        "steps": rollout.steps,
        "food_collected": rollout.food_collected,
        "hazard_contacts": rollout.hazard_contacts,
        "wall_collisions": rollout.wall_collisions,
        "final_position": [float(rollout.final_position[0]), float(rollout.final_position[1])],
        "termination": rollout.termination,
    }


def print_summary(summary: dict[str, object]) -> None:
    print("PHASE 1 VALIDATION SUMMARY")
    print(f"config: {summary['config']}")
    print(f"map: {summary['map']}")
    print(f"base_seed: {summary['base_seed']}")
    print(f"observation_dim: {summary['observation_dim']}")
    print(f"random_mlp_parameter_count: {summary['random_mlp_parameter_count']}")
    print("\ndemo rollouts:")
    for rollout in summary["demo_rollouts"]:  # type: ignore[index]
        print(
            "  {name}: fitness={fitness:.3f}, steps={steps}, food={food_collected}, "
            "hazards={hazard_contacts}, walls={wall_collisions}, final=({x:.3f},{y:.3f}), "
            "term={termination}".format(
                x=rollout["final_position"][0],
                y=rollout["final_position"][1],
                **rollout,
            )
        )
    degenerate = summary["degenerate_policy_check"]  # type: ignore[index]
    print(
        "\ndegenerate policy check: {status} "
        "(stay={stay_still_fitness:.3f}, safe_corner={safe_corner_fitness:.3f}, "
        "best_active={best_active_fitness:.3f}, "
        "margin={margin_best_active_minus_degenerate:.3f})".format(**degenerate)
    )
    speed = summary["speed_benchmark"]  # type: ignore[index]
    print(
        "\nspeed benchmark: {evaluations} evals in {elapsed_seconds:.3f}s = "
        "{evaluations_per_second:.2f} eval/s; mean fitness={mean_fitness:.3f}".format(**speed)
    )
    diagnostic = summary["phase1_5_distance_diagnostic"]  # type: ignore[index]
    print(
        "\nPhase 1.5 diagnostic: {rollouts} rollouts, {pair_count} pairs, "
        "Pearson r={pearson_correlation:.3f}, "
        "median ratio={median_geodesic_to_euclidean_ratio:.3f}, "
        "p95 ratio={p95_geodesic_to_euclidean_ratio:.3f}, "
        "max ratio={max_geodesic_to_euclidean_ratio:.3f}, "
        "ratio>3 pairs={detour_pair_count_ratio_gt_3}".format(**diagnostic)
    )
    print(
        "  by region: horseshoe n={horseshoe_region_pair_count}, "
        "r={horseshoe_region_pearson_correlation:.3f}; "
        "open n={open_field_pair_count}, r={open_field_pearson_correlation:.3f}".format(
            **diagnostic
        )
    )
    print("\nlandmark detour checks:")
    for pair in summary["landmark_diagnostic_pairs"]:  # type: ignore[index]
        print(
            "  {name}: euclidean={euclidean:.3f}, geodesic={geodesic:.3f}, "
            "ratio={geodesic_to_euclidean_ratio:.2f}".format(**pair)
        )
    print("\noutput files:")
    for output_file in summary["output_files"]:  # type: ignore[index]
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
