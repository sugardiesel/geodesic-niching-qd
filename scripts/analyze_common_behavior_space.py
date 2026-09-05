"""Post-hoc common behavior-space comparison for learned-BD archives.

This analysis deliberately ignores each condition's learned latent coordinates. It reconstructs the
final archive elites from saved archive/evaluation CSVs, maps them into a fixed hand-defined raw
behavior space, and computes coverage/QD/pairwise metrics in that shared space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon

QD_SCORE_FLOOR = -5.0
GRID_BINS = (8, 8, 8)
LOOP_SCORE_MAX = 12.0
FOOD_MAX = 8.0

DATASETS = [
    {
        "map": "horseshoe",
        "baseline_dir": Path("results/phase5/baseline_b_learned_bd_euclidean"),
        "contribution_dir": Path("results/phase5/contribution_geodesic_niching"),
    },
    {
        "map": "open_robustness",
        "baseline_dir": Path("results/phase6_open_robustness/baseline_b_learned_bd_euclidean"),
        "contribution_dir": Path("results/phase6_open_robustness/contribution_geodesic_niching"),
    },
]

CONDITION_LABELS = {
    "baseline_b_learned_bd_euclidean": "Baseline B: learned BD Euclidean",
    "contribution_geodesic_niching": "Contribution: geodesic niching",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase6_common_behavior_space")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elite_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for condition, root in [
            ("baseline_b_learned_bd_euclidean", dataset["baseline_dir"]),
            ("contribution_geodesic_niching", dataset["contribution_dir"]),
        ]:
            seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
            if not seed_dirs:
                raise FileNotFoundError(f"No seed directories under {root}")
            for seed_dir in seed_dirs:
                seed = int(seed_dir.name.replace("seed_", ""))
                result = analyze_seed(
                    map_name=str(dataset["map"]),
                    condition=condition,
                    seed=seed,
                    seed_dir=seed_dir,
                )
                elite_rows.extend(result["elite_rows"])
                metric_rows.append(result["metrics"])

    summary_rows = summarize_conditions(metric_rows)
    test_rows = statistical_tests(metric_rows)

    elite_csv = output_dir / "common_behavior_elites.csv"
    metrics_csv = output_dir / "common_behavior_metrics_by_seed.csv"
    summary_csv = output_dir / "common_behavior_condition_summary.csv"
    tests_csv = output_dir / "common_behavior_wilcoxon_paired_tests.csv"
    summary_json = output_dir / "common_behavior_summary.json"
    write_csv(elite_csv, elite_rows)
    write_csv(metrics_csv, metric_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(tests_csv, test_rows)
    plot_files = save_plots(metric_rows, elite_rows, output_dir / "common_behavior_space")

    summary = {
        "analysis": "Representation-independent common behavior-space archive comparison",
        "feature_space": {
            "features": [
                {
                    "name": "path_efficiency",
                    "definition": (
                        "displacement / path_length, clipped to [0, 1]; set to 0 for "
                        "zero-length paths"
                    ),
                    "reason": (
                        "captures direct movement versus wandering/looping without depending on "
                        "final x,y position"
                    ),
                },
                {
                    "name": "loop_score",
                    "definition": "total absolute turn divided by 2*pi, clipped to [0, 12]",
                    "reason": (
                        "captures cyclic/recurrent movement strategies directly from the raw "
                        "trajectory"
                    ),
                },
                {
                    "name": "food_collected",
                    "definition": "raw food count, clipped to [0, 8]",
                    "reason": (
                        "captures task/foraging success as a behaviorally meaningful outcome "
                        "dimension"
                    ),
                },
            ],
            "grid_bins": list(GRID_BINS),
            "grid_cell_count": int(np.prod(GRID_BINS)),
            "distance_space": (
                "Euclidean distance after normalizing the three fixed features to [0, 1]"
            ),
            "qd_score_floor": QD_SCORE_FLOOR,
        },
        "elite_recovery_note": (
            "No search was rerun. Recovery uses recorded final-stage insertions, fitness and "
            "episode counters, and descriptors in the final encoder space only. Baseline B "
            "pre-retraining ties use verified checkpoint replays; Contribution uses its saved "
            "final-space codes. Unresolved matches raise an error rather than choosing a "
            "trajectory by closeness in incompatible encoder spaces."
        ),
        "condition_summaries": summary_rows,
        "statistical_tests": test_rows,
        "output_files": [
            str(elite_csv),
            str(metrics_csv),
            str(summary_csv),
            str(tests_csv),
            *plot_files,
            str(summary_json),
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def analyze_seed(map_name: str, condition: str, seed: int, seed_dir: Path) -> dict[str, Any]:
    archive_rows = read_csv(seed_dir / "archive_cells.csv")
    evaluation_rows = read_csv(seed_dir / "evaluations.csv")
    context = recovery_context(seed_dir, evaluation_rows, condition)
    matched_rows, recovery = recover_elite_evaluation_rows(archive_rows, evaluation_rows, **context)

    elite_rows: list[dict[str, Any]] = []
    cell_best: dict[tuple[int, int, int], float] = {}
    feature_points: list[np.ndarray] = []
    for archive_row, eval_row, match_info in matched_rows:
        features = common_features(eval_row)
        cell = common_cell(features)
        normalized = normalized_features(features)
        fitness = float(eval_row["fitness"])
        cell_best[cell] = max(cell_best.get(cell, -math.inf), fitness)
        feature_points.append(normalized)
        elite_rows.append(
            {
                "map": map_name,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "seed": seed,
                "archive_cell_x": int(archive_row["cell_x"]),
                "archive_cell_y": int(archive_row["cell_y"]),
                "evaluation": int(eval_row["evaluation"]),
                "fitness": fitness,
                "common_cell_0": cell[0],
                "common_cell_1": cell[1],
                "common_cell_2": cell[2],
                "path_efficiency": features["path_efficiency"],
                "loop_score_raw": float(eval_row["loop_score"]),
                "loop_score_clipped": features["loop_score"],
                "food_collected_raw": float(eval_row["food_collected"]),
                "food_collected_clipped": features["food_collected"],
                "path_length": float(eval_row["path_length"]),
                "displacement": float(eval_row["displacement"]),
                "bbox_area": float(eval_row["bbox_area"]),
                "is_loop": parse_bool(eval_row["is_loop"]),
                "is_near_still": parse_bool(eval_row["is_near_still"]),
                "match_candidate_count": match_info["candidate_count"],
                "match_descriptor_distance": match_info["descriptor_distance"],
                "match_status": match_info["status"],
            }
        )

    feature_array = np.asarray(feature_points, dtype=np.float64)
    occupied_cells = len(cell_best)
    metrics = {
        "map": map_name,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "seed": seed,
        "archive_elites": len(archive_rows),
        "recovered_elites": len(elite_rows),
        "unmatched_elites": recovery["unmatched"],
        "ambiguous_elites": recovery["ambiguous"],
        "descriptor_tiebreaks": recovery["descriptor_tiebreaks"],
        "common_grid_cells": int(np.prod(GRID_BINS)),
        "common_occupied_cells": occupied_cells,
        "common_coverage": occupied_cells / int(np.prod(GRID_BINS)),
        "common_coverage_percent": 100.0 * occupied_cells / int(np.prod(GRID_BINS)),
        "common_qd_score": shifted_qd_score(np.asarray(list(cell_best.values()), dtype=np.float64)),
        "common_raw_qd_score": float(
            np.sum(np.asarray(list(cell_best.values()), dtype=np.float64))
        ),
        "common_mean_pairwise_distance": mean_pairwise_distance(feature_array),
        "mean_path_efficiency": finite_mean([row["path_efficiency"] for row in elite_rows]),
        "mean_loop_score_clipped": finite_mean([row["loop_score_clipped"] for row in elite_rows]),
        "mean_food_collected": finite_mean([row["food_collected_clipped"] for row in elite_rows]),
        "loop_fraction": finite_mean([1.0 if row["is_loop"] else 0.0 for row in elite_rows]),
        "near_still_fraction": finite_mean(
            [1.0 if row["is_near_still"] else 0.0 for row in elite_rows]
        ),
    }
    return {"elite_rows": elite_rows, "metrics": metrics}


def recover_elite_evaluation_rows(
    archive_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
    *,
    final_latents: dict[int, np.ndarray] | None = None,
    retrain_evaluation: int | None = None,
    final_insertions: dict[tuple[int, int], int] | None = None,
) -> tuple[list[tuple[dict[str, str], dict[str, str], dict[str, Any]]], dict[str, int]]:
    index: dict[tuple[int, int, int, int], list[dict[str, str]]] = {}
    for row in evaluation_rows:
        key = discrete_key(row)
        index.setdefault(key, []).append(row)

    matched: list[tuple[dict[str, str], dict[str, str], dict[str, Any]]] = []
    recovery = {"unmatched": 0, "ambiguous": 0, "descriptor_tiebreaks": 0}
    for archive_row in archive_rows:
        key = discrete_key(archive_row)
        candidates = [
            row
            for row in index.get(key, [])
            if math.isclose(
                float(row["fitness"]), float(archive_row["fitness"]), rel_tol=0.0, abs_tol=1e-9
            )
        ]
        candidate_count = len(candidates)
        if candidate_count > 1:
            recovery["ambiguous"] += 1
        cell = (int(archive_row["cell_x"]), int(archive_row["cell_y"]))
        status = "unique"
        if final_insertions is not None:
            recorded = final_insertions.get(cell)
            if recorded is not None:
                candidates = [row for row in candidates if int(row["evaluation"]) == recorded]
                status = "recorded_final_stage_insertion"
            else:
                candidates = [
                    row for row in candidates if int(row["evaluation"]) <= int(retrain_evaluation)
                ]
                status = "retained_from_retrain"
        if not candidates:
            raise ValueError(f"No verified evaluation matches final archive cell {cell}.")
        descriptor_distance = float("nan")
        if len(candidates) > 1:
            codes = final_latents or {}
            missing = [
                int(row["evaluation"]) for row in candidates if int(row["evaluation"]) not in codes
            ]
            if missing:
                raise ValueError(
                    f"Cell {cell} requires final-encoder codes for evaluations {missing}."
                )
            archive_descriptor = np.asarray(
                [float(archive_row["descriptor_x"]), float(archive_row["descriptor_y"])]
            )
            distances = [
                float(np.linalg.norm(codes[int(row["evaluation"])] - archive_descriptor))
                for row in candidates
            ]
            matching = [i for i, distance in enumerate(distances) if distance <= 1e-4]
            if not matching:
                raise ValueError(
                    f"Cell {cell} has no matching final-space descriptor: {distances}."
                )
            reference = normalized_features(common_features(candidates[matching[0]]))
            if any(
                not np.allclose(
                    normalized_features(common_features(candidates[i])),
                    reference,
                    atol=1e-12,
                    rtol=0.0,
                )
                for i in matching[1:]
            ):
                raise ValueError(
                    f"Cell {cell} has unresolved final-space ties with different behaviors."
                )
            selected_index = min(matching, key=lambda i: int(candidates[i]["evaluation"]))
            selected = candidates[selected_index]
            descriptor_distance = distances[selected_index]
            recovery["descriptor_tiebreaks"] += 1
            status = "verified_final_space_descriptor"
        else:
            selected = candidates[0]
        matched.append(
            (
                archive_row,
                selected,
                {
                    "candidate_count": candidate_count,
                    "descriptor_distance": descriptor_distance,
                    "status": status,
                },
            )
        )
    return matched, recovery


def recovery_context(
    seed_dir: Path, evaluation_rows: list[dict[str, str]], condition: str
) -> dict[str, Any]:
    if condition == "contribution_geodesic_niching":
        rows = read_csv(seed_dir / "visited_latents_final_space.csv")
        return {
            "final_latents": {
                int(row["index"]) + 1: np.asarray([float(row["latent_0"]), float(row["latent_1"])])
                for row in rows
            }
        }
    summary = json.loads((seed_dir / "phase3_summary.json").read_text(encoding="utf-8"))
    retrain = max([int(summary["bootstrap_evaluations"]), *summary["retrain_evaluations"]])
    insertions = {
        (int(row["cell_x"]), int(row["cell_y"])): int(row["evaluation"])
        for row in evaluation_rows
        if int(row["evaluation"]) > retrain and parse_bool(row["inserted"])
    }
    path = seed_dir / "recovered_pre_retrain_latents.csv"
    codes = (
        {
            int(row["evaluation"]): np.asarray([float(row["latent_0"]), float(row["latent_1"])])
            for row in read_csv(path)
        }
        if path.exists()
        else {}
    )
    return {"final_latents": codes, "retrain_evaluation": retrain, "final_insertions": insertions}


def discrete_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(float(row["steps"])),
        int(float(row["food_collected"])),
        int(float(row["wall_collisions"])),
        int(float(row["hazard_contacts"])),
    )


def common_features(row: dict[str, str]) -> dict[str, float]:
    path_length = float(row["path_length"])
    displacement = float(row["displacement"])
    if path_length <= 1e-12:
        path_efficiency = 0.0
    else:
        path_efficiency = displacement / path_length
    return {
        "path_efficiency": float(np.clip(path_efficiency, 0.0, 1.0)),
        "loop_score": float(np.clip(float(row["loop_score"]), 0.0, LOOP_SCORE_MAX)),
        "food_collected": float(np.clip(float(row["food_collected"]), 0.0, FOOD_MAX)),
    }


def normalized_features(features: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            features["path_efficiency"],
            features["loop_score"] / LOOP_SCORE_MAX,
            features["food_collected"] / FOOD_MAX,
        ],
        dtype=np.float64,
    )


def common_cell(features: dict[str, float]) -> tuple[int, int, int]:
    normalized = normalized_features(features)
    indices = np.floor(normalized * np.asarray(GRID_BINS, dtype=np.float64)).astype(int)
    indices = np.clip(indices, 0, np.asarray(GRID_BINS, dtype=int) - 1)
    return (int(indices[0]), int(indices[1]), int(indices[2]))


def shifted_qd_score(fitnesses: np.ndarray) -> float:
    if len(fitnesses) == 0:
        return 0.0
    return float(np.sum(np.maximum(fitnesses - QD_SCORE_FLOOR, 0.0)))


def mean_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    delta = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    tri = np.triu_indices(len(points), k=1)
    return float(np.mean(distances[tri]))


def summarize_conditions(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        "common_coverage_percent",
        "common_qd_score",
        "common_raw_qd_score",
        "common_mean_pairwise_distance",
        "common_occupied_cells",
        "archive_elites",
        "ambiguous_elites",
        "unmatched_elites",
    ]
    for map_name in sorted({row["map"] for row in metric_rows}):
        for condition in ["baseline_b_learned_bd_euclidean", "contribution_geodesic_niching"]:
            rows = [
                row
                for row in metric_rows
                if row["map"] == map_name and row["condition"] == condition
            ]
            if not rows:
                continue
            summary: dict[str, Any] = {
                "map": map_name,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "seed_count": len(rows),
            }
            for metric in metrics:
                values_array = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                summary[f"{metric}_mean"] = float(np.mean(values_array))
                summary[f"{metric}_std"] = (
                    float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
                )
            output.append(summary)
    return output


def statistical_tests(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        ("common_coverage_percent", "Common-space coverage (%)"),
        ("common_qd_score", "Common-space shifted QD-score"),
        ("common_mean_pairwise_distance", "Common-space mean pairwise distance"),
    ]
    for map_name in sorted({row["map"] for row in metric_rows}):
        baseline = {
            int(row["seed"]): row
            for row in metric_rows
            if row["map"] == map_name and row["condition"] == "baseline_b_learned_bd_euclidean"
        }
        contribution = {
            int(row["seed"]): row
            for row in metric_rows
            if row["map"] == map_name and row["condition"] == "contribution_geodesic_niching"
        }
        if set(baseline) != set(contribution):
            raise ValueError(f"Common-space paired comparison has mismatched seeds for {map_name}.")
        seeds = sorted(baseline)
        for metric, label in metrics:
            left = np.asarray([float(baseline[seed][metric]) for seed in seeds], dtype=np.float64)
            right = np.asarray(
                [float(contribution[seed][metric]) for seed in seeds], dtype=np.float64
            )
            differences = right - left
            nonzero = differences[np.abs(differences) > 1e-12]
            if len(nonzero):
                result = wilcoxon(right, left, alternative="two-sided", method="auto")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
                ranks = rankdata(np.abs(nonzero))
                positive = float(np.sum(ranks[nonzero > 0.0]))
                negative = float(np.sum(ranks[nonzero < 0.0]))
                rank_biserial = (positive - negative) / (positive + negative)
            else:
                statistic = 0.0
                p_value = 1.0
                rank_biserial = 0.0
            output.append(
                {
                    "map": map_name,
                    "comparison": "Baseline B vs Contribution",
                    "metric": metric,
                    "metric_label": label,
                    "baseline_values_by_seed": ";".join(
                        f"{seed}:{float(baseline[seed][metric]):.6f}" for seed in seeds
                    ),
                    "contribution_values_by_seed": ";".join(
                        f"{seed}:{float(contribution[seed][metric]):.6f}" for seed in seeds
                    ),
                    "baseline_mean": float(np.mean(left)),
                    "contribution_mean": float(np.mean(right)),
                    "mean_difference_contribution_minus_baseline": float(
                        np.mean(right) - np.mean(left)
                    ),
                    "paired_differences_contribution_minus_baseline": ";".join(
                        f"{seed}:{difference:.6f}"
                        for seed, difference in zip(seeds, differences, strict=True)
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial_effect_size": float(rank_biserial),
                }
            )
    return output


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    greater = 0
    less = 0
    for a in x:
        greater += int(np.count_nonzero(a > y))
        less += int(np.count_nonzero(a < y))
    denom = len(x) * len(y)
    return float((greater - less) / denom) if denom else float("nan")


def save_plots(
    metric_rows: list[dict[str, Any]], elite_rows: list[dict[str, Any]], output_stem: Path
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), constrained_layout=True)
    metrics = [
        ("common_coverage_percent", "Coverage (%)"),
        ("common_qd_score", "Shifted QD"),
        ("common_mean_pairwise_distance", "Mean pairwise distance"),
    ]
    colors = {
        "baseline_b_learned_bd_euclidean": "#2563eb",
        "contribution_geodesic_niching": "#dc2626",
    }
    maps = sorted({row["map"] for row in metric_rows})
    conditions = ["baseline_b_learned_bd_euclidean", "contribution_geodesic_niching"]
    group_labels = [
        f"{map_name}\n{condition.split('_')[0] if condition.startswith('baseline') else 'contrib'}"
        for map_name in maps
        for condition in conditions
    ]
    x = np.arange(len(group_labels))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        values_list = []
        color_list = []
        for map_name in maps:
            for condition in conditions:
                rows = [
                    row
                    for row in metric_rows
                    if row["map"] == map_name and row["condition"] == condition
                ]
                values_list.append([float(row[metric]) for row in rows])
                color_list.append(colors[condition])
        ax.boxplot(values_list, positions=x, widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(ax.patches, color_list, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
        for idx, vals in enumerate(values_list):
            ax.scatter(
                np.full(len(vals), idx), vals, s=18, c=color_list[idx], alpha=0.8, linewidths=0
            )
        ax.set_xticks(x, group_labels, rotation=0, fontsize=8)
        ax.set_title(title)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)

    scatter_paths = save_feature_scatter(
        elite_rows, output_stem.with_name(output_stem.name + "_elite_scatter")
    )
    return [str(path) for path in paths] + scatter_paths


def save_feature_scatter(elite_rows: list[dict[str, Any]], output_stem: Path) -> list[str]:
    maps = sorted({row["map"] for row in elite_rows})
    fig, axes = plt.subplots(1, len(maps), figsize=(5.2 * len(maps), 4.1), constrained_layout=True)
    if len(maps) == 1:
        axes = [axes]
    colors = {
        "baseline_b_learned_bd_euclidean": "#2563eb",
        "contribution_geodesic_niching": "#dc2626",
    }
    for ax, map_name in zip(axes, maps, strict=True):
        for condition in ["baseline_b_learned_bd_euclidean", "contribution_geodesic_niching"]:
            rows = [
                row
                for row in elite_rows
                if row["map"] == map_name and row["condition"] == condition
            ]
            ax.scatter(
                [row["path_efficiency"] for row in rows],
                [row["loop_score_clipped"] for row in rows],
                s=13,
                alpha=0.35,
                c=colors[condition],
                label=CONDITION_LABELS[condition],
                linewidths=0,
            )
        ax.set_xlabel("path efficiency")
        ax.set_ylabel("loop score (clipped)")
        ax.set_title(map_name.replace("_", " "))
        ax.legend(frameon=False, fontsize=8)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def print_summary(summary: dict[str, Any]) -> None:
    print("COMMON BEHAVIOR-SPACE ANALYSIS")
    print(
        "features: path_efficiency=displacement/path_length, loop_score clipped to [0, 12], "
        "food_collected clipped to [0, 8]; grid=8x8x8"
    )
    print("condition summaries:")
    for row in summary["condition_summaries"]:
        print(
            "  {map} {condition}: coverage={common_coverage_percent_mean:.3f}% +/- "
            "{common_coverage_percent_std:.3f}, QD={common_qd_score_mean:.3f} +/- "
            "{common_qd_score_std:.3f}, pairwise={common_mean_pairwise_distance_mean:.4f} +/- "
            "{common_mean_pairwise_distance_std:.4f}, unmatched={unmatched_elites_mean:.1f}, "
            "ambiguous={ambiguous_elites_mean:.1f}".format(**row)
        )
    print("tests:")
    for row in summary["statistical_tests"]:
        print(
            "  {map} {metric}: B={baseline_mean:.6f}, C={contribution_mean:.6f}, "
            "diff={mean_difference_contribution_minus_baseline:.6f}, W={wilcoxon_statistic:.3f}, "
            "p={p_value:.6g}, rank-biserial={rank_biserial_effect_size:.3f}".format(**row)
        )
    print("output files:")
    for path in summary["output_files"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
