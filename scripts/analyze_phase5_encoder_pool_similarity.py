"""Compare Baseline B and Contribution trajectory pools before the eval-10000 retrain."""

from __future__ import annotations

import argparse
import csv
import json
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
from scipy.stats import ks_2samp, wasserstein_distance

NUMERIC_FEATURES = [
    "fitness",
    "steps",
    "food_collected",
    "wall_collisions",
    "hazard_contacts",
    "path_length",
    "displacement",
    "loop_score",
    "bbox_area",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="results/phase5/baseline_b_learned_bd_euclidean")
    parser.add_argument(
        "--contribution-dir", default="results/phase5/contribution_geodesic_niching"
    )
    parser.add_argument("--output-dir", default="results/phase6_encoder_pool_similarity")
    parser.add_argument("--pool-start", type=int, default=2001)
    parser.add_argument("--pool-end", type=int, default=10000)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    contribution_dir = Path(args.contribution_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(path for path in baseline_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories under {baseline_dir}")

    seed_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for baseline_seed_dir in seed_dirs:
        seed = int(baseline_seed_dir.name.replace("seed_", ""))
        contribution_seed_dir = contribution_dir / baseline_seed_dir.name
        if not contribution_seed_dir.exists():
            raise FileNotFoundError(contribution_seed_dir)
        result = analyze_seed(
            seed=seed,
            baseline_csv=baseline_seed_dir / "evaluations.csv",
            contribution_csv=contribution_seed_dir / "evaluations.csv",
            pool_start=int(args.pool_start),
            pool_end=int(args.pool_end),
        )
        seed_rows.append(result["seed_summary"])
        feature_rows.extend(result["feature_rows"])
        distribution_rows.extend(result["distribution_rows"])

    aggregate_rows = aggregate_seed_summaries(seed_rows)
    feature_aggregate_rows = aggregate_feature_rows(feature_rows)
    seed_csv = output_dir / "seed_pool_similarity_summary.csv"
    aggregate_csv = output_dir / "aggregate_pool_similarity_summary.csv"
    feature_csv = output_dir / "feature_distribution_comparison.csv"
    feature_aggregate_csv = output_dir / "aggregate_feature_distribution_comparison.csv"
    distribution_csv = output_dir / "category_distribution_by_seed.csv"
    summary_json = output_dir / "encoder_pool_similarity_summary.json"
    write_csv(seed_csv, seed_rows)
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(feature_csv, feature_rows)
    write_csv(feature_aggregate_csv, feature_aggregate_rows)
    write_csv(distribution_csv, distribution_rows)
    plot_files = save_plots(seed_rows, feature_rows, output_dir / "encoder_pool_similarity")

    summary = {
        "analysis": "Baseline B vs Contribution eval-10000 autoencoder retrain-pool similarity",
        "pool_definition": (
            "condition-specific archive-driven trajectories from evaluations "
            f"{int(args.pool_start)} "
            f"through {int(args.pool_end)}, excluding the identical 1..2000 bootstrap pool"
        ),
        "seed_count": len(seed_rows),
        "seed_summaries": seed_rows,
        "aggregate": aggregate_rows,
        "feature_aggregates": feature_aggregate_rows,
        "interpretation_note": (
            "Boolean category overlap is sum(min(p_baseline, p_contribution)) over near-still, "
            "loop, and other categories; 1.0 means identical category proportions. Numeric "
            "histogram overlap is computed with shared quantile bins per seed and feature."
        ),
        "output_files": [
            str(seed_csv),
            str(aggregate_csv),
            str(feature_csv),
            str(feature_aggregate_csv),
            str(distribution_csv),
            *plot_files,
            str(summary_json),
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def analyze_seed(
    seed: int,
    baseline_csv: Path,
    contribution_csv: Path,
    pool_start: int,
    pool_end: int,
) -> dict[str, Any]:
    baseline_all = read_rows(baseline_csv)
    contribution_all = read_rows(contribution_csv)
    bootstrap_baseline = filter_range(baseline_all, 1, pool_start - 1)
    bootstrap_contribution = filter_range(contribution_all, 1, pool_start - 1)
    baseline_pool = filter_range(baseline_all, pool_start, pool_end)
    contribution_pool = filter_range(contribution_all, pool_start, pool_end)
    baseline_full_retrain = filter_range(baseline_all, 1, pool_end)
    contribution_full_retrain = filter_range(contribution_all, 1, pool_end)

    bootstrap_identical = exact_diagnostic_match_fraction(
        bootstrap_baseline, bootstrap_contribution
    )
    pool_exact_match = exact_diagnostic_match_fraction(baseline_pool, contribution_pool)
    full_exact_match = exact_diagnostic_match_fraction(
        baseline_full_retrain, contribution_full_retrain
    )

    baseline_counts = category_counts(baseline_pool)
    contribution_counts = category_counts(contribution_pool)
    baseline_dist = category_distribution(baseline_counts)
    contribution_dist = category_distribution(contribution_counts)
    category_overlap = distribution_overlap(baseline_dist, contribution_dist)
    baseline_full_counts = category_counts(baseline_full_retrain)
    contribution_full_counts = category_counts(contribution_full_retrain)
    baseline_full_dist = category_distribution(baseline_full_counts)
    contribution_full_dist = category_distribution(contribution_full_counts)
    full_category_overlap = distribution_overlap(baseline_full_dist, contribution_full_dist)

    feature_rows: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        baseline_values = values(baseline_pool, feature)
        contribution_values = values(contribution_pool, feature)
        feature_rows.append(compare_feature(seed, feature, baseline_values, contribution_values))

    path_row = next(row for row in feature_rows if row["feature"] == "path_length")
    loop_score_row = next(row for row in feature_rows if row["feature"] == "loop_score")
    fitness_row = next(row for row in feature_rows if row["feature"] == "fitness")
    seed_summary = {
        "seed": seed,
        "pool_start": pool_start,
        "pool_end": pool_end,
        "baseline_pool_count": len(baseline_pool),
        "contribution_pool_count": len(contribution_pool),
        "bootstrap_exact_diagnostic_match_fraction": bootstrap_identical,
        "archive_driven_exact_diagnostic_match_fraction": pool_exact_match,
        "full_retrain_exact_diagnostic_match_fraction": full_exact_match,
        "baseline_loop_fraction": safe_fraction(baseline_counts["loop"], len(baseline_pool)),
        "contribution_loop_fraction": safe_fraction(
            contribution_counts["loop"], len(contribution_pool)
        ),
        "loop_fraction_difference_contribution_minus_baseline": safe_fraction(
            contribution_counts["loop"], len(contribution_pool)
        )
        - safe_fraction(baseline_counts["loop"], len(baseline_pool)),
        "baseline_near_still_fraction": safe_fraction(
            baseline_counts["near_still"], len(baseline_pool)
        ),
        "contribution_near_still_fraction": safe_fraction(
            contribution_counts["near_still"], len(contribution_pool)
        ),
        "near_still_fraction_difference_contribution_minus_baseline": safe_fraction(
            contribution_counts["near_still"], len(contribution_pool)
        )
        - safe_fraction(baseline_counts["near_still"], len(baseline_pool)),
        "baseline_other_fraction": safe_fraction(baseline_counts["other"], len(baseline_pool)),
        "contribution_other_fraction": safe_fraction(
            contribution_counts["other"], len(contribution_pool)
        ),
        "category_overlap_fraction": category_overlap,
        "category_dissimilarity_fraction": 1.0 - category_overlap,
        "full_retrain_baseline_loop_fraction": safe_fraction(
            baseline_full_counts["loop"], len(baseline_full_retrain)
        ),
        "full_retrain_contribution_loop_fraction": safe_fraction(
            contribution_full_counts["loop"], len(contribution_full_retrain)
        ),
        "full_retrain_category_overlap_fraction": full_category_overlap,
        "full_retrain_category_dissimilarity_fraction": 1.0 - full_category_overlap,
        "path_length_baseline_mean": float(np.mean(values(baseline_pool, "path_length"))),
        "path_length_contribution_mean": float(np.mean(values(contribution_pool, "path_length"))),
        "path_length_cohens_d": path_row["cohens_d_contribution_minus_baseline"],
        "path_length_histogram_overlap": path_row["histogram_overlap"],
        "loop_score_baseline_mean": float(np.mean(values(baseline_pool, "loop_score"))),
        "loop_score_contribution_mean": float(np.mean(values(contribution_pool, "loop_score"))),
        "loop_score_cohens_d": loop_score_row["cohens_d_contribution_minus_baseline"],
        "loop_score_histogram_overlap": loop_score_row["histogram_overlap"],
        "fitness_baseline_mean": float(np.mean(values(baseline_pool, "fitness"))),
        "fitness_contribution_mean": float(np.mean(values(contribution_pool, "fitness"))),
        "fitness_cohens_d": fitness_row["cohens_d_contribution_minus_baseline"],
        "fitness_histogram_overlap": fitness_row["histogram_overlap"],
        "mean_numeric_histogram_overlap": float(
            np.mean([row["histogram_overlap"] for row in feature_rows])
        ),
        "mean_absolute_cohens_d": float(
            np.mean([abs(row["cohens_d_contribution_minus_baseline"]) for row in feature_rows])
        ),
        "full_retrain_mean_numeric_histogram_overlap": float(
            np.mean(
                [
                    histogram_overlap(
                        values(baseline_full_retrain, feature),
                        values(contribution_full_retrain, feature),
                    )
                    for feature in NUMERIC_FEATURES
                ]
            )
        ),
    }

    distribution_rows = []
    for category in ["near_still", "loop", "other"]:
        distribution_rows.append(
            {
                "seed": seed,
                "category": category,
                "baseline_fraction": baseline_dist[category],
                "contribution_fraction": contribution_dist[category],
                "absolute_difference": abs(baseline_dist[category] - contribution_dist[category]),
            }
        )

    return {
        "seed_summary": seed_summary,
        "feature_rows": feature_rows,
        "distribution_rows": distribution_rows,
    }


def compare_feature(
    seed: int, feature: str, baseline: np.ndarray, contribution: np.ndarray
) -> dict[str, Any]:
    ks = ks_2samp(baseline, contribution, alternative="two-sided", method="asymp")
    pooled_std = pooled_standard_deviation(baseline, contribution)
    mean_diff = float(np.mean(contribution) - np.mean(baseline))
    return {
        "seed": seed,
        "feature": feature,
        "baseline_mean": float(np.mean(baseline)),
        "contribution_mean": float(np.mean(contribution)),
        "mean_difference_contribution_minus_baseline": mean_diff,
        "baseline_median": float(np.median(baseline)),
        "contribution_median": float(np.median(contribution)),
        "median_difference_contribution_minus_baseline": float(
            np.median(contribution) - np.median(baseline)
        ),
        "baseline_std": float(np.std(baseline)),
        "contribution_std": float(np.std(contribution)),
        "cohens_d_contribution_minus_baseline": mean_diff / pooled_std if pooled_std > 0.0 else 0.0,
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "wasserstein_distance": float(wasserstein_distance(baseline, contribution)),
        "wasserstein_distance_over_pooled_std": float(
            wasserstein_distance(baseline, contribution) / pooled_std
        )
        if pooled_std > 0.0
        else 0.0,
        "histogram_overlap": histogram_overlap(baseline, contribution),
    }


def aggregate_seed_summaries(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "bootstrap_exact_diagnostic_match_fraction",
        "archive_driven_exact_diagnostic_match_fraction",
        "full_retrain_exact_diagnostic_match_fraction",
        "baseline_loop_fraction",
        "contribution_loop_fraction",
        "loop_fraction_difference_contribution_minus_baseline",
        "baseline_near_still_fraction",
        "contribution_near_still_fraction",
        "near_still_fraction_difference_contribution_minus_baseline",
        "category_overlap_fraction",
        "category_dissimilarity_fraction",
        "full_retrain_baseline_loop_fraction",
        "full_retrain_contribution_loop_fraction",
        "full_retrain_category_overlap_fraction",
        "full_retrain_category_dissimilarity_fraction",
        "path_length_baseline_mean",
        "path_length_contribution_mean",
        "path_length_cohens_d",
        "path_length_histogram_overlap",
        "loop_score_baseline_mean",
        "loop_score_contribution_mean",
        "loop_score_cohens_d",
        "loop_score_histogram_overlap",
        "fitness_baseline_mean",
        "fitness_contribution_mean",
        "fitness_cohens_d",
        "fitness_histogram_overlap",
        "mean_numeric_histogram_overlap",
        "mean_absolute_cohens_d",
        "full_retrain_mean_numeric_histogram_overlap",
    ]
    return [
        summary_row("seed_mean", seed_rows, keys),
        summary_row("seed_std", seed_rows, keys, std=True),
    ]


def aggregate_feature_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        rows = [row for row in feature_rows if row["feature"] == feature]
        output.append(
            {
                "feature": feature,
                "baseline_mean_across_seeds": finite_mean([row["baseline_mean"] for row in rows]),
                "contribution_mean_across_seeds": finite_mean(
                    [row["contribution_mean"] for row in rows]
                ),
                "mean_difference_contribution_minus_baseline": finite_mean(
                    [row["mean_difference_contribution_minus_baseline"] for row in rows]
                ),
                "mean_cohens_d": finite_mean(
                    [row["cohens_d_contribution_minus_baseline"] for row in rows]
                ),
                "mean_absolute_cohens_d": finite_mean(
                    [abs(row["cohens_d_contribution_minus_baseline"]) for row in rows]
                ),
                "mean_ks_statistic": finite_mean([row["ks_statistic"] for row in rows]),
                "mean_wasserstein_over_pooled_std": finite_mean(
                    [row["wasserstein_distance_over_pooled_std"] for row in rows]
                ),
                "mean_histogram_overlap": finite_mean([row["histogram_overlap"] for row in rows]),
            }
        )
    return output


def summary_row(
    label: str, rows: list[dict[str, Any]], keys: list[str], std: bool = False
) -> dict[str, Any]:
    output: dict[str, Any] = {"summary": label, "seed_count": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        output[key] = float(np.std(vals, ddof=1)) if std and len(vals) > 1 else float(np.mean(vals))
    return output


def filter_range(rows: list[dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    return [row for row in rows if start <= int(row["evaluation"]) <= end]


def read_rows(path: Path) -> list[dict[str, str]]:
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


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def category_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"near_still": 0, "loop": 0, "other": 0}
    for row in rows:
        if parse_bool(row["is_near_still"]):
            counts["near_still"] += 1
        elif parse_bool(row["is_loop"]):
            counts["loop"] += 1
        else:
            counts["other"] += 1
    return counts


def category_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: safe_fraction(value, total) for key, value in counts.items()}


def distribution_overlap(left: dict[str, float], right: dict[str, float]) -> float:
    return float(sum(min(left[key], right[key]) for key in left))


def exact_diagnostic_match_fraction(
    left: list[dict[str, str]], right: list[dict[str, str]]
) -> float:
    if not left or len(left) != len(right):
        return float("nan")
    keys = [
        "fitness",
        "steps",
        "food_collected",
        "wall_collisions",
        "hazard_contacts",
        "path_length",
        "displacement",
        "loop_score",
        "bbox_area",
        "is_loop",
        "is_near_still",
    ]
    matches = 0
    for left_row, right_row in zip(left, right, strict=True):
        if all(str(left_row[key]) == str(right_row[key]) for key in keys):
            matches += 1
    return safe_fraction(matches, len(left))


def pooled_standard_deviation(left: np.ndarray, right: np.ndarray) -> float:
    left_var = float(np.var(left, ddof=1)) if len(left) > 1 else 0.0
    right_var = float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    denom = max(1, len(left) + len(right) - 2)
    pooled = ((len(left) - 1) * left_var + (len(right) - 1) * right_var) / denom
    return float(np.sqrt(max(pooled, 0.0)))


def histogram_overlap(left: np.ndarray, right: np.ndarray, bins: int = 20) -> float:
    pooled = np.concatenate([left, right])
    edges = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return 1.0
    left_counts, _ = np.histogram(left, bins=edges)
    right_counts, _ = np.histogram(right, bins=edges)
    left_prob = left_counts / max(1, np.sum(left_counts))
    right_prob = right_counts / max(1, np.sum(right_counts))
    return float(np.sum(np.minimum(left_prob, right_prob)))


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def finite_mean(values_list: list[float]) -> float:
    arr = np.asarray(values_list, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def save_plots(
    seed_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], output_stem: Path
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    seeds = [str(row["seed"]) for row in seed_rows]
    x = np.arange(len(seeds))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        [row["baseline_loop_fraction"] for row in seed_rows],
        width,
        label="Baseline B",
    )
    axes[0].bar(
        x + width / 2,
        [row["contribution_loop_fraction"] for row in seed_rows],
        width,
        label="Contribution",
    )
    axes[0].set_xticks(x, seeds)
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("loop fraction")
    axes[0].set_title("Eval 2001-10000 loop rates")
    axes[0].legend(frameon=False, fontsize=8)

    selected = ["path_length", "loop_score", "fitness", "bbox_area"]
    positions = np.arange(len(selected))
    means = [
        finite_mean(
            [
                abs(row["cohens_d_contribution_minus_baseline"])
                for row in feature_rows
                if row["feature"] == feature
            ]
        )
        for feature in selected
    ]
    overlaps = [
        finite_mean([row["histogram_overlap"] for row in feature_rows if row["feature"] == feature])
        for feature in selected
    ]
    axes[1].bar(positions - width / 2, means, width, label="|Cohen's d|")
    axes[1].bar(positions + width / 2, overlaps, width, label="hist. overlap")
    axes[1].set_xticks(positions, selected, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Distribution similarity")
    axes[1].legend(frameon=False, fontsize=8)

    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def print_summary(summary: dict[str, Any]) -> None:
    print("PHASE 5 ENCODER RETRAIN-POOL SIMILARITY")
    print(summary["pool_definition"])
    aggregate = {row["summary"]: row for row in summary["aggregate"]}
    mean = aggregate["seed_mean"]
    std = aggregate["seed_std"]
    print(
        "category overlap: {category_overlap_fraction:.3%} +/- {std_overlap:.3%}; "
        "dissimilarity={category_dissimilarity_fraction:.3%}; "
        "full retrain overlap={full_retrain_category_overlap_fraction:.3%}".format(
            std_overlap=std["category_overlap_fraction"],
            **mean,
        )
    )
    print(
        "loop fractions: baseline={baseline_loop_fraction:.3%}, "
        "contribution={contribution_loop_fraction:.3%}, "
        "diff={loop_fraction_difference_contribution_minus_baseline:.3%}".format(**mean)
    )
    print(
        "near-still fractions: baseline={baseline_near_still_fraction:.3%}, "
        "contribution={contribution_near_still_fraction:.3%}, "
        "diff={near_still_fraction_difference_contribution_minus_baseline:.3%}".format(**mean)
    )
    print(
        "path length means: baseline={path_length_baseline_mean:.3f}, "
        "contribution={path_length_contribution_mean:.3f}, Cohen d={path_length_cohens_d:.3f}, "
        "hist overlap={path_length_histogram_overlap:.3f}".format(**mean)
    )
    print(
        "mean numeric histogram overlap={mean_numeric_histogram_overlap:.3f}, "
        "mean |Cohen d|={mean_absolute_cohens_d:.3f}".format(**mean)
    )
    print("per-seed:")
    for row in summary["seed_summaries"]:
        print(
            "  seed {seed}: category_overlap={category_overlap_fraction:.3%}, "
            "loop B={baseline_loop_fraction:.3%} C={contribution_loop_fraction:.3%}, "
            "path d={path_length_cohens_d:.3f}, "
            "numeric_overlap={mean_numeric_histogram_overlap:.3f}".format(**row)
        )
    print("output files:")
    for path in summary["output_files"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
