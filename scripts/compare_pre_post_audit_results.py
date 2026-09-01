"""Compare archived buggy Contribution results with corrected post-audit runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon

DATASETS = {
    "horseshoe": (
        Path("results_pre_audit_fix/phase5"),
        Path("results/phase5"),
    ),
    "open_robustness": (
        Path("results_pre_audit_fix/phase6_open_robustness"),
        Path("results/phase6_open_robustness"),
    ),
}
CONTRIBUTION = "contribution_geodesic_niching"
BASELINE_B = "baseline_b_learned_bd_euclidean"
METRICS = [
    "coverage_percent",
    "qd_score",
    "raw_qd_score",
    "max_fitness",
    "occupancy_entropy",
    "mean_pairwise_descriptor_euclidean",
    "mean_pairwise_descriptor_knn_geodesic",
    "elapsed_seconds",
    "evaluations_per_second",
]


def main() -> None:
    output_dir = Path("results/audit_fix/pre_post_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []

    for dataset, (before_root, after_root) in DATASETS.items():
        before_conditions = rows_by_key(
            read_csv(before_root / "phase5_condition_summary.csv"), "condition"
        )
        after_conditions = rows_by_key(
            read_csv(after_root / "phase5_condition_summary.csv"), "condition"
        )
        for metric in METRICS:
            before = before_conditions[CONTRIBUTION]
            after = after_conditions[CONTRIBUTION]
            before_mean = float(before[f"{metric}_mean"])
            after_mean = float(after[f"{metric}_mean"])
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "pre_fix_mean": before_mean,
                    "pre_fix_std": float(before[f"{metric}_std"]),
                    "post_fix_mean": after_mean,
                    "post_fix_std": float(after[f"{metric}_std"]),
                    "post_minus_pre": after_mean - before_mean,
                    "relative_change_percent": (
                        100.0 * (after_mean - before_mean) / abs(before_mean)
                        if abs(before_mean) > 1e-12
                        else float("nan")
                    ),
                }
            )

        for implementation, root in [("pre_fix", before_root), ("post_fix", after_root)]:
            run_rows = read_csv(root / "phase5_run_summary.csv")
            paired_rows.extend(paired_baseline_comparison(dataset, implementation, run_rows))
            graph_rows.extend(graph_diagnostics(dataset, implementation, root))

    write_csv(output_dir / "contribution_pre_post_metrics.csv", aggregate_rows)
    write_csv(output_dir / "baseline_b_vs_contribution_paired_tests.csv", paired_rows)
    write_csv(output_dir / "geodesic_graph_pre_post_diagnostics.csv", graph_rows)
    summary = {
        "purpose": "GPT-5.6 Sol Ultra audit remediation before/after comparison",
        "pairing": "Same seeds and 20,000-evaluation budget before and after the graph fix.",
        "metric_caveat": (
            "Pre-fix pairwise geodesic reporting used k=5; post-fix reporting uses corrected k=20. "
            "Quality metrics and Euclidean descriptor distance are directly comparable."
        ),
        "aggregate_rows": aggregate_rows,
        "paired_tests": paired_rows,
        "graph_diagnostics": graph_rows,
        "output_files": [
            str(output_dir / "contribution_pre_post_metrics.csv"),
            str(output_dir / "baseline_b_vs_contribution_paired_tests.csv"),
            str(output_dir / "geodesic_graph_pre_post_diagnostics.csv"),
            str(output_dir / "pre_post_summary.json"),
        ],
    }
    (output_dir / "pre_post_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_summary(aggregate_rows, paired_rows, graph_rows)


def paired_baseline_comparison(
    dataset: str,
    implementation: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    baseline = {int(row["seed"]): row for row in rows if row["condition"] == BASELINE_B}
    contribution = {int(row["seed"]): row for row in rows if row["condition"] == CONTRIBUTION}
    seeds = sorted(set(baseline) & set(contribution))
    output = []
    for metric in METRICS:
        left = np.asarray([float(baseline[seed][metric]) for seed in seeds], dtype=np.float64)
        right = np.asarray([float(contribution[seed][metric]) for seed in seeds], dtype=np.float64)
        differences = right - left
        nonzero = differences[np.abs(differences) > 1e-12]
        if len(nonzero):
            result = wilcoxon(right, left, alternative="two-sided", method="auto")
            ranks = rankdata(np.abs(nonzero))
            positive = float(np.sum(ranks[nonzero > 0.0]))
            negative = float(np.sum(ranks[nonzero < 0.0]))
            effect = (positive - negative) / (positive + negative)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        else:
            statistic = 0.0
            p_value = 1.0
            effect = 0.0
        output.append(
            {
                "dataset": dataset,
                "implementation": implementation,
                "metric": metric,
                "seed_count": len(seeds),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_effect_size": float(effect),
                "mean_contribution_minus_baseline_b": float(np.mean(differences)),
                "paired_differences": json.dumps(differences.tolist()),
            }
        )
    return output


def graph_diagnostics(dataset: str, implementation: str, root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed_dir in sorted((root / CONTRIBUTION).glob("seed_*")):
        summary = json.loads((seed_dir / "phase4_summary.json").read_text(encoding="utf-8"))
        graph = summary["geodesic_graph"]
        rows.append(
            {
                "dataset": dataset,
                "implementation": implementation,
                "seed": int(summary["seed"]),
                "graph_k": int(graph["graph_k"]),
                "connected_components_final": int(graph["connected_components"]),
                "finite_centroid_fraction_final": float(graph["finite_centroid_fraction"]),
                "penalized_centroid_distances_final": int(graph["penalized_centroid_distances"]),
                "cumulative_penalized_centroid_distances": int(
                    graph.get("cumulative_penalized_centroid_distances", 0)
                ),
                "refreshes_with_penalty": int(graph.get("refreshes_with_penalty", 0)),
            }
        )
    return rows


def rows_by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    aggregate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> None:
    print("PRE/POST AUDIT COMPARISON")
    for dataset in DATASETS:
        print(dataset)
        for metric in ["coverage_percent", "qd_score", "max_fitness", "elapsed_seconds"]:
            row = next(
                item
                for item in aggregate_rows
                if item["dataset"] == dataset and item["metric"] == metric
            )
            print(
                "  {metric}: {pre_fix_mean:.3f} -> {post_fix_mean:.3f} "
                "(delta {post_minus_pre:+.3f})".format(**row)
            )
        for implementation in ["pre_fix", "post_fix"]:
            row = next(
                item
                for item in paired_rows
                if item["dataset"] == dataset
                and item["implementation"] == implementation
                and item["metric"] == "qd_score"
            )
            print(
                f"  {implementation} Baseline B vs Contribution QD: "
                f"p={row['p_value']:.6f}, rank-biserial={row['rank_biserial_effect_size']:.3f}"
            )
    post_graph = [row for row in graph_rows if row["implementation"] == "post_fix"]
    print(
        "post-fix graph: final finite fraction mean="
        f"{np.mean([row['finite_centroid_fraction_final'] for row in post_graph]):.3f}, "
        f"components range={min(row['connected_components_final'] for row in post_graph)}-"
        f"{max(row['connected_components_final'] for row in post_graph)}"
    )


if __name__ == "__main__":
    main()
