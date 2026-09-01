"""Compare high-blowup vs typical geodesic-assignment disagreements."""

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
from scipy.stats import mannwhitneyu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disagreement-dir",
        default="results/phase5/geodesic_disagreement_analysis_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase6_high_blowup_arrival_analysis",
    )
    parser.add_argument("--high-threshold", type=float, default=2.0)
    parser.add_argument("--typical-threshold", type=float, default=1.5)
    parser.add_argument("--first-few", type=int, default=5)
    parser.add_argument("--min-evaluation", type=int, default=10001)
    args = parser.parse_args()

    disagreement_dir = Path(args.disagreement_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_candidate_rows(disagreement_dir)
    rows = [row for row in rows if int(row["evaluation"]) >= int(args.min_evaluation)]
    enriched = add_geodesic_arrival_ranks(rows)
    disagreement_rows = [row for row in enriched if row["assignment_disagrees"]]
    high_rows = [
        row for row in disagreement_rows if row["blowup_ratio"] >= float(args.high_threshold)
    ]
    typical_rows = [
        row for row in disagreement_rows if row["blowup_ratio"] < float(args.typical_threshold)
    ]

    comparison_rows = [
        summarize_group("high_blowup_ge_2", high_rows, first_few=int(args.first_few)),
        summarize_group("typical_blowup_lt_1_5", typical_rows, first_few=int(args.first_few)),
    ]
    test_rows = statistical_tests(high_rows, typical_rows)

    csv_path = output_dir / "high_vs_typical_disagreement_summary.csv"
    tests_path = output_dir / "high_vs_typical_disagreement_tests.csv"
    enriched_path = output_dir / "disagreement_arrival_ranks.csv"
    json_path = output_dir / "high_blowup_arrival_summary.json"
    plot_files = save_plots(high_rows, typical_rows, output_dir / "high_vs_typical_arrival")

    write_csv(csv_path, comparison_rows)
    write_csv(tests_path, test_rows)
    write_csv(enriched_path, [row_to_csv(row) for row in disagreement_rows])

    summary = {
        "analysis": "High-blowup disagreement arrival timing and niche-newness analysis",
        "input_dir": str(disagreement_dir),
        "high_blowup_definition": f"blowup_ratio >= {float(args.high_threshold)}",
        "typical_definition": f"blowup_ratio < {float(args.typical_threshold)}",
        "first_few_definition": (
            f"geodesic assignment rank within seed/cell <= {int(args.first_few)}"
        ),
        "analysis_scope": f"online post-retrain evaluations >= {int(args.min_evaluation)} only",
        "mutation_distance_status": (
            "Not recoverable from existing Phase 5 artifacts: evaluation CSVs do not store parent "
            "IDs, parent genomes, or candidate genomes for every evaluation."
        ),
        "group_summaries": comparison_rows,
        "statistical_tests": test_rows,
        "output_files": [
            str(csv_path),
            str(tests_path),
            str(enriched_path),
            *plot_files,
            str(json_path),
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def load_candidate_rows(disagreement_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(disagreement_dir.glob("seed_*")):
        path = seed_dir / "candidate_assignment_disagreement.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row = parse_row(raw)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No candidate disagreement CSVs found under {disagreement_dir}")
    return rows


def parse_row(raw: dict[str, str]) -> dict[str, Any]:
    return {
        "seed": int(raw["seed"]),
        "evaluation": int(raw["evaluation"]),
        "fitness": float(raw["fitness"]),
        "source": raw["source"],
        "actual_run_inserted": parse_bool(raw["actual_run_inserted"]),
        "actual_run_replaced": parse_bool(raw["actual_run_replaced"]),
        "geodesic_cell_x": int(raw["geodesic_cell_x"]),
        "geodesic_cell_y": int(raw["geodesic_cell_y"]),
        "assignment_disagrees": parse_bool(raw["assignment_disagrees"]),
        "blowup_ratio": float(raw["geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"]),
        "centroid_cell_displacement": float(raw["centroid_cell_displacement"]),
    }


def add_geodesic_arrival_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["geodesic_cell_x"]), int(row["geodesic_cell_y"]))
        grouped.setdefault(key, []).append(row)
    for key_rows in grouped.values():
        key_rows.sort(key=lambda row: int(row["evaluation"]))
        for rank, row in enumerate(key_rows, start=1):
            row["geodesic_cell_arrival_rank"] = rank
            row["geodesic_cell_first_evaluation"] = int(key_rows[0]["evaluation"])
            row["evaluations_since_first_geodesic_cell_arrival"] = int(row["evaluation"]) - int(
                key_rows[0]["evaluation"]
            )
    return rows


def summarize_group(name: str, rows: list[dict[str, Any]], first_few: int) -> dict[str, Any]:
    evaluations = values(rows, "evaluation")
    ranks = values(rows, "geodesic_cell_arrival_rank")
    since_first = values(rows, "evaluations_since_first_geodesic_cell_arrival")
    fitness = values(rows, "fitness")
    return {
        "group": name,
        "count": len(rows),
        "mean_evaluation": finite_mean(evaluations),
        "median_evaluation": finite_median(evaluations),
        "fraction_eval_le_2000": fraction(rows, lambda row: int(row["evaluation"]) <= 2000),
        "fraction_eval_le_5000": fraction(rows, lambda row: int(row["evaluation"]) <= 5000),
        "fraction_eval_le_10000": fraction(rows, lambda row: int(row["evaluation"]) <= 10000),
        "fraction_post_retrain": fraction(rows, lambda row: int(row["evaluation"]) > 10000),
        "mean_geodesic_cell_arrival_rank": finite_mean(ranks),
        "median_geodesic_cell_arrival_rank": finite_median(ranks),
        "fraction_first_assignment_to_geodesic_cell": fraction(
            rows, lambda row: int(row["geodesic_cell_arrival_rank"]) == 1
        ),
        "fraction_first_few_to_geodesic_cell": fraction(
            rows, lambda row: int(row["geodesic_cell_arrival_rank"]) <= first_few
        ),
        "mean_evaluations_since_first_geodesic_cell_arrival": finite_mean(since_first),
        "median_evaluations_since_first_geodesic_cell_arrival": finite_median(since_first),
        "mean_fitness": finite_mean(fitness),
        "median_fitness": finite_median(fitness),
        "inserted_rate": fraction(rows, lambda row: bool(row["actual_run_inserted"])),
        "replaced_rate": fraction(rows, lambda row: bool(row["actual_run_replaced"])),
        "mutated_elite_fraction": fraction(rows, lambda row: row["source"] == "mutated_elite"),
        "bootstrap_random_fraction": fraction(
            rows, lambda row: row["source"] == "bootstrap_random"
        ),
        "mutation_distance_available": False,
    }


def statistical_tests(
    high_rows: list[dict[str, Any]], typical_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tests = [
        ("evaluation", "evaluation index"),
        ("geodesic_cell_arrival_rank", "arrival rank in geodesic niche"),
        (
            "evaluations_since_first_geodesic_cell_arrival",
            "evaluations since first geodesic niche arrival",
        ),
        ("fitness", "candidate fitness"),
    ]
    output: list[dict[str, Any]] = []
    for key, label in tests:
        high = values(high_rows, key)
        typical = values(typical_rows, key)
        result = mannwhitneyu(high, typical, alternative="two-sided", method="asymptotic")
        output.append(
            {
                "metric": key,
                "label": label,
                "high_mean": finite_mean(high),
                "typical_mean": finite_mean(typical),
                "high_median": finite_median(high),
                "typical_median": finite_median(typical),
                "mannwhitney_u": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return output


def save_plots(
    high_rows: list[dict[str, Any]], typical_rows: list[dict[str, Any]], output_stem: Path
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].hist(
        [values(typical_rows, "evaluation"), values(high_rows, "evaluation")],
        bins=np.arange(0, 20001, 1000),
        label=["typical blowup < 1.5", "high blowup >= 2"],
        color=["#64748b", "#dc2626"],
        alpha=0.72,
        density=True,
    )
    axes[0].set_xlabel("evaluation")
    axes[0].set_ylabel("density")
    axes[0].set_title("When disagreements appear")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(
        [
            np.clip(values(typical_rows, "geodesic_cell_arrival_rank"), 1, 50),
            np.clip(values(high_rows, "geodesic_cell_arrival_rank"), 1, 50),
        ],
        bins=np.arange(1, 52, 2),
        label=["typical blowup < 1.5", "high blowup >= 2"],
        color=["#64748b", "#dc2626"],
        alpha=0.72,
        density=True,
    )
    axes[1].set_xlabel("arrival rank in geodesic niche (clipped at 50)")
    axes[1].set_ylabel("density")
    axes[1].set_title("How new the target niche is")
    axes[1].legend(frameon=False, fontsize=8)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def row_to_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": row["seed"],
        "evaluation": row["evaluation"],
        "fitness": row["fitness"],
        "source": row["source"],
        "blowup_ratio": row["blowup_ratio"],
        "geodesic_cell_x": row["geodesic_cell_x"],
        "geodesic_cell_y": row["geodesic_cell_y"],
        "geodesic_cell_arrival_rank": row["geodesic_cell_arrival_rank"],
        "geodesic_cell_first_evaluation": row["geodesic_cell_first_evaluation"],
        "evaluations_since_first_geodesic_cell_arrival": row[
            "evaluations_since_first_geodesic_cell_arrival"
        ],
        "actual_run_inserted": row["actual_run_inserted"],
        "actual_run_replaced": row["actual_run_replaced"],
    }


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def fraction(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return float("nan")
    return float(sum(1 for row in rows if predicate(row)) / len(rows))


def finite_mean(values_array: np.ndarray) -> float:
    finite = values_array[np.isfinite(values_array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def finite_median(values_array: np.ndarray) -> float:
    finite = values_array[np.isfinite(values_array)]
    return float(np.median(finite)) if len(finite) else float("nan")


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


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


def print_summary(summary: dict[str, Any]) -> None:
    print("HIGH-BLOWUP DISAGREEMENT ARRIVAL ANALYSIS")
    for row in summary["group_summaries"]:
        print(
            "{group}: n={count}, median_eval={median_evaluation:.1f}, "
            "median_arrival_rank={median_geodesic_cell_arrival_rank:.1f}, "
            "first_few={fraction_first_few_to_geodesic_cell:.3%}, "
            "median_fitness={median_fitness:.3f}, replaced={replaced_rate:.3%}".format(**row)
        )
    print(summary["mutation_distance_status"])
    print("tests:")
    for row in summary["statistical_tests"]:
        print(
            "  {metric}: high_median={high_median:.3f}, typical_median={typical_median:.3f}, "
            "p={p_value:.6g}".format(**row)
        )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
