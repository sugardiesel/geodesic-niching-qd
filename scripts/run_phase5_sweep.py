"""Run and aggregate the Phase 5 production sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
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
import yaml
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

CONDITIONS: dict[str, dict[str, str]] = {
    "baseline_a_handcrafted_map_elites": {
        "label": "Baseline A: hand-crafted BD",
        "base_config": "configs/phase2_handcrafted_map_elites.yaml",
        "script": "scripts/run_phase2_map_elites.py",
        "summary": "phase2_summary.json",
        "heatmap_stem": "archive_heatmap_final_xy",
    },
    "baseline_b_learned_bd_euclidean": {
        "label": "Baseline B: learned BD Euclidean",
        "base_config": "configs/phase3_aurora_euclidean.yaml",
        "script": "scripts/run_phase3_aurora_euclidean.py",
        "summary": "phase3_summary.json",
        "heatmap_stem": "archive_heatmap_learned_latent",
    },
    "contribution_geodesic_niching": {
        "label": "Contribution: geodesic niching",
        "base_config": "configs/phase4_geodesic_niching.yaml",
        "script": "scripts/run_phase4_geodesic_niching.py",
        "summary": "phase4_summary.json",
        "heatmap_stem": "archive_heatmap_geodesic_latent",
    },
}

METRICS: dict[str, str] = {
    "coverage_percent": "Coverage (%)",
    "qd_score": "Shifted QD-score",
    "raw_qd_score": "Raw QD-score",
    "max_fitness": "Max fitness",
    "occupancy_entropy": "Occupancy entropy (coverage-derived)",
    "mean_pairwise_descriptor_euclidean": "Mean pairwise elite descriptor distance (Euclidean)",
    "mean_pairwise_descriptor_knn_geodesic": (
        "Mean pairwise elite descriptor distance (k-NN geodesic)"
    ),
    "elapsed_seconds": "Wall-clock runtime (s)",
    "evaluations_per_second": "Throughput (eval/s)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/production_protocol.yaml")
    parser.add_argument("--output-dir", default="results/phase5")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError(f"{args.protocol} must contain a mapping.")
    protocol["protocol_path"] = args.protocol
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = build_run_specs(protocol, output_dir)
    if not args.aggregate_only:
        for spec in run_specs:
            run_condition_seed(spec, resume=not args.no_resume)
    aggregate_phase5(run_specs, output_dir, protocol_path=args.protocol)


def build_run_specs(protocol: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    budget = protocol["production_budget"]
    seeds = [int(seed) for seed in budget["seeds"]]
    total_evaluations = int(budget["total_evaluations_per_seed"])
    specs: list[dict[str, Any]] = []
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for condition in budget["conditions"]:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown Phase 5 condition: {condition}")
        condition_info = CONDITIONS[condition]
        for seed in seeds:
            run_dir = output_dir / condition / f"seed_{seed}"
            config_path = config_dir / f"{condition}_seed_{seed}.yaml"
            config = make_run_config(
                condition,
                condition_info,
                seed,
                run_dir,
                total_evaluations,
                protocol=protocol,
                protocol_path=str(
                    protocol.get("protocol_path", "configs/production_protocol.yaml")
                ),
            )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            specs.append(
                {
                    "condition": condition,
                    "label": condition_info["label"],
                    "seed": seed,
                    "script": condition_info["script"],
                    "config_path": config_path,
                    "run_dir": run_dir,
                    "summary_path": run_dir / condition_info["summary"],
                    "heatmap_stem": condition_info["heatmap_stem"],
                }
            )
    return specs


def make_run_config(
    condition: str,
    condition_info: dict[str, str],
    seed: int,
    run_dir: Path,
    total_evaluations: int,
    protocol: dict[str, Any] | None = None,
    protocol_path: str = "configs/production_protocol.yaml",
) -> dict[str, Any]:
    base_config = yaml.safe_load(Path(condition_info["base_config"]).read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError(f"{condition_info['base_config']} must contain a mapping.")
    config = json.loads(json.dumps(base_config))
    config["experiment"]["name"] = f"phase5_{condition}_seed_{seed}"
    config["experiment"]["seed"] = int(seed)
    config["experiment"]["output_dir"] = str(run_dir)
    config["experiment"]["production_protocol"] = protocol_path
    config["experiment"]["plot_title"] = f"Phase 5 {condition_info['label']} seed {seed}"
    if protocol is not None and protocol.get("env_config_override"):
        config["experiment"]["env_config"] = str(protocol["env_config_override"])
    if protocol is not None and protocol.get("experiment_name_prefix"):
        prefix = str(protocol["experiment_name_prefix"])
        config["experiment"]["name"] = f"{prefix}_{condition}_seed_{seed}"
        config["experiment"]["plot_title"] = f"{prefix} {condition_info['label']} seed {seed}"
    config["algorithm"]["total_evaluations"] = int(total_evaluations)
    reporting_k = int(
        (protocol or {})
        .get("metrics", {})
        .get(
            "pairwise_geodesic_k",
            config["metrics"]["pairwise_geodesic_k"],
        )
    )
    config["metrics"]["pairwise_geodesic_k"] = reporting_k
    if condition == "contribution_geodesic_niching":
        geodesic_protocol = (protocol or {}).get("phase4_geodesic_design", {})
        config["geodesic"]["graph_k"] = int(
            geodesic_protocol.get("graph_k", config["geodesic"]["graph_k"])
        )
        config["geodesic"]["assignment_k"] = int(
            geodesic_protocol.get("assignment_k", config["geodesic"]["assignment_k"])
        )
        config["geodesic"]["max_graph_points"] = 1000
        config["geodesic"]["reservoir_strategy"] = "rolling_recent_buffer"
    return config


def run_condition_seed(spec: dict[str, Any], resume: bool) -> None:
    summary_path = Path(spec["summary_path"])
    if resume and summary_path.exists():
        print(f"[phase5] skip existing {spec['condition']} seed {spec['seed']}: {summary_path}")
        return
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "phase5_run.log"
    command = [sys.executable, spec["script"], "--config", str(spec["config_path"])]
    print(f"[phase5] start {spec['condition']} seed {spec['seed']}")
    print(f"[phase5] command: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Phase 5 run failed: {spec['condition']} seed {spec['seed']}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Run completed but missing summary: {summary_path}")
    print(f"[phase5] complete {spec['condition']} seed {spec['seed']}")


def aggregate_phase5(run_specs: list[dict[str, Any]], output_dir: Path, protocol_path: str) -> None:
    rows = [load_run_row(spec) for spec in run_specs]
    write_csv(output_dir / "phase5_run_summary.csv", rows)
    condition_rows = aggregate_conditions(rows)
    write_csv(output_dir / "phase5_condition_summary.csv", condition_rows)
    stats_rows = wilcoxon_rows(rows)
    write_csv(output_dir / "phase5_wilcoxon_paired_tests.csv", stats_rows)
    unpaired_reference_rows = mann_whitney_rows(rows)
    write_csv(output_dir / "phase5_mannwhitney_unpaired_reference.csv", unpaired_reference_rows)
    curve_files = save_aggregate_curves(
        run_specs, output_dir / "phase5_aggregate_qd_coverage_curves"
    )
    heatmap_rows = copy_representative_heatmaps(
        rows, run_specs, output_dir / "representative_heatmaps"
    )
    write_csv(output_dir / "phase5_representative_heatmaps.csv", heatmap_rows)
    summary = {
        "protocol": protocol_path,
        "fresh_phase5_only": True,
        "excluded_prior_runs_note": (
            "Aggregation only reads summaries under this Phase 5 output directory. Earlier sanity "
            "runs and k=10 single-seed artifacts are not included."
        ),
        "seed_policy": "The same fixed seed list is reused for all three conditions.",
        "conditions": list(CONDITIONS),
        "metrics": METRICS,
        "output_files": [
            str(output_dir / "phase5_run_summary.csv"),
            str(output_dir / "phase5_condition_summary.csv"),
            str(output_dir / "phase5_wilcoxon_paired_tests.csv"),
            str(output_dir / "phase5_mannwhitney_unpaired_reference.csv"),
            *curve_files,
            str(output_dir / "phase5_representative_heatmaps.csv"),
            str(output_dir / "phase5_summary.json"),
        ],
    }
    (output_dir / "phase5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_phase5_summary(condition_rows, stats_rows, summary["output_files"])


def load_run_row(spec: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads(Path(spec["summary_path"]).read_text(encoding="utf-8"))
    final = summary["final_metrics"]
    return {
        "condition": spec["condition"],
        "label": spec["label"],
        "seed": int(spec["seed"]),
        "run_dir": str(spec["run_dir"]),
        "coverage_percent": float(final["coverage"]) * 100.0,
        "qd_score": float(final["qd_score"]),
        "raw_qd_score": float(final["raw_qd_score"]),
        "max_fitness": float(final["max_fitness"]),
        "occupancy_entropy": float(final["occupancy_entropy"]),
        "mean_pairwise_descriptor_euclidean": float(final["mean_pairwise_descriptor_euclidean"]),
        "mean_pairwise_descriptor_knn_geodesic": float(
            final["mean_pairwise_descriptor_knn_geodesic"]
        ),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "evaluations_per_second": float(summary["evaluations_per_second"]),
        "pairwise_geodesic_k": int(final["pairwise_geodesic_k"]),
        "total_evaluations": int(summary["total_evaluations"]),
    }


def aggregate_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition, label in [(key, value["label"]) for key, value in CONDITIONS.items()]:
        condition_rows = [row for row in rows if row["condition"] == condition]
        aggregate: dict[str, Any] = {
            "condition": condition,
            "label": label,
            "seed_count": len(condition_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in condition_rows], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(aggregate)
    return output


def mann_whitney_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("baseline_b_learned_bd_euclidean", "contribution_geodesic_niching"),
        ("baseline_a_handcrafted_map_elites", "baseline_b_learned_bd_euclidean"),
    ]
    output: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_rows = [row for row in rows if row["condition"] == left]
        right_rows = [row for row in rows if row["condition"] == right]
        for metric in METRICS:
            left_values = np.asarray([float(row[metric]) for row in left_rows], dtype=np.float64)
            right_values = np.asarray([float(row[metric]) for row in right_rows], dtype=np.float64)
            result = mannwhitneyu(
                left_values, right_values, alternative="two-sided", method="exact"
            )
            output.append(
                {
                    "comparison": f"{left} vs {right}",
                    "left_condition": left,
                    "right_condition": right,
                    "metric": metric,
                    "metric_label": METRICS[metric],
                    "left_mean": float(np.mean(left_values)),
                    "right_mean": float(np.mean(right_values)),
                    "mean_difference_right_minus_left": float(
                        np.mean(right_values) - np.mean(left_values)
                    ),
                    "u_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "note": (
                        "Unpaired reference only; the matched-seed Wilcoxon table is authoritative."
                    ),
                }
            )
    return output


def wilcoxon_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("baseline_b_learned_bd_euclidean", "contribution_geodesic_niching"),
        ("baseline_a_handcrafted_map_elites", "baseline_b_learned_bd_euclidean"),
    ]
    output: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_by_seed = {int(row["seed"]): row for row in rows if row["condition"] == left}
        right_by_seed = {int(row["seed"]): row for row in rows if row["condition"] == right}
        shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
        if set(left_by_seed) != set(right_by_seed):
            raise ValueError(f"Paired comparison requires identical seed sets: {left} vs {right}")
        for metric in METRICS:
            left_values = np.asarray(
                [float(left_by_seed[seed][metric]) for seed in shared_seeds],
                dtype=np.float64,
            )
            right_values = np.asarray(
                [float(right_by_seed[seed][metric]) for seed in shared_seeds],
                dtype=np.float64,
            )
            differences = right_values - left_values
            nonzero = differences[np.abs(differences) > 1e-12]
            if len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0
                rank_biserial = 0.0
            else:
                result = wilcoxon(
                    right_values,
                    left_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
                ranks = rankdata(np.abs(nonzero))
                positive_rank = float(np.sum(ranks[nonzero > 0.0]))
                negative_rank = float(np.sum(ranks[nonzero < 0.0]))
                rank_biserial = (positive_rank - negative_rank) / (positive_rank + negative_rank)
            output.append(
                {
                    "comparison": f"{left} vs {right}",
                    "left_condition": left,
                    "right_condition": right,
                    "metric": metric,
                    "metric_label": METRICS[metric],
                    "paired_seed_count": len(shared_seeds),
                    "seeds": json.dumps(shared_seeds),
                    "left_values_by_seed": json.dumps(left_values.tolist()),
                    "right_values_by_seed": json.dumps(right_values.tolist()),
                    "paired_differences_right_minus_left": json.dumps(differences.tolist()),
                    "left_mean": float(np.mean(left_values)),
                    "right_mean": float(np.mean(right_values)),
                    "mean_paired_difference_right_minus_left": float(np.mean(differences)),
                    "median_paired_difference_right_minus_left": float(np.median(differences)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial_effect_size": float(rank_biserial),
                    "note": (
                        "Authoritative paired shared-seed comparison; low n limits statistical "
                        "power."
                    ),
                }
            )
    return output


def save_aggregate_curves(run_specs: list[dict[str, Any]], output_stem: Path) -> list[str]:
    grid = np.arange(2000, 20001, 100, dtype=np.float64)
    colors = {
        "baseline_a_handcrafted_map_elites": "#2563eb",
        "baseline_b_learned_bd_euclidean": "#16a34a",
        "contribution_geodesic_niching": "#dc2626",
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True, constrained_layout=True)
    for condition, info in CONDITIONS.items():
        specs = [spec for spec in run_specs if spec["condition"] == condition]
        qd_curves = []
        coverage_curves = []
        for spec in specs:
            metrics_path = Path(spec["run_dir"]) / "metrics.csv"
            evaluations, qd_score, coverage = read_curve(metrics_path)
            qd_curves.append(np.interp(grid, evaluations, qd_score))
            coverage_curves.append(np.interp(grid, evaluations, coverage * 100.0))
        qd = np.asarray(qd_curves, dtype=np.float64)
        coverage_arr = np.asarray(coverage_curves, dtype=np.float64)
        plot_mean_std(axes[0], grid, qd, info["label"], colors[condition])
        plot_mean_std(axes[1], grid, coverage_arr, info["label"], colors[condition])
    axes[0].set_ylabel("shifted QD-score")
    axes[1].set_ylabel("coverage (%)")
    axes[1].set_xlabel("evaluations")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Phase 5 aggregate QD-score and coverage curves (mean +/- std)")
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    evaluations = np.asarray([float(row["evaluation"]) for row in rows], dtype=np.float64)
    qd_score = np.asarray([float(row["qd_score"]) for row in rows], dtype=np.float64)
    coverage = np.asarray([float(row["coverage"]) for row in rows], dtype=np.float64)
    order = np.argsort(evaluations)
    return evaluations[order], qd_score[order], coverage[order]


def plot_mean_std(ax: plt.Axes, x: np.ndarray, curves: np.ndarray, label: str, color: str) -> None:
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0.0)


def copy_representative_heatmaps(
    rows: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = sorted(
            [row for row in rows if row["condition"] == condition],
            key=lambda row: float(row["qd_score"]),
        )
        median_row = condition_rows[len(condition_rows) // 2]
        spec = next(
            item
            for item in run_specs
            if item["condition"] == condition and int(item["seed"]) == int(median_row["seed"])
        )
        stem = spec["heatmap_stem"]
        copied_files: list[str] = []
        for suffix in [".png", ".pdf"]:
            source = Path(spec["run_dir"]) / f"{stem}{suffix}"
            if source.exists():
                target = output_dir / f"{condition}_seed_{median_row['seed']}_{stem}{suffix}"
                shutil.copy2(source, target)
                copied_files.append(str(target))
        output.append(
            {
                "condition": condition,
                "label": spec["label"],
                "representative_seed": int(median_row["seed"]),
                "selection_rule": "median seed by shifted QD-score",
                "qd_score": float(median_row["qd_score"]),
                "files": ";".join(copied_files),
            }
        )
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def print_phase5_summary(
    condition_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    output_files: list[str],
) -> None:
    print("PHASE 5 SWEEP SUMMARY")
    for row in condition_rows:
        print(
            "{label}: coverage={coverage_percent_mean:.2f}±{coverage_percent_std:.2f}%, "
            "QD={qd_score_mean:.2f}±{qd_score_std:.2f}, "
            "max_fit={max_fitness_mean:.2f}±{max_fitness_std:.2f}, "
            "geo_dist={mean_pairwise_descriptor_knn_geodesic_mean:.2f}±"
            "{mean_pairwise_descriptor_knn_geodesic_std:.2f}".format(**row)
        )
    print("Paired Wilcoxon p-values:")
    for row in stats_rows:
        if row["metric"] in {
            "coverage_percent",
            "qd_score",
            "mean_pairwise_descriptor_knn_geodesic",
        }:
            print(f"  {row['comparison']} {row['metric']}: p={row['p_value']:.6f}")
    print("output files:")
    for output_file in output_files:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
