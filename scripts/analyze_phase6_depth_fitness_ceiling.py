"""Test whether geodesically deep or high-blowup-target niches have lower final fitness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
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
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon
from scipy.stats import t as student_t

from algorithms.knn_graph import (
    build_endpoint_manifold_graph,
    endpoint_shortest_path_distances,
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    archive_dir: Path
    disagreement_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase6_depth_fitness_ceiling_analysis")
    parser.add_argument("--high-blowup-threshold", type=float, default=2.0)
    parser.add_argument("--depth-match-key", default="bulk_depth")
    parser.add_argument("--min-touch-evaluation", type=int, default=10001)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        DatasetSpec(
            label="horseshoe",
            archive_dir=Path("results/phase5/contribution_geodesic_niching"),
            disagreement_dir=Path("results/phase5/geodesic_disagreement_analysis_v2"),
        ),
        DatasetSpec(
            label="open_robustness",
            archive_dir=Path("results/phase6_open_robustness/contribution_geodesic_niching"),
            disagreement_dir=Path("results/phase6_open_robustness/geodesic_disagreement_analysis"),
        ),
    ]

    niche_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        result = analyze_dataset(
            dataset,
            high_threshold=float(args.high_blowup_threshold),
            min_touch_evaluation=int(args.min_touch_evaluation),
        )
        niche_rows.extend(result["niche_rows"])
        run_rows.extend(result["run_rows"])

    correlation_rows = correlation_summary(niche_rows)
    match_pairs_depth = nearest_depth_matches(
        niche_rows, depth_key=str(args.depth_match_key), include_assignment_count=False
    )
    match_pairs_depth_count = nearest_depth_matches(
        niche_rows,
        depth_key=str(args.depth_match_key),
        include_assignment_count=True,
    )
    match_summary_rows = [
        *matching_summary(match_pairs_depth, "nearest_bulk_depth"),
        *matching_summary(match_pairs_depth_count, "nearest_bulk_depth_plus_assignment_count"),
    ]
    bin_rows = depth_bin_comparison(niche_rows, depth_key=str(args.depth_match_key), bin_count=5)
    regression_rows = regression_summary(niche_rows)
    group_rows = high_target_group_summary(niche_rows)

    niche_csv = output_dir / "niche_depth_fitness_rows.csv"
    run_csv = output_dir / "run_depth_fitness_summary.csv"
    corr_csv = output_dir / "depth_fitness_correlations.csv"
    match_csv = output_dir / "nearest_depth_matched_pairs.csv"
    match_count_csv = output_dir / "nearest_depth_plus_count_matched_pairs.csv"
    match_summary_csv = output_dir / "controlled_matching_summary.csv"
    bin_csv = output_dir / "depth_bin_controlled_comparison.csv"
    regression_csv = output_dir / "depth_touch_regression.csv"
    group_csv = output_dir / "high_blowup_target_group_summary.csv"
    summary_json = output_dir / "depth_fitness_ceiling_summary.json"

    write_csv(niche_csv, niche_rows)
    write_csv(run_csv, run_rows)
    write_csv(corr_csv, correlation_rows)
    write_csv(match_csv, match_pairs_depth)
    write_csv(match_count_csv, match_pairs_depth_count)
    write_csv(match_summary_csv, match_summary_rows)
    write_csv(bin_csv, bin_rows)
    write_csv(regression_csv, regression_rows)
    write_csv(group_csv, group_rows)

    plot_files = [
        *save_depth_scatter(niche_rows, output_dir / "depth_vs_final_fitness"),
        *save_match_plot(
            match_pairs_depth, output_dir / "nearest_depth_matched_fitness_differences"
        ),
    ]

    summary = {
        "analysis": "Phase 6 structural fitness-ceiling diagnostic",
        "method": (
            "Final filled Contribution archive cells only. A high-blowup target is a final cell "
            "that was ever the geodesic-assigned target of a Euclidean/geodesic assignment "
            f"disagreement with blowup ratio >= {float(args.high_blowup_threshold)}. Home depth "
            "is final-graph geodesic distance from the centroid nearest the mean bootstrap latent. "
            "Bulk depth is mean final-graph geodesic distance from the cell centroid to all other "
            "filled archive centroids in the same seed."
        ),
        "touch_scope": f"online post-retrain evaluations >= {int(args.min_touch_evaluation)} only",
        "datasets": [dataset.label for dataset in datasets],
        "niche_count": len(niche_rows),
        "run_count": len(run_rows),
        "correlations": correlation_rows,
        "group_summaries": group_rows,
        "controlled_matching": match_summary_rows,
        "regressions": regression_rows,
        "output_files": [
            str(niche_csv),
            str(run_csv),
            str(corr_csv),
            str(match_csv),
            str(match_count_csv),
            str(match_summary_csv),
            str(bin_csv),
            str(regression_csv),
            str(group_csv),
            *plot_files,
            str(summary_json),
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def analyze_dataset(
    dataset: DatasetSpec,
    high_threshold: float,
    min_touch_evaluation: int,
) -> dict[str, list[dict[str, Any]]]:
    seed_dirs = sorted(path for path in dataset.archive_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {dataset.archive_dir}")

    all_niche_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed_result = analyze_seed(
            dataset,
            seed_dir,
            high_threshold=high_threshold,
            min_touch_evaluation=min_touch_evaluation,
        )
        all_niche_rows.extend(seed_result["niche_rows"])
        run_rows.append(seed_result["run_row"])
    return {"niche_rows": all_niche_rows, "run_rows": run_rows}


def analyze_seed(
    dataset: DatasetSpec,
    seed_dir: Path,
    high_threshold: float,
    min_touch_evaluation: int,
) -> dict[str, Any]:
    summary = json.loads((seed_dir / "phase4_summary.json").read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    bins = (int(summary["grid_bins"][0]), int(summary["grid_bins"][1]))
    graph_k = int(summary["geodesic_graph"]["graph_k"])
    bootstrap_evaluations = int(summary["bootstrap_evaluations"])

    support_points, support_kinds = read_support_points(
        seed_dir / "geodesic_graph_support_points.csv"
    )
    centroid_support_indices = np.asarray(
        [idx for idx, kind in enumerate(support_kinds) if kind == "centroid"],
        dtype=np.int64,
    )
    if len(centroid_support_indices) != bins[0] * bins[1]:
        raise ValueError(
            f"{seed_dir}: expected {bins[0] * bins[1]} centroids, "
            f"found {len(centroid_support_indices)}"
        )
    centroids = support_points[centroid_support_indices]

    endpoint_mask = np.asarray([kind == "centroid" for kind in support_kinds], dtype=bool)
    graph = build_endpoint_manifold_graph(support_points, endpoint_mask, graph_k)
    centroid_to_support = endpoint_shortest_path_distances(graph, endpoint_mask)
    centroid_to_support = apply_disconnection_penalty(centroid_to_support, support_points)
    centroid_geodesic = centroid_to_support[:, centroid_support_indices]

    archive_cells = read_archive_cells(seed_dir / "archive_cells.csv")
    filled_indices = np.asarray(
        [cell_to_index(int(row["cell_x"]), int(row["cell_y"]), bins) for row in archive_cells],
        dtype=np.int64,
    )
    latents = read_latents(seed_dir / "visited_latents_final_space.csv")
    home_latent = np.mean(latents[: min(bootstrap_evaluations, len(latents))], axis=0)
    home_centroid_index = int(np.argmin(np.linalg.norm(centroids - home_latent[None, :], axis=1)))

    touch_info = read_touch_info(
        dataset.disagreement_dir / seed_dir.name / "candidate_assignment_disagreement.csv",
        bins=bins,
        high_threshold=high_threshold,
        min_evaluation=min_touch_evaluation,
    )

    niche_rows: list[dict[str, Any]] = []
    filled_centroid_distances = centroid_geodesic[np.ix_(filled_indices, filled_indices)]
    for archive_pos, raw_cell in enumerate(archive_cells):
        cell_x = int(raw_cell["cell_x"])
        cell_y = int(raw_cell["cell_y"])
        centroid_index = cell_to_index(cell_x, cell_y, bins)
        centroid_error = float(
            np.linalg.norm(
                centroids[centroid_index]
                - np.asarray(
                    [float(raw_cell["centroid_x"]), float(raw_cell["centroid_y"])], dtype=np.float64
                )
            )
        )
        if centroid_error > 1e-6:
            raise ValueError(
                f"{seed_dir}: centroid mismatch for cell {(cell_x, cell_y)}: {centroid_error}"
            )

        row_in_filled = filled_centroid_distances[archive_pos]
        if len(row_in_filled) > 1:
            bulk_depth = float(np.mean(np.delete(row_in_filled, archive_pos)))
            median_bulk_depth = float(np.median(np.delete(row_in_filled, archive_pos)))
        else:
            bulk_depth = float("nan")
            median_bulk_depth = float("nan")
        cell_info = touch_info.get(centroid_index, {})
        niche_rows.append(
            {
                "dataset": dataset.label,
                "seed": seed,
                "cell_x": cell_x,
                "cell_y": cell_y,
                "centroid_index": centroid_index,
                "fitness": float(raw_cell["fitness"]),
                "home_depth": float(centroid_geodesic[home_centroid_index, centroid_index]),
                "bulk_depth": bulk_depth,
                "median_bulk_depth": median_bulk_depth,
                "centroid_to_home_euclidean": float(
                    np.linalg.norm(centroids[centroid_index] - centroids[home_centroid_index])
                ),
                "high_blowup_target": bool(cell_info.get("high_blowup_target", False)),
                "high_blowup_touch_count": int(cell_info.get("high_blowup_touch_count", 0)),
                "any_disagreement_target": bool(cell_info.get("any_disagreement_target", False)),
                "any_disagreement_touch_count": int(
                    cell_info.get("any_disagreement_touch_count", 0)
                ),
                "geodesic_assignment_count": int(cell_info.get("geodesic_assignment_count", 0)),
                "log_geodesic_assignment_count": float(
                    math.log1p(int(cell_info.get("geodesic_assignment_count", 0)))
                ),
            }
        )

    run_row = {
        "dataset": dataset.label,
        "seed": seed,
        "filled_niches": len(archive_cells),
        "home_centroid_index": home_centroid_index,
        "home_centroid_cell_x": home_centroid_index // bins[1],
        "home_centroid_cell_y": home_centroid_index % bins[1],
        "high_blowup_target_filled_niches": int(
            sum(1 for row in niche_rows if row["high_blowup_target"])
        ),
        "high_blowup_target_filled_fraction": fraction(
            niche_rows, lambda row: bool(row["high_blowup_target"])
        ),
        "any_disagreement_target_filled_niches": int(
            sum(1 for row in niche_rows if row["any_disagreement_target"])
        ),
        "any_disagreement_target_filled_fraction": fraction(
            niche_rows, lambda row: bool(row["any_disagreement_target"])
        ),
        "graph_k": graph_k,
        "support_points": len(support_points),
        "filled_centroid_finite_fraction": float(np.mean(np.isfinite(filled_centroid_distances))),
        "mean_final_elite_fitness": finite_mean(values(niche_rows, "fitness")),
        "mean_home_depth": finite_mean(values(niche_rows, "home_depth")),
        "mean_bulk_depth": finite_mean(values(niche_rows, "bulk_depth")),
    }
    return {"niche_rows": niche_rows, "run_row": run_row}


def read_touch_info(
    path: Path,
    bins: tuple[int, int],
    high_threshold: float,
    min_evaluation: int,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    info: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if int(raw["evaluation"]) < min_evaluation:
                continue
            cell_x = int(raw["geodesic_cell_x"])
            cell_y = int(raw["geodesic_cell_y"])
            centroid_index = cell_to_index(cell_x, cell_y, bins)
            row = info.setdefault(
                centroid_index,
                {
                    "geodesic_assignment_count": 0,
                    "any_disagreement_touch_count": 0,
                    "high_blowup_touch_count": 0,
                    "any_disagreement_target": False,
                    "high_blowup_target": False,
                },
            )
            row["geodesic_assignment_count"] += 1
            if parse_bool(raw["assignment_disagrees"]):
                row["any_disagreement_touch_count"] += 1
                row["any_disagreement_target"] = True
                blowup = float(raw["geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"])
                if blowup >= high_threshold:
                    row["high_blowup_touch_count"] += 1
                    row["high_blowup_target"] = True
    return info


def correlation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = grouped_rows(rows, ["dataset", "seed"])
    for key, group_rows in groups.items():
        dataset, seed = key
        for depth_key in ["home_depth", "bulk_depth", "median_bulk_depth"]:
            output.append(correlation_row(dataset, int(seed), "per_seed", depth_key, group_rows))

    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        for depth_key in ["home_depth", "bulk_depth", "median_bulk_depth"]:
            output.append(correlation_row(dataset, -1, "pooled_raw", depth_key, dataset_rows))
            output.append(
                correlation_row(
                    dataset,
                    -1,
                    "pooled_within_seed_z",
                    depth_key,
                    within_seed_z_rows(dataset_rows, depth_key),
                )
            )

    for depth_key in ["home_depth", "bulk_depth", "median_bulk_depth"]:
        output.append(correlation_row("all_maps", -1, "pooled_raw", depth_key, rows))
        output.append(
            correlation_row(
                "all_maps",
                -1,
                "pooled_within_seed_z",
                depth_key,
                within_seed_z_rows(rows, depth_key),
            )
        )
    return output


def correlation_row(
    dataset: str,
    seed: int,
    scope: str,
    depth_key: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    x = values(rows, depth_key)
    y = values(rows, "fitness")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    else:
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        pearson_r = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)
        spearman_r = float(spearman.statistic)
        spearman_p = float(spearman.pvalue)
    return {
        "dataset": dataset,
        "seed": seed if seed >= 0 else "",
        "scope": scope,
        "depth_metric": depth_key,
        "n": int(len(x)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "mean_depth": finite_mean(x),
        "mean_fitness": finite_mean(y),
    }


def within_seed_z_rows(rows: list[dict[str, Any]], depth_key: str) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for _, group in grouped_rows(rows, ["dataset", "seed"]).items():
        depths = values(group, depth_key)
        fitness = values(group, "fitness")
        depth_z = zscore(depths)
        fitness_z = zscore(fitness)
        for row, dz, fz in zip(group, depth_z, fitness_z, strict=False):
            if not math.isfinite(float(dz)) or not math.isfinite(float(fz)):
                continue
            copied = dict(row)
            copied[depth_key] = float(dz)
            copied["fitness"] = float(fz)
            transformed.append(copied)
    return transformed


def nearest_depth_matches(
    rows: list[dict[str, Any]],
    depth_key: str,
    include_assignment_count: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for (dataset, seed), group in grouped_rows(rows, ["dataset", "seed"]).items():
        targets = [row for row in group if bool(row["high_blowup_target"])]
        controls = [row for row in group if not bool(row["high_blowup_target"])]
        if not targets or not controls:
            continue
        depth_scale = finite_std(values(group, depth_key))
        count_scale = finite_std(values(group, "log_geodesic_assignment_count"))
        for target in targets:
            best_control: dict[str, Any] | None = None
            best_distance = float("inf")
            for control in controls:
                depth_distance = (
                    abs(float(target[depth_key]) - float(control[depth_key])) / depth_scale
                )
                if include_assignment_count:
                    count_distance = (
                        abs(
                            float(target["log_geodesic_assignment_count"])
                            - float(control["log_geodesic_assignment_count"])
                        )
                        / count_scale
                    )
                    distance = math.sqrt(
                        depth_distance * depth_distance + count_distance * count_distance
                    )
                else:
                    distance = depth_distance
                if distance < best_distance:
                    best_distance = distance
                    best_control = control
            if best_control is None:
                continue
            pairs.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "match_type": "depth_plus_assignment_count"
                    if include_assignment_count
                    else "depth_only",
                    "target_cell_x": int(target["cell_x"]),
                    "target_cell_y": int(target["cell_y"]),
                    "control_cell_x": int(best_control["cell_x"]),
                    "control_cell_y": int(best_control["cell_y"]),
                    "target_fitness": float(target["fitness"]),
                    "control_fitness": float(best_control["fitness"]),
                    "fitness_difference_target_minus_control": float(target["fitness"])
                    - float(best_control["fitness"]),
                    "target_depth": float(target[depth_key]),
                    "control_depth": float(best_control[depth_key]),
                    "absolute_depth_difference": abs(
                        float(target[depth_key]) - float(best_control[depth_key])
                    ),
                    "target_assignment_count": int(target["geodesic_assignment_count"]),
                    "control_assignment_count": int(best_control["geodesic_assignment_count"]),
                    "match_distance_standardized": float(best_distance),
                }
            )
    return pairs


def matching_summary(pairs: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_label, group_rows in [
        *[
            (dataset, [row for row in pairs if row["dataset"] == dataset])
            for dataset in sorted({row["dataset"] for row in pairs})
        ],
        ("all_maps", pairs),
    ]:
        if not group_rows:
            continue
        diffs = values(group_rows, "fitness_difference_target_minus_control")
        try:
            wilcoxon_result = wilcoxon(diffs, alternative="two-sided")
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        rows.append(
            {
                "comparison": label,
                "dataset": group_label,
                "matched_pair_count": len(group_rows),
                "mean_target_minus_control_fitness": finite_mean(diffs),
                "median_target_minus_control_fitness": finite_median(diffs),
                "fraction_targets_lower_than_control": float(np.mean(diffs < 0.0)),
                "mean_absolute_depth_difference": finite_mean(
                    values(group_rows, "absolute_depth_difference")
                ),
                "median_absolute_depth_difference": finite_median(
                    values(group_rows, "absolute_depth_difference")
                ),
                "wilcoxon_p": wilcoxon_p,
            }
        )
    return rows


def depth_bin_comparison(
    rows: list[dict[str, Any]], depth_key: str, bin_count: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (dataset, seed), group in grouped_rows(rows, ["dataset", "seed"]).items():
        depths = values(group, depth_key)
        finite = np.isfinite(depths)
        if np.count_nonzero(finite) < bin_count:
            continue
        edges = np.quantile(depths[finite], np.linspace(0.0, 1.0, bin_count + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            continue
        for bin_index in range(len(edges) - 1):
            low = float(edges[bin_index])
            high = float(edges[bin_index + 1])
            if bin_index == len(edges) - 2:
                members = [row for row in group if low <= float(row[depth_key]) <= high]
            else:
                members = [row for row in group if low <= float(row[depth_key]) < high]
            touched = [row for row in members if bool(row["high_blowup_target"])]
            controls = [row for row in members if not bool(row["high_blowup_target"])]
            if not touched or not controls:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "depth_metric": depth_key,
                    "bin_index": bin_index,
                    "depth_low": low,
                    "depth_high": high,
                    "touched_count": len(touched),
                    "control_count": len(controls),
                    "touched_mean_fitness": finite_mean(values(touched, "fitness")),
                    "control_mean_fitness": finite_mean(values(controls, "fitness")),
                    "difference_touched_minus_control": finite_mean(values(touched, "fitness"))
                    - finite_mean(values(controls, "fitness")),
                    "touched_mean_assignment_count": finite_mean(
                        values(touched, "geodesic_assignment_count")
                    ),
                    "control_mean_assignment_count": finite_mean(
                        values(controls, "geodesic_assignment_count")
                    ),
                }
            )
    return output


def regression_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for dataset_label, dataset_rows in [
        *[
            (dataset, [row for row in rows if row["dataset"] == dataset])
            for dataset in sorted({row["dataset"] for row in rows})
        ],
        ("all_maps", rows),
    ]:
        for depth_key in ["home_depth", "bulk_depth"]:
            transformed = within_seed_z_rows(dataset_rows, depth_key)
            result = fit_linear_model(
                transformed,
                response_key="fitness",
                predictor_keys=[depth_key, "high_blowup_target", "log_geodesic_assignment_count"],
            )
            for coefficient in result:
                coefficient.update({"dataset": dataset_label, "depth_metric": depth_key})
                outputs.append(coefficient)
    return outputs


def fit_linear_model(
    rows: list[dict[str, Any]],
    response_key: str,
    predictor_keys: list[str],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    y = values(rows, response_key)
    columns = [np.ones(len(rows), dtype=np.float64)]
    names = ["intercept"]
    for key in predictor_keys:
        if key == "high_blowup_target":
            column = np.asarray([1.0 if bool(row[key]) else 0.0 for row in rows], dtype=np.float64)
        else:
            column = zscore(values(rows, key))
        columns.append(column)
        names.append(key)
    x = np.column_stack(columns)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = y[mask]
    x = x[mask]
    n, p = x.shape
    if n <= p:
        return []
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    dof = max(1, n - p)
    sigma2 = float(np.sum(residuals * residuals) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    t_values = beta / np.maximum(se, 1e-12)
    p_values = 2.0 * student_t.sf(np.abs(t_values), df=dof)
    return [
        {
            "term": name,
            "n": int(n),
            "coefficient": float(coef),
            "standard_error": float(err),
            "t_value": float(tval),
            "p_value": float(pval),
        }
        for name, coef, err, tval, pval in zip(names, beta, se, t_values, p_values, strict=True)
    ]


def high_target_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label, group in [
        *[
            (dataset, [row for row in rows if row["dataset"] == dataset])
            for dataset in sorted({row["dataset"] for row in rows})
        ],
        ("all_maps", rows),
    ]:
        touched = [row for row in group if bool(row["high_blowup_target"])]
        controls = [row for row in group if not bool(row["high_blowup_target"])]
        fitness_p = mann_whitney_p(values(touched, "fitness"), values(controls, "fitness"))
        depth_p = mann_whitney_p(values(touched, "bulk_depth"), values(controls, "bulk_depth"))
        output.append(
            {
                "dataset": label,
                "n_total": len(group),
                "n_high_blowup_target": len(touched),
                "n_control": len(controls),
                "high_blowup_target_fraction": safe_fraction(len(touched), len(group)),
                "target_mean_fitness": finite_mean(values(touched, "fitness")),
                "control_mean_fitness": finite_mean(values(controls, "fitness")),
                "target_median_fitness": finite_median(values(touched, "fitness")),
                "control_median_fitness": finite_median(values(controls, "fitness")),
                "uncontrolled_fitness_difference_mean": finite_mean(values(touched, "fitness"))
                - finite_mean(values(controls, "fitness")),
                "uncontrolled_fitness_mannwhitney_p": fitness_p,
                "target_mean_bulk_depth": finite_mean(values(touched, "bulk_depth")),
                "control_mean_bulk_depth": finite_mean(values(controls, "bulk_depth")),
                "target_minus_control_mean_bulk_depth": finite_mean(values(touched, "bulk_depth"))
                - finite_mean(values(controls, "bulk_depth")),
                "bulk_depth_mannwhitney_p": depth_p,
                "target_mean_assignment_count": finite_mean(
                    values(touched, "geodesic_assignment_count")
                ),
                "control_mean_assignment_count": finite_mean(
                    values(controls, "geodesic_assignment_count")
                ),
            }
        )
    return output


def save_depth_scatter(rows: list[dict[str, Any]], output_stem: Path) -> list[str]:
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.2 * len(datasets), 4.2), constrained_layout=True
    )
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets, strict=True):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        controls = [row for row in dataset_rows if not bool(row["high_blowup_target"])]
        targets = [row for row in dataset_rows if bool(row["high_blowup_target"])]
        ax.scatter(
            values(controls, "bulk_depth"),
            values(controls, "fitness"),
            s=14,
            alpha=0.45,
            c="#64748b",
            label="not high-blowup target",
            linewidths=0,
        )
        ax.scatter(
            values(targets, "bulk_depth"),
            values(targets, "fitness"),
            s=20,
            alpha=0.70,
            c="#dc2626",
            label="high-blowup target",
            linewidths=0,
        )
        ax.set_title(dataset.replace("_", " "))
        ax.set_xlabel("bulk geodesic depth")
        ax.set_ylabel("final elite fitness")
        ax.legend(frameon=False, fontsize=8)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def save_match_plot(pairs: list[dict[str, Any]], output_stem: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    datasets = sorted({row["dataset"] for row in pairs})
    data = [
        values(
            [row for row in pairs if row["dataset"] == dataset],
            "fitness_difference_target_minus_control",
        )
        for dataset in datasets
    ]
    if data:
        ax.boxplot(
            data, tick_labels=[dataset.replace("_", "\n") for dataset in datasets], showfliers=False
        )
        for idx, values_array in enumerate(data, start=1):
            x = np.full(len(values_array), idx, dtype=np.float64)
            ax.scatter(x, values_array, s=8, alpha=0.18, c="#334155", linewidths=0)
    ax.axhline(0.0, color="#0f172a", linewidth=1.0, linestyle="--")
    ax.set_ylabel("target fitness - nearest-depth control fitness")
    ax.set_title("High-blowup target niches vs depth-matched controls")
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def read_archive_cells(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def read_support_points(path: Path) -> tuple[np.ndarray, list[str]]:
    rows = read_csv(path)
    points = np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in rows], dtype=np.float64
    )
    kinds = [row["kind"] for row in rows]
    return points, kinds


def read_latents(path: Path) -> np.ndarray:
    rows = read_csv(path)
    return np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in rows], dtype=np.float64
    )


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


def apply_disconnection_penalty(distances: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    finite = np.isfinite(distances)
    if np.all(finite):
        return distances
    positive_finite = distances[finite & (distances > 0.0)]
    finite_scale = float(np.max(positive_finite)) if len(positive_finite) else 0.0
    if len(support_points) >= 2:
        span = support_points[:, None, :] - support_points[None, :, :]
        support_scale = float(np.max(np.linalg.norm(span, axis=2)))
    else:
        support_scale = 0.5
    penalty = max(finite_scale, support_scale, 0.5) * 2.0
    repaired = distances.copy()
    repaired[~finite] = penalty
    return repaired


def cell_to_index(cell_x: int, cell_y: int, bins: tuple[int, int]) -> int:
    return int(cell_x) * int(bins[1]) + int(cell_y)


def grouped_rows(
    rows: list[dict[str, Any]], keys: list[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def zscore(values_array: np.ndarray) -> np.ndarray:
    output = np.asarray(values_array, dtype=np.float64).copy()
    finite = np.isfinite(output)
    if not np.any(finite):
        return np.full_like(output, np.nan, dtype=np.float64)
    mean = float(np.mean(output[finite]))
    std = float(np.std(output[finite]))
    if std <= 1e-12:
        result = np.zeros_like(output, dtype=np.float64)
        result[~finite] = np.nan
        return result
    result = (output - mean) / std
    result[~finite] = np.nan
    return result


def finite_mean(values_array: np.ndarray) -> float:
    finite = values_array[np.isfinite(values_array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def finite_median(values_array: np.ndarray) -> float:
    finite = values_array[np.isfinite(values_array)]
    return float(np.median(finite)) if len(finite) else float("nan")


def finite_std(values_array: np.ndarray) -> float:
    finite = values_array[np.isfinite(values_array)]
    std = float(np.std(finite)) if len(finite) else 0.0
    return std if std > 1e-12 else 1.0


def fraction(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return float("nan")
    return float(sum(1 for row in rows if predicate(row)) / len(rows))


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def mann_whitney_p(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    return float(mannwhitneyu(left, right, alternative="two-sided", method="asymptotic").pvalue)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def print_summary(summary: dict[str, Any]) -> None:
    print("PHASE 6 STRUCTURAL FITNESS-CEILING ANALYSIS")
    print(f"niches={summary['niche_count']}, runs={summary['run_count']}")
    print("depth-fitness correlations (pooled within-seed z):")
    for row in summary["correlations"]:
        if row["scope"] == "pooled_within_seed_z" and row["depth_metric"] in {
            "home_depth",
            "bulk_depth",
        }:
            print(
                "  {dataset} {depth_metric}: n={n}, Pearson r={pearson_r:.3f} "
                "(p={pearson_p:.3g}), Spearman rho={spearman_rho:.3f} "
                "(p={spearman_p:.3g})".format(**row)
            )
    print("high-blowup target group summaries:")
    for row in summary["group_summaries"]:
        print(
            "  {dataset}: targets={n_high_blowup_target}/{n_total} "
            "({high_blowup_target_fraction:.1%}), "
            "raw mean diff={uncontrolled_fitness_difference_mean:.3f}, "
            "target-control bulk depth diff={target_minus_control_mean_bulk_depth:.3f}".format(
                **row
            )
        )
    print("controlled matching:")
    for row in summary["controlled_matching"]:
        if row["comparison"] == "nearest_bulk_depth":
            print(
                "  {dataset}: pairs={matched_pair_count}, mean target-control fitness="
                "{mean_target_minus_control_fitness:.3f}, "
                "median={median_target_minus_control_fitness:.3f}, "
                "targets_lower={fraction_targets_lower_than_control:.1%}, "
                "p={wilcoxon_p:.3g}".format(**row)
            )
    print("regression high_blowup_target coefficients:")
    for row in summary["regressions"]:
        if row["term"] == "high_blowup_target" and row["depth_metric"] == "bulk_depth":
            print(
                "  {dataset}: coef={coefficient:.3f} SD fitness, p={p_value:.3g} "
                "(controls: bulk depth + assignment count)".format(**row)
            )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
