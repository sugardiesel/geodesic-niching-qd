"""Analyze Euclidean-vs-geodesic centroid assignment disagreement in Phase 5 runs."""

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
from scipy.stats import pearsonr, spearmanr

from algorithms.geodesic_archive import GeodesicGridArchive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-dir", default="results/phase5/contribution_geodesic_niching")
    parser.add_argument("--output-dir", default="results/phase5/geodesic_disagreement_analysis_v2")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--example-count", type=int, default=12)
    args = parser.parse_args()

    phase5_dir = Path(args.phase5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(path for path in phase5_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {phase5_dir}")

    seed_summaries: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    all_disagreement_rows: list[dict[str, Any]] = []
    generated_files: list[str] = []

    for seed_dir in seed_dirs:
        seed_output_dir = output_dir / seed_dir.name
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        result = analyze_seed(
            seed_dir=seed_dir,
            output_dir=seed_output_dir,
            window_size=int(args.window_size),
            example_count=int(args.example_count),
        )
        seed_summaries.append(result["summary"])
        window_rows.extend(result["window_rows"])
        example_rows.extend(result["examples"])
        all_disagreement_rows.extend(result["disagreement_rows"])
        generated_files.extend(result["files"])

    example_rows = globally_rank_examples(example_rows)
    aggregate = aggregate_summaries(seed_summaries)
    aggregate.update(pooled_disagreement_outcome_stats(all_disagreement_rows))
    fitness_bin_rows = blowup_fitness_bin_rows(all_disagreement_rows)
    summary_csv = output_dir / "disagreement_seed_summary.csv"
    window_csv = output_dir / "disagreement_over_time.csv"
    examples_csv = output_dir / "disagreement_examples.csv"
    fitness_bins_csv = output_dir / "disagreement_fitness_by_blowup_bin.csv"
    aggregate_json = output_dir / "disagreement_summary.json"
    write_csv(summary_csv, seed_summaries)
    write_csv(window_csv, window_rows)
    write_csv(examples_csv, example_rows)
    write_csv(fitness_bins_csv, fitness_bin_rows)
    aggregate_plot_files = save_aggregate_curve(
        window_rows, output_dir / "disagreement_rate_over_time_all_seeds"
    )
    aggregate["output_files"] = [
        str(summary_csv),
        str(window_csv),
        str(examples_csv),
        str(fitness_bins_csv),
        *aggregate_plot_files,
        *generated_files,
        str(aggregate_json),
    ]
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print_summary(aggregate, seed_summaries, example_rows)


def analyze_seed(
    seed_dir: Path, output_dir: Path, window_size: int, example_count: int
) -> dict[str, Any]:
    summary = json.loads((seed_dir / "phase4_summary.json").read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    total_evaluations = int(summary["total_evaluations"])
    retrain_evaluation = int(summary["retrain_evaluations"][0])
    graph_k = int(summary["geodesic_graph"]["graph_k"])
    assignment_k = int(summary["geodesic_graph"]["assignment_k"])
    max_graph_points = int(summary["geodesic_graph"]["max_graph_points"])
    refresh_interval = int(summary["geodesic_graph"]["refresh_interval"])
    bins = (int(summary["grid_bins"][0]), int(summary["grid_bins"][1]))

    latents = read_latents(seed_dir / "visited_latents_final_space.csv")
    eval_rows = read_csv(seed_dir / "evaluations.csv")
    if len(latents) != total_evaluations:
        raise ValueError(f"{seed_dir}: expected {total_evaluations} latents, found {len(latents)}")
    centroids = read_centroids(seed_dir / "geodesic_graph_support_points.csv")
    archive_cell_map = read_archive_cell_map(seed_dir / "archive_cells.csv")
    bd_min, bd_max = bounds_from_centroids(centroids, bins)

    evaluations = np.arange(1, total_evaluations + 1, dtype=np.int64)
    state_evaluations = np.asarray(
        [
            graph_state_evaluation(
                evaluation=int(evaluation),
                retrain_evaluation=retrain_evaluation,
                refresh_interval=refresh_interval,
            )
            for evaluation in evaluations
        ],
        dtype=np.int64,
    )

    euclidean_indices = np.full(total_evaluations, -1, dtype=np.int64)
    geodesic_indices = np.full(total_evaluations, -1, dtype=np.int64)
    euclidean_selected_distances = np.full(total_evaluations, np.nan, dtype=np.float64)
    euclidean_distance_to_geodesic = np.full(total_evaluations, np.nan, dtype=np.float64)
    geodesic_selected_distances = np.full(total_evaluations, np.nan, dtype=np.float64)
    geodesic_distance_to_euclidean = np.full(total_evaluations, np.nan, dtype=np.float64)
    state_component_counts: dict[int, int] = {}
    state_penalized_counts: dict[int, int] = {}

    for state_evaluation in sorted(set(int(value) for value in state_evaluations)):
        state_mask = state_evaluations == state_evaluation
        indices = np.flatnonzero(state_mask)
        rolling_start = max(0, state_evaluation - max_graph_points)
        rolling = latents[rolling_start:state_evaluation]
        archive = make_archive(
            bins=bins,
            bd_min=bd_min,
            bd_max=bd_max,
            graph_k=graph_k,
            assignment_k=assignment_k,
            max_graph_points=max_graph_points,
            rolling_latents=rolling,
        )
        centroid_error = float(np.max(np.linalg.norm(archive.centroids - centroids, axis=1)))
        if centroid_error > 1e-8:
            raise ValueError(f"{seed_dir}: reconstructed centroids differ by {centroid_error}")
        graph_stats = archive.graph_stats()
        state_component_counts[state_evaluation] = int(graph_stats.connected_components)
        state_penalized_counts[state_evaluation] = int(graph_stats.penalized_centroid_distances)

        block = assign_block(archive, latents[indices])
        euclidean_indices[indices] = block["euclidean_indices"]
        geodesic_indices[indices] = block["geodesic_indices"]
        euclidean_selected_distances[indices] = block["euclidean_selected_distances"]
        euclidean_distance_to_geodesic[indices] = block["euclidean_distance_to_geodesic"]
        geodesic_selected_distances[indices] = block["geodesic_selected_distances"]
        geodesic_distance_to_euclidean[indices] = block["geodesic_distance_to_euclidean"]

    disagreement = euclidean_indices != geodesic_indices
    geodesic_gap = geodesic_distance_to_euclidean - geodesic_selected_distances
    blowup_ratio = geodesic_distance_to_euclidean / np.maximum(geodesic_selected_distances, 1e-12)
    displacement_cells = centroid_cell_distance(euclidean_indices, geodesic_indices, bins)
    strong_blowup = disagreement & (blowup_ratio >= 2.0)
    very_strong_blowup = disagreement & (blowup_ratio >= 5.0)
    post_retrain_disagreement = disagreement & (evaluations > retrain_evaluation)

    detail_rows = candidate_detail_rows(
        seed=seed,
        evaluations=evaluations,
        latents=latents,
        eval_rows=eval_rows,
        state_evaluations=state_evaluations,
        euclidean_indices=euclidean_indices,
        geodesic_indices=geodesic_indices,
        euclidean_selected_distances=euclidean_selected_distances,
        euclidean_distance_to_geodesic=euclidean_distance_to_geodesic,
        geodesic_selected_distances=geodesic_selected_distances,
        geodesic_distance_to_euclidean=geodesic_distance_to_euclidean,
        geodesic_gap=geodesic_gap,
        blowup_ratio=blowup_ratio,
        displacement_cells=displacement_cells,
        bins=bins,
        disagreement=disagreement,
        archive_cell_map=archive_cell_map,
    )
    disagreement_rows = [row for row in detail_rows if bool(row["assignment_disagrees"])]
    details_csv = output_dir / "candidate_assignment_disagreement.csv"
    write_csv(details_csv, detail_rows)

    window_rows = summarize_windows(
        seed=seed,
        evaluations=evaluations,
        latents=latents,
        disagreement=disagreement,
        blowup_ratio=blowup_ratio,
        geodesic_gap=geodesic_gap,
        window_size=window_size,
    )
    examples = top_disagreement_examples(
        detail_rows=detail_rows,
        example_count=example_count,
    )
    region = summarize_regions(latents, disagreement, bins=10)
    summary_row = {
        "seed": seed,
        "candidate_count": int(total_evaluations),
        "disagreement_count": int(np.count_nonzero(disagreement)),
        "disagreement_rate": float(np.mean(disagreement)),
        "post_retrain_candidate_count": int(np.count_nonzero(evaluations > retrain_evaluation)),
        "post_retrain_disagreement_count": int(np.count_nonzero(post_retrain_disagreement)),
        "post_retrain_disagreement_rate": safe_fraction(
            int(np.count_nonzero(post_retrain_disagreement)),
            int(np.count_nonzero(evaluations > retrain_evaluation)),
        ),
        "strong_blowup_count_ratio_ge_2": int(np.count_nonzero(strong_blowup)),
        "strong_blowup_fraction_of_disagreements_ratio_ge_2": safe_fraction(
            int(np.count_nonzero(strong_blowup)),
            int(np.count_nonzero(disagreement)),
        ),
        "very_strong_blowup_count_ratio_ge_5": int(np.count_nonzero(very_strong_blowup)),
        "very_strong_blowup_fraction_of_disagreements_ratio_ge_5": safe_fraction(
            int(np.count_nonzero(very_strong_blowup)),
            int(np.count_nonzero(disagreement)),
        ),
        "mean_geodesic_blowup_ratio_disagreements": finite_mean(blowup_ratio[disagreement]),
        "median_geodesic_blowup_ratio_disagreements": finite_median(blowup_ratio[disagreement]),
        "max_geodesic_blowup_ratio_disagreements": finite_max(blowup_ratio[disagreement]),
        "mean_geodesic_gap_disagreements": finite_mean(geodesic_gap[disagreement]),
        "median_geodesic_gap_disagreements": finite_median(geodesic_gap[disagreement]),
        "mean_cell_displacement_disagreements": finite_mean(displacement_cells[disagreement]),
        "median_cell_displacement_disagreements": finite_median(displacement_cells[disagreement]),
        "max_cell_displacement_disagreements": finite_max(displacement_cells[disagreement]),
        **disagreement_outcome_stats(disagreement_rows, retrain_evaluation=retrain_evaluation),
        **fitness_blowup_correlation_stats(disagreement_rows, prefix=""),
        **fitness_blowup_correlation_stats(
            [row for row in disagreement_rows if int(row["evaluation"]) > retrain_evaluation],
            prefix="post_retrain_",
        ),
        "region_bins": 10,
        **region,
        "graph_state_count": len(state_component_counts),
        "max_graph_components_seen": int(max(state_component_counts.values())),
        "total_penalized_centroid_distances_across_states": int(
            sum(state_penalized_counts.values())
        ),
        "analysis_space": "saved final one-retrain latent space",
        "graph_reconstruction_note": (
            "Candidates up to the eval-10000 retrain are assigned using the rebuilt final-space "
            "graph state at eval 10000; later candidates use the same 250-eval rolling-buffer "
            "refresh schedule as Phase 4."
        ),
    }

    seed_plot_files = [
        *save_seed_curve(window_rows, output_dir / "disagreement_rate_over_time"),
        *save_seed_scatter(
            latents=latents,
            disagreement=disagreement,
            blowup_ratio=blowup_ratio,
            output_stem=output_dir / "latent_disagreement_scatter",
            title=f"Seed {seed}: geodesic vs Euclidean assignment disagreements",
        ),
        *save_region_heatmap(
            latents=latents,
            disagreement=disagreement,
            output_stem=output_dir / "latent_disagreement_region_heatmap",
            title=f"Seed {seed}: disagreement rate by latent region",
        ),
        str(details_csv),
    ]
    return {
        "summary": summary_row,
        "window_rows": window_rows,
        "examples": examples,
        "disagreement_rows": disagreement_rows,
        "files": seed_plot_files,
    }


def graph_state_evaluation(evaluation: int, retrain_evaluation: int, refresh_interval: int) -> int:
    if evaluation <= retrain_evaluation:
        return retrain_evaluation
    since_retrain = evaluation - retrain_evaluation
    if since_retrain % refresh_interval == 0:
        return evaluation
    return retrain_evaluation + (since_retrain // refresh_interval) * refresh_interval


def make_archive(
    bins: tuple[int, int],
    bd_min: np.ndarray,
    bd_max: np.ndarray,
    graph_k: int,
    assignment_k: int,
    max_graph_points: int,
    rolling_latents: np.ndarray,
) -> GeodesicGridArchive:
    archive = GeodesicGridArchive(
        bins=bins,
        bd_min=bd_min,
        bd_max=bd_max,
        genome_size=1,
        graph_k=graph_k,
        assignment_k=assignment_k,
        pairwise_geodesic_k=graph_k,
        max_graph_points=max_graph_points,
        refresh_interval=0,
    )
    archive.add_visited_latents(rolling_latents)
    return archive


def assign_block(
    archive: GeodesicGridArchive, points: np.ndarray, chunk_size: int = 512
) -> dict[str, np.ndarray]:
    n = len(points)
    euclidean_indices = np.full(n, -1, dtype=np.int64)
    geodesic_indices = np.full(n, -1, dtype=np.int64)
    euclidean_selected_distances = np.full(n, np.nan, dtype=np.float64)
    euclidean_distance_to_geodesic = np.full(n, np.nan, dtype=np.float64)
    geodesic_selected_distances = np.full(n, np.nan, dtype=np.float64)
    geodesic_distance_to_euclidean = np.full(n, np.nan, dtype=np.float64)
    if archive._tree is None:
        raise ValueError("Disagreement analysis requires at least one visited routing point.")
    query_k = min(max(1, archive.assignment_k), len(archive._routing_support_indices))

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = points[start:stop]
        euclidean = np.linalg.norm(archive.centroids[None, :, :] - block[:, None, :], axis=2)
        euc_idx = np.argmin(euclidean, axis=1).astype(np.int64)
        distances, indices = archive._tree.query(block, k=query_k)
        distances = np.atleast_2d(distances).astype(np.float64)
        tree_indices = np.atleast_2d(indices).astype(np.int64)
        if len(block) == 1:
            distances = distances.reshape(1, -1)
            tree_indices = tree_indices.reshape(1, -1)
        indices = archive._routing_support_indices[tree_indices]
        geodesic_to_centroids = np.min(
            archive._centroid_distances[:, indices] + distances[None, :, :],
            axis=2,
        ).T
        geo_idx = np.nanargmin(geodesic_to_centroids, axis=1).astype(np.int64)
        row_indices = np.arange(len(block), dtype=np.int64)
        target = slice(start, stop)
        euclidean_indices[target] = euc_idx
        geodesic_indices[target] = geo_idx
        euclidean_selected_distances[target] = euclidean[row_indices, euc_idx]
        euclidean_distance_to_geodesic[target] = euclidean[row_indices, geo_idx]
        geodesic_selected_distances[target] = geodesic_to_centroids[row_indices, geo_idx]
        geodesic_distance_to_euclidean[target] = geodesic_to_centroids[row_indices, euc_idx]

    return {
        "euclidean_indices": euclidean_indices,
        "geodesic_indices": geodesic_indices,
        "euclidean_selected_distances": euclidean_selected_distances,
        "euclidean_distance_to_geodesic": euclidean_distance_to_geodesic,
        "geodesic_selected_distances": geodesic_selected_distances,
        "geodesic_distance_to_euclidean": geodesic_distance_to_euclidean,
    }


def candidate_detail_rows(
    seed: int,
    evaluations: np.ndarray,
    latents: np.ndarray,
    eval_rows: list[dict[str, str]],
    state_evaluations: np.ndarray,
    euclidean_indices: np.ndarray,
    geodesic_indices: np.ndarray,
    euclidean_selected_distances: np.ndarray,
    euclidean_distance_to_geodesic: np.ndarray,
    geodesic_selected_distances: np.ndarray,
    geodesic_distance_to_euclidean: np.ndarray,
    geodesic_gap: np.ndarray,
    blowup_ratio: np.ndarray,
    displacement_cells: np.ndarray,
    bins: tuple[int, int],
    disagreement: np.ndarray,
    archive_cell_map: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, evaluation in enumerate(evaluations):
        euc_cell = index_to_cell(int(euclidean_indices[idx]), bins)
        geo_cell = index_to_cell(int(geodesic_indices[idx]), bins)
        source_row = eval_rows[idx] if idx < len(eval_rows) else {}
        euc_archive_cell = archive_cell_map.get(euc_cell)
        geo_archive_cell = archive_cell_map.get(geo_cell)
        fitness = float(source_row.get("fitness", "nan"))
        rows.append(
            {
                "seed": seed,
                "evaluation": int(evaluation),
                "graph_state_evaluation": int(state_evaluations[idx]),
                "latent_0": float(latents[idx, 0]),
                "latent_1": float(latents[idx, 1]),
                "fitness": fitness,
                "source": source_row.get("source", ""),
                "actual_run_inserted": parse_bool(source_row.get("inserted", "")),
                "actual_run_replaced": parse_bool(source_row.get("replaced", "")),
                "actual_run_cell_x": int(float(source_row.get("cell_x", -1))),
                "actual_run_cell_y": int(float(source_row.get("cell_y", -1))),
                "euclidean_centroid_index": int(euclidean_indices[idx]),
                "euclidean_cell_x": int(euc_cell[0]),
                "euclidean_cell_y": int(euc_cell[1]),
                "geodesic_centroid_index": int(geodesic_indices[idx]),
                "geodesic_cell_x": int(geo_cell[0]),
                "geodesic_cell_y": int(geo_cell[1]),
                "assignment_disagrees": bool(disagreement[idx]),
                "euclidean_distance_to_euclidean_choice": float(euclidean_selected_distances[idx]),
                "euclidean_distance_to_geodesic_choice": float(euclidean_distance_to_geodesic[idx]),
                "geodesic_distance_to_geodesic_choice": float(geodesic_selected_distances[idx]),
                "geodesic_distance_to_euclidean_choice": float(geodesic_distance_to_euclidean[idx]),
                "geodesic_gap_euclidean_choice_minus_geodesic_choice": float(geodesic_gap[idx]),
                "geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice": float(
                    blowup_ratio[idx]
                ),
                "centroid_cell_displacement": float(displacement_cells[idx]),
                "synthetic_like_large_blowup_ratio_ge_2": bool(
                    disagreement[idx] and blowup_ratio[idx] >= 2.0
                ),
                "synthetic_like_very_large_blowup_ratio_ge_5": bool(
                    disagreement[idx] and blowup_ratio[idx] >= 5.0
                ),
                "final_euclidean_cell_occupied": euc_archive_cell is not None,
                "final_euclidean_cell_fitness": float(euc_archive_cell["fitness"])
                if euc_archive_cell
                else float("nan"),
                "candidate_minus_final_euclidean_cell_fitness": (
                    fitness - float(euc_archive_cell["fitness"])
                    if euc_archive_cell
                    else float("nan")
                ),
                "final_geodesic_cell_occupied": geo_archive_cell is not None,
                "final_geodesic_cell_fitness": float(geo_archive_cell["fitness"])
                if geo_archive_cell
                else float("nan"),
                "candidate_minus_final_geodesic_cell_fitness": (
                    fitness - float(geo_archive_cell["fitness"])
                    if geo_archive_cell
                    else float("nan")
                ),
            }
        )
    return rows


def summarize_windows(
    seed: int,
    evaluations: np.ndarray,
    latents: np.ndarray,
    disagreement: np.ndarray,
    blowup_ratio: np.ndarray,
    geodesic_gap: np.ndarray,
    window_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start_eval in range(1, int(evaluations[-1]) + 1, window_size):
        end_eval = min(start_eval + window_size - 1, int(evaluations[-1]))
        mask = (evaluations >= start_eval) & (evaluations <= end_eval)
        disagree_mask = mask & disagreement
        rows.append(
            {
                "seed": seed,
                "window_start": start_eval,
                "window_end": end_eval,
                "window_midpoint": (start_eval + end_eval) / 2.0,
                "candidate_count": int(np.count_nonzero(mask)),
                "disagreement_count": int(np.count_nonzero(disagree_mask)),
                "disagreement_rate": float(np.mean(disagreement[mask]))
                if np.any(mask)
                else float("nan"),
                "mean_latent_0": finite_mean(latents[mask, 0]),
                "mean_latent_1": finite_mean(latents[mask, 1]),
                "mean_blowup_ratio_disagreements": finite_mean(blowup_ratio[disagree_mask]),
                "mean_geodesic_gap_disagreements": finite_mean(geodesic_gap[disagree_mask]),
            }
        )
    return rows


def summarize_regions(
    latents: np.ndarray, disagreement: np.ndarray, bins: int
) -> dict[str, float | int]:
    x_edges = np.linspace(float(np.min(latents[:, 0])), float(np.max(latents[:, 0])), bins + 1)
    y_edges = np.linspace(float(np.min(latents[:, 1])), float(np.max(latents[:, 1])), bins + 1)
    all_counts, _, _ = np.histogram2d(latents[:, 0], latents[:, 1], bins=[x_edges, y_edges])
    disagreement_counts, _, _ = np.histogram2d(
        latents[disagreement, 0],
        latents[disagreement, 1],
        bins=[x_edges, y_edges],
    )
    total_disagreements = int(np.sum(disagreement_counts))
    if total_disagreements == 0:
        return {
            "occupied_region_bins_all_candidates": int(np.count_nonzero(all_counts)),
            "occupied_region_bins_disagreements": 0,
            "top_5_region_fraction_of_disagreements": 0.0,
            "top_10_region_fraction_of_disagreements": 0.0,
            "top_10_region_fraction_of_all_candidates": 0.0,
            "disagreement_region_entropy_normalized": 0.0,
            "max_region_disagreement_rate": 0.0,
        }
    flat_disagreement = np.sort(disagreement_counts.ravel())[::-1]
    flat_all = np.sort(all_counts.ravel())[::-1]
    probabilities = flat_disagreement[flat_disagreement > 0] / total_disagreements
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / np.log(bins * bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        region_rates = np.divide(
            disagreement_counts,
            all_counts,
            out=np.zeros_like(disagreement_counts),
            where=all_counts > 0,
        )
    return {
        "occupied_region_bins_all_candidates": int(np.count_nonzero(all_counts)),
        "occupied_region_bins_disagreements": int(np.count_nonzero(disagreement_counts)),
        "top_5_region_fraction_of_disagreements": float(
            np.sum(flat_disagreement[:5]) / total_disagreements
        ),
        "top_10_region_fraction_of_disagreements": float(
            np.sum(flat_disagreement[:10]) / total_disagreements
        ),
        "top_10_region_fraction_of_all_candidates": float(np.sum(flat_all[:10]) / len(latents)),
        "disagreement_region_entropy_normalized": normalized_entropy,
        "max_region_disagreement_rate": float(np.max(region_rates)),
    }


def top_disagreement_examples(
    detail_rows: list[dict[str, Any]], example_count: int
) -> list[dict[str, Any]]:
    disagreements = [row for row in detail_rows if bool(row["assignment_disagrees"])]
    disagreements.sort(
        key=lambda row: (
            float(row["geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"]),
            float(row["geodesic_gap_euclidean_choice_minus_geodesic_choice"]),
        ),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(disagreements[:example_count], start=1):
        selected = {
            "rank_within_seed": rank,
            "seed": row["seed"],
            "evaluation": row["evaluation"],
            "fitness": row["fitness"],
            "latent_0": row["latent_0"],
            "latent_1": row["latent_1"],
            "euclidean_cell": f"({row['euclidean_cell_x']},{row['euclidean_cell_y']})",
            "geodesic_cell": f"({row['geodesic_cell_x']},{row['geodesic_cell_y']})",
            "euclidean_centroid_index": row["euclidean_centroid_index"],
            "geodesic_centroid_index": row["geodesic_centroid_index"],
            "euclidean_distance_to_euclidean_choice": row["euclidean_distance_to_euclidean_choice"],
            "euclidean_distance_to_geodesic_choice": row["euclidean_distance_to_geodesic_choice"],
            "geodesic_distance_to_geodesic_choice": row["geodesic_distance_to_geodesic_choice"],
            "geodesic_distance_to_euclidean_choice": row["geodesic_distance_to_euclidean_choice"],
            "geodesic_gap_euclidean_choice_minus_geodesic_choice": row[
                "geodesic_gap_euclidean_choice_minus_geodesic_choice"
            ],
            "geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice": row[
                "geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"
            ],
            "centroid_cell_displacement": row["centroid_cell_displacement"],
            "actual_run_inserted": row["actual_run_inserted"],
            "actual_run_replaced": row["actual_run_replaced"],
            "final_euclidean_cell_occupied": row["final_euclidean_cell_occupied"],
            "final_euclidean_cell_fitness": row["final_euclidean_cell_fitness"],
            "candidate_minus_final_euclidean_cell_fitness": row[
                "candidate_minus_final_euclidean_cell_fitness"
            ],
            "final_geodesic_cell_occupied": row["final_geodesic_cell_occupied"],
            "final_geodesic_cell_fitness": row["final_geodesic_cell_fitness"],
            "candidate_minus_final_geodesic_cell_fitness": row[
                "candidate_minus_final_geodesic_cell_fitness"
            ],
        }
        output.append(selected)
    return output


def disagreement_outcome_stats(
    rows: list[dict[str, Any]], retrain_evaluation: int
) -> dict[str, Any]:
    all_stats = disagreement_outcome_stats_for_scope(rows)
    post_rows = [row for row in rows if int(row["evaluation"]) > retrain_evaluation]
    post_stats = disagreement_outcome_stats_for_scope(post_rows)
    return {
        "disagreement_inserted_count": all_stats["inserted_count"],
        "disagreement_inserted_rate": all_stats["inserted_rate"],
        "disagreement_replaced_count": all_stats["replaced_count"],
        "disagreement_replaced_rate": all_stats["replaced_rate"],
        "disagreement_geodesic_cell_match_count": all_stats["geodesic_cell_match_count"],
        "disagreement_geodesic_cell_match_rate": all_stats["geodesic_cell_match_rate"],
        "disagreement_replaced_matching_geodesic_cell_count": all_stats[
            "replaced_matching_geodesic_cell_count"
        ],
        "disagreement_replaced_matching_geodesic_cell_rate": all_stats[
            "replaced_matching_geodesic_cell_rate"
        ],
        "post_retrain_disagreement_count": post_stats["count"],
        "post_retrain_disagreement_inserted_count": post_stats["inserted_count"],
        "post_retrain_disagreement_inserted_rate": post_stats["inserted_rate"],
        "post_retrain_disagreement_replaced_count": post_stats["replaced_count"],
        "post_retrain_disagreement_replaced_rate": post_stats["replaced_rate"],
    }


def disagreement_outcome_stats_for_scope(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    inserted = [row for row in rows if bool(row["actual_run_inserted"])]
    replaced = [row for row in rows if bool(row["actual_run_replaced"])]
    cell_match = [row for row in rows if actual_cell_matches_geodesic_cell(row)]
    replaced_matching = [row for row in replaced if actual_cell_matches_geodesic_cell(row)]
    return {
        "count": count,
        "inserted_count": len(inserted),
        "inserted_rate": safe_fraction(len(inserted), count),
        "replaced_count": len(replaced),
        "replaced_rate": safe_fraction(len(replaced), count),
        "geodesic_cell_match_count": len(cell_match),
        "geodesic_cell_match_rate": safe_fraction(len(cell_match), count),
        "replaced_matching_geodesic_cell_count": len(replaced_matching),
        "replaced_matching_geodesic_cell_rate": safe_fraction(len(replaced_matching), count),
    }


def actual_cell_matches_geodesic_cell(row: dict[str, Any]) -> bool:
    return int(row["actual_run_cell_x"]) == int(row["geodesic_cell_x"]) and int(
        row["actual_run_cell_y"]
    ) == int(row["geodesic_cell_y"])


def fitness_blowup_correlation_stats(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, float | int]:
    if len(rows) < 3:
        return {
            f"{prefix}fitness_blowup_correlation_count": len(rows),
            f"{prefix}fitness_blowup_pearson_r": float("nan"),
            f"{prefix}fitness_blowup_pearson_p": float("nan"),
            f"{prefix}fitness_blowup_spearman_r": float("nan"),
            f"{prefix}fitness_blowup_spearman_p": float("nan"),
        }
    fitness = np.asarray([float(row["fitness"]) for row in rows], dtype=np.float64)
    blowup = np.asarray(
        [float(row["geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"]) for row in rows],
        dtype=np.float64,
    )
    finite = np.isfinite(fitness) & np.isfinite(blowup)
    if np.count_nonzero(finite) < 3:
        return {
            f"{prefix}fitness_blowup_correlation_count": int(np.count_nonzero(finite)),
            f"{prefix}fitness_blowup_pearson_r": float("nan"),
            f"{prefix}fitness_blowup_pearson_p": float("nan"),
            f"{prefix}fitness_blowup_spearman_r": float("nan"),
            f"{prefix}fitness_blowup_spearman_p": float("nan"),
        }
    pearson = pearsonr(blowup[finite], fitness[finite])
    spearman = spearmanr(blowup[finite], fitness[finite])
    return {
        f"{prefix}fitness_blowup_correlation_count": int(np.count_nonzero(finite)),
        f"{prefix}fitness_blowup_pearson_r": float(pearson.statistic),
        f"{prefix}fitness_blowup_pearson_p": float(pearson.pvalue),
        f"{prefix}fitness_blowup_spearman_r": float(spearman.statistic),
        f"{prefix}fitness_blowup_spearman_p": float(spearman.pvalue),
    }


def blowup_fitness_bin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("[1.00,1.25)", 1.0, 1.25),
        ("[1.25,1.50)", 1.25, 1.50),
        ("[1.50,2.00)", 1.50, 2.00),
        ("[2.00,5.00)", 2.00, 5.00),
        ("[5.00,inf)", 5.00, float("inf")),
    ]
    output: list[dict[str, Any]] = []
    for scope, scope_rows in [
        ("all_disagreements", rows),
        ("post_retrain_disagreements", [row for row in rows if int(row["evaluation"]) > 10000]),
    ]:
        total = len(scope_rows)
        for label, lower, upper in bins:
            bin_rows = [
                row
                for row in scope_rows
                if lower
                <= float(row["geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"])
                < upper
            ]
            fitness = np.asarray([float(row["fitness"]) for row in bin_rows], dtype=np.float64)
            output.append(
                {
                    "scope": scope,
                    "blowup_ratio_bin": label,
                    "lower_inclusive": lower,
                    "upper_exclusive": upper,
                    "count": len(bin_rows),
                    "fraction_of_scope": safe_fraction(len(bin_rows), total),
                    "mean_fitness": finite_mean(fitness),
                    "median_fitness": finite_median(fitness),
                    "std_fitness": finite_std(fitness),
                    "inserted_count": sum(
                        1 for row in bin_rows if bool(row["actual_run_inserted"])
                    ),
                    "inserted_rate": safe_fraction(
                        sum(1 for row in bin_rows if bool(row["actual_run_inserted"])),
                        len(bin_rows),
                    ),
                    "replaced_count": sum(
                        1 for row in bin_rows if bool(row["actual_run_replaced"])
                    ),
                    "replaced_rate": safe_fraction(
                        sum(1 for row in bin_rows if bool(row["actual_run_replaced"])),
                        len(bin_rows),
                    ),
                }
            )
    return output


def globally_rank_examples(example_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        example_rows,
        key=lambda row: (
            float(row["geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice"]),
            float(row["geodesic_gap_euclidean_choice_minus_geodesic_choice"]),
        ),
        reverse=True,
    )
    for global_rank, row in enumerate(ranked, start=1):
        row["global_rank"] = global_rank
    return ranked


def centroid_cell_distance(
    left: np.ndarray, right: np.ndarray, bins: tuple[int, int]
) -> np.ndarray:
    left_x = left // bins[1]
    left_y = left % bins[1]
    right_x = right // bins[1]
    right_y = right % bins[1]
    return np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2).astype(np.float64)


def index_to_cell(index: int, bins: tuple[int, int]) -> tuple[int, int]:
    return int(index // bins[1]), int(index % bins[1])


def bounds_from_centroids(
    centroids: np.ndarray, bins: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.unique(np.round(centroids[:, 0], decimals=12))
    ys = np.unique(np.round(centroids[:, 1], decimals=12))
    if len(xs) != bins[0] or len(ys) != bins[1]:
        raise ValueError(f"Expected {bins} centroid grid, found {len(xs)} x {len(ys)}")
    step_x = float(np.median(np.diff(xs)))
    step_y = float(np.median(np.diff(ys)))
    bd_min = np.array([float(xs[0] - 0.5 * step_x), float(ys[0] - 0.5 * step_y)], dtype=np.float64)
    bd_max = np.array(
        [float(xs[-1] + 0.5 * step_x), float(ys[-1] + 0.5 * step_y)], dtype=np.float64
    )
    return bd_min, bd_max


def read_latents(path: Path) -> np.ndarray:
    rows = read_csv(path)
    points = [[float(row["latent_0"]), float(row["latent_1"])] for row in rows]
    return np.asarray(points, dtype=np.float64)


def read_centroids(path: Path) -> np.ndarray:
    rows = read_csv(path)
    points = [
        [float(row["latent_0"]), float(row["latent_1"])]
        for row in rows
        if row.get("kind") == "centroid"
    ]
    return np.asarray(points, dtype=np.float64)


def read_archive_cell_map(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    rows = read_csv(path)
    cell_map: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        cell = (int(row["cell_x"]), int(row["cell_y"]))
        cell_map[cell] = {
            "fitness": float(row["fitness"]),
            "descriptor_x": float(row["descriptor_x"]),
            "descriptor_y": float(row["descriptor_y"]),
            "centroid_x": float(row["centroid_x"]),
            "centroid_y": float(row["centroid_y"]),
        }
    return cell_map


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def aggregate_summaries(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "disagreement_rate",
        "post_retrain_disagreement_rate",
        "strong_blowup_fraction_of_disagreements_ratio_ge_2",
        "very_strong_blowup_fraction_of_disagreements_ratio_ge_5",
        "mean_geodesic_blowup_ratio_disagreements",
        "median_geodesic_blowup_ratio_disagreements",
        "max_geodesic_blowup_ratio_disagreements",
        "mean_geodesic_gap_disagreements",
        "mean_cell_displacement_disagreements",
        "top_10_region_fraction_of_disagreements",
        "top_10_region_fraction_of_all_candidates",
        "disagreement_region_entropy_normalized",
        "max_region_disagreement_rate",
        "disagreement_inserted_rate",
        "disagreement_replaced_rate",
        "disagreement_replaced_matching_geodesic_cell_rate",
        "post_retrain_disagreement_inserted_rate",
        "post_retrain_disagreement_replaced_rate",
        "fitness_blowup_pearson_r",
        "fitness_blowup_spearman_r",
        "post_retrain_fitness_blowup_pearson_r",
        "post_retrain_fitness_blowup_spearman_r",
    ]
    aggregate: dict[str, Any] = {
        "analysis": "Phase 5 Contribution Euclidean-vs-geodesic centroid assignment disagreement",
        "seed_count": len(seed_summaries),
        "seeds": [int(row["seed"]) for row in seed_summaries],
        "candidate_count_total": int(sum(int(row["candidate_count"]) for row in seed_summaries)),
        "disagreement_count_total": int(
            sum(int(row["disagreement_count"]) for row in seed_summaries)
        ),
        "disagreement_rate_pooled": safe_fraction(
            int(sum(int(row["disagreement_count"]) for row in seed_summaries)),
            int(sum(int(row["candidate_count"]) for row in seed_summaries)),
        ),
        "primary_post_retrain_candidate_count": int(
            sum(int(row["post_retrain_candidate_count"]) for row in seed_summaries)
        ),
        "primary_post_retrain_disagreement_count": int(
            sum(int(row["post_retrain_disagreement_count"]) for row in seed_summaries)
        ),
        "primary_post_retrain_disagreement_rate": safe_fraction(
            int(sum(int(row["post_retrain_disagreement_count"]) for row in seed_summaries)),
            int(sum(int(row["post_retrain_candidate_count"]) for row in seed_summaries)),
        ),
        "headline_scope": "online_post_retrain_evaluations_10001_to_20000",
        "pre_retrain_scope": (
            "secondary post-hoc approximation in the final retrained latent coordinate system"
        ),
        "method_note": (
            "No new evaluations are run. The primary analysis uses saved final one-retrain latent "
            "codes and reconstructs the corrected endpoint-only centroid plus rolling-buffer graph "
            "states used online after the eval-10000 rebuild. Earlier evaluations are "
            "reconstructed in final latent coordinates and are secondary only."
        ),
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in seed_summaries], dtype=np.float64)
        values = values[np.isfinite(values)]
        aggregate[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
        aggregate[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return aggregate


def pooled_disagreement_outcome_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_outcomes = disagreement_outcome_stats_for_scope(rows)
    post_rows = [row for row in rows if int(row["evaluation"]) > 10000]
    post_outcomes = disagreement_outcome_stats_for_scope(post_rows)
    correlations = fitness_blowup_correlation_stats(rows, prefix="pooled_")
    post_correlations = fitness_blowup_correlation_stats(post_rows, prefix="pooled_post_retrain_")
    return {
        "primary_post_retrain_elite_replacement_count": post_outcomes["replaced_count"],
        "primary_post_retrain_elite_replacement_rate_among_disagreements": post_outcomes[
            "replaced_rate"
        ],
        "pooled_disagreement_inserted_count": all_outcomes["inserted_count"],
        "pooled_disagreement_inserted_rate": all_outcomes["inserted_rate"],
        "pooled_disagreement_replaced_count": all_outcomes["replaced_count"],
        "pooled_disagreement_replaced_rate": all_outcomes["replaced_rate"],
        "pooled_disagreement_geodesic_cell_match_count": all_outcomes["geodesic_cell_match_count"],
        "pooled_disagreement_geodesic_cell_match_rate": all_outcomes["geodesic_cell_match_rate"],
        "pooled_disagreement_replaced_matching_geodesic_cell_count": all_outcomes[
            "replaced_matching_geodesic_cell_count"
        ],
        "pooled_disagreement_replaced_matching_geodesic_cell_rate": all_outcomes[
            "replaced_matching_geodesic_cell_rate"
        ],
        "pooled_post_retrain_disagreement_count": post_outcomes["count"],
        "pooled_post_retrain_disagreement_inserted_count": post_outcomes["inserted_count"],
        "pooled_post_retrain_disagreement_inserted_rate": post_outcomes["inserted_rate"],
        "pooled_post_retrain_disagreement_replaced_count": post_outcomes["replaced_count"],
        "pooled_post_retrain_disagreement_replaced_rate": post_outcomes["replaced_rate"],
        **correlations,
        **post_correlations,
    }


def save_seed_curve(rows: list[dict[str, Any]], output_stem: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    xs = [float(row["window_midpoint"]) for row in rows]
    ys = [100.0 * float(row["disagreement_rate"]) for row in rows]
    ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, color="#dc2626")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("assignment disagreement (%)")
    ax.set_title("Euclidean vs geodesic niche assignment disagreement over time")
    ax.grid(alpha=0.25)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def save_aggregate_curve(rows: list[dict[str, Any]], output_stem: Path) -> list[str]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if not by_seed:
        return []
    first_seed = next(iter(by_seed))
    xs = np.asarray(
        [float(row["window_midpoint"]) for row in by_seed[first_seed]], dtype=np.float64
    )
    curves = []
    for seed_rows in by_seed.values():
        seed_rows = sorted(seed_rows, key=lambda row: float(row["window_midpoint"]))
        curves.append([100.0 * float(row["disagreement_rate"]) for row in seed_rows])
    values = np.asarray(curves, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(xs, mean, color="#dc2626", linewidth=2.0, label="mean")
    ax.fill_between(
        xs, mean - std, mean + std, color="#dc2626", alpha=0.18, linewidth=0.0, label="+/- std"
    )
    ax.set_xlabel("evaluation")
    ax.set_ylabel("assignment disagreement (%)")
    ax.set_title("Phase 5 Contribution assignment disagreement over time")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def save_seed_scatter(
    latents: np.ndarray,
    disagreement: np.ndarray,
    blowup_ratio: np.ndarray,
    output_stem: Path,
    title: str,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    ax.scatter(
        latents[:, 0],
        latents[:, 1],
        s=3.0,
        c="#94a3b8",
        alpha=0.18,
        linewidths=0,
        label="all candidates",
    )
    if np.any(disagreement):
        colors = np.clip(blowup_ratio[disagreement], 1.0, 10.0)
        sc = ax.scatter(
            latents[disagreement, 0],
            latents[disagreement, 1],
            s=6.0,
            c=colors,
            cmap="magma",
            alpha=0.75,
            linewidths=0,
            label="disagreement",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("geodesic blowup ratio (clipped at 10)")
    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.15)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def save_region_heatmap(
    latents: np.ndarray,
    disagreement: np.ndarray,
    output_stem: Path,
    title: str,
    bins: int = 10,
) -> list[str]:
    x_edges = np.linspace(float(np.min(latents[:, 0])), float(np.max(latents[:, 0])), bins + 1)
    y_edges = np.linspace(float(np.min(latents[:, 1])), float(np.max(latents[:, 1])), bins + 1)
    all_counts, _, _ = np.histogram2d(latents[:, 0], latents[:, 1], bins=[x_edges, y_edges])
    disagreement_counts, _, _ = np.histogram2d(
        latents[disagreement, 0],
        latents[disagreement, 1],
        bins=[x_edges, y_edges],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.divide(
            disagreement_counts,
            all_counts,
            out=np.zeros_like(disagreement_counts),
            where=all_counts > 0,
        )
    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    image = ax.imshow(
        rates.T * 100.0,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="magma",
        vmin=0.0,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("disagreement rate (%)")
    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title(title)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


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


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if len(finite) else float("nan")


def finite_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0


def print_summary(
    aggregate: dict[str, Any],
    seed_summaries: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> None:
    print("PHASE 5 GEODESIC ASSIGNMENT DISAGREEMENT ANALYSIS")
    print(
        "PRIMARY online post-retrain disagreement: {primary_post_retrain_disagreement_count}/"
        "{primary_post_retrain_candidate_count} = {primary_post_retrain_disagreement_rate:.3%}; "
        "elite replacements among disagreements={primary_post_retrain_elite_replacement_count} "
        "({primary_post_retrain_elite_replacement_rate_among_disagreements:.3%})".format(
            **aggregate
        )
    )
    print(
        "SECONDARY pooled final-space reconstruction: {disagreement_count_total}/"
        "{candidate_count_total} = {disagreement_rate_pooled:.3%}".format(**aggregate)
    )
    print(
        "per-seed disagreement: mean={disagreement_rate_mean:.3%}, "
        "std={disagreement_rate_std:.3%}".format(**aggregate)
    )
    print(
        "synthetic-like blowup among disagreements: ratio>=2 mean="
        "{strong_blowup_fraction_of_disagreements_ratio_ge_2_mean:.3%}, ratio>=5 mean="
        "{very_strong_blowup_fraction_of_disagreements_ratio_ge_5_mean:.3%}".format(**aggregate)
    )
    print(
        "elite impact among disagreements: inserted={pooled_disagreement_inserted_count}/"
        "{disagreement_count_total} ({pooled_disagreement_inserted_rate:.3%}), "
        "replaced={pooled_disagreement_replaced_count}/{disagreement_count_total} "
        "({pooled_disagreement_replaced_rate:.3%})".format(**aggregate)
    )
    print(
        "post-retrain exact-space disagreements: n={pooled_post_retrain_disagreement_count}, "
        "inserted={pooled_post_retrain_disagreement_inserted_rate:.3%}, "
        "replaced={pooled_post_retrain_disagreement_replaced_rate:.3%}".format(**aggregate)
    )
    print(
        "fitness vs blowup among disagreements: Pearson r={pooled_fitness_blowup_pearson_r:.4f} "
        "(p={pooled_fitness_blowup_pearson_p:.4g}), Spearman rho="
        "{pooled_fitness_blowup_spearman_r:.4f} (p={pooled_fitness_blowup_spearman_p:.4g})".format(
            **aggregate
        )
    )
    print(
        "region concentration: top10 disagreement bins mean="
        "{top_10_region_fraction_of_disagreements_mean:.3%}; all candidates top10 mean="
        "{top_10_region_fraction_of_all_candidates_mean:.3%}; entropy mean="
        "{disagreement_region_entropy_normalized_mean:.3f}".format(**aggregate)
    )
    print("per seed:")
    for row in seed_summaries:
        print(
            "  seed {seed}: disagreement={disagreement_rate:.3%} "
            "({disagreement_count}/{candidate_count}), "
            "ratio>=2={strong_blowup_fraction_of_disagreements_ratio_ge_2:.3%}, "
            "top10_bins={top_10_region_fraction_of_disagreements:.3%}, "
            "max_region_rate={max_region_disagreement_rate:.3%}".format(**row)
        )
    print("top examples:")
    for row in example_rows[:10]:
        print(
            "  seed {seed} eval {evaluation}: euc {euclidean_cell} -> geo {geodesic_cell}, "
            "geo_choice={geodesic_distance_to_geodesic_choice:.3f}, "
            "euc_choice_geo_dist={geodesic_distance_to_euclidean_choice:.3f}, "
            "ratio={geodesic_blowup_ratio_euclidean_choice_over_geodesic_choice:.2f}".format(**row)
        )
    print("output files:")
    for output_file in aggregate["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
