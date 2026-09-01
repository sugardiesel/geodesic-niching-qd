"""Audit whether disconnected geodesic endpoints starved corrected production archives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist

from algorithms.knn_graph import build_endpoint_manifold_graph, endpoint_shortest_path_distances


@dataclass(frozen=True)
class RunSpec:
    map_name: str
    root: Path
    seed: int

    @property
    def run_dir(self) -> Path:
        return self.root / "contribution_geodesic_niching" / f"seed_{self.seed}"

    @property
    def config_path(self) -> Path:
        return self.root / "configs" / f"contribution_geodesic_niching_seed_{self.seed}.yaml"


@dataclass
class GraphState:
    centroids: np.ndarray
    rolling: np.ndarray
    raw_distances: np.ndarray
    repaired_distances: np.ndarray
    tree: cKDTree
    reachable_centroids: np.ndarray
    component_count: int
    finite_pair_fraction: float
    penalty_distance: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-dir", default="results/phase5")
    parser.add_argument("--open-dir", default="results/phase6_open_robustness")
    parser.add_argument("--dynamic-dir", default="results/audit_connectivity_verification")
    parser.add_argument(
        "--output-dir", default="results/audit_connectivity_verification/penalty_audit"
    )
    args = parser.parse_args()

    specs = [
        *[
            RunSpec("horseshoe", Path(args.phase5_dir), seed)
            for seed in [1001, 1002, 1003, 1004, 1005]
        ],
        *[RunSpec("open", Path(args.open_dir), seed) for seed in [1001, 1002, 1003]],
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    refresh_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    for spec in specs:
        seed_summary, run_refreshes, run_centroids = audit_run(spec)
        seed_rows.append(seed_summary)
        refresh_rows.extend(run_refreshes)
        centroid_rows.extend(run_centroids)
        print_seed_summary(seed_summary)

    aggregate_rows = aggregate_seed_rows(seed_rows)
    dynamic_rows = aggregate_dynamic_tables(Path(args.dynamic_dir))
    static_rows = combine_static_tables(Path(args.dynamic_dir))

    write_csv(output_dir / "penalty_audit_by_seed.csv", seed_rows)
    write_csv(output_dir / "penalty_audit_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "penalty_audit_refreshes.csv", refresh_rows)
    write_csv(output_dir / "penalty_audit_centroids.csv", centroid_rows)
    write_csv(output_dir / "dynamic_k_sensitivity_aggregate.csv", dynamic_rows)
    write_csv(output_dir / "static_k_sensitivity_by_map.csv", static_rows)
    plot_paths = save_connectivity_plots(refresh_rows, centroid_rows, output_dir)

    summary = {
        "analysis": "Corrected graph k-sensitivity and actual disconnection-penalty audit",
        "scope": (
            "No search reruns. Pre-retrain and post-retrain online assignments are replayed in "
            "their exact saved encoder spaces. Eval-10000 archive rebuild assignments are audited "
            "separately in the final encoder space."
        ),
        "penalty_assignment_definition": (
            "A candidate uses the penalty only when the production-selected centroid has no finite "
            "raw shortest path through any of that query's assignment-k routing neighbors."
        ),
        "seed_rows": seed_rows,
        "aggregate_rows": aggregate_rows,
        "output_files": [
            str(output_dir / "penalty_audit_by_seed.csv"),
            str(output_dir / "penalty_audit_aggregate.csv"),
            str(output_dir / "penalty_audit_refreshes.csv"),
            str(output_dir / "penalty_audit_centroids.csv"),
            str(output_dir / "dynamic_k_sensitivity_aggregate.csv"),
            str(output_dir / "static_k_sensitivity_by_map.csv"),
            *plot_paths,
            str(output_dir / "connectivity_penalty_audit_summary.json"),
        ],
    }
    summary_path = output_dir / "connectivity_penalty_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nAGGREGATE")
    for row in aggregate_rows:
        print(
            "{map}: exact assignments={exact_online_assignment_count}, mismatches="
            "{exact_online_assignment_mismatch_count}, penalty selected="
            "{all_assignment_penalty_count}/{all_assignment_count} "
            "({all_assignment_penalty_rate:.3%}), "
            "post centroids never reachable={post_never_reachable_centroids_mean:.1f}/625, "
            "final occupied among never reachable="
            "{post_never_reachable_final_occupied_mean:.1f}".format(**row)
        )


def audit_run(
    spec: RunSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{spec.config_path} must contain a mapping.")
    algo = config["algorithm"]
    autoencoder = config["autoencoder"]
    geodesic = config["geodesic"]
    bins = (int(algo["grid_bins"][0]), int(algo["grid_bins"][1]))
    bootstrap = int(algo["bootstrap_evaluations"])
    retrain = min(int(value) for value in algo["retrain_evaluations"])
    total = int(algo["total_evaluations"])
    graph_k = int(geodesic["graph_k"])
    assignment_k = int(geodesic["assignment_k"])
    rolling_size = int(geodesic["max_graph_points"])
    refresh_interval = int(geodesic["refresh_interval"])
    padding = float(autoencoder["latent_bounds_padding_fraction"])

    evaluation_rows = read_evaluations(spec.run_dir / "evaluations.csv")
    online_latents = np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in evaluation_rows],
        dtype=np.float64,
    )
    final_latents = read_points(spec.run_dir / "visited_latents_final_space.csv")
    saved_cells = np.asarray(
        [[int(row["cell_x"]), int(row["cell_y"])] for row in evaluation_rows], dtype=np.int64
    )
    final_occupied = read_occupied_indices(spec.run_dir / "archive_cells.csv", bins)

    pre_centroids = make_centroids(*latent_bounds(online_latents[:bootstrap], padding), bins=bins)
    post_centroids = make_centroids(*latent_bounds(final_latents[:retrain], padding), bins=bins)

    pre = audit_stage(
        map_name=spec.map_name,
        seed=spec.seed,
        stage="pre_retrain",
        latents=online_latents,
        centroids=pre_centroids,
        initial_evaluation=bootstrap,
        end_evaluation=retrain,
        graph_k=graph_k,
        assignment_k=assignment_k,
        rolling_size=rolling_size,
        refresh_interval=refresh_interval,
        initial_query_count=bootstrap,
        saved_cells=saved_cells,
        compare_online=True,
    )
    post = audit_stage(
        map_name=spec.map_name,
        seed=spec.seed,
        stage="post_retrain",
        latents=final_latents,
        centroids=post_centroids,
        initial_evaluation=retrain,
        end_evaluation=total,
        graph_k=graph_k,
        assignment_k=assignment_k,
        rolling_size=rolling_size,
        refresh_interval=refresh_interval,
        initial_query_count=retrain,
        saved_cells=saved_cells,
        compare_online=True,
    )

    post_reachable_fraction = post["reachable_refresh_count"] / post["refresh_count"]
    never_reachable = post_reachable_fraction == 0.0
    unreachable_majority = post_reachable_fraction < 0.5
    nearly_always_unreachable = post_reachable_fraction <= 0.1
    post_assignments = post["assignment_count"]
    post_real_assignments = post["real_assignment_count"]
    post_penalty_assignments = post["penalty_assignment_count"]
    only_penalty_assigned = (post_assignments > 0) & (post_real_assignments == 0)

    centroid_rows: list[dict[str, Any]] = []
    for centroid_index, centroid in enumerate(post_centroids):
        centroid_rows.append(
            {
                "map": spec.map_name,
                "seed": spec.seed,
                "centroid_index": centroid_index,
                "cell_x": centroid_index // bins[1],
                "cell_y": centroid_index % bins[1],
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "reachable_refresh_fraction": float(post_reachable_fraction[centroid_index]),
                "never_reachable": bool(never_reachable[centroid_index]),
                "unreachable_more_than_half": bool(unreachable_majority[centroid_index]),
                "unreachable_at_least_90_percent": bool(nearly_always_unreachable[centroid_index]),
                "assignment_count_rebuild_plus_post": int(post_assignments[centroid_index]),
                "real_assignment_count": int(post_real_assignments[centroid_index]),
                "penalty_assignment_count": int(post_penalty_assignments[centroid_index]),
                "only_ever_assigned_via_penalty": bool(only_penalty_assigned[centroid_index]),
                "final_occupied": bool(centroid_index in final_occupied),
            }
        )

    all_assignment_count = pre["all_assignment_count"] + post["all_assignment_count"]
    all_penalty_count = pre["all_penalty_count"] + post["all_penalty_count"]
    exact_online_count = pre["online_assignment_count"] + post["online_assignment_count"]
    exact_mismatch_count = pre["online_mismatch_count"] + post["online_mismatch_count"]
    summary = {
        "map": spec.map_name,
        "seed": spec.seed,
        "graph_k": graph_k,
        "assignment_k": assignment_k,
        "pre_refresh_count": pre["refresh_count"],
        "post_refresh_count": post["refresh_count"],
        "pre_refreshes_with_unreachable_centroids": pre["refreshes_with_unreachable"],
        "post_refreshes_with_unreachable_centroids": post["refreshes_with_unreachable"],
        "pre_mean_reachable_centroid_fraction": pre["mean_reachable_fraction"],
        "post_mean_reachable_centroid_fraction": post["mean_reachable_fraction"],
        "post_final_reachable_centroid_fraction": post["final_reachable_fraction"],
        "pre_assignment_count": pre["all_assignment_count"],
        "pre_penalty_assignment_count": pre["all_penalty_count"],
        "pre_penalty_assignment_rate": safe_ratio(
            pre["all_penalty_count"], pre["all_assignment_count"]
        ),
        "retrain_rebuild_assignment_count": post["initial_assignment_count"],
        "retrain_rebuild_penalty_assignment_count": post["initial_penalty_count"],
        "retrain_rebuild_penalty_assignment_rate": safe_ratio(
            post["initial_penalty_count"], post["initial_assignment_count"]
        ),
        "post_online_assignment_count": post["online_assignment_count"],
        "post_online_penalty_assignment_count": post["online_penalty_count"],
        "post_online_penalty_assignment_rate": safe_ratio(
            post["online_penalty_count"], post["online_assignment_count"]
        ),
        "all_assignment_count": all_assignment_count,
        "all_assignment_penalty_count": all_penalty_count,
        "all_assignment_penalty_rate": safe_ratio(all_penalty_count, all_assignment_count),
        "exact_online_assignment_count": exact_online_count,
        "exact_online_assignment_mismatch_count": exact_mismatch_count,
        "exact_online_assignment_match_rate": 1.0
        - safe_ratio(exact_mismatch_count, exact_online_count),
        "post_never_reachable_centroids": int(np.count_nonzero(never_reachable)),
        "post_unreachable_more_than_half_centroids": int(np.count_nonzero(unreachable_majority)),
        "post_unreachable_at_least_90_percent_centroids": int(
            np.count_nonzero(nearly_always_unreachable)
        ),
        "post_only_penalty_assigned_centroids": int(np.count_nonzero(only_penalty_assigned)),
        "post_never_assigned_centroids": int(np.count_nonzero(post_assignments == 0)),
        "post_never_reachable_final_occupied": int(
            sum(int(index in final_occupied) for index in np.flatnonzero(never_reachable))
        ),
        "post_never_reachable_final_unoccupied": int(
            sum(int(index not in final_occupied) for index in np.flatnonzero(never_reachable))
        ),
        "final_occupied_centroids": len(final_occupied),
        "post_distinct_assigned_centroids": int(np.count_nonzero(post_assignments > 0)),
        "post_distinct_real_assigned_centroids": int(np.count_nonzero(post_real_assignments > 0)),
    }
    return summary, [*pre["refresh_rows"], *post["refresh_rows"]], centroid_rows


def audit_stage(
    *,
    map_name: str,
    seed: int,
    stage: str,
    latents: np.ndarray,
    centroids: np.ndarray,
    initial_evaluation: int,
    end_evaluation: int,
    graph_k: int,
    assignment_k: int,
    rolling_size: int,
    refresh_interval: int,
    initial_query_count: int,
    saved_cells: np.ndarray,
    compare_online: bool,
) -> dict[str, Any]:
    centroid_count = len(centroids)
    reachable_refresh_count = np.zeros(centroid_count, dtype=np.int64)
    assignment_count = np.zeros(centroid_count, dtype=np.int64)
    real_assignment_count = np.zeros(centroid_count, dtype=np.int64)
    penalty_assignment_count = np.zeros(centroid_count, dtype=np.int64)
    refresh_rows: list[dict[str, Any]] = []
    initial_assignment_count = 0
    initial_penalty_count = 0
    online_assignment_count = 0
    online_penalty_count = 0
    online_mismatch_count = 0

    next_refresh = ((initial_evaluation // refresh_interval) + 1) * refresh_interval
    refresh_evaluations = [initial_evaluation]
    refresh_evaluations.extend(range(next_refresh, end_evaluation + 1, refresh_interval))

    for refresh_index, refresh_evaluation in enumerate(refresh_evaluations):
        rolling = latents[max(0, refresh_evaluation - rolling_size) : refresh_evaluation]
        state = build_graph_state(rolling, centroids, graph_k)
        reachable_refresh_count += state.reachable_centroids.astype(np.int64)
        refresh_rows.append(
            {
                "map": map_name,
                "seed": seed,
                "stage": stage,
                "refresh_index": refresh_index,
                "evaluation": refresh_evaluation,
                "component_count": state.component_count,
                "support_finite_pair_fraction": state.finite_pair_fraction,
                "reachable_centroid_count": int(np.count_nonzero(state.reachable_centroids)),
                "reachable_centroid_fraction": float(np.mean(state.reachable_centroids)),
                "unreachable_centroid_count": int(
                    centroid_count - np.count_nonzero(state.reachable_centroids)
                ),
                "penalty_distance": state.penalty_distance,
            }
        )

        if refresh_index == 0 and initial_query_count > 0:
            for query in latents[:initial_query_count]:
                chosen, used_penalty = assign_query(state, query, assignment_k)
                assignment_count[chosen] += 1
                if used_penalty:
                    penalty_assignment_count[chosen] += 1
                    initial_penalty_count += 1
                else:
                    real_assignment_count[chosen] += 1
                initial_assignment_count += 1

        interval_start = initial_evaluation + 1 if refresh_index == 0 else refresh_evaluation
        interval_end = (
            refresh_evaluations[refresh_index + 1] - 1
            if refresh_index + 1 < len(refresh_evaluations)
            else end_evaluation
        )
        for evaluation in range(interval_start, interval_end + 1):
            chosen, used_penalty = assign_query(state, latents[evaluation - 1], assignment_k)
            assignment_count[chosen] += 1
            online_assignment_count += 1
            if used_penalty:
                penalty_assignment_count[chosen] += 1
                online_penalty_count += 1
            else:
                real_assignment_count[chosen] += 1
            if compare_online:
                expected = saved_cells[evaluation - 1]
                side = int(round(math.sqrt(centroid_count)))
                reconstructed = np.asarray([chosen // side, chosen % side], dtype=np.int64)
                if not np.array_equal(expected, reconstructed):
                    online_mismatch_count += 1

    reachable_fractions = [float(row["reachable_centroid_fraction"]) for row in refresh_rows]
    return {
        "refresh_count": len(refresh_rows),
        "refreshes_with_unreachable": sum(
            int(int(row["unreachable_centroid_count"]) > 0) for row in refresh_rows
        ),
        "mean_reachable_fraction": float(np.mean(reachable_fractions)),
        "final_reachable_fraction": reachable_fractions[-1],
        "reachable_refresh_count": reachable_refresh_count,
        "assignment_count": assignment_count,
        "real_assignment_count": real_assignment_count,
        "penalty_assignment_count": penalty_assignment_count,
        "initial_assignment_count": initial_assignment_count,
        "initial_penalty_count": initial_penalty_count,
        "online_assignment_count": online_assignment_count,
        "online_penalty_count": online_penalty_count,
        "online_mismatch_count": online_mismatch_count,
        "all_assignment_count": initial_assignment_count + online_assignment_count,
        "all_penalty_count": initial_penalty_count + online_penalty_count,
        "refresh_rows": refresh_rows,
    }


def build_graph_state(rolling: np.ndarray, centroids: np.ndarray, graph_k: int) -> GraphState:
    support = np.vstack([rolling, centroids])
    endpoint_mask = np.concatenate(
        [np.zeros(len(rolling), dtype=bool), np.ones(len(centroids), dtype=bool)]
    )
    graph = build_endpoint_manifold_graph(support, endpoint_mask, graph_k)
    routing_graph = graph[: len(rolling), : len(rolling)]
    component_count, _ = connected_components(routing_graph, directed=False)
    raw = endpoint_shortest_path_distances(graph, endpoint_mask)
    route_raw = raw[:, : len(rolling)]
    reachable = np.any(np.isfinite(route_raw), axis=1)
    penalty = disconnection_penalty(raw, support)
    repaired = raw.copy()
    repaired[~np.isfinite(repaired)] = penalty

    sample_indices = np.linspace(0, len(support) - 1, min(220, len(support)), dtype=np.int64)
    sampled = raw[:, sample_indices]
    finite_pair_fraction = float(np.mean(np.isfinite(sampled)))
    return GraphState(
        centroids=centroids,
        rolling=rolling,
        raw_distances=raw,
        repaired_distances=repaired,
        tree=cKDTree(rolling),
        reachable_centroids=reachable,
        component_count=int(component_count),
        finite_pair_fraction=finite_pair_fraction,
        penalty_distance=penalty,
    )


def assign_query(state: GraphState, query: np.ndarray, assignment_k: int) -> tuple[int, bool]:
    query_k = min(max(1, assignment_k), len(state.rolling))
    query_distances, route_indices = state.tree.query(query, k=query_k)
    query_distances = np.atleast_1d(query_distances).astype(np.float64)
    route_indices = np.atleast_1d(route_indices).astype(np.int64)
    repaired = state.repaired_distances[:, route_indices] + query_distances[None, :]
    centroid_distances = np.min(repaired, axis=1)
    chosen = int(np.argmin(centroid_distances))
    repaired_choice = repaired[chosen]
    selected_neighbor = int(np.argmin(repaired_choice))
    used_penalty = not bool(
        np.isfinite(state.raw_distances[chosen, route_indices[selected_neighbor]])
    )
    return chosen, used_penalty


def disconnection_penalty(raw: np.ndarray, support: np.ndarray) -> float:
    finite_positive = raw[np.isfinite(raw) & (raw > 0.0)]
    finite_scale = float(np.max(finite_positive)) if len(finite_positive) else 0.0
    support_scale = support_diameter(support)
    return max(finite_scale, support_scale, 0.5) * 2.0


def support_diameter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.5
    try:
        hull_points = points[ConvexHull(points).vertices]
    except Exception:
        hull_points = points
    distances = pdist(hull_points)
    return float(np.max(distances)) if len(distances) else 0.5


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for map_name in sorted({str(row["map"]) for row in rows}):
        subset = [row for row in rows if row["map"] == map_name]
        result: dict[str, Any] = {"map": map_name, "seed_count": len(subset)}
        for key in [
            "exact_online_assignment_count",
            "exact_online_assignment_mismatch_count",
            "all_assignment_count",
            "all_assignment_penalty_count",
        ]:
            result[key] = int(sum(int(row[key]) for row in subset))
        result["all_assignment_penalty_rate"] = safe_ratio(
            result["all_assignment_penalty_count"], result["all_assignment_count"]
        )
        for key in [
            "pre_mean_reachable_centroid_fraction",
            "post_mean_reachable_centroid_fraction",
            "post_final_reachable_centroid_fraction",
            "post_never_reachable_centroids",
            "post_unreachable_more_than_half_centroids",
            "post_unreachable_at_least_90_percent_centroids",
            "post_only_penalty_assigned_centroids",
            "post_never_assigned_centroids",
            "post_never_reachable_final_occupied",
            "post_never_reachable_final_unoccupied",
            "final_occupied_centroids",
            "post_distinct_assigned_centroids",
            "post_distinct_real_assigned_centroids",
        ]:
            values = np.asarray([float(row[key]) for row in subset], dtype=np.float64)
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_min"] = float(np.min(values))
            result[f"{key}_max"] = float(np.max(values))
        output.append(result)
    return output


def aggregate_dynamic_tables(root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(root.glob("dynamic_*_seed_*/phase4_dynamic_graph_k_sensitivity.csv")):
        map_name = "horseshoe" if "dynamic_horseshoe" in str(path) else "open"
        for row in read_csv(path):
            grouped[(map_name, int(row["snapshot_evaluation"]), int(row["k"]))].append(row)

    output: list[dict[str, Any]] = []
    for (map_name, snapshot, k), rows in sorted(grouped.items()):
        components = np.asarray([int(row["component_count"]) for row in rows], dtype=np.int64)
        finite = np.asarray(
            [float(row["support_finite_pair_fraction"]) for row in rows], dtype=np.float64
        )
        ratios = np.asarray(
            [float(row["support_geodesic_to_euclidean_ratio"]) for row in rows],
            dtype=np.float64,
        )
        output.append(
            {
                "map": map_name,
                "snapshot_evaluation": snapshot,
                "k": k,
                "seed_count": len(rows),
                "connected_seed_count": int(np.count_nonzero(components == 1)),
                "component_count_mean": float(np.mean(components)),
                "component_count_min": int(np.min(components)),
                "component_count_max": int(np.max(components)),
                "support_finite_pair_fraction_mean": float(np.mean(finite)),
                "support_finite_pair_fraction_min": float(np.min(finite)),
                "support_finite_pair_fraction_max": float(np.max(finite)),
                "support_geodesic_to_euclidean_ratio_mean": float(np.mean(ratios)),
                "support_geodesic_to_euclidean_ratio_std": float(
                    np.std(ratios, ddof=1) if len(ratios) > 1 else 0.0
                ),
                "support_geodesic_to_euclidean_ratio_min": float(np.min(ratios)),
                "support_geodesic_to_euclidean_ratio_max": float(np.max(ratios)),
            }
        )
    return output


def combine_static_tables(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for map_name, directory in [
        ("horseshoe", "static_horseshoe_seed_1001"),
        ("open", "static_open_seed_1001"),
    ]:
        path = root / directory / "phase3_knn_k_sensitivity.csv"
        for row in read_csv(path):
            output.append({"map": map_name, **row})
    return output


def save_connectivity_plots(
    refresh_rows: list[dict[str, Any]],
    centroid_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for map_name, color in [("horseshoe", "#c2410c"), ("open", "#2563eb")]:
        subset = [
            row for row in refresh_rows if row["map"] == map_name and row["stage"] == "post_retrain"
        ]
        evaluations = sorted({int(row["evaluation"]) for row in subset})
        means = [
            float(
                np.mean(
                    [
                        float(row["reachable_centroid_fraction"])
                        for row in subset
                        if int(row["evaluation"]) == evaluation
                    ]
                )
            )
            for evaluation in evaluations
        ]
        axes[0].plot(evaluations, means, color=color, label=map_name)
    axes[0].set_xlabel("Evaluation")
    axes[0].set_ylabel("Reachable centroid fraction")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for position, map_name in enumerate(["horseshoe", "open"]):
        values = [
            float(row["reachable_refresh_fraction"])
            for row in centroid_rows
            if row["map"] == map_name
        ]
        axes[1].hist(
            values,
            bins=np.linspace(0.0, 1.0, 21),
            alpha=0.55,
            label=map_name,
            color=["#c2410c", "#2563eb"][position],
        )
    axes[1].set_xlabel("Per-centroid reachable refresh fraction")
    axes[1].set_ylabel("Centroids across seeds")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    paths = [
        output_dir / "centroid_connectivity_over_run.png",
        output_dir / "centroid_connectivity_over_run.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def latent_bounds(points: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    span = np.maximum(high - low, 1e-6)
    return low - span * padding, high + span * padding


def make_centroids(bd_min: np.ndarray, bd_max: np.ndarray, bins: tuple[int, int]) -> np.ndarray:
    xs = np.linspace(bd_min[0], bd_max[0], bins[0], endpoint=False, dtype=np.float64)
    ys = np.linspace(bd_min[1], bd_max[1], bins[1], endpoint=False, dtype=np.float64)
    step = (bd_max - bd_min) / np.asarray(bins, dtype=np.float64)
    return np.asarray(
        [[x + step[0] * 0.5, y + step[1] * 0.5] for x in xs for y in ys],
        dtype=np.float64,
    )


def read_evaluations(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def read_points(path: Path) -> np.ndarray:
    rows = read_csv(path)
    return np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in rows],
        dtype=np.float64,
    )


def read_occupied_indices(path: Path, bins: tuple[int, int]) -> set[int]:
    return {int(row["cell_x"]) * bins[1] + int(row["cell_y"]) for row in read_csv(path)}


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def print_seed_summary(row: dict[str, Any]) -> None:
    print(
        "{map} seed {seed}: exact_match={exact_online_assignment_match_rate:.3%}, "
        "penalty_selected={all_assignment_penalty_count}/{all_assignment_count} "
        "({all_assignment_penalty_rate:.3%}), post_reachable_mean="
        "{post_mean_reachable_centroid_fraction:.3%}, never_reachable="
        "{post_never_reachable_centroids}, final_occupied={final_occupied_centroids}".format(**row)
    )


if __name__ == "__main__":
    main()
