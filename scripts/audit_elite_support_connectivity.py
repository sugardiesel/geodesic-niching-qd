"""Gate elite-plus-recent geodesic graph support using saved production streams."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
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
import yaml
from scipy.sparse.csgraph import connected_components

from algorithms.archive_metrics import _pairwise_euclidean_matrix
from algorithms.geodesic_archive import compose_real_routing_support
from algorithms.knn_graph import (
    build_endpoint_manifold_graph,
    endpoint_safe_all_pairs_distances,
)
from scripts.audit_geodesic_connectivity_penalty import assign_query, build_graph_state
from scripts.phase3_knn_k_sensitivity import run_k_sensitivity


@dataclass(frozen=True)
class RunSpec:
    map_name: str
    root: Path
    seed: int

    @property
    def contribution_dir(self) -> Path:
        return self.root / "contribution_geodesic_niching" / f"seed_{self.seed}"

    @property
    def config_path(self) -> Path:
        return self.root / "configs" / f"contribution_geodesic_niching_seed_{self.seed}.yaml"

    @property
    def baseline_sample(self) -> Path:
        return (
            self.root
            / "baseline_b_learned_bd_euclidean"
            / f"seed_{self.seed}"
            / "representative_trajectory_sample.npz"
        )


@dataclass(frozen=True)
class RunData:
    rows: list[dict[str, str]]
    online_latents: np.ndarray
    final_latents: np.ndarray
    fitness: np.ndarray
    bins: tuple[int, int]
    bootstrap: int
    retrain: int
    total: int
    padding: float
    old_graph_k: int
    old_assignment_k: int
    old_rolling_size: int
    refresh_interval: int
    expected_retrain_elites: int
    expected_final_elites: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-dir", default="results/phase5")
    parser.add_argument("--open-dir", default="results/phase6_open_robustness")
    parser.add_argument("--output-dir", default="results/elite_support_connectivity_diagnostic")
    parser.add_argument(
        "--rolling-only-audit",
        default=("results/audit_connectivity_verification/penalty_audit/penalty_audit_by_seed.csv"),
    )
    parser.add_argument("--rolling-buffer-size", type=int, default=500)
    parser.add_argument("--k-values", default="3,5,10,20,30")
    parser.add_argument("--snapshots", default="2000,10000,20000")
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    snapshots = parse_int_list(args.snapshots)
    if not k_values or not snapshots:
        raise ValueError("--k-values and --snapshots must be non-empty.")
    if args.rolling_buffer_size <= 0:
        raise ValueError("--rolling-buffer-size must be positive.")

    specs = [
        *[RunSpec("horseshoe", Path(args.phase5_dir), seed) for seed in range(1001, 1006)],
        *[RunSpec("open", Path(args.open_dir), seed) for seed in range(1001, 1004)],
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    static_rows: list[dict[str, Any]] = []
    refresh_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_static(spec, k_values, static_rows)
        run_refreshes, run_snapshots, run_seeds = audit_run(
            spec,
            k_values=k_values,
            snapshots=snapshots,
            rolling_buffer_size=int(args.rolling_buffer_size),
        )
        refresh_rows.extend(run_refreshes)
        snapshot_rows.extend(run_snapshots)
        seed_rows.extend(run_seeds)
        print_run_summary(spec, run_seeds)

    static_aggregate = aggregate_static(static_rows)
    snapshot_aggregate = aggregate_snapshots(snapshot_rows)
    seed_aggregate = aggregate_seed_rows(seed_rows)
    comparison_rows = compare_with_rolling_only(
        seed_rows,
        read_csv(Path(args.rolling_only_audit)),
        comparison_k=20,
    )
    write_csv(output_dir / "static_k_sensitivity_by_seed.csv", static_rows)
    write_csv(output_dir / "static_k_sensitivity_aggregate.csv", static_aggregate)
    write_csv(output_dir / "dynamic_snapshot_k_sensitivity_by_seed.csv", snapshot_rows)
    write_csv(output_dir / "dynamic_snapshot_k_sensitivity_aggregate.csv", snapshot_aggregate)
    write_csv(output_dir / "refresh_connectivity_by_seed.csv", seed_rows)
    write_csv(output_dir / "refresh_connectivity_aggregate.csv", seed_aggregate)
    write_csv(output_dir / "refresh_connectivity_states.csv", refresh_rows)
    write_csv(output_dir / "rolling_only_vs_elite_support_k20.csv", comparison_rows)
    plot_paths = save_plots(seed_aggregate, snapshot_aggregate, output_dir)

    summary = {
        "diagnostic": "Elite-plus-recent real-point support connectivity gate",
        "search_rerun": False,
        "support_definition": (
            "Stable deduplicated union of every current archive elite latent and the most recent "
            f"{args.rolling_buffer_size} candidate latents. Elite points take precedence when a "
            "candidate is present in both sets. Grid centroids remain endpoints; centroid-to-"
            "centroid edges and centroid transit are forbidden."
        ),
        "history_definition": (
            "The saved corrected Contribution candidate streams and their historical archive "
            "assignments are held fixed to isolate support composition. The eval-10000 archive "
            "rebuild is reconstructed exactly in final-encoder coordinates."
        ),
        "precommitted_gate": (
            "A k is connectivity-safe only if every routing graph has one component across every "
            "seed and post-retrain refresh, no centroid is never reachable, and mean reachable-"
            "centroid fraction is at least 0.95 on both maps. Production search is not authorized "
            "unless reachability improves substantially and a defensible k passes or nearly "
            "passes these conditions without collapsing the distance ratio to 1."
        ),
        "k_values": k_values,
        "snapshots": snapshots,
        "rolling_buffer_size": int(args.rolling_buffer_size),
        "static_aggregate": static_aggregate,
        "snapshot_aggregate": snapshot_aggregate,
        "refresh_aggregate": seed_aggregate,
        "rolling_only_k20_comparison": comparison_rows,
        "output_files": [
            str(output_dir / "static_k_sensitivity_by_seed.csv"),
            str(output_dir / "static_k_sensitivity_aggregate.csv"),
            str(output_dir / "dynamic_snapshot_k_sensitivity_by_seed.csv"),
            str(output_dir / "dynamic_snapshot_k_sensitivity_aggregate.csv"),
            str(output_dir / "refresh_connectivity_by_seed.csv"),
            str(output_dir / "refresh_connectivity_aggregate.csv"),
            str(output_dir / "refresh_connectivity_states.csv"),
            str(output_dir / "rolling_only_vs_elite_support_k20.csv"),
            *plot_paths,
            str(output_dir / "elite_support_connectivity_summary.json"),
        ],
    }
    (output_dir / "elite_support_connectivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def audit_run(
    spec: RunSpec,
    *,
    k_values: list[int],
    snapshots: list[int],
    rolling_buffer_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = load_run(spec)
    pre_centroids = make_centroids(
        *latent_bounds(data.online_latents[: data.bootstrap], data.padding), bins=data.bins
    )
    post_centroids = make_centroids(
        *latent_bounds(data.final_latents[: data.retrain], data.padding), bins=data.bins
    )
    pre_elites = reconstruct_pre_elites(
        data, max(snapshot for snapshot in snapshots if snapshot < data.retrain)
    )
    post_elites = reconstruct_post_retrain_elites(data, post_centroids)
    if len(post_elites[data.retrain]) != data.expected_retrain_elites:
        raise RuntimeError(
            f"{spec.map_name} seed {spec.seed}: reconstructed retrain archive has "
            f"{len(post_elites[data.retrain])} elites, expected {data.expected_retrain_elites}."
        )
    if len(post_elites[data.total]) != data.expected_final_elites:
        raise RuntimeError(
            f"{spec.map_name} seed {spec.seed}: reconstructed final archive has "
            f"{len(post_elites[data.total])} elites, expected {data.expected_final_elites}."
        )

    refresh_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    reachability: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    components: dict[int, list[int]] = {k: [] for k in k_values}

    for snapshot in snapshots:
        if snapshot < data.retrain:
            latents = data.online_latents
            centroids = pre_centroids
            elite_indices = pre_elites[snapshot]
            encoder_space = "pre_retrain_encoder"
        else:
            latents = data.final_latents
            centroids = post_centroids
            elite_indices = post_elites[snapshot]
            encoder_space = "final_one_retrain_encoder"
        routing, kinds = support_at(
            latents,
            elite_indices,
            snapshot,
            rolling_buffer_size,
        )
        for k in k_values:
            state = evaluate_support(routing, kinds, centroids, k, include_distances=True)
            state.pop("reachable_mask")
            snapshot_rows.append(
                {
                    "map": spec.map_name,
                    "seed": spec.seed,
                    "snapshot_evaluation": snapshot,
                    "encoder_space": encoder_space,
                    "k": k,
                    **state,
                }
            )

    post_refreshes = list(range(data.retrain, data.total + 1, data.refresh_interval))
    for evaluation in post_refreshes:
        routing, kinds = support_at(
            data.final_latents,
            post_elites[evaluation],
            evaluation,
            rolling_buffer_size,
        )
        for k in k_values:
            state = evaluate_support(routing, kinds, post_centroids, k, include_distances=False)
            reachable = np.asarray(state.pop("reachable_mask"), dtype=bool)
            reachability[k].append(reachable)
            components[k].append(int(state["component_count"]))
            refresh_rows.append(
                {
                    "map": spec.map_name,
                    "seed": spec.seed,
                    "evaluation": evaluation,
                    "k": k,
                    **state,
                }
            )

    seed_rows: list[dict[str, Any]] = []
    for k in k_values:
        reachable_matrix = np.asarray(reachability[k], dtype=bool)
        per_centroid = np.mean(reachable_matrix, axis=0)
        component_values = np.asarray(components[k], dtype=np.int64)
        seed_rows.append(
            {
                "map": spec.map_name,
                "seed": spec.seed,
                "k": k,
                "refresh_count": len(post_refreshes),
                "mean_reachable_centroid_fraction": float(np.mean(reachable_matrix)),
                "min_refresh_reachable_centroid_fraction": float(
                    np.min(np.mean(reachable_matrix, axis=1))
                ),
                "final_reachable_centroid_fraction": float(np.mean(reachable_matrix[-1])),
                "never_reachable_centroids": int(np.count_nonzero(per_centroid == 0.0)),
                "unreachable_more_than_half_centroids": int(np.count_nonzero(per_centroid < 0.5)),
                "unreachable_at_least_90_percent_centroids": int(
                    np.count_nonzero(per_centroid <= 0.1)
                ),
                "connected_refresh_count": int(np.count_nonzero(component_values == 1)),
                "connected_refresh_fraction": float(np.mean(component_values == 1)),
                "component_count_mean": float(np.mean(component_values)),
                "component_count_max": int(np.max(component_values)),
            }
        )
    return refresh_rows, snapshot_rows, seed_rows


def load_run(spec: RunSpec) -> RunData:
    config = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
    rows = read_csv(spec.contribution_dir / "evaluations.csv")
    run_summary = json.loads(
        (spec.contribution_dir / "phase4_summary.json").read_text(encoding="utf-8")
    )
    algo = config["algorithm"]
    geodesic = config["geodesic"]
    return RunData(
        rows=rows,
        online_latents=np.asarray(
            [[float(row["latent_0"]), float(row["latent_1"])] for row in rows],
            dtype=np.float64,
        ),
        final_latents=read_points(spec.contribution_dir / "visited_latents_final_space.csv"),
        fitness=np.asarray([float(row["fitness"]) for row in rows], dtype=np.float64),
        bins=(int(algo["grid_bins"][0]), int(algo["grid_bins"][1])),
        bootstrap=int(algo["bootstrap_evaluations"]),
        retrain=min(int(value) for value in algo["retrain_evaluations"]),
        total=int(algo["total_evaluations"]),
        padding=float(config["autoencoder"]["latent_bounds_padding_fraction"]),
        old_graph_k=int(geodesic["graph_k"]),
        old_assignment_k=int(geodesic["assignment_k"]),
        old_rolling_size=int(geodesic["max_graph_points"]),
        refresh_interval=int(geodesic["refresh_interval"]),
        expected_retrain_elites=int(
            run_summary["retrain_events"][0]["rebuilt_archive_filled_cells"]
        ),
        expected_final_elites=int(run_summary["final_metrics"]["filled_cells"]),
    )


def reconstruct_pre_elites(data: RunData, maximum: int) -> dict[int, np.ndarray]:
    requested = {value for value in [2000, maximum] if value <= maximum}
    elites: dict[int, int] = {}
    output: dict[int, np.ndarray] = {}
    for index in range(maximum):
        update_elite(elites, saved_cell_index(data.rows[index], data.bins), index, data.fitness)
        evaluation = index + 1
        if evaluation in requested:
            output[evaluation] = np.asarray(sorted(elites.values()), dtype=np.int64)
    return output


def reconstruct_post_retrain_elites(
    data: RunData,
    centroids: np.ndarray,
) -> dict[int, np.ndarray]:
    rolling = data.final_latents[max(0, data.retrain - data.old_rolling_size) : data.retrain]
    state = build_graph_state(rolling, centroids, data.old_graph_k)
    elites: dict[int, int] = {}
    for index in range(data.retrain):
        chosen, _ = assign_query(state, data.final_latents[index], data.old_assignment_k)
        update_elite(elites, chosen, index, data.fitness)

    output = {data.retrain: np.asarray(sorted(elites.values()), dtype=np.int64)}
    next_refresh = data.retrain + data.refresh_interval
    for index in range(data.retrain, data.total):
        update_elite(elites, saved_cell_index(data.rows[index], data.bins), index, data.fitness)
        evaluation = index + 1
        if evaluation == next_refresh:
            output[evaluation] = np.asarray(sorted(elites.values()), dtype=np.int64)
            next_refresh += data.refresh_interval
    if data.total not in output:
        output[data.total] = np.asarray(sorted(elites.values()), dtype=np.int64)
    return output


def update_elite(
    elites: dict[int, int],
    cell: int,
    candidate_index: int,
    fitness: np.ndarray,
) -> None:
    if cell < 0:
        return
    incumbent = elites.get(cell)
    if incumbent is None or fitness[candidate_index] > fitness[incumbent]:
        elites[cell] = candidate_index


def support_at(
    latents: np.ndarray,
    elite_indices: np.ndarray,
    evaluation: int,
    rolling_buffer_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    elites = latents[np.asarray(elite_indices, dtype=np.int64)]
    rolling = latents[max(0, evaluation - rolling_buffer_size) : evaluation]
    return compose_real_routing_support(elites, rolling)


def evaluate_support(
    routing: np.ndarray,
    routing_kinds: np.ndarray,
    centroids: np.ndarray,
    k: int,
    *,
    include_distances: bool,
) -> dict[str, Any]:
    support = np.vstack([routing, centroids])
    endpoint_mask = np.concatenate(
        [np.zeros(len(routing), dtype=bool), np.ones(len(centroids), dtype=bool)]
    )
    graph = build_endpoint_manifold_graph(support, endpoint_mask, k)
    routing_graph = graph[: len(routing), : len(routing)]
    component_count, labels = connected_components(routing_graph, directed=False)
    component_sizes = np.bincount(labels)
    attachments = graph[len(routing) :, : len(routing)].tocsr()
    reachable = np.diff(attachments.indptr) > 0
    result: dict[str, Any] = {
        "routing_point_count": len(routing),
        "archive_elite_point_count": int(np.count_nonzero(routing_kinds == "archive_elite")),
        "rolling_point_count": int(np.count_nonzero(routing_kinds == "rolling_buffer")),
        "component_count": int(component_count),
        "largest_component_fraction": float(np.max(component_sizes) / len(routing)),
        "reachable_centroid_count": int(np.count_nonzero(reachable)),
        "reachable_centroid_fraction": float(np.mean(reachable)),
        "reachable_mask": reachable,
    }
    if include_distances:
        geodesic = endpoint_safe_all_pairs_distances(graph, endpoint_mask)
        euclidean = _pairwise_euclidean_matrix(support)
        tri = np.triu_indices(len(support), k=1)
        finite = np.isfinite(geodesic[tri])
        mean_euclidean = float(np.mean(euclidean[tri][finite]))
        mean_geodesic = float(np.mean(geodesic[tri][finite]))
        result.update(
            {
                "support_finite_pair_fraction": float(np.mean(finite)),
                "support_mean_euclidean_finite": mean_euclidean,
                "support_mean_geodesic_finite": mean_geodesic,
                "support_geodesic_to_euclidean_ratio": mean_geodesic / mean_euclidean,
            }
        )
    return result


def run_static(spec: RunSpec, k_values: list[int], output: list[dict[str, Any]]) -> None:
    sample = np.load(spec.baseline_sample)
    rows, _ = run_k_sensitivity(
        sample["latents"].astype(np.float64),
        sample["evaluations"].astype(np.int64),
        k_values,
    )
    output.extend({"map": spec.map_name, "seed": spec.seed, **row} for row in rows)


def aggregate_static(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_numeric_rows(
        rows,
        group_keys=["map", "k"],
        value_keys=[
            "component_count",
            "finite_pair_fraction",
            "ratio_mean_geodesic_to_mean_euclidean",
        ],
    )


def aggregate_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_numeric_rows(
        rows,
        group_keys=["map", "snapshot_evaluation", "k"],
        value_keys=[
            "component_count",
            "support_finite_pair_fraction",
            "support_geodesic_to_euclidean_ratio",
            "reachable_centroid_fraction",
            "archive_elite_point_count",
            "rolling_point_count",
        ],
    )


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_numeric_rows(
        rows,
        group_keys=["map", "k"],
        value_keys=[
            "mean_reachable_centroid_fraction",
            "min_refresh_reachable_centroid_fraction",
            "final_reachable_centroid_fraction",
            "never_reachable_centroids",
            "unreachable_more_than_half_centroids",
            "unreachable_at_least_90_percent_centroids",
            "connected_refresh_fraction",
            "component_count_mean",
            "component_count_max",
        ],
    )


def compare_with_rolling_only(
    revised_rows: list[dict[str, Any]],
    old_rows: list[dict[str, str]],
    *,
    comparison_k: int,
) -> list[dict[str, Any]]:
    old_lookup = {(row["map"], int(row["seed"])): row for row in old_rows}
    output: list[dict[str, Any]] = []
    for row in revised_rows:
        if int(row["k"]) != comparison_k:
            continue
        old = old_lookup[(str(row["map"]), int(row["seed"]))]
        result: dict[str, Any] = {
            "map": row["map"],
            "seed": row["seed"],
            "k": comparison_k,
        }
        for revised_key, old_key in [
            (
                "mean_reachable_centroid_fraction",
                "post_mean_reachable_centroid_fraction",
            ),
            ("never_reachable_centroids", "post_never_reachable_centroids"),
            (
                "unreachable_more_than_half_centroids",
                "post_unreachable_more_than_half_centroids",
            ),
        ]:
            old_value = float(old[old_key])
            revised_value = float(row[revised_key])
            result[f"rolling_only_{revised_key}"] = old_value
            result[f"elite_support_{revised_key}"] = revised_value
            result[f"delta_{revised_key}"] = revised_value - old_value
        output.append(result)
    return output


def aggregate_numeric_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    value_keys: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, subset in sorted(groups.items()):
        result = {key: value for key, value in zip(group_keys, group, strict=True)}
        result["seed_count"] = len(subset)
        for key in value_keys:
            values = np.asarray([float(row[key]) for row in subset], dtype=np.float64)
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            result[f"{key}_min"] = float(np.min(values))
            result[f"{key}_max"] = float(np.max(values))
        output.append(result)
    return output


def save_plots(
    seed_aggregate: list[dict[str, Any]],
    snapshot_aggregate: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    colors = {"horseshoe": "#c2410c", "open": "#2563eb"}
    for map_name in ["horseshoe", "open"]:
        rows = [row for row in seed_aggregate if row["map"] == map_name]
        k = [int(row["k"]) for row in rows]
        axes[0, 0].plot(
            k,
            [row["mean_reachable_centroid_fraction_mean"] for row in rows],
            marker="o",
            color=colors[map_name],
            label=map_name,
        )
        axes[0, 1].plot(
            k,
            [row["never_reachable_centroids_mean"] for row in rows],
            marker="o",
            color=colors[map_name],
            label=map_name,
        )
        axes[1, 0].plot(
            k,
            [row["unreachable_more_than_half_centroids_mean"] for row in rows],
            marker="o",
            color=colors[map_name],
            label=map_name,
        )
        final_rows = [
            row
            for row in snapshot_aggregate
            if row["map"] == map_name and int(row["snapshot_evaluation"]) == 20000
        ]
        axes[1, 1].plot(
            [int(row["k"]) for row in final_rows],
            [row["support_geodesic_to_euclidean_ratio_mean"] for row in final_rows],
            marker="o",
            color=colors[map_name],
            label=map_name,
        )
    axes[0, 0].set_ylabel("Mean reachable-centroid fraction")
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 1].set_ylabel("Never-reachable centroids")
    axes[1, 0].set_ylabel("Unreachable >50% of refreshes")
    axes[1, 1].set_ylabel("20k geodesic / Euclidean ratio")
    for axis in axes.flat:
        axis.set_xlabel("k")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    paths = [
        output_dir / "elite_support_connectivity_gate.png",
        output_dir / "elite_support_connectivity_gate.pdf",
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


def make_centroids(
    bd_min: np.ndarray,
    bd_max: np.ndarray,
    *,
    bins: tuple[int, int],
) -> np.ndarray:
    xs = np.linspace(bd_min[0], bd_max[0], bins[0], endpoint=False)
    ys = np.linspace(bd_min[1], bd_max[1], bins[1], endpoint=False)
    step = (bd_max - bd_min) / np.asarray(bins, dtype=np.float64)
    return np.asarray([[x + step[0] / 2, y + step[1] / 2] for x in xs for y in ys])


def saved_cell_index(row: dict[str, str], bins: tuple[int, int]) -> int:
    cell_x = int(row["cell_x"])
    cell_y = int(row["cell_y"])
    if cell_x < 0 or cell_y < 0:
        return -1
    return cell_x * bins[1] + cell_y


def read_points(path: Path) -> np.ndarray:
    return np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in read_csv(path)],
        dtype=np.float64,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def print_run_summary(spec: RunSpec, rows: list[dict[str, Any]]) -> None:
    print(f"{spec.map_name} seed {spec.seed}")
    for row in rows:
        print(
            "  k={k:>2}: reachable={mean_reachable_centroid_fraction:.3%}, "
            "never={never_reachable_centroids}, >50%={unreachable_more_than_half_centroids}, "
            "connected={connected_refresh_count}/{refresh_count}".format(**row)
        )


if __name__ == "__main__":
    main()
