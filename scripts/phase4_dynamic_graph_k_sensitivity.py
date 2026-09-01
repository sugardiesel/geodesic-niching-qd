"""Simulate Phase 4 rolling-buffer graph k-sensitivity at saved run snapshots."""

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
import yaml
from scipy.sparse.csgraph import connected_components

from algorithms.knn_graph import (
    build_endpoint_manifold_graph,
    endpoint_safe_all_pairs_distances,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-dir", default="results/phase4_geodesic_niching_seed_1001")
    parser.add_argument("--config", default="configs/phase4_geodesic_niching.yaml")
    parser.add_argument("--output-dir", default="results/phase4_dynamic_graph_k_sensitivity")
    parser.add_argument("--snapshots", default="2000,10000,20000")
    parser.add_argument("--k-values", default="3,5,10,20,30")
    parser.add_argument("--rolling-buffer-size", type=int, default=1000)
    args = parser.parse_args()

    phase4_dir = Path(args.phase4_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{args.config} must contain a mapping.")

    snapshots = [int(value.strip()) for value in args.snapshots.split(",") if value.strip()]
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    if not snapshots or not k_values:
        raise ValueError("--snapshots and --k-values must both be non-empty.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_retrain_latents = read_evaluation_latents(phase4_dir / "evaluations.csv")
    final_space_latents = read_point_csv(phase4_dir / "visited_latents_final_space.csv")
    rows, summary = run_dynamic_sensitivity(
        pre_retrain_latents=pre_retrain_latents,
        final_space_latents=final_space_latents,
        config=config,
        snapshots=snapshots,
        k_values=k_values,
        rolling_buffer_size=int(args.rolling_buffer_size),
    )

    csv_path = output_dir / "phase4_dynamic_graph_k_sensitivity.csv"
    json_path = output_dir / "phase4_dynamic_graph_k_sensitivity_summary.json"
    plot_paths = save_plot(rows, output_dir / "phase4_dynamic_graph_k_sensitivity")
    write_csv(csv_path, rows)
    summary.update(
        {
            "phase4_dir": str(phase4_dir),
            "config": str(args.config),
            "output_files": [str(csv_path), *plot_paths, str(json_path)],
        }
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, rows)


def run_dynamic_sensitivity(
    pre_retrain_latents: np.ndarray,
    final_space_latents: np.ndarray,
    config: dict[str, Any],
    snapshots: list[int],
    k_values: list[int],
    rolling_buffer_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algo = config["algorithm"]
    autoencoder_cfg = config["autoencoder"]
    retrain_points = [int(value) for value in algo.get("retrain_evaluations", [])]
    first_retrain = min(retrain_points) if retrain_points else int(algo["total_evaluations"]) + 1
    bins = (int(algo["grid_bins"][0]), int(algo["grid_bins"][1]))
    padding = float(autoencoder_cfg["latent_bounds_padding_fraction"])

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        latent_stream, bounds_basis, encoder_space = snapshot_latents(
            snapshot=snapshot,
            first_retrain=first_retrain,
            pre_retrain_latents=pre_retrain_latents,
            final_space_latents=final_space_latents,
        )
        if len(latent_stream) == 0:
            raise ValueError(f"No latent points available for snapshot {snapshot}.")
        rolling = latent_stream[max(0, len(latent_stream) - rolling_buffer_size) :]
        bd_min, bd_max = latent_bounds(bounds_basis, padding)
        centroids = make_centroids(bd_min, bd_max, bins)
        support = np.vstack([rolling, centroids])
        support_kind = np.concatenate(
            [np.zeros(len(rolling), dtype=bool), np.ones(len(centroids), dtype=bool)]
        )

        for k in k_values:
            rows.append(
                evaluate_graph(
                    snapshot=snapshot,
                    encoder_space=encoder_space,
                    k=k,
                    support=support,
                    support_is_centroid=support_kind,
                    rolling_count=len(rolling),
                    centroid_count=len(centroids),
                )
            )

    summary = {
        "diagnostic": "Phase 4 dynamic rolling-buffer graph k-sensitivity",
        "snapshots": snapshots,
        "k_values": k_values,
        "rolling_buffer_size": int(rolling_buffer_size),
        "distance_definition": (
            "Each snapshot support set is fixed archive grid centroids plus the last "
            "rolling-buffer "
            "latents available at that point. For 2k the pre-retrain encoder latents are used; "
            "for 10k/20k the saved final one-retrain latent stream is used. Geodesic distances "
            "are shortest paths on a deduplicated, union-symmetrized k-NN graph. Centroids are "
            "endpoint "
            "nodes and cannot act as transit points through unvisited regions."
        ),
        "rows": rows,
    }
    return rows, summary


def snapshot_latents(
    snapshot: int,
    first_retrain: int,
    pre_retrain_latents: np.ndarray,
    final_space_latents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if snapshot < first_retrain:
        stream = pre_retrain_latents[:snapshot]
        return stream, stream, "pre_retrain_encoder"
    stream = final_space_latents[:snapshot]
    bounds_basis = final_space_latents[:first_retrain]
    return stream, bounds_basis, "final_one_retrain_encoder"


def evaluate_graph(
    snapshot: int,
    encoder_space: str,
    k: int,
    support: np.ndarray,
    support_is_centroid: np.ndarray,
    rolling_count: int,
    centroid_count: int,
) -> dict[str, Any]:
    graph = build_endpoint_manifold_graph(support, support_is_centroid, k)
    routing_indices = np.flatnonzero(~support_is_centroid)
    routing_graph = graph[routing_indices][:, routing_indices]
    component_count, component_labels = connected_components(routing_graph, directed=False)
    component_sizes = np.bincount(component_labels)
    geodesic = endpoint_safe_all_pairs_distances(graph, support_is_centroid)
    euclidean = pairwise_euclidean(support)

    support_stats = distance_stats(euclidean, geodesic, np.arange(len(support)))
    rolling_stats = distance_stats(euclidean, geodesic, np.arange(rolling_count))
    centroid_indices = np.flatnonzero(support_is_centroid)
    centroid_stats = distance_stats(euclidean, geodesic, centroid_indices)
    return {
        "snapshot_evaluation": int(snapshot),
        "encoder_space": encoder_space,
        "k": int(min(max(1, k), len(support) - 1)),
        "support_point_count": int(len(support)),
        "rolling_buffer_points": int(rolling_count),
        "centroid_points": int(centroid_count),
        "component_count": int(component_count),
        "largest_component_fraction": float(np.max(component_sizes) / len(routing_indices)),
        "support_finite_pair_fraction": support_stats["finite_pair_fraction"],
        "support_mean_euclidean": support_stats["mean_euclidean"],
        "support_mean_geodesic": support_stats["mean_geodesic"],
        "support_geodesic_to_euclidean_ratio": support_stats["ratio"],
        "rolling_finite_pair_fraction": rolling_stats["finite_pair_fraction"],
        "rolling_geodesic_to_euclidean_ratio": rolling_stats["ratio"],
        "centroid_finite_pair_fraction": centroid_stats["finite_pair_fraction"],
        "centroid_geodesic_to_euclidean_ratio": centroid_stats["ratio"],
    }


def pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def distance_stats(
    euclidean: np.ndarray, geodesic: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    if len(indices) < 2:
        return {
            "finite_pair_fraction": 0.0,
            "mean_euclidean": float("nan"),
            "mean_geodesic": float("nan"),
            "ratio": float("nan"),
        }
    sub_euclidean = euclidean[np.ix_(indices, indices)]
    sub_geodesic = geodesic[np.ix_(indices, indices)]
    tri = np.triu_indices(len(indices), k=1)
    finite = np.isfinite(sub_geodesic[tri])
    if not np.any(finite):
        return {
            "finite_pair_fraction": 0.0,
            "mean_euclidean": float("nan"),
            "mean_geodesic": float("nan"),
            "ratio": float("nan"),
        }
    mean_euclidean = float(np.mean(sub_euclidean[tri][finite]))
    mean_geodesic = float(np.mean(sub_geodesic[tri][finite]))
    return {
        "finite_pair_fraction": float(np.mean(finite)),
        "mean_euclidean": mean_euclidean,
        "mean_geodesic": mean_geodesic,
        "ratio": mean_geodesic / mean_euclidean if mean_euclidean > 0.0 else float("nan"),
    }


def latent_bounds(latents: np.ndarray, padding_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.min(latents, axis=0)
    maximum = np.max(latents, axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    padding = span * float(padding_fraction)
    return minimum - padding, maximum + padding


def make_centroids(bd_min: np.ndarray, bd_max: np.ndarray, bins: tuple[int, int]) -> np.ndarray:
    xs = np.linspace(bd_min[0], bd_max[0], bins[0], endpoint=False, dtype=np.float64)
    ys = np.linspace(bd_min[1], bd_max[1], bins[1], endpoint=False, dtype=np.float64)
    step = (bd_max - bd_min) / np.asarray(bins, dtype=np.float64)
    xs = xs + step[0] * 0.5
    ys = ys + step[1] * 0.5
    return np.asarray([[x, y] for x in xs for y in ys], dtype=np.float64)


def read_evaluation_latents(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["latent_0"] == "" or row["latent_1"] == "":
                continue
            points.append([float(row["latent_0"]), float(row["latent_1"])])
    return np.asarray(points, dtype=np.float64)


def read_point_csv(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append([float(row["latent_0"]), float(row["latent_1"])])
    return np.asarray(points, dtype=np.float64)


def save_plot(rows: list[dict[str, Any]], output_stem: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    snapshots = sorted({int(row["snapshot_evaluation"]) for row in rows})
    colors = {2000: "#2563eb", 10000: "#dc2626", 20000: "#16a34a"}
    for snapshot in snapshots:
        subset = [row for row in rows if int(row["snapshot_evaluation"]) == snapshot]
        subset = sorted(subset, key=lambda row: int(row["k"]))
        k_values = [int(row["k"]) for row in subset]
        ratios = [float(row["support_geodesic_to_euclidean_ratio"]) for row in subset]
        ax.plot(
            k_values,
            ratios,
            marker="o",
            color=colors.get(snapshot, None),
            label=f"eval {snapshot}",
        )
    ax.axhline(1.0, color="#64748b", linewidth=1.0, linestyle="--")
    ax.set_xlabel("k in k-NN graph")
    ax.set_ylabel("support geodesic / Euclidean ratio")
    ax.set_title("Dynamic Phase 4 graph k-sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print("PHASE 4 DYNAMIC GRAPH K-SENSITIVITY")
    print(f"phase4_dir: {summary['phase4_dir']}")
    print(f"rolling_buffer_size: {summary['rolling_buffer_size']}")
    print("snapshot | k | finite | components | support ratio | rolling ratio | centroid ratio")
    for row in rows:
        print(
            "{snapshot_evaluation:>8} | {k:>2} | {support_finite_pair_fraction:.3f} | "
            "{component_count:>2} | {support_geodesic_to_euclidean_ratio:.3f} | "
            "{rolling_geodesic_to_euclidean_ratio:.3f} | "
            "{centroid_geodesic_to_euclidean_ratio:.3f}".format(**row)
        )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
