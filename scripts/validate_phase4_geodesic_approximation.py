"""Validate Phase 4 approximate geodesic distances against the full visited-point graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from algorithms.knn_graph import (
    build_endpoint_manifold_graph,
    build_undirected_knn_graph,
    endpoint_shortest_path_distances,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-dir", default="results/phase4_geodesic_niching_seed_1001")
    parser.add_argument("--config", default="configs/phase4_geodesic_niching.yaml")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    output_dir = Path(args.phase4_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    graph_k = int(config["geodesic"]["graph_k"])
    attach_k = int(config["geodesic"]["assignment_k"])

    archive_points = read_points(output_dir / "archive_cells.csv", "descriptor_x", "descriptor_y")
    support_points, support_endpoint_mask = read_support_points(
        output_dir / "geodesic_graph_support_points.csv"
    )
    full_points = read_points(
        output_dir / "visited_latents_final_space.csv", "latent_0", "latent_1"
    )

    start = time.perf_counter()
    pair_indices = sample_pairs(len(archive_points), int(args.pairs), int(args.seed))
    unique_indices = np.unique(pair_indices.reshape(-1))
    query_points = archive_points[unique_indices]
    archive_to_query = {
        int(archive_idx): int(query_idx) for query_idx, archive_idx in enumerate(unique_indices)
    }

    approximate_distances = query_pair_distances(
        support_points,
        query_points,
        pair_indices,
        archive_to_query,
        graph_k=graph_k,
        attach_k=attach_k,
        endpoint_mask=support_endpoint_mask,
    )
    exact_distances = query_pair_distances(
        full_points,
        query_points,
        pair_indices,
        archive_to_query,
        graph_k=graph_k,
        attach_k=attach_k,
        endpoint_mask=np.zeros(len(full_points), dtype=bool),
    )
    elapsed = time.perf_counter() - start

    rows = []
    for row_idx, ((left, right), approximate, exact) in enumerate(
        zip(pair_indices, approximate_distances, exact_distances, strict=True)
    ):
        rows.append(
            {
                "pair_index": row_idx,
                "archive_i": int(left),
                "archive_j": int(right),
                "approx_geodesic": float(approximate),
                "exact_full_graph_geodesic": float(exact),
                "absolute_error": float(abs(approximate - exact)),
                "relative_error": float(abs(approximate - exact) / exact) if exact > 1e-12 else 0.0,
            }
        )

    summary = summarize(rows)
    summary.update(
        {
            "phase4_dir": str(output_dir),
            "config": args.config,
            "sampled_pairs": int(len(pair_indices)),
            "sample_seed": int(args.seed),
            "archive_points": int(len(archive_points)),
            "unique_query_points": int(len(query_points)),
            "approx_support_points": int(len(support_points)),
            "full_graph_points": int(len(full_points)),
            "graph_k": graph_k,
            "attach_k": attach_k,
            "elapsed_seconds": elapsed,
            "interpretation": (
                "Approximate distances are computed on the same saved reservoir+centroid support "
                "graph used by the Phase 4 run. Exact distances are computed on a full k-NN graph "
                "over all final-space visited latent points."
            ),
        }
    )

    pairs_csv = output_dir / "geodesic_approximation_validation_pairs.csv"
    summary_json = output_dir / "geodesic_approximation_validation_summary.json"
    write_csv(pairs_csv, rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, pairs_csv, summary_json)


def read_points(path: Path, x_key: str, y_key: str) -> np.ndarray:
    rows = read_csv(path)
    points = [[float(row[x_key]), float(row[y_key])] for row in rows]
    return np.asarray(points, dtype=np.float64)


def read_support_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    points = np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in rows],
        dtype=np.float64,
    )
    endpoint_mask = np.asarray([row["kind"] == "centroid" for row in rows], dtype=bool)
    return points, endpoint_mask


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_pairs(point_count: int, pair_count: int, seed: int) -> np.ndarray:
    if point_count < 2:
        raise ValueError("Need at least two archive points for pair validation.")
    all_pairs = np.asarray(
        [(left, right) for left in range(point_count) for right in range(left + 1, point_count)],
        dtype=np.int32,
    )
    rng = np.random.default_rng(seed)
    count = min(pair_count, len(all_pairs))
    chosen = rng.choice(len(all_pairs), size=count, replace=False)
    return all_pairs[chosen]


def query_pair_distances(
    base_points: np.ndarray,
    query_points: np.ndarray,
    pair_indices: np.ndarray,
    archive_to_query: dict[int, int],
    graph_k: int,
    attach_k: int,
    endpoint_mask: np.ndarray,
) -> np.ndarray:
    if np.any(endpoint_mask):
        base_graph = build_endpoint_manifold_graph(base_points, endpoint_mask, graph_k)
    else:
        base_graph = build_undirected_knn_graph(base_points, graph_k)
    combined, combined_endpoint_mask = attach_queries(
        base_points,
        base_graph,
        endpoint_mask,
        query_points,
        attach_k,
    )
    base_count = len(base_points)
    endpoint_indices = np.flatnonzero(combined_endpoint_mask)
    endpoint_distances = endpoint_shortest_path_distances(combined, combined_endpoint_mask)
    endpoint_row = {int(node): row for row, node in enumerate(endpoint_indices)}
    values = []
    for left, right in pair_indices:
        left_query = archive_to_query[int(left)]
        right_query = archive_to_query[int(right)]
        left_node = base_count + left_query
        right_node = base_count + right_query
        values.append(float(endpoint_distances[endpoint_row[left_node], right_node]))
    return np.asarray(values, dtype=np.float64)


def attach_queries(
    base_points: np.ndarray,
    base_graph: csr_matrix,
    base_endpoint_mask: np.ndarray,
    query_points: np.ndarray,
    attach_k: int,
) -> tuple[csr_matrix, np.ndarray]:
    base_count = len(base_points)
    query_count = len(query_points)
    routing_indices = np.flatnonzero(~base_endpoint_mask)
    routing_points = base_points[routing_indices]
    tree = cKDTree(routing_points)
    k = min(max(1, int(attach_k)), len(routing_points))
    distances, indices = tree.query(query_points, k=k)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    if query_count == 1:
        distances = distances.reshape(1, -1)
        indices = indices.reshape(1, -1)

    coo = base_graph.tocoo()
    rows = coo.row.astype(np.int64).tolist()
    cols = coo.col.astype(np.int64).tolist()
    data = coo.data.astype(np.float64).tolist()
    for query_idx in range(query_count):
        query_node = base_count + query_idx
        for distance, tree_idx in zip(distances[query_idx], indices[query_idx], strict=False):
            base_idx = int(routing_indices[int(tree_idx)])
            rows.append(query_node)
            cols.append(base_idx)
            data.append(float(distance))
            rows.append(base_idx)
            cols.append(query_node)
            data.append(float(distance))
    total = base_count + query_count
    combined = csr_matrix((data, (rows, cols)), shape=(total, total), dtype=np.float64)
    endpoint_mask = np.concatenate((base_endpoint_mask, np.ones(query_count, dtype=bool)))
    return combined, endpoint_mask


def summarize(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    approximate = np.asarray([float(row["approx_geodesic"]) for row in rows], dtype=np.float64)
    exact = np.asarray([float(row["exact_full_graph_geodesic"]) for row in rows], dtype=np.float64)
    finite = np.isfinite(approximate) & np.isfinite(exact)
    approximate = approximate[finite]
    exact = exact[finite]
    errors = np.abs(approximate - exact)
    relative = errors / np.maximum(exact, 1e-12)
    if len(approximate) >= 2:
        pearson = float(np.corrcoef(approximate, exact)[0, 1])
    else:
        pearson = float("nan")
    return {
        "finite_pair_fraction": float(np.mean(finite)) if len(finite) else 0.0,
        "pearson_r": pearson,
        "mean_absolute_error": float(np.mean(errors)) if len(errors) else float("nan"),
        "median_absolute_error": float(np.median(errors)) if len(errors) else float("nan"),
        "root_mean_squared_error": float(np.sqrt(np.mean(errors**2)))
        if len(errors)
        else float("nan"),
        "mean_relative_error": float(np.mean(relative)) if len(relative) else float("nan"),
        "median_relative_error": float(np.median(relative)) if len(relative) else float("nan"),
        "max_absolute_error": float(np.max(errors)) if len(errors) else float("nan"),
        "approx_mean": float(np.mean(approximate)) if len(approximate) else float("nan"),
        "exact_mean": float(np.mean(exact)) if len(exact) else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any], pairs_csv: Path, summary_json: Path) -> None:
    print("PHASE 4 GEODESIC APPROXIMATION VALIDATION")
    print(f"sampled pairs: {summary['sampled_pairs']}")
    print(f"archive points: {summary['archive_points']}")
    print(f"approx support points: {summary['approx_support_points']}")
    print(f"full graph points: {summary['full_graph_points']}")
    print(f"finite pair fraction: {summary['finite_pair_fraction']:.3f}")
    print(f"Pearson r: {summary['pearson_r']:.4f}")
    print(f"MAE: {summary['mean_absolute_error']:.4f}")
    print(f"RMSE: {summary['root_mean_squared_error']:.4f}")
    print(f"mean relative error: {summary['mean_relative_error']:.3%}")
    print(f"median relative error: {summary['median_relative_error']:.3%}")
    print(f"max absolute error: {summary['max_absolute_error']:.4f}")
    print(f"elapsed seconds: {summary['elapsed_seconds']:.3f}")
    print(f"pairs csv: {pairs_csv}")
    print(f"summary json: {summary_json}")


if __name__ == "__main__":
    main()
