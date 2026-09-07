"""Check k-NN graph curvature sensitivity on locked Baseline B latent samples."""

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
from scipy.sparse.csgraph import connected_components, dijkstra

from algorithms.archive_metrics import _pairwise_euclidean_matrix
from algorithms.knn_graph import build_undirected_knn_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-npz",
        default=(
            "results/phase5/baseline_b_learned_bd_euclidean/seed_1001/"
            "representative_trajectory_sample.npz"
        ),
    )
    parser.add_argument("--output-dir", default="reproduced/phase3_knn_k_sensitivity")
    parser.add_argument("--k-values", default="3,5,10,20,30")
    args = parser.parse_args()

    sample_npz = Path(args.sample_npz)
    if not sample_npz.exists():
        raise FileNotFoundError(
            f"Missing representative sample: {sample_npz}. "
            "Pass --sample-npz for another completed run."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    if not k_values:
        raise ValueError("--k-values must contain at least one integer.")

    sample = np.load(sample_npz)
    latents = sample["latents"].astype(np.float64)
    evaluations = sample["evaluations"].astype(np.int64)
    rows, summary = run_k_sensitivity(latents, evaluations, k_values)

    csv_path = output_dir / "phase3_knn_k_sensitivity.csv"
    json_path = output_dir / "phase3_knn_k_sensitivity_summary.json"
    plot_paths = save_plot(rows, output_dir / "phase3_knn_k_sensitivity")
    write_csv(csv_path, rows)
    summary["sample_npz"] = str(sample_npz)
    summary["output_files"] = [str(csv_path), *plot_paths, str(json_path)]
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, rows)


def run_k_sensitivity(
    latents: np.ndarray,
    evaluations: np.ndarray,
    k_values: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    distances = _pairwise_euclidean_matrix(latents)
    tri = np.triu_indices(len(latents), k=1)
    total_pairs = len(distances[tri])
    rows: list[dict[str, Any]] = []
    for k in k_values:
        graph = build_undirected_knn_graph(latents, k)
        geodesic = dijkstra(graph, directed=False)
        finite = np.isfinite(geodesic[tri])
        finite_euclidean = distances[tri][finite]
        finite_geodesic = geodesic[tri][finite]
        per_pair_ratio = finite_geodesic / np.maximum(finite_euclidean, 1e-12)
        _, component_labels = connected_components(graph, directed=False)
        component_sizes = np.bincount(component_labels).tolist()
        mean_euclidean_finite = (
            float(np.mean(finite_euclidean)) if len(finite_euclidean) else float("nan")
        )
        mean_geodesic_finite = (
            float(np.mean(finite_geodesic)) if len(finite_geodesic) else float("nan")
        )
        rows.append(
            {
                "k": int(min(max(1, k), len(latents) - 1)),
                "point_count": int(len(latents)),
                "total_pair_count": int(total_pairs),
                "finite_pair_count": int(np.count_nonzero(finite)),
                "disconnected_pair_count": int(np.count_nonzero(~finite)),
                "finite_pair_fraction": float(np.mean(finite)),
                "component_count": int(len(component_sizes)),
                "largest_component_fraction": float(max(component_sizes) / len(latents)),
                "mean_euclidean_all_pairs": float(np.mean(distances[tri])),
                "mean_euclidean_finite_pairs": mean_euclidean_finite,
                "mean_knn_geodesic_finite_pairs": mean_geodesic_finite,
                "ratio_mean_geodesic_to_mean_euclidean": (
                    mean_geodesic_finite / mean_euclidean_finite
                    if np.isfinite(mean_euclidean_finite) and mean_euclidean_finite > 0.0
                    else float("nan")
                ),
                "mean_per_pair_geodesic_to_euclidean_ratio": (
                    float(np.mean(per_pair_ratio)) if len(per_pair_ratio) else float("nan")
                ),
                "median_per_pair_geodesic_to_euclidean_ratio": (
                    float(np.median(per_pair_ratio)) if len(per_pair_ratio) else float("nan")
                ),
                "p90_per_pair_geodesic_to_euclidean_ratio": (
                    float(np.quantile(per_pair_ratio, 0.90))
                    if len(per_pair_ratio)
                    else float("nan")
                ),
                "p95_per_pair_geodesic_to_euclidean_ratio": (
                    float(np.quantile(per_pair_ratio, 0.95))
                    if len(per_pair_ratio)
                    else float("nan")
                ),
                "max_per_pair_geodesic_to_euclidean_ratio": (
                    float(np.max(per_pair_ratio)) if len(per_pair_ratio) else float("nan")
                ),
            }
        )
    summary = {
        "diagnostic": "Phase 3 locked Baseline B representative latent k-NN sensitivity",
        "point_count": int(len(latents)),
        "evaluation_min": int(np.min(evaluations)),
        "evaluation_max": int(np.max(evaluations)),
        "k_values": [int(row["k"]) for row in rows],
        "distance_definition": (
            "Euclidean distances are direct latent-code distances. Geodesic distances are shortest "
            "paths on the same undirected symmetrized k-NN graph convention used by archive "
            "metrics. Ratios are computed over finite connected pairs for each k."
        ),
        "rows": rows,
    }
    return rows, summary


def connected_component_sizes(graph: list[list[tuple[int, float]]]) -> list[int]:
    seen = np.zeros(len(graph), dtype=bool)
    sizes: list[int] = []
    for start in range(len(graph)):
        if bool(seen[start]):
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor, _weight in graph[node]:
                if not bool(seen[neighbor]):
                    seen[neighbor] = True
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def save_plot(rows: list[dict[str, Any]], output_stem: Path) -> list[str]:
    k_values = np.asarray([int(row["k"]) for row in rows], dtype=int)
    ratio = np.asarray(
        [float(row["ratio_mean_geodesic_to_mean_euclidean"]) for row in rows], dtype=float
    )
    finite_fraction = np.asarray([float(row["finite_pair_fraction"]) for row in rows], dtype=float)
    p95 = np.asarray(
        [float(row["p95_per_pair_geodesic_to_euclidean_ratio"]) for row in rows], dtype=float
    )

    fig, ax_ratio = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax_ratio.plot(k_values, ratio, marker="o", color="#2563eb", label="ratio of means")
    ax_ratio.plot(k_values, p95, marker="s", color="#dc2626", label="p95 pair ratio")
    ax_ratio.axhline(1.0, color="#64748b", linewidth=1.0, linestyle="--")
    ax_ratio.set_xlabel("k in k-NN graph")
    ax_ratio.set_ylabel("geodesic / Euclidean distance ratio")
    ax_ratio.set_title("Baseline B representative latent graph k-sensitivity")
    ax_ratio.grid(alpha=0.25)

    ax_finite = ax_ratio.twinx()
    ax_finite.plot(
        k_values, finite_fraction, marker="^", color="#16a34a", label="finite pair fraction"
    )
    ax_finite.set_ylabel("finite pair fraction")
    ax_finite.set_ylim(0.0, 1.05)

    lines, labels = ax_ratio.get_legend_handles_labels()
    lines2, labels2 = ax_finite.get_legend_handles_labels()
    ax_ratio.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

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
    print("PHASE 3 K-NN LATENT GEODESIC SENSITIVITY")
    print(f"sample: {summary['sample_npz']}")
    print(
        f"points: {summary['point_count']} "
        f"(evaluations {summary['evaluation_min']} to {summary['evaluation_max']})"
    )
    print(
        "k | finite | components | mean_euc | mean_geo | geo/euc | p95 pair ratio | max pair ratio"
    )
    for row in rows:
        print(
            "{k:>2} | {finite_pair_fraction:.3f} | {component_count:>2} | "
            "{mean_euclidean_finite_pairs:.3f} | {mean_knn_geodesic_finite_pairs:.3f} | "
            "{ratio_mean_geodesic_to_mean_euclidean:.3f} | "
            "{p95_per_pair_geodesic_to_euclidean_ratio:.3f} | "
            "{max_per_pair_geodesic_to_euclidean_ratio:.3f}".format(**row)
        )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
