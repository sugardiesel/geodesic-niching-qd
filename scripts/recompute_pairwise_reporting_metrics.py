"""Recompute final-archive pairwise reporting metrics under one shared k."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from algorithms.archive_metrics import pairwise_distance_metrics

SUMMARY_FILES = {
    "baseline_a_handcrafted_map_elites": "phase2_summary.json",
    "baseline_b_learned_bd_euclidean": "phase3_summary.json",
    "contribution_geodesic_niching": "phase4_summary.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["results/phase5", "results/phase6_open_robustness"],
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--audit-csv",
        default="results/audit_fix/pairwise_reporting_k20_recompute.csv",
    )
    args = parser.parse_args()

    audit_rows: list[dict[str, Any]] = []
    for root_value in args.roots:
        root = Path(root_value)
        for condition, summary_name in SUMMARY_FILES.items():
            condition_dir = root / condition
            for seed_dir in sorted(condition_dir.glob("seed_*")):
                audit_rows.append(recompute_seed(root, condition, seed_dir, summary_name, args.k))
    write_csv(Path(args.audit_csv), audit_rows)
    for row in audit_rows:
        print(
            "{dataset} {condition} seed={seed}: k {old_k}->{new_k}, "
            "geodesic {old_geodesic:.6f}->{new_geodesic:.6f}".format(**row)
        )
    print(f"audit table: {args.audit_csv}")


def recompute_seed(
    root: Path,
    condition: str,
    seed_dir: Path,
    summary_name: str,
    k: int,
) -> dict[str, Any]:
    points = read_descriptors(seed_dir / "archive_cells.csv")
    pairwise = pairwise_distance_metrics(points, k)
    summary_path = seed_dir / summary_name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final = summary["final_metrics"]
    row = {
        "dataset": root.name,
        "condition": condition,
        "seed": int(seed_dir.name.removeprefix("seed_")),
        "elite_count": len(points),
        "old_k": int(final["pairwise_geodesic_k"]),
        "new_k": int(pairwise.knn_k),
        "old_euclidean": float(final["mean_pairwise_descriptor_euclidean"]),
        "new_euclidean": pairwise.mean_euclidean,
        "old_geodesic": float(final["mean_pairwise_descriptor_knn_geodesic"]),
        "new_geodesic": pairwise.mean_knn_geodesic,
        "finite_fraction": pairwise.finite_geodesic_fraction,
        "disconnected_pairs": pairwise.disconnected_pair_count,
    }
    final.update(
        {
            "mean_pairwise_descriptor_euclidean": pairwise.mean_euclidean,
            "mean_pairwise_descriptor_knn_geodesic": pairwise.mean_knn_geodesic,
            "pairwise_geodesic_k": pairwise.knn_k,
            "pairwise_geodesic_finite_fraction": pairwise.finite_geodesic_fraction,
            "pairwise_geodesic_disconnected_pairs": pairwise.disconnected_pair_count,
        }
    )
    summary["pairwise_reporting_metric_recomputed_posthoc"] = {
        "k": int(pairwise.knn_k),
        "source": "final archive_cells.csv descriptors",
        "reason": "Shared post-audit reporting convention across all conditions.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return row


def read_descriptors(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray(
        [[float(row["descriptor_x"]), float(row["descriptor_y"])] for row in rows],
        dtype=np.float64,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
