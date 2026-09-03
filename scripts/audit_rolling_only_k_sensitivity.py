# ruff: noqa: I001
"""Recompute full-refresh k sensitivity for rolling-only graph support."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_elite_support_connectivity import (
    RunSpec,
    aggregate_numeric_rows,
    evaluate_support,
    latent_bounds,
    load_run,
    make_centroids,
    read_csv,
    write_csv,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "audit_connectivity_verification"
    / "rolling_only_full_refresh_k_sensitivity"
)
SUMMARY_DATA = (
    ROOT
    / "results"
    / "final_summary"
    / "data"
    / "rolling_only_k_sensitivity_report.csv"
)
RATIO_SOURCE = (
    ROOT
    / "results"
    / "audit_connectivity_verification"
    / "penalty_audit"
    / "dynamic_k_sensitivity_aggregate.csv"
)
AUTHORITATIVE_K20 = (
    ROOT
    / "results"
    / "audit_connectivity_verification"
    / "penalty_audit"
    / "penalty_audit_aggregate.csv"
)
K_VALUES = [3, 5, 10, 20, 30]


def audit_run(spec: RunSpec) -> list[dict[str, Any]]:
    data = load_run(spec)
    centroids = make_centroids(
        *latent_bounds(data.final_latents[: data.retrain], data.padding), bins=data.bins
    )
    refreshes = list(range(data.retrain, data.total + 1, data.refresh_interval))
    reachability: dict[int, list[np.ndarray]] = {k: [] for k in K_VALUES}

    for evaluation in refreshes:
        routing = data.final_latents[
            max(0, evaluation - data.old_rolling_size) : evaluation
        ]
        kinds = np.full(len(routing), "rolling_buffer", dtype="U16")
        for k in K_VALUES:
            state = evaluate_support(
                routing,
                kinds,
                centroids,
                k,
                include_distances=False,
            )
            reachability[k].append(np.asarray(state["reachable_mask"], dtype=bool))

    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        reachable = np.asarray(reachability[k], dtype=bool)
        per_centroid = np.mean(reachable, axis=0)
        rows.append(
            {
                "map": spec.map_name,
                "seed": spec.seed,
                "k": k,
                "refresh_count": len(refreshes),
                "rolling_buffer_size": data.old_rolling_size,
                "mean_reachable_centroid_fraction": float(np.mean(reachable)),
                "never_reachable_centroids": int(np.count_nonzero(per_centroid == 0.0)),
                "unreachable_more_than_half_centroids": int(
                    np.count_nonzero(per_centroid < 0.5)
                ),
            }
        )
    return rows


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_numeric_rows(
        rows,
        group_keys=["map", "k"],
        value_keys=[
            "mean_reachable_centroid_fraction",
            "never_reachable_centroids",
            "unreachable_more_than_half_centroids",
        ],
    )


def merge_final_ratios(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ratio_rows = [
        row for row in read_csv(RATIO_SOURCE) if int(row["snapshot_evaluation"]) == 20000
    ]
    ratios = {(row["map"], int(row["k"])): row for row in ratio_rows}
    merged: list[dict[str, Any]] = []
    for row in aggregate:
        ratio = ratios[(str(row["map"]), int(row["k"]))]
        merged.append(
            {
                **row,
                "geodesic_to_euclidean_ratio_at_20000_mean": float(
                    ratio["support_geodesic_to_euclidean_ratio_mean"]
                ),
                "geodesic_to_euclidean_ratio_at_20000_std": float(
                    ratio["support_geodesic_to_euclidean_ratio_std"]
                ),
                "geodesic_to_euclidean_ratio_at_20000_min": float(
                    ratio["support_geodesic_to_euclidean_ratio_min"]
                ),
                "geodesic_to_euclidean_ratio_at_20000_max": float(
                    ratio["support_geodesic_to_euclidean_ratio_max"]
                ),
            }
        )
    return merged


def validate_k20(aggregate: list[dict[str, Any]]) -> None:
    expected = {row["map"]: row for row in read_csv(AUTHORITATIVE_K20)}
    for row in aggregate:
        if int(row["k"]) != 20:
            continue
        source = expected[str(row["map"])]
        checks = {
            "mean_reachable_centroid_fraction_mean": float(
                source["post_mean_reachable_centroid_fraction_mean"]
            ),
            "never_reachable_centroids_mean": float(
                source["post_never_reachable_centroids_mean"]
            ),
        }
        for key, expected_value in checks.items():
            actual = float(row[key])
            if not np.isclose(actual, expected_value, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"{row['map']} k=20 {key}: {actual} != authoritative {expected_value}"
                )


def register_summary_artifact(path: Path) -> None:
    manifest_path = ROOT / "results" / "final_summary" / "artifact_manifest.csv"
    rows = read_csv(manifest_path) if manifest_path.exists() else []
    relative_path = path.relative_to(ROOT / "results" / "final_summary").as_posix()
    rows = [row for row in rows if row["file"] != relative_path]
    rows.append(
        {
            "category": "data",
            "file": relative_path,
            "source": (
                "results/audit_connectivity_verification/rolling_only_full_refresh_"
                "k_sensitivity + penalty_audit/dynamic_k_sensitivity_aggregate.csv"
            ),
            "purpose": (
                "Authoritative rolling-only permanent reachability and final distance ratios."
            ),
        }
    )
    write_csv(manifest_path, rows)


def main() -> None:
    specs = [
        *[RunSpec("horseshoe", ROOT / "results" / "phase5", seed) for seed in range(1001, 1006)],
        *[
            RunSpec("open", ROOT / "results" / "phase6_open_robustness", seed)
            for seed in range(1001, 1004)
        ],
    ]
    seed_rows: list[dict[str, Any]] = []
    for spec in specs:
        seed_rows.extend(audit_run(spec))
        print(f"completed {spec.map_name} seed {spec.seed}", flush=True)

    aggregate = aggregate_seed_rows(seed_rows)
    validate_k20(aggregate)
    report_rows = merge_final_ratios(aggregate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DATA.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "rolling_only_k_sensitivity_by_seed.csv", seed_rows)
    write_csv(OUTPUT_DIR / "rolling_only_k_sensitivity_aggregate.csv", aggregate)
    write_csv(SUMMARY_DATA, report_rows)
    register_summary_artifact(SUMMARY_DATA)

    summary = {
        "analysis": "Full-refresh k sensitivity for authoritative rolling-only graph support",
        "search_rerun": False,
        "support": "The most recent 1,000 final-encoder latent codes only; no elite retention.",
        "refreshes": "Every 250 evaluations from 10,000 through 20,000 inclusive.",
        "k_values": K_VALUES,
        "k20_validation": (
            "Exact match to penalty_audit_aggregate.csv for mean reachable-centroid fraction "
            "and never-reachable-centroid count."
        ),
        "report_rows": report_rows,
    }
    (OUTPUT_DIR / "rolling_only_k_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for row in report_rows:
        print(
            "{map:9s} k={k:2d} reachable={mean_reachable_centroid_fraction_mean:.9f} "
            "never={never_reachable_centroids_mean:.3f} "
            "ratio={geodesic_to_euclidean_ratio_at_20000_mean:.9f}".format(**row)
        )


if __name__ == "__main__":
    main()
