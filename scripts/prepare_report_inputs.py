# ruff: noqa: I001
from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.phase1_detour_figure import generate_horseshoe_detour_artifacts


OUTPUT_DIR = ROOT / "results" / "final_summary"
DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"

MAP_SOURCES = {
    "horseshoe": ROOT / "results" / "phase5",
    "open": ROOT / "results" / "phase6_open_robustness",
}

RUN_COLUMNS = [
    "condition",
    "label",
    "seed",
    "coverage_percent",
    "qd_score",
    "raw_qd_score",
    "max_fitness",
    "occupancy_entropy",
    "mean_pairwise_descriptor_euclidean",
    "mean_pairwise_descriptor_knn_geodesic",
    "elapsed_seconds",
    "evaluations_per_second",
    "pairwise_geodesic_k",
    "total_evaluations",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_artifact(
    source: Path,
    destination: Path,
    category: str,
    purpose: str,
    manifest: list[dict[str, str]],
) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest.append(
        {
            "category": category,
            "file": destination.relative_to(OUTPUT_DIR).as_posix(),
            "source": source.relative_to(ROOT).as_posix(),
            "purpose": purpose,
        }
    )


def build_combined_tables(manifest: list[dict[str, str]]) -> None:
    combined_runs: list[dict[str, str]] = []
    combined_summaries: list[dict[str, str]] = []
    combined_tests: list[dict[str, str]] = []

    for map_name, source_dir in MAP_SOURCES.items():
        run_rows = read_csv(source_dir / "phase5_run_summary.csv")
        for row in run_rows:
            combined_runs.append({"map": map_name, **{key: row[key] for key in RUN_COLUMNS}})

        for row in read_csv(source_dir / "phase5_condition_summary.csv"):
            combined_summaries.append({"map": map_name, **row})

        for row in read_csv(source_dir / "phase5_wilcoxon_paired_tests.csv"):
            combined_tests.append({"map": map_name, **row})

        for source_name in [
            "phase5_run_summary.csv",
            "phase5_condition_summary.csv",
            "phase5_wilcoxon_paired_tests.csv",
            "phase5_representative_heatmaps.csv",
            "phase5_summary.json",
        ]:
            destination_name = f"{map_name}_{source_name}"
            copy_artifact(
                source_dir / source_name,
                DATA_DIR / destination_name,
                "data",
                f"Unmodified authoritative {map_name}-map source output.",
                manifest,
            )

    combined_run_path = DATA_DIR / "corrected_k20_per_seed_metrics.csv"
    write_csv(combined_run_path, combined_runs)
    manifest.append(
        {
            "category": "data",
            "file": combined_run_path.relative_to(OUTPUT_DIR).as_posix(),
            "source": "combined from both authoritative phase5_run_summary.csv files",
            "purpose": "All per-seed metrics for both maps and all three conditions.",
        }
    )

    combined_summary_path = DATA_DIR / "corrected_k20_condition_mean_std.csv"
    write_csv(combined_summary_path, combined_summaries)
    manifest.append(
        {
            "category": "data",
            "file": combined_summary_path.relative_to(OUTPUT_DIR).as_posix(),
            "source": "combined from both authoritative phase5_condition_summary.csv files",
            "purpose": "Condition-level mean and sample standard deviation for both maps.",
        }
    )

    combined_test_path = DATA_DIR / "corrected_k20_wilcoxon_paired_tests.csv"
    write_csv(combined_test_path, combined_tests)
    manifest.append(
        {
            "category": "data",
            "file": combined_test_path.relative_to(OUTPUT_DIR).as_posix(),
            "source": "combined from both authoritative phase5_wilcoxon_paired_tests.csv files",
            "purpose": "All paired shared-seed Wilcoxon outputs, including exact per-seed arrays.",
        }
    )

    baseline_b_vs_contribution = [
        row
        for row in combined_tests
        if row["left_condition"] == "baseline_b_learned_bd_euclidean"
        and row["right_condition"] == "contribution_geodesic_niching"
    ]
    comparison_path = DATA_DIR / "corrected_k20_baseline_b_vs_contribution_wilcoxon.csv"
    write_csv(comparison_path, baseline_b_vs_contribution)
    manifest.append(
        {
            "category": "data",
            "file": comparison_path.relative_to(OUTPUT_DIR).as_posix(),
            "source": "filtered from corrected_k20_wilcoxon_paired_tests.csv",
            "purpose": "Headline Baseline B versus Contribution paired tests for both maps.",
        }
    )


def archive_grid(path: Path, bins: int = 25) -> np.ndarray:
    grid = np.full((bins, bins), np.nan, dtype=np.float64)
    for row in read_csv(path):
        cell_x = int(row["cell_x"])
        cell_y = int(row["cell_y"])
        grid[cell_y, cell_x] = float(row["fitness"])
    return grid


def metric_row(condition: str, seed: int) -> dict[str, str]:
    rows = read_csv(MAP_SOURCES["horseshoe"] / "phase5_run_summary.csv")
    return next(
        row
        for row in rows
        if row["condition"] == condition and int(row["seed"]) == seed
    )


def make_paired_heatmap(manifest: list[dict[str, str]], seed: int = 1002) -> None:
    baseline_condition = "baseline_b_learned_bd_euclidean"
    contribution_condition = "contribution_geodesic_niching"
    baseline_cells = (
        MAP_SOURCES["horseshoe"] / baseline_condition / f"seed_{seed}" / "archive_cells.csv"
    )
    contribution_cells = (
        MAP_SOURCES["horseshoe"] / contribution_condition / f"seed_{seed}" / "archive_cells.csv"
    )
    baseline_grid = archive_grid(baseline_cells)
    contribution_grid = archive_grid(contribution_cells)
    finite_values = np.concatenate(
        [
            baseline_grid[np.isfinite(baseline_grid)],
            contribution_grid[np.isfinite(contribution_grid)],
        ]
    )
    baseline_metrics = metric_row(baseline_condition, seed)
    contribution_metrics = metric_row(contribution_condition, seed)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), constrained_layout=True)
    images = []
    panels = [
        (axes[0], baseline_grid, "Baseline B", baseline_metrics),
        (axes[1], contribution_grid, "Contribution (k=20)", contribution_metrics),
    ]
    for ax, grid, label, metrics in panels:
        image = ax.imshow(
            grid,
            origin="lower",
            cmap="viridis",
            vmin=float(np.min(finite_values)),
            vmax=float(np.max(finite_values)),
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        ax.set_title(
            f"{label}, seed {seed}\n"
            f"coverage={float(metrics['coverage_percent']):.2f}%, "
            f"QD={float(metrics['qd_score']):.1f}",
            fontsize=10,
        )
        ax.set_xlabel("archive cell x")
        ax.set_ylabel("archive cell y")
        ax.set_xlim(-0.5, 24.5)
        ax.set_ylim(-0.5, 24.5)
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("elite fitness")
    fig.suptitle(
        "Corrected final archives on the horseshoe map\n"
        "same seed; condition-specific learned latent grids",
        fontsize=12,
    )

    stem = FIGURE_DIR / "horseshoe_archive_heatmaps_baseline_b_vs_contribution_seed1002"
    for suffix in [".png", ".pdf"]:
        output_path = stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=240)
        manifest.append(
            {
                "category": "figure",
                "file": output_path.relative_to(OUTPUT_DIR).as_posix(),
                "source": (
                    "results/phase5/{baseline_b_learned_bd_euclidean,"
                    "contribution_geodesic_niching}/seed_1002/archive_cells.csv"
                ),
                "purpose": (
                    "Same-seed representative archive comparison with a shared fitness scale."
                ),
            }
        )
    plt.close(fig)


def copy_figures(manifest: list[dict[str, str]]) -> None:
    figure_sources = [
        (
            ROOT
            / "results"
            / "phase1_distance_diagnostic_corrected"
            / "phase1_report_euclidean_vs_geodesic_by_region",
            "phase1_5_region_split_distance_scatter",
            "Corrected Phase 1.5 Euclidean versus true maze-geodesic diagnostic.",
        ),
        (
            MAP_SOURCES["horseshoe"] / "phase5_aggregate_qd_coverage_curves",
            "horseshoe_corrected_qd_coverage_curves",
            "Corrected horseshoe-map aggregate shifted-QD and coverage curves.",
        ),
        (
            MAP_SOURCES["open"] / "phase5_aggregate_qd_coverage_curves",
            "open_corrected_qd_coverage_curves",
            "Corrected open-map aggregate shifted-QD and coverage curves.",
        ),
        (
            ROOT
            / "results"
            / "audit_connectivity_verification"
            / "penalty_audit"
            / "centroid_connectivity_over_run",
            "corrected_rolling_centroid_connectivity_over_run",
            "Rolling-only corrected graph reachability over complete production runs.",
        ),
    ]
    for source_stem, destination_stem, purpose in figure_sources:
        for suffix in [".png", ".pdf"]:
            copy_artifact(
                source_stem.with_suffix(suffix),
                (FIGURE_DIR / destination_stem).with_suffix(suffix),
                "figure",
                purpose,
                manifest,
            )


def make_horseshoe_detour_figure(manifest: list[dict[str, str]]) -> None:
    stem = FIGURE_DIR / "phase1_horseshoe_u_bottom_detour"
    provenance = DATA_DIR / "phase1_horseshoe_u_bottom_detour_provenance.csv"
    generate_horseshoe_detour_artifacts(
        config_path=ROOT / "configs" / "phase1_horseshoe.yaml",
        legacy_summary_path=ROOT / "results" / "phase1_environment" / "phase1_summary.json",
        corrected_summary_path=(
            ROOT / "results" / "phase1_distance_diagnostic_corrected" / "phase1_summary.json"
        ),
        output_stem=stem,
        provenance_path=provenance,
        pair_name="u_bottom_detour",
    )
    source = (
        "configs/phase1_horseshoe.yaml + envs/maze_distance.py + "
        "results/phase1_environment/phase1_summary.json + "
        "results/phase1_distance_diagnostic_corrected/phase1_summary.json"
    )
    purpose = (
        "Literal horseshoe map with the corrected Dijkstra path for the u_bottom_detour "
        "landmark."
    )
    for suffix in [".png", ".pdf"]:
        output_path = stem.with_suffix(suffix)
        manifest.append(
            {
                "category": "figure",
                "file": output_path.relative_to(OUTPUT_DIR).as_posix(),
                "source": source,
                "purpose": purpose,
            }
        )
    manifest.append(
        {
            "category": "data",
            "file": provenance.relative_to(OUTPUT_DIR).as_posix(),
            "source": source,
            "purpose": "Old-versus-corrected Phase 1.5 landmark provenance and impact check.",
        }
    )

    connectivity_tables = [
        (
            ROOT
            / "results"
            / "audit_connectivity_verification"
            / "penalty_audit"
            / "static_k_sensitivity_by_map.csv",
            "corrected_rolling_static_k_sensitivity_by_map.csv",
        ),
        (
            ROOT
            / "results"
            / "audit_connectivity_verification"
            / "penalty_audit"
            / "dynamic_k_sensitivity_aggregate.csv",
            "corrected_rolling_dynamic_k_sensitivity_aggregate.csv",
        ),
        (
            ROOT
            / "results"
            / "elite_support_connectivity_diagnostic"
            / "refresh_connectivity_aggregate.csv",
            "elite_retention_refresh_connectivity_aggregate.csv",
        ),
        (
            ROOT
            / "results"
            / "elite_support_connectivity_diagnostic"
            / "rolling_only_vs_elite_support_k20.csv",
            "rolling_only_vs_elite_support_k20.csv",
        ),
    ]
    for source, destination_name in connectivity_tables:
        copy_artifact(
            source,
            DATA_DIR / destination_name,
            "data",
            "Numerical source table for the final connectivity/starvation diagnostics.",
            manifest,
        )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    build_combined_tables(manifest)
    copy_figures(manifest)
    make_paired_heatmap(manifest)
    make_horseshoe_detour_figure(manifest)
    write_csv(OUTPUT_DIR / "artifact_manifest.csv", manifest)
    print(f"Prepared {len(manifest)} final result artifacts in {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"final-result preparation failed: {exc}", file=sys.stderr)
        raise
