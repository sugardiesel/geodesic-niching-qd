"""Compare learned latent distances with true maze geodesic distances."""

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

from algorithms.trajectory_autoencoder import (
    encode_sequences,
    load_autoencoder_checkpoint,
    trajectory_to_sequence,
)
from envs.forage_maze import ForageMaze2D
from envs.maze_distance import GridShortestPath
from scripts.run_phase1_validation import run_diagnostic_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase3-config",
        default="results/phase5/configs/baseline_b_learned_bd_euclidean_seed_1001.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Defaults to <Phase 3 output_dir>/autoencoder_final.pt.",
    )
    parser.add_argument(
        "--sample-npz",
        default="",
        help="Defaults to <Phase 3 output_dir>/representative_trajectory_sample.npz.",
    )
    parser.add_argument("--archive-summary", default="")
    parser.add_argument("--pairs-csv", default="")
    parser.add_argument("--output-dir", default="reproduced/phase3_latent_diagnostic")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    phase3_config = yaml.safe_load(Path(args.phase3_config).read_text(encoding="utf-8"))
    if not isinstance(phase3_config, dict):
        raise ValueError(f"{args.phase3_config} must contain a mapping.")
    run_dir = Path(str(phase3_config["experiment"]["output_dir"]).replace("\\", "/"))
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "autoencoder_final.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing locked Baseline B checkpoint: {checkpoint}. "
            "Use the committed production checkpoint or pass --checkpoint for another run."
        )
    sample_npz = (
        Path(args.sample_npz)
        if args.sample_npz
        else run_dir / "representative_trajectory_sample.npz"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ForageMaze2D.from_config_path(phase3_config["experiment"]["env_config"])
    training, metadata = load_autoencoder_checkpoint(checkpoint, device=args.device)
    if sample_npz.exists():
        summary = run_representative_sample_diagnostic(
            env=env,
            phase3_config=phase3_config,
            checkpoint=checkpoint,
            sample_npz=sample_npz,
            output_dir=output_dir,
            training=training,
            metadata=metadata,
            archive_summary_path=(
                Path(args.archive_summary)
                if args.archive_summary
                else run_dir / "phase3_summary.json"
            ),
        )
    else:
        if not args.pairs_csv:
            raise FileNotFoundError(
                f"Missing representative sample: {sample_npz}. "
                "Pass --sample-npz for a saved run, or --pairs-csv for the "
                "legacy Phase 1.5 diagnostic."
            )
        summary = run_legacy_phase1_pair_diagnostic(
            env=env,
            phase3_config=phase3_config,
            checkpoint=checkpoint,
            pairs_csv=Path(args.pairs_csv),
            output_dir=output_dir,
            training=training,
            metadata=metadata,
        )
    summary["phase3_config"] = args.phase3_config
    summary["output_files"].append(str(output_dir / "phase3_latent_diagnostic_summary.json"))
    (output_dir / "phase3_latent_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_summary(summary)


def run_representative_sample_diagnostic(
    env: ForageMaze2D,
    phase3_config: dict[str, Any],
    checkpoint: Path,
    sample_npz: Path,
    output_dir: Path,
    training: Any,
    metadata: dict[str, Any],
    archive_summary_path: Path,
) -> dict[str, Any]:
    sample = np.load(sample_npz)
    sequences = sample["sequences"].astype(np.float32)
    final_positions = sample["final_positions"].astype(np.float64)
    saved_latents = sample["latents"].astype(np.float64)
    evaluations = sample["evaluations"].astype(np.int64)
    fitnesses = sample["fitnesses"].astype(np.float64)
    is_loop = sample["is_loop"].astype(bool)
    is_near_still = sample["is_near_still"].astype(bool)

    latents = encode_sequences(training.model, training.normalizer, sequences, training.device)
    latent_reencode_max_abs_diff = (
        float(np.max(np.abs(latents - saved_latents))) if len(latents) else 0.0
    )

    pair_data = build_geodesic_pair_data(final_positions, latents, evaluations, env)
    metrics = correlation_summary(pair_data)
    scale = latent_scale_summary(
        latents, evaluations, fitnesses, is_loop, is_near_still, archive_summary_path
    )
    plot_files = save_latent_scatter(
        pair_data,
        metrics,
        output_dir / "phase3_representative_latent_vs_maze_geodesic_by_region",
        title="Representative Baseline B latent distance vs maze geodesic",
    )
    enriched_csv = output_dir / "phase3_representative_latent_vs_maze_geodesic_pairs.csv"
    write_csv(enriched_csv, pair_data)
    return {
        "diagnostic_mode": "representative_locked_baseline_b_run_sample",
        "checkpoint": str(checkpoint),
        "sample_npz": str(sample_npz),
        "env_config": phase3_config["experiment"]["env_config"],
        "sample_count": int(len(sequences)),
        "evaluation_min": int(np.min(evaluations)),
        "evaluation_max": int(np.max(evaluations)),
        "loop_fraction": float(np.mean(is_loop)),
        "near_still_fraction": float(np.mean(is_near_still)),
        "pair_count": len(pair_data),
        "device": training.device,
        "sequence_length": int(metadata["sequence_length"]),
        "feature_names": list(metadata["feature_names"]),
        "latent_reencode_max_abs_diff": latent_reencode_max_abs_diff,
        "latent_distance_vs_maze_geodesic": metrics,
        "latent_scale_check": scale,
        "output_files": [str(enriched_csv), *plot_files],
    }


def run_legacy_phase1_pair_diagnostic(
    env: ForageMaze2D,
    phase3_config: dict[str, Any],
    checkpoint: Path,
    pairs_csv: Path,
    output_dir: Path,
    training: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    base_seed = int(env.config["experiment"]["base_seed"])
    random_policy_count = int(env.config["validation"]["random_policy_count"])
    rollouts = run_diagnostic_rollouts(env, base_seed, random_policy_count)
    pair_rows = read_pair_csv(pairs_csv)
    sequence_length = int(metadata["sequence_length"])
    feature_names = list(metadata["feature_names"])
    sequences = np.stack(
        [
            trajectory_to_sequence(rollout.trajectory, sequence_length, feature_names)
            for rollout in rollouts
        ],
        axis=0,
    )
    latents = encode_sequences(training.model, training.normalizer, sequences, training.device)
    pair_data = build_pair_data(rollouts, latents, pair_rows)

    metrics = correlation_summary(pair_data)
    plot_files = save_latent_scatter(
        pair_data,
        metrics,
        output_dir / "phase3_latent_vs_maze_geodesic_by_region",
        title="Locked Baseline B latent distance vs maze geodesic",
    )
    enriched_csv = output_dir / "phase3_latent_vs_maze_geodesic_pairs.csv"
    write_csv(enriched_csv, pair_data)
    return {
        "diagnostic_mode": "legacy_phase1_5_validation_rollouts",
        "checkpoint": str(checkpoint),
        "pairs_csv": str(pairs_csv),
        "env_config": phase3_config["experiment"]["env_config"],
        "rollout_count": len(rollouts),
        "pair_count": len(pair_data),
        "device": training.device,
        "sequence_length": sequence_length,
        "feature_names": feature_names,
        "baseline_phase1_5_reference": {
            "raw_final_xy_overall_r": 0.814461465531535,
            "raw_final_xy_horseshoe_r": 0.4723962242386472,
            "raw_final_xy_open_field_r": 0.9986711109086749,
        },
        "latent_distance_vs_maze_geodesic": metrics,
        "output_files": [str(enriched_csv), *plot_files],
    }


def read_pair_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "euclidean_final_xy": float(row["euclidean_final_xy"]),
                    "maze_geodesic_shortest_path": float(row["maze_geodesic_shortest_path"]),
                    "ratio": float(row["ratio"]),
                    "shortest_path_touches_horseshoe_region": row[
                        "shortest_path_touches_horseshoe_region"
                    ]
                    .strip()
                    .lower()
                    == "true",
                }
            )
    return rows


def build_pair_data(
    rollouts: list[Any],
    latents: np.ndarray,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_points = np.asarray([rollout.final_position for rollout in rollouts], dtype=np.float64)
    expected_count = len(final_points) * (len(final_points) - 1) // 2
    if expected_count != len(pair_rows):
        raise ValueError(
            f"Pair CSV has {len(pair_rows)} rows, but {len(final_points)} rollouts imply "
            f"{expected_count}."
        )

    rows: list[dict[str, Any]] = []
    pair_index = 0
    max_euclidean_mismatch = 0.0
    for left in range(len(final_points)):
        for right in range(left + 1, len(final_points)):
            saved = pair_rows[pair_index]
            final_xy_distance = float(np.linalg.norm(final_points[left] - final_points[right]))
            max_euclidean_mismatch = max(
                max_euclidean_mismatch,
                abs(final_xy_distance - float(saved["euclidean_final_xy"])),
            )
            latent_distance = float(np.linalg.norm(latents[left] - latents[right]))
            rows.append(
                {
                    "pair_index": pair_index,
                    "rollout_i": left,
                    "rollout_j": right,
                    "rollout_i_name": rollouts[left].name,
                    "rollout_j_name": rollouts[right].name,
                    "euclidean_final_xy": float(saved["euclidean_final_xy"]),
                    "maze_geodesic_shortest_path": float(saved["maze_geodesic_shortest_path"]),
                    "latent_euclidean_distance": latent_distance,
                    "shortest_path_touches_horseshoe_region": bool(
                        saved["shortest_path_touches_horseshoe_region"]
                    ),
                }
            )
            pair_index += 1
    if max_euclidean_mismatch > 1e-5:
        raise ValueError(
            "Regenerated diagnostic rollout pair order does not match the saved Phase 1.5 pair "
            "CSV. "
            f"Max Euclidean mismatch: {max_euclidean_mismatch:.8f}."
        )
    return rows


def build_geodesic_pair_data(
    final_positions: np.ndarray,
    latents: np.ndarray,
    evaluations: np.ndarray,
    env: ForageMaze2D,
) -> list[dict[str, Any]]:
    shortest = GridShortestPath(
        env,
        grid_size=int(env.config["shortest_path"]["grid_size"]),
        connectivity=int(env.config["shortest_path"]["connectivity"]),
    )
    bounds = env.config.get("diagnostic", {}).get("horseshoe_region")
    if bounds is None:
        region = None
    else:
        region = (
            float(bounds["x_min"]),
            float(bounds["y_min"]),
            float(bounds["x_max"]),
            float(bounds["y_max"]),
        )
    cells = [shortest.point_to_cell(point) for point in final_positions]
    rows: list[dict[str, Any]] = []
    pair_index = 0
    for left in range(len(final_positions)):
        distances = shortest.distances_from_cell(cells[left])
        for right in range(left + 1, len(final_positions)):
            path = shortest.shortest_path(final_positions[left], final_positions[right])
            touches_horseshoe = path_touches_region(path, region) if region is not None else False
            rows.append(
                {
                    "pair_index": pair_index,
                    "sample_i": left,
                    "sample_j": right,
                    "evaluation_i": int(evaluations[left]),
                    "evaluation_j": int(evaluations[right]),
                    "euclidean_final_xy": float(
                        np.linalg.norm(final_positions[left] - final_positions[right])
                    ),
                    "maze_geodesic_shortest_path": float(distances[cells[right]]),
                    "latent_euclidean_distance": float(
                        np.linalg.norm(latents[left] - latents[right])
                    ),
                    "shortest_path_touches_horseshoe_region": bool(touches_horseshoe),
                }
            )
            pair_index += 1
    return rows


def path_touches_region(path: np.ndarray, region: tuple[float, float, float, float]) -> bool:
    if len(path) == 0:
        return False
    x_min, y_min, x_max, y_max = region
    inside_x = (path[:, 0] >= x_min) & (path[:, 0] <= x_max)
    inside_y = (path[:, 1] >= y_min) & (path[:, 1] <= y_max)
    return bool(np.any(inside_x & inside_y))


def correlation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    geodesic = np.asarray(
        [float(row["maze_geodesic_shortest_path"]) for row in rows], dtype=np.float64
    )
    latent = np.asarray([float(row["latent_euclidean_distance"]) for row in rows], dtype=np.float64)
    horseshoe = np.asarray(
        [bool(row["shortest_path_touches_horseshoe_region"]) for row in rows], dtype=bool
    )
    return {
        "overall_pair_count": int(len(rows)),
        "overall_pearson_r": pearson(geodesic, latent),
        "horseshoe_region_pair_count": int(np.count_nonzero(horseshoe)),
        "horseshoe_region_pearson_r": pearson(geodesic[horseshoe], latent[horseshoe]),
        "open_field_pair_count": int(np.count_nonzero(~horseshoe)),
        "open_field_pearson_r": pearson(geodesic[~horseshoe], latent[~horseshoe]),
        "latent_distance_mean": float(np.mean(latent)),
        "latent_distance_std": float(np.std(latent)),
        "maze_geodesic_mean": float(np.mean(geodesic)),
        "maze_geodesic_std": float(np.std(geodesic)),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_latent_scatter(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_stem: Path,
    title: str,
) -> list[str]:
    geodesic = np.asarray(
        [float(row["maze_geodesic_shortest_path"]) for row in rows], dtype=np.float64
    )
    latent = np.asarray([float(row["latent_euclidean_distance"]) for row in rows], dtype=np.float64)
    horseshoe = np.asarray(
        [bool(row["shortest_path_touches_horseshoe_region"]) for row in rows], dtype=bool
    )

    fig, ax = plt.subplots(figsize=(6.7, 5.0), constrained_layout=True)
    ax.scatter(
        geodesic[~horseshoe],
        latent[~horseshoe],
        s=6,
        alpha=0.28,
        color="#2563eb",
        label=f"open field (n={metrics['open_field_pair_count']})",
        edgecolors="none",
        rasterized=True,
    )
    ax.scatter(
        geodesic[horseshoe],
        latent[horseshoe],
        s=7,
        alpha=0.32,
        color="#dc2626",
        marker="^",
        label=f"horseshoe region (n={metrics['horseshoe_region_pair_count']})",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel("true maze geodesic distance")
    ax.set_ylabel("latent Euclidean distance")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    text = (
        f"overall r = {metrics['overall_pearson_r']:.3f}\n"
        f"horseshoe r = {metrics['horseshoe_region_pearson_r']:.3f}\n"
        f"open-field r = {metrics['open_field_pearson_r']:.3f}"
    )
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.92,
        },
    )
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]


def latent_scale_summary(
    latents: np.ndarray,
    evaluations: np.ndarray,
    fitnesses: np.ndarray,
    is_loop: np.ndarray,
    is_near_still: np.ndarray,
    archive_summary_path: Path,
) -> dict[str, Any]:
    latent_distances = pairwise_upper_distances(latents)
    archive_mean = float("nan")
    archive_descriptors = np.asarray([], dtype=np.float64).reshape(0, 2)
    phase3_summary: dict[str, Any] = {}
    if archive_summary_path.exists():
        phase3_summary = json.loads(archive_summary_path.read_text(encoding="utf-8"))
        archive_mean = float(phase3_summary["final_metrics"]["mean_pairwise_descriptor_euclidean"])
        archive_csv = archive_summary_path.with_name("archive_cells.csv")
        if archive_csv.exists():
            archive_descriptors = read_archive_descriptors(archive_csv)
    sample_mean = float(np.mean(latent_distances)) if len(latent_distances) else float("nan")
    bootstrap_evaluations = int(phase3_summary.get("bootstrap_evaluations", 2000))
    retrain_events = phase3_summary.get("retrain_events", [])
    retrain_evaluation = (
        int(retrain_events[0]["evaluation"]) if retrain_events else bootstrap_evaluations
    )
    top_10_threshold = float(np.quantile(fitnesses, 0.9)) if len(fitnesses) else float("nan")
    return {
        "sample_mean_pairwise_latent_euclidean": sample_mean,
        "sample_pairwise_distribution": distance_distribution_stats(latents),
        "archive_mean_pairwise_elite_descriptor_euclidean": archive_mean,
        "archive_pairwise_distribution": distance_distribution_stats(archive_descriptors),
        "sample_to_archive_mean_ratio": sample_mean / archive_mean
        if np.isfinite(archive_mean) and archive_mean > 0
        else float("nan"),
        "sample_loop_only_pairwise_distribution": distance_distribution_stats(latents[is_loop]),
        "sample_non_loop_pairwise_distribution": distance_distribution_stats(latents[~is_loop]),
        "sample_near_still_pairwise_distribution": distance_distribution_stats(
            latents[is_near_still]
        ),
        "sample_non_near_still_pairwise_distribution": distance_distribution_stats(
            latents[~is_near_still]
        ),
        "sample_early_bootstrap_pairwise_distribution": distance_distribution_stats(
            latents[evaluations <= bootstrap_evaluations]
        ),
        "sample_middle_pre_retrain_pairwise_distribution": distance_distribution_stats(
            latents[(evaluations > bootstrap_evaluations) & (evaluations <= retrain_evaluation)]
        ),
        "sample_late_post_retrain_pairwise_distribution": distance_distribution_stats(
            latents[evaluations > retrain_evaluation]
        ),
        "sample_top_10_percent_fitness_pairwise_distribution": distance_distribution_stats(
            latents[fitnesses >= top_10_threshold]
        ),
        "rough_archive_cell_coverage_by_sample": rough_archive_cell_coverage(
            latents, archive_descriptors, phase3_summary
        ),
        "scale_diagnosis": (
            "The checkpoint re-encoding path is internally consistent; remaining scale difference "
            "is expected when comparing a random cross-section of evaluated policies with the "
            "final archive elites, because the archive metric is computed only over filled-cell "
            "elites and therefore emphasizes descriptor spread."
        ),
        "scale_interpretation": (
            "Comparable if ratio is close to 1; large deviations mean sampled evaluated policies "
            "occupy a different latent spread than final elites."
        ),
    }


def read_archive_descriptors(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append([float(row["descriptor_x"]), float(row["descriptor_y"])])
    return np.asarray(rows, dtype=np.float64)


def distance_distribution_stats(points: np.ndarray) -> dict[str, Any]:
    distances = pairwise_upper_distances(points)
    if len(distances) == 0:
        return {
            "point_count": int(len(points)),
            "pair_count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "max": float("nan"),
        }
    return {
        "point_count": int(len(points)),
        "pair_count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "q75": float(np.quantile(distances, 0.75)),
        "q90": float(np.quantile(distances, 0.90)),
        "max": float(np.max(distances)),
    }


def rough_archive_cell_coverage(
    sample_latents: np.ndarray,
    archive_descriptors: np.ndarray,
    phase3_summary: dict[str, Any],
) -> dict[str, Any]:
    if len(sample_latents) == 0 or len(archive_descriptors) == 0:
        return {
            "sample_points_inside_archive_descriptor_range": 0,
            "sample_occupied_rough_cells": 0,
        }
    bins = phase3_summary.get("grid_bins", [25, 25])
    bins_array = np.asarray([int(bins[0]), int(bins[1])], dtype=int)
    descriptor_min = np.min(archive_descriptors, axis=0)
    descriptor_max = np.max(archive_descriptors, axis=0)
    span = np.maximum(descriptor_max - descriptor_min, 1e-12)
    scaled = (sample_latents - descriptor_min) / span
    cells = np.floor(scaled * bins_array).astype(int)
    valid = np.all((cells >= 0) & (cells < bins_array), axis=1)
    occupied = {tuple(cell) for cell in cells[valid]}
    return {
        "note": (
            "Uses archive elite descriptor min/max as a rough scale check, not the exact padded "
            "archive bounds."
        ),
        "archive_elite_count": int(len(archive_descriptors)),
        "sample_count": int(len(sample_latents)),
        "sample_points_inside_archive_descriptor_range": int(np.count_nonzero(valid)),
        "sample_occupied_rough_cells": int(len(occupied)),
    }


def pairwise_upper_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=np.float64)
    delta = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    return distances[np.triu_indices(len(points), k=1)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    metrics = summary["latent_distance_vs_maze_geodesic"]
    print("PHASE 3 LATENT MANIFOLD DIAGNOSTIC")
    print(f"mode: {summary['diagnostic_mode']}")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"pairs: {summary['pair_count']}")
    print(f"device: {summary['device']}")
    print(
        "latent vs maze geodesic: overall r={overall_pearson_r:.3f}, "
        "horseshoe r={horseshoe_region_pearson_r:.3f}, "
        "open-field r={open_field_pearson_r:.3f}".format(**metrics)
    )
    if "latent_scale_check" in summary:
        scale = summary["latent_scale_check"]
        print(
            "latent scale: sample mean={sample_mean_pairwise_latent_euclidean:.3f}, "
            "archive elite mean={archive_mean_pairwise_elite_descriptor_euclidean:.3f}, "
            "ratio={sample_to_archive_mean_ratio:.3f}".format(**scale)
        )
        print(f"latent re-encode max abs diff: {summary['latent_reencode_max_abs_diff']:.6g}")
    if "baseline_phase1_5_reference" in summary:
        reference = summary["baseline_phase1_5_reference"]
        print(
            "Phase 1.5 raw final-x/y reference: overall r={raw_final_xy_overall_r:.3f}, "
            "horseshoe r={raw_final_xy_horseshoe_r:.3f}, "
            "open-field r={raw_final_xy_open_field_r:.3f}".format(**reference)
        )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
