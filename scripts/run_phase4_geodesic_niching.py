"""Run Phase 4: geodesic-niching QD over the learned trajectory latent space."""

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

from algorithms.geodesic_archive import GeodesicGridArchive
from algorithms.policies import MLPPolicyGenome
from algorithms.trajectory_autoencoder import (
    AutoencoderTrainingResult,
    encode_sequences,
    reconstruct_sequences,
    train_autoencoder,
    trajectory_to_sequence,
)
from analysis.phase2_plots import save_metric_curves
from analysis.phase3_plots import (
    save_autoencoder_loss_plot,
    save_latent_archive_heatmap,
    save_reconstruction_examples,
)
from envs.forage_maze import ForageMaze2D, load_env_config
from scripts.run_phase3_aurora_euclidean import (
    autoencoder_loss_note,
    candidate_diag,
    evaluation_row,
    latent_bounds,
    latent_is_outside_bounds,
    loss_ylabel,
    reconstruction_metrics,
    select_reconstruction_examples,
    stack_sequences,
    summarize_latent_clipping,
    summarize_trajectory_diagnostics,
    trajectory_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_geodesic_niching.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{args.config} must contain a mapping.")

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ForageMaze2D(load_env_config(config["experiment"]["env_config"]))

    start = time.perf_counter()
    result = run_phase4(env, config)
    elapsed = time.perf_counter() - start

    archive: GeodesicGridArchive = result["archive"]
    metrics: list[dict[str, Any]] = result["metrics"]
    evaluations: list[dict[str, Any]] = result["evaluations"]
    final_training: AutoencoderTrainingResult = result["final_training"]

    archive_csv = output_dir / "archive_cells.csv"
    metrics_csv = output_dir / "metrics.csv"
    evaluations_csv = output_dir / "evaluations.csv"
    visited_latents_csv = output_dir / "visited_latents_final_space.csv"
    support_points_csv = output_dir / "geodesic_graph_support_points.csv"
    loss_csv = output_dir / "autoencoder_loss.csv"
    write_csv(archive_csv, archive.cell_rows())
    write_csv(metrics_csv, metrics)
    write_csv(evaluations_csv, evaluations)
    write_csv(visited_latents_csv, archive.visited_latent_rows())
    write_csv(support_points_csv, archive.support_point_rows())
    write_csv(
        loss_csv,
        [
            {"epoch": idx + 1, "train_loss": train, "validation_loss": validation}
            for idx, (train, validation) in enumerate(
                zip(final_training.train_losses, final_training.validation_losses, strict=False)
            )
        ],
    )

    loss_files = save_autoencoder_loss_plot(
        final_training.train_losses,
        final_training.validation_losses,
        output_dir / "autoencoder_loss",
        ylabel=loss_ylabel(config),
    )
    recon_files = save_reconstruction_examples(
        result["reconstruction_real"],
        result["reconstruction_pred"],
        output_dir / "trajectory_reconstruction_examples",
        config["trajectory"]["features"],
        labels=result["reconstruction_labels"],
    )
    heatmap_files = save_latent_archive_heatmap(
        archive,
        output_dir / "archive_heatmap_geodesic_latent",
        "Phase 4 geodesic-niching archive: learned latent BD",
    )
    curve_files = save_metric_curves(
        metrics,
        output_dir / "qd_score_coverage_curves",
        title=result["plot_title"],
    )

    final_metrics = archive.metrics(include_pairwise=True)
    graph_stats = archive.graph_stats()
    summary = {
        "config": args.config,
        "env_config": config["experiment"]["env_config"],
        "seed": int(config["experiment"]["seed"]),
        "descriptor": config["algorithm"]["descriptor"],
        "archive_assignment": config["algorithm"]["archive_assignment"],
        "grid_bins": config["algorithm"]["grid_bins"],
        "hidden_dim": int(config["algorithm"]["hidden_dim"]),
        "policy_parameter_count": int(archive.genome_size),
        "total_evaluations": int(config["algorithm"]["total_evaluations"]),
        "bootstrap_evaluations": int(config["algorithm"]["bootstrap_evaluations"]),
        "retrain_evaluations": list(config["algorithm"].get("retrain_evaluations", [])),
        "autoencoder_device": final_training.device,
        "autoencoder_latent_dim": int(config["autoencoder"]["latent_dim"]),
        "autoencoder_normalize": bool(config["autoencoder"].get("normalize", True)),
        "autoencoder_derivative_loss_weight": float(
            config["autoencoder"].get("derivative_loss_weight", 0.0)
        ),
        "autoencoder_loss_note": autoencoder_loss_note(config),
        "autoencoder_final_train_loss": float(final_training.train_losses[-1]),
        "autoencoder_final_validation_loss": float(final_training.validation_losses[-1]),
        "autoencoder_normalization_audit": dict(final_training.normalization_audit),
        "autoencoder_bootstrap_normalization_audit": result["bootstrap_normalization_audit"],
        "elapsed_seconds": elapsed,
        "evaluations_per_second": int(config["algorithm"]["total_evaluations"]) / elapsed,
        "final_metrics": final_metrics,
        "trajectory_diagnostics": result["trajectory_diagnostics"],
        "latent_outside_bounds": result["latent_outside_bounds"],
        "reconstruction_quality": result["reconstruction_quality"],
        "retrain_events": result["retrain_events"],
        "geodesic_graph": {
            "assignment": (
                "nearest latent grid centroid by approximate k-NN graph shortest-path distance"
            ),
            "parent_selection": "archive elites sampled by normalized geodesic novelty score",
            "graph_k": graph_stats.knn_k,
            "assignment_k": int(config["geodesic"]["assignment_k"]),
            "max_graph_points": int(config["geodesic"]["max_graph_points"]),
            "reservoir_points": graph_stats.reservoir_points,
            "rolling_buffer_points": graph_stats.rolling_buffer_points,
            "archive_elite_points": graph_stats.archive_elite_points,
            "include_archive_elites_in_support": bool(
                config["geodesic"].get("include_archive_elites_in_support", False)
            ),
            "reservoir_strategy": graph_stats.reservoir_strategy,
            "reservoir_seed": graph_stats.reservoir_seed,
            "refresh_interval": int(config["geodesic"]["refresh_interval"]),
            "support_points": graph_stats.support_points,
            "visited_points": graph_stats.visited_points,
            "centroid_count": graph_stats.centroid_count,
            "refresh_count": graph_stats.refresh_count,
            "finite_centroid_fraction": graph_stats.finite_centroid_fraction,
            "connected_components": graph_stats.connected_components,
            "largest_component_fraction": graph_stats.largest_component_fraction,
            "penalized_centroid_distances": graph_stats.penalized_centroid_distances,
            "cumulative_penalized_centroid_distances": (
                graph_stats.cumulative_penalized_centroid_distances
            ),
            "refreshes_with_penalty": graph_stats.refreshes_with_penalty,
            "penalty_distance": graph_stats.penalty_distance,
        },
        "output_files": [
            str(archive_csv),
            str(metrics_csv),
            str(evaluations_csv),
            str(visited_latents_csv),
            str(support_points_csv),
            str(loss_csv),
            *loss_files,
            *recon_files,
            *heatmap_files,
            *curve_files,
            str(output_dir / "phase4_summary.json"),
        ],
    }
    (output_dir / "phase4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def run_phase4(env: ForageMaze2D, config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["experiment"]["seed"])
    algo = config["algorithm"]
    trajectory_cfg = config["trajectory"]
    autoencoder_cfg = config["autoencoder"]
    metrics_cfg = config["metrics"]
    rng = np.random.default_rng(seed)

    obs_dim = env.observation_dim
    hidden_dim = int(algo["hidden_dim"])
    total_evaluations = int(algo["total_evaluations"])
    bootstrap_evaluations = int(algo["bootstrap_evaluations"])
    sequence_length = int(trajectory_cfg["sequence_length"])
    feature_names = list(trajectory_cfg["features"])
    retrain_evaluations = {int(value) for value in algo.get("retrain_evaluations", [])}

    candidates: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []

    for evaluation in range(bootstrap_evaluations):
        genome = MLPPolicyGenome.random(obs_dim, hidden_dim, rng, scale=float(algo["init_scale"]))
        rollout = env.rollout(
            genome.to_policy(),
            seed=seed + evaluation * 1009,
            name=f"phase4_bootstrap_{evaluation}",
            record=True,
        )
        sequence = trajectory_to_sequence(rollout.trajectory, sequence_length, feature_names)
        diag = trajectory_diagnostics(sequence)
        candidates.append(
            {
                "genome": genome,
                "rollout": rollout,
                "sequence": sequence,
                "source": "bootstrap_random",
                **diag,
            }
        )
        evaluation_rows.append(evaluation_row(evaluation + 1, "bootstrap_random", rollout, diag))

    training = train_autoencoder(stack_sequences(candidates), autoencoder_cfg, seed=seed)
    bootstrap_normalization_audit = dict(training.normalization_audit)
    archive, latents, outside_flags = build_geodesic_archive_from_candidates(
        candidates, training, env, config
    )
    update_rows_with_geodesic(evaluation_rows, latents, archive, outside_flags)

    metrics_log = [
        {
            "evaluation": bootstrap_evaluations,
            "event": "bootstrap",
            **archive.metrics(include_pairwise=True),
        }
    ]
    pairwise_log_interval = int(metrics_cfg.get("pairwise_log_interval", 1000))
    log_interval = int(algo["log_interval"])
    retrain_events: list[dict[str, Any]] = []
    all_outside_flags = list(outside_flags)

    for evaluation in range(bootstrap_evaluations, total_evaluations):
        parent = archive.sample_elite(rng, obs_dim=obs_dim, hidden_dim=hidden_dim)
        genome = parent.mutate(
            rng,
            sigma=float(algo["mutation_sigma"]),
            mutation_probability=float(algo["mutation_probability"]),
        )
        rollout = env.rollout(
            genome.to_policy(),
            seed=seed + evaluation * 1009,
            name=f"phase4_eval_{evaluation}",
            record=True,
        )
        sequence = trajectory_to_sequence(rollout.trajectory, sequence_length, feature_names)
        diag = trajectory_diagnostics(sequence)
        latent = encode_sequences(
            training.model, training.normalizer, sequence[None, :, :], training.device
        )[0]
        outside_bounds = latent_is_outside_bounds(latent, archive)
        inserted, replaced, cell, geodesic_distance = archive.add(
            genome,
            rollout,
            descriptor=latent,
            evaluation=evaluation + 1,
        )
        candidates.append(
            {
                "genome": genome,
                "rollout": rollout,
                "sequence": sequence,
                "source": "mutated_elite",
                **diag,
            }
        )
        row = evaluation_row(
            evaluation + 1,
            "mutated_elite",
            rollout,
            diag,
            latent,
            cell,
            inserted,
            replaced,
            outside_bounds,
        )
        row["assignment_geodesic_distance"] = geodesic_distance
        evaluation_rows.append(row)
        all_outside_flags.append(outside_bounds)

        if evaluation + 1 in retrain_evaluations:
            training = train_autoencoder(
                stack_sequences(candidates), autoencoder_cfg, seed=seed + evaluation + 1
            )
            archive, rebuilt_latents, rebuilt_outside = build_geodesic_archive_from_candidates(
                candidates, training, env, config
            )
            retrain_events.append(
                {
                    "evaluation": evaluation + 1,
                    "train_loss": float(training.train_losses[-1]),
                    "validation_loss": float(training.validation_losses[-1]),
                    "candidate_count": len(candidates),
                    "rebuilt_archive_filled_cells": archive.occupied_count,
                    "rebuilt_outside_bounds_fraction": float(np.mean(rebuilt_outside))
                    if rebuilt_outside
                    else 0.0,
                    "normalization_audit": dict(training.normalization_audit),
                    "graph_support_points": archive.graph_stats().support_points,
                    "graph_refresh_count": archive.graph_stats().refresh_count,
                }
            )
            metrics_log.append(
                {
                    "evaluation": evaluation + 1,
                    "event": "retrain",
                    **archive.metrics(include_pairwise=True),
                }
            )

        if (evaluation + 1) % log_interval == 0 or evaluation + 1 == total_evaluations:
            is_final = evaluation + 1 == total_evaluations
            include_pairwise = is_final or (
                pairwise_log_interval > 0 and (evaluation + 1) % pairwise_log_interval == 0
            )
            metrics_log.append(
                {
                    "evaluation": evaluation + 1,
                    "event": "log",
                    **archive.metrics(include_pairwise=include_pairwise),
                }
            )

    final_sequences = stack_sequences(candidates)
    final_reconstruction = reconstruct_sequences(
        training.model, training.normalizer, final_sequences, training.device
    )
    diagnostics = [candidate_diag(candidate) for candidate in candidates]
    selected_indices, selected_labels = select_reconstruction_examples(diagnostics)
    return {
        "archive": archive,
        "metrics": metrics_log,
        "evaluations": evaluation_rows,
        "final_training": training,
        "reconstruction_real": final_sequences[selected_indices],
        "reconstruction_pred": final_reconstruction[selected_indices],
        "reconstruction_labels": selected_labels,
        "trajectory_diagnostics": summarize_trajectory_diagnostics(
            diagnostics, bootstrap_evaluations
        ),
        "latent_outside_bounds": summarize_latent_clipping(
            all_outside_flags, bootstrap_evaluations
        ),
        "reconstruction_quality": reconstruction_metrics(
            final_sequences, final_reconstruction, diagnostics
        ),
        "plot_title": str(
            config["experiment"].get("plot_title", "Phase 4 geodesic-niching progress")
        ),
        "retrain_events": retrain_events,
        "bootstrap_normalization_audit": bootstrap_normalization_audit,
    }


def build_geodesic_archive_from_candidates(
    candidates: list[dict[str, Any]],
    training: AutoencoderTrainingResult,
    env: ForageMaze2D,
    config: dict[str, Any],
) -> tuple[GeodesicGridArchive, np.ndarray, list[bool]]:
    algo = config["algorithm"]
    metrics_cfg = config["metrics"]
    autoencoder_cfg = config["autoencoder"]
    geodesic_cfg = config["geodesic"]
    sequences = stack_sequences(candidates)
    latents = encode_sequences(training.model, training.normalizer, sequences, training.device)
    latent_min, latent_max = latent_bounds(
        latents, float(autoencoder_cfg["latent_bounds_padding_fraction"])
    )
    archive = GeodesicGridArchive(
        bins=(int(algo["grid_bins"][0]), int(algo["grid_bins"][1])),
        bd_min=latent_min,
        bd_max=latent_max,
        genome_size=MLPPolicyGenome.parameter_count_for(
            env.observation_dim, int(algo["hidden_dim"])
        ),
        qd_score_fitness_floor=float(metrics_cfg["qd_score_fitness_floor"]),
        pairwise_geodesic_k=int(metrics_cfg["pairwise_geodesic_k"]),
        graph_k=int(geodesic_cfg["graph_k"]),
        assignment_k=int(geodesic_cfg["assignment_k"]),
        max_graph_points=int(geodesic_cfg["max_graph_points"]),
        include_archive_elites_in_support=bool(
            geodesic_cfg.get("include_archive_elites_in_support", False)
        ),
        reservoir_seed=int(geodesic_cfg.get("reservoir_seed") or config["experiment"]["seed"]),
        refresh_interval=int(geodesic_cfg["refresh_interval"]),
        selection_novelty_neighbors=int(geodesic_cfg["selection_novelty_neighbors"]),
        selection_novelty_floor=float(geodesic_cfg["selection_novelty_floor"]),
    )
    archive.add_visited_latents(latents)
    outside_flags: list[bool] = []
    for candidate, latent in zip(candidates, latents, strict=True):
        outside_flags.append(latent_is_outside_bounds(latent, archive))
        archive.add(
            candidate["genome"], candidate["rollout"], descriptor=latent, record_visited=False
        )
    if bool(geodesic_cfg.get("include_archive_elites_in_support", False)):
        archive.refresh_graph(force=True)
    return archive, latents, outside_flags


def update_rows_with_geodesic(
    rows: list[dict[str, Any]],
    latents: np.ndarray,
    archive: GeodesicGridArchive,
    outside_flags: list[bool],
) -> None:
    for row, latent, outside in zip(rows, latents, outside_flags, strict=True):
        cell, geodesic_distance = archive.cell_for(latent)
        row.update(
            {
                "latent_0": float(latent[0]),
                "latent_1": float(latent[1]),
                "cell_x": -1 if cell is None else int(cell[0]),
                "cell_y": -1 if cell is None else int(cell[1]),
                "latent_clipped": bool(outside),
                "assignment_geodesic_distance": geodesic_distance,
            }
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    metrics = summary["final_metrics"]
    diagnostics = summary["trajectory_diagnostics"]
    outside = summary["latent_outside_bounds"]
    reconstruction = summary["reconstruction_quality"]
    graph = summary["geodesic_graph"]
    print("PHASE 4 GEODESIC-NICHING SUMMARY")
    print(f"config: {summary['config']}")
    print(f"seed: {summary['seed']}")
    print(f"archive_assignment: {summary['archive_assignment']}")
    print(
        "autoencoder final loss: train={autoencoder_final_train_loss:.6f}, "
        "validation={autoencoder_final_validation_loss:.6f}".format(**summary)
    )
    print(
        "runtime: {total_evaluations} evals in {elapsed_seconds:.3f}s = "
        "{evaluations_per_second:.2f} eval/s".format(**summary)
    )
    print(
        "final archive: filled={filled_cells}, coverage={coverage:.3%}, "
        "QD-score={qd_score:.3f}, raw QD-score={raw_qd_score:.3f}, "
        "max fitness={max_fitness:.3f}, mean fitness={mean_fitness:.3f}, "
        "occupancy_entropy={occupancy_entropy:.3f}".format(**metrics)
    )
    print(
        "pairwise descriptor distances: euclidean={mean_pairwise_descriptor_euclidean:.3f}, "
        "knn-geodesic={mean_pairwise_descriptor_knn_geodesic:.3f}, "
        "finite={pairwise_geodesic_finite_fraction:.3f}".format(**metrics)
    )
    print(
        "trajectory loops: bootstrap={bootstrap_loop_count}/{bootstrap_count} "
        "({bootstrap_loop_fraction:.3%}), "
        "later={later_loop_count}/{later_count} ({later_loop_fraction:.3%})".format(**diagnostics)
    )
    print(
        "latent outside bounds: overall={overall_clipped_fraction:.3%}, "
        "post-bootstrap={post_bootstrap_clipped_fraction:.3%}".format(**outside)
    )
    print(
        "reconstruction MSE: overall={overall_mse:.6f}, near-still={near_still_mse:.6f}, "
        "loop={loop_mse:.6f}".format(**reconstruction)
    )
    print(
        "geodesic graph: support={support_points}, visited={visited_points}, "
        "centroids={centroid_count}, "
        "rolling_buffer={rolling_buffer_points}, k={graph_k}, refreshes={refresh_count}, "
        "finite_centroid_fraction={finite_centroid_fraction:.3f}, "
        "components={connected_components}, "
        "penalized={penalized_centroid_distances}".format(**graph)
    )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
