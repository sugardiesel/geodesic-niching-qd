"""Run Phase 3: AURORA-style learned behavior descriptor with Euclidean archive."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from algorithms.map_elites import GridArchive
from algorithms.policies import MLPPolicyGenome
from algorithms.trajectory_autoencoder import (
    AutoencoderTrainingResult,
    encode_sequences,
    reconstruct_sequences,
    save_autoencoder_checkpoint,
    train_autoencoder,
    trajectory_to_sequence,
)
from analysis.phase2_plots import save_metric_curves
from analysis.phase3_plots import (
    save_autoencoder_loss_plot,
    save_latent_archive_heatmap,
    save_reconstruction_examples,
)
from envs.forage_maze import ForageMaze2D, RolloutResult, load_env_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_aurora_euclidean.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{args.config} must contain a mapping.")

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ForageMaze2D(load_env_config(config["experiment"]["env_config"]))

    start = time.perf_counter()
    result = run_phase3(env, config)
    elapsed = time.perf_counter() - start

    archive: GridArchive = result["archive"]
    metrics: list[dict[str, Any]] = result["metrics"]
    evaluations: list[dict[str, Any]] = result["evaluations"]
    final_training: AutoencoderTrainingResult = result["final_training"]
    reconstruction_real: np.ndarray = result["reconstruction_real"]
    reconstruction_pred: np.ndarray = result["reconstruction_pred"]
    reconstruction_labels: list[str] = result["reconstruction_labels"]

    archive_csv = output_dir / "archive_cells.csv"
    metrics_csv = output_dir / "metrics.csv"
    evaluations_csv = output_dir / "evaluations.csv"
    loss_csv = output_dir / "autoencoder_loss.csv"
    diagnostics_csv = output_dir / "latent_dim_sanity.csv"
    checkpoint_path = output_dir / "autoencoder_final.pt"
    representative_npz = output_dir / "representative_trajectory_sample.npz"
    representative_csv = output_dir / "representative_trajectory_sample.csv"
    write_csv(archive_csv, archive.cell_rows())
    write_csv(metrics_csv, metrics)
    write_csv(evaluations_csv, evaluations)
    write_csv(
        loss_csv,
        [
            {"epoch": idx + 1, "train_loss": train, "validation_loss": validation}
            for idx, (train, validation) in enumerate(
                zip(final_training.train_losses, final_training.validation_losses, strict=False)
            )
        ],
    )
    write_csv(diagnostics_csv, result["latent_dim_sanity"])
    representative_files: list[str] = []
    if result["representative_sample"]:
        sample = result["representative_sample"]
        np.savez_compressed(
            representative_npz,
            sequences=sample["sequences"],
            final_positions=sample["final_positions"],
            latents=sample["latents"],
            evaluations=sample["evaluations"],
            fitnesses=sample["fitnesses"],
            path_lengths=sample["path_lengths"],
            loop_scores=sample["loop_scores"],
            is_loop=sample["is_loop"],
            is_near_still=sample["is_near_still"],
        )
        write_csv(representative_csv, sample["rows"])
        representative_files = [str(representative_npz), str(representative_csv)]
    save_autoencoder_checkpoint(
        final_training,
        checkpoint_path,
        sequence_length=int(config["trajectory"]["sequence_length"]),
        feature_names=list(config["trajectory"]["features"]),
        config=dict(config["autoencoder"]),
    )

    loss_files = save_autoencoder_loss_plot(
        final_training.train_losses,
        final_training.validation_losses,
        output_dir / "autoencoder_loss",
        ylabel=loss_ylabel(config),
    )
    recon_files = save_reconstruction_examples(
        reconstruction_real,
        reconstruction_pred,
        output_dir / "trajectory_reconstruction_examples",
        config["trajectory"]["features"],
        labels=reconstruction_labels,
    )
    heatmap_files = save_latent_archive_heatmap(
        archive,
        output_dir / "archive_heatmap_learned_latent",
        "Phase 3 AURORA-style archive: learned latent BD",
    )
    curve_files = save_metric_curves(
        metrics,
        output_dir / "qd_score_coverage_curves",
        title=result["plot_title"],
    )

    final_metrics = archive.metrics(include_pairwise=True)
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
        "latent_clipping": result["latent_clipping"],
        "reconstruction_quality": result["reconstruction_quality"],
        "latent_dim_sanity": result["latent_dim_sanity"],
        "retrain_events": result["retrain_events"],
        "representative_sample": result["representative_sample_summary"],
        "output_files": [
            str(archive_csv),
            str(metrics_csv),
            str(evaluations_csv),
            str(loss_csv),
            str(diagnostics_csv),
            str(checkpoint_path),
            *representative_files,
            *loss_files,
            *recon_files,
            *heatmap_files,
            *curve_files,
            str(output_dir / "phase3_summary.json"),
        ],
    }
    (output_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def run_phase3(env: ForageMaze2D, config: dict[str, Any]) -> dict[str, Any]:
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
            name=f"phase3_bootstrap_{evaluation}",
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
    archive, latents, clip_flags = build_archive_from_candidates(candidates, training, env, config)
    update_rows_with_latents(evaluation_rows, latents, archive, clip_flags)

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
    all_clip_flags = list(clip_flags)

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
            name=f"phase3_eval_{evaluation}",
            record=True,
        )
        sequence = trajectory_to_sequence(rollout.trajectory, sequence_length, feature_names)
        diag = trajectory_diagnostics(sequence)
        latent = encode_sequences(
            training.model, training.normalizer, sequence[None, :, :], training.device
        )[0]
        clipped = latent_is_outside_bounds(latent, archive)
        inserted, replaced, cell = archive.add(genome, rollout, descriptor=latent)
        candidates.append(
            {
                "genome": genome,
                "rollout": rollout,
                "sequence": sequence,
                "source": "mutated_elite",
                **diag,
            }
        )
        evaluation_rows.append(
            evaluation_row(
                evaluation + 1,
                "mutated_elite",
                rollout,
                diag,
                latent,
                cell,
                inserted,
                replaced,
                clipped,
            )
        )
        all_clip_flags.append(clipped)

        if evaluation + 1 in retrain_evaluations:
            training = train_autoencoder(
                stack_sequences(candidates), autoencoder_cfg, seed=seed + evaluation + 1
            )
            archive, rebuilt_latents, rebuilt_clip_flags = build_archive_from_candidates(
                candidates, training, env, config
            )
            retrain_events.append(
                {
                    "evaluation": evaluation + 1,
                    "train_loss": float(training.train_losses[-1]),
                    "validation_loss": float(training.validation_losses[-1]),
                    "candidate_count": len(candidates),
                    "rebuilt_archive_filled_cells": archive.occupied_count,
                    "rebuilt_clip_fraction": float(np.mean(rebuilt_clip_flags))
                    if rebuilt_clip_flags
                    else 0.0,
                    "normalization_audit": dict(training.normalization_audit),
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
    reconstruction_real = final_sequences[selected_indices]
    reconstruction_pred = final_reconstruction[selected_indices]
    reconstruction_quality = reconstruction_metrics(
        final_sequences, final_reconstruction, diagnostics
    )
    latent_dim_sanity = run_latent_dim_sanity(final_sequences, diagnostics, config, seed)
    representative_sample = build_representative_sample(
        candidates, final_sequences, diagnostics, training, config, seed
    )

    return {
        "archive": archive,
        "metrics": metrics_log,
        "evaluations": evaluation_rows,
        "final_training": training,
        "autoencoder_device": training.device,
        "reconstruction_real": reconstruction_real,
        "reconstruction_pred": reconstruction_pred,
        "reconstruction_labels": selected_labels,
        "trajectory_diagnostics": summarize_trajectory_diagnostics(
            diagnostics, bootstrap_evaluations
        ),
        "latent_clipping": summarize_latent_clipping(all_clip_flags, bootstrap_evaluations),
        "reconstruction_quality": reconstruction_quality,
        "latent_dim_sanity": latent_dim_sanity,
        "plot_title": str(
            config["experiment"].get(
                "plot_title", "Phase 3 AURORA-style Euclidean MAP-Elites progress"
            )
        ),
        "retrain_events": retrain_events,
        "bootstrap_normalization_audit": bootstrap_normalization_audit,
        "representative_sample": representative_sample,
        "representative_sample_summary": summarize_representative_sample(representative_sample),
    }


def autoencoder_loss_note(config: dict[str, Any]) -> str:
    normalize = bool(config["autoencoder"].get("normalize", True))
    derivative = float(config["autoencoder"].get("derivative_loss_weight", 0.0))
    parts = ["validation loss is MSE in the autoencoder training space"]
    if normalize:
        parts.append("trajectory features are normalized before training")
    else:
        parts.append("trajectory features are left in raw environment units")
    if derivative > 0.0:
        parts.append(f"training loss adds derivative MSE with weight {derivative:g}")
    return "; ".join(parts)


def loss_ylabel(config: dict[str, Any]) -> str:
    if bool(config["autoencoder"].get("normalize", True)):
        return "normalized reconstruction loss"
    return "raw reconstruction loss"


def build_archive_from_candidates(
    candidates: list[dict[str, Any]],
    training: AutoencoderTrainingResult,
    env: ForageMaze2D,
    config: dict[str, Any],
) -> tuple[GridArchive, np.ndarray, list[bool]]:
    algo = config["algorithm"]
    metrics_cfg = config["metrics"]
    autoencoder_cfg = config["autoencoder"]
    sequences = stack_sequences(candidates)
    latents = encode_sequences(training.model, training.normalizer, sequences, training.device)
    latent_min, latent_max = latent_bounds(
        latents, float(autoencoder_cfg["latent_bounds_padding_fraction"])
    )
    archive = GridArchive(
        bins=(int(algo["grid_bins"][0]), int(algo["grid_bins"][1])),
        bd_min=latent_min,
        bd_max=latent_max,
        genome_size=MLPPolicyGenome.parameter_count_for(
            env.observation_dim, int(algo["hidden_dim"])
        ),
        qd_score_fitness_floor=float(metrics_cfg["qd_score_fitness_floor"]),
        pairwise_geodesic_k=int(metrics_cfg["pairwise_geodesic_k"]),
        clip_descriptors=True,
    )
    clip_flags: list[bool] = []
    for candidate, latent in zip(candidates, latents, strict=True):
        clip_flags.append(latent_is_outside_bounds(latent, archive))
        archive.add(candidate["genome"], candidate["rollout"], descriptor=latent)
    return archive, latents, clip_flags


def stack_sequences(candidates: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([candidate["sequence"] for candidate in candidates], axis=0)


def latent_bounds(latents: np.ndarray, padding_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.min(latents, axis=0)
    high = np.max(latents, axis=0)
    span = np.maximum(high - low, 1e-6)
    padding = span * padding_fraction
    return low - padding, high + padding


def latent_is_outside_bounds(latent: np.ndarray, archive: GridArchive) -> bool:
    return bool(np.any(latent < archive.bd_min) or np.any(latent > archive.bd_max))


def trajectory_diagnostics(sequence: np.ndarray) -> dict[str, float | bool]:
    xy = sequence[:, :2]
    diffs = np.diff(xy, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(segment_lengths))
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    valid = segment_lengths > 1e-6
    total_abs_turn = 0.0
    if np.count_nonzero(valid) >= 3:
        vectors = diffs[valid]
        angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        total_abs_turn = float(np.sum(np.abs(np.diff(angles))))
    loop_score = total_abs_turn / (2.0 * math.pi)
    bbox = np.ptp(xy, axis=0)
    bbox_area = float(bbox[0] * bbox[1])
    is_loop = bool(path_length >= 1.0 and loop_score >= 1.0 and bbox_area >= 0.02)
    is_near_still = bool(path_length <= 0.20 and displacement <= 0.10)
    return {
        "path_length": path_length,
        "displacement": displacement,
        "total_abs_turn": total_abs_turn,
        "loop_score": loop_score,
        "bbox_area": bbox_area,
        "is_loop": is_loop,
        "is_near_still": is_near_still,
    }


def candidate_diag(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": candidate["source"],
        "path_length": float(candidate["path_length"]),
        "displacement": float(candidate["displacement"]),
        "total_abs_turn": float(candidate["total_abs_turn"]),
        "loop_score": float(candidate["loop_score"]),
        "bbox_area": float(candidate["bbox_area"]),
        "is_loop": bool(candidate["is_loop"]),
        "is_near_still": bool(candidate["is_near_still"]),
    }


def summarize_trajectory_diagnostics(
    diagnostics: list[dict[str, Any]],
    bootstrap_evaluations: int,
) -> dict[str, Any]:
    bootstrap = diagnostics[:bootstrap_evaluations]
    later = diagnostics[bootstrap_evaluations:]
    return {
        "loop_definition": (
            "path_length >= 1.0 and total_abs_turn/(2*pi) >= 1.0 and bbox_area >= 0.02"
        ),
        "near_still_definition": "path_length <= 0.20 and displacement <= 0.10",
        "bootstrap_count": len(bootstrap),
        "bootstrap_loop_count": count_bool(bootstrap, "is_loop"),
        "bootstrap_loop_fraction": fraction_bool(bootstrap, "is_loop"),
        "bootstrap_near_still_count": count_bool(bootstrap, "is_near_still"),
        "bootstrap_near_still_fraction": fraction_bool(bootstrap, "is_near_still"),
        "later_count": len(later),
        "later_loop_count": count_bool(later, "is_loop"),
        "later_loop_fraction": fraction_bool(later, "is_loop"),
        "later_near_still_count": count_bool(later, "is_near_still"),
        "later_near_still_fraction": fraction_bool(later, "is_near_still"),
        "max_loop_score": max((float(row["loop_score"]) for row in diagnostics), default=0.0),
        "median_path_length": float(np.median([float(row["path_length"]) for row in diagnostics])),
    }


def count_bool(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(1 for row in rows if bool(row[key])))


def fraction_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return count_bool(rows, key) / len(rows)


def summarize_latent_clipping(clip_flags: list[bool], bootstrap_evaluations: int) -> dict[str, Any]:
    flags = np.asarray(clip_flags, dtype=bool)
    later = (
        flags[bootstrap_evaluations:]
        if len(flags) > bootstrap_evaluations
        else np.asarray([], dtype=bool)
    )
    return {
        "overall_clipped_count": int(np.count_nonzero(flags)),
        "overall_clipped_fraction": float(np.mean(flags)) if len(flags) else 0.0,
        "post_bootstrap_clipped_count": int(np.count_nonzero(later)),
        "post_bootstrap_clipped_fraction": float(np.mean(later)) if len(later) else 0.0,
    }


def reconstruction_metrics(
    real: np.ndarray,
    reconstructed: np.ndarray,
    diagnostics: list[dict[str, Any]],
) -> dict[str, float | int]:
    errors = np.mean((real - reconstructed) ** 2, axis=(1, 2))
    xy_errors = np.mean((real[:, :, :2] - reconstructed[:, :, :2]) ** 2, axis=(1, 2))
    loop_mask = np.asarray([bool(row["is_loop"]) for row in diagnostics])
    still_mask = np.asarray([bool(row["is_near_still"]) for row in diagnostics])
    return {
        "overall_mse": float(np.mean(errors)),
        "overall_xy_mse": float(np.mean(xy_errors)),
        "loop_count": int(np.count_nonzero(loop_mask)),
        "loop_mse": masked_mean(errors, loop_mask),
        "loop_xy_mse": masked_mean(xy_errors, loop_mask),
        "near_still_count": int(np.count_nonzero(still_mask)),
        "near_still_mse": masked_mean(errors, still_mask),
        "near_still_xy_mse": masked_mean(xy_errors, still_mask),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def select_reconstruction_examples(
    diagnostics: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    path_lengths = np.asarray([float(row["path_length"]) for row in diagnostics])
    loop_scores = np.asarray([float(row["loop_score"]) for row in diagnostics])
    near_indices = list(np.argsort(path_lengths)[:2])
    loop_indices = list(np.argsort(-loop_scores)[:2])
    selected: list[int] = []
    labels: list[str] = []
    for idx in near_indices:
        if idx not in selected:
            selected.append(int(idx))
            labels.append(
                f"near-still eval {idx + 1}, path={path_lengths[idx]:.3f}, "
                f"loop={loop_scores[idx]:.2f}"
            )
    for idx in loop_indices:
        if idx not in selected:
            selected.append(int(idx))
            labels.append(
                f"high-loop eval {idx + 1}, path={path_lengths[idx]:.3f}, "
                f"loop={loop_scores[idx]:.2f}"
            )
    return np.asarray(selected, dtype=int), labels


def run_latent_dim_sanity(
    sequences: np.ndarray,
    diagnostics: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    sanity_cfg = config.get("autoencoder_sanity", {})
    if not sanity_cfg or not bool(sanity_cfg.get("enabled", False)):
        return []
    dims = [int(value) for value in sanity_cfg.get("latent_dims", [])]
    epochs = int(sanity_cfg.get("epochs", 30))
    bootstrap_count = int(config["algorithm"]["bootstrap_evaluations"])
    base_cfg = copy.deepcopy(config["autoencoder"])
    base_cfg["epochs"] = epochs
    rows: list[dict[str, Any]] = []
    train_sets = {
        "bootstrap_only": sequences[:bootstrap_count],
        "all_evaluations": sequences,
    }
    for train_name, train_sequences in train_sets.items():
        for latent_dim in dims:
            ae_cfg = copy.deepcopy(base_cfg)
            ae_cfg["latent_dim"] = latent_dim
            training = train_autoencoder(
                train_sequences, ae_cfg, seed=seed + latent_dim + len(train_name)
            )
            reconstructed = reconstruct_sequences(
                training.model, training.normalizer, sequences, training.device
            )
            metrics = reconstruction_metrics(sequences, reconstructed, diagnostics)
            rows.append(
                {
                    "train_set": train_name,
                    "latent_dim": latent_dim,
                    "epochs": epochs,
                    "final_train_loss": float(training.train_losses[-1]),
                    "final_validation_loss": float(training.validation_losses[-1]),
                    **metrics,
                }
            )
    return rows


def build_representative_sample(
    candidates: list[dict[str, Any]],
    sequences: np.ndarray,
    diagnostics: list[dict[str, Any]],
    training: AutoencoderTrainingResult,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    sample_cfg = config.get("representative_sample", {})
    if not sample_cfg or not bool(sample_cfg.get("enabled", False)):
        return {}
    sample_count = min(int(sample_cfg.get("sample_count", 400)), len(candidates))
    strata = max(1, min(int(sample_cfg.get("strata", 20)), sample_count))
    rng = np.random.default_rng(int(sample_cfg.get("seed", seed + 2026)))
    indices = stratified_indices(len(candidates), sample_count, strata, rng)
    sample_sequences = sequences[indices]
    sample_latents = encode_sequences(
        training.model, training.normalizer, sample_sequences, training.device
    )
    final_positions = np.asarray(
        [candidates[int(idx)]["rollout"].final_position for idx in indices],
        dtype=np.float64,
    )
    fitnesses = np.asarray(
        [float(candidates[int(idx)]["rollout"].fitness) for idx in indices], dtype=np.float64
    )
    path_lengths = np.asarray(
        [float(diagnostics[int(idx)]["path_length"]) for idx in indices], dtype=np.float64
    )
    loop_scores = np.asarray(
        [float(diagnostics[int(idx)]["loop_score"]) for idx in indices], dtype=np.float64
    )
    is_loop = np.asarray([bool(diagnostics[int(idx)]["is_loop"]) for idx in indices], dtype=bool)
    is_near_still = np.asarray(
        [bool(diagnostics[int(idx)]["is_near_still"]) for idx in indices], dtype=bool
    )
    rows: list[dict[str, Any]] = []
    for sample_idx, eval_idx in enumerate(indices):
        rollout = candidates[int(eval_idx)]["rollout"]
        rows.append(
            {
                "sample_index": sample_idx,
                "evaluation": int(eval_idx) + 1,
                "source": candidates[int(eval_idx)]["source"],
                "fitness": float(rollout.fitness),
                "final_x": float(rollout.final_position[0]),
                "final_y": float(rollout.final_position[1]),
                "latent_0": float(sample_latents[sample_idx, 0]),
                "latent_1": float(sample_latents[sample_idx, 1]),
                "path_length": float(path_lengths[sample_idx]),
                "loop_score": float(loop_scores[sample_idx]),
                "is_loop": bool(is_loop[sample_idx]),
                "is_near_still": bool(is_near_still[sample_idx]),
            }
        )
    return {
        "indices": indices,
        "sequences": sample_sequences,
        "final_positions": final_positions,
        "latents": sample_latents,
        "evaluations": indices + 1,
        "fitnesses": fitnesses,
        "path_lengths": path_lengths,
        "loop_scores": loop_scores,
        "is_loop": is_loop,
        "is_near_still": is_near_still,
        "rows": rows,
    }


def stratified_indices(
    total_count: int,
    sample_count: int,
    strata: int,
    rng: np.random.Generator,
) -> np.ndarray:
    edges = np.linspace(0, total_count, strata + 1, dtype=int)
    base = sample_count // strata
    remainder = sample_count % strata
    selected: list[int] = []
    for stratum in range(strata):
        start = int(edges[stratum])
        end = int(edges[stratum + 1])
        if end <= start:
            continue
        quota = base + (1 if stratum < remainder else 0)
        quota = min(quota, end - start)
        selected.extend(
            rng.choice(np.arange(start, end), size=quota, replace=False).astype(int).tolist()
        )
    return np.asarray(sorted(selected), dtype=int)


def summarize_representative_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if not sample:
        return {"enabled": False}
    latents = np.asarray(sample["latents"], dtype=np.float64)
    pairwise = pairwise_upper_distances(latents)
    evaluations = np.asarray(sample["evaluations"], dtype=np.int64)
    return {
        "enabled": True,
        "sample_count": int(len(evaluations)),
        "evaluation_min": int(np.min(evaluations)),
        "evaluation_max": int(np.max(evaluations)),
        "loop_fraction": float(np.mean(sample["is_loop"])),
        "near_still_fraction": float(np.mean(sample["is_near_still"])),
        "mean_pairwise_latent_euclidean": float(np.mean(pairwise))
        if len(pairwise)
        else float("nan"),
        "median_pairwise_latent_euclidean": float(np.median(pairwise))
        if len(pairwise)
        else float("nan"),
        "max_pairwise_latent_euclidean": float(np.max(pairwise)) if len(pairwise) else float("nan"),
    }


def pairwise_upper_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=np.float64)
    delta = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    tri = np.triu_indices(len(points), k=1)
    return distances[tri]


def update_rows_with_latents(
    rows: list[dict[str, Any]],
    latents: np.ndarray,
    archive: GridArchive,
    clip_flags: list[bool],
) -> None:
    for row, latent, clipped in zip(rows, latents, clip_flags, strict=True):
        cell = archive.cell_for(latent)
        row.update(
            {
                "latent_0": float(latent[0]),
                "latent_1": float(latent[1]),
                "cell_x": -1 if cell is None else int(cell[0]),
                "cell_y": -1 if cell is None else int(cell[1]),
                "latent_clipped": bool(clipped),
            }
        )


def evaluation_row(
    evaluation: int,
    source: str,
    rollout: RolloutResult,
    diag: dict[str, Any],
    latent: np.ndarray | None = None,
    cell: tuple[int, int] | None = None,
    inserted: bool = False,
    replaced: bool = False,
    latent_clipped: bool = False,
) -> dict[str, Any]:
    return {
        "evaluation": evaluation,
        "source": source,
        "fitness": rollout.fitness,
        "final_x": float(rollout.final_position[0]),
        "final_y": float(rollout.final_position[1]),
        "latent_0": "" if latent is None else float(latent[0]),
        "latent_1": "" if latent is None else float(latent[1]),
        "cell_x": "" if cell is None else int(cell[0]),
        "cell_y": "" if cell is None else int(cell[1]),
        "inserted": bool(inserted),
        "replaced": bool(replaced),
        "latent_clipped": bool(latent_clipped),
        "steps": rollout.steps,
        "food_collected": rollout.food_collected,
        "wall_collisions": rollout.wall_collisions,
        "hazard_contacts": rollout.hazard_contacts,
        "path_length": float(diag["path_length"]),
        "displacement": float(diag["displacement"]),
        "loop_score": float(diag["loop_score"]),
        "bbox_area": float(diag["bbox_area"]),
        "is_loop": bool(diag["is_loop"]),
        "is_near_still": bool(diag["is_near_still"]),
    }


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
    clipping = summary["latent_clipping"]
    reconstruction = summary["reconstruction_quality"]
    print("PHASE 3 AURORA-STYLE EUCLIDEAN BASELINE SUMMARY")
    print(f"config: {summary['config']}")
    print(f"env_config: {summary['env_config']}")
    print(f"seed: {summary['seed']}")
    print(f"descriptor: {summary['descriptor']}")
    print(f"archive_assignment: {summary['archive_assignment']}")
    print(f"autoencoder_device: {summary['autoencoder_device']}")
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
        "max fitness={max_fitness:.3f}, mean fitness={mean_fitness:.3f}".format(**metrics)
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
        "latent clipping: overall={overall_clipped_fraction:.3%}, "
        "post-bootstrap={post_bootstrap_clipped_fraction:.3%}".format(**clipping)
    )
    print(
        "reconstruction MSE: overall={overall_mse:.6f}, near-still={near_still_mse:.6f}, "
        "loop={loop_mse:.6f}".format(**reconstruction)
    )
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
