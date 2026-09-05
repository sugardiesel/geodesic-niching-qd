"""Compare saved policies under historical and corrected food-respawn timing.

Genome recovery follows recorded archive outcomes and the original RNG stream.
It never evaluates new candidates or changes the recorded search history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.geodesic_archive import GeodesicGridArchive
from algorithms.policies import MLPPolicyGenome
from algorithms.trajectory_autoencoder import trajectory_to_sequence
from envs.forage_maze import ForageMaze2D, load_env_config
from scripts.analyze_common_behavior_space import read_csv, write_csv
from scripts.correct_phase1_distance_diagnostic import HistoricalPhase1ReplayMaze
from scripts.recover_baseline_elite_latents import restore_genomes
from scripts.run_phase3_aurora_euclidean import latent_bounds, trajectory_diagnostics


def restore_geodesic_genomes(rows, config, obs_dim, requested):
    algo = config["algorithm"]
    bootstrap = int(algo["bootstrap_evaluations"])
    if not requested or max(requested) >= min(algo["retrain_evaluations"]):
        raise ValueError("Only pre-retraining history can be reconstructed here.")
    latents = np.asarray(
        [[float(row["latent_0"]), float(row["latent_1"])] for row in rows], dtype=np.float32
    )
    low, high = latent_bounds(
        latents[:bootstrap], float(config["autoencoder"]["latent_bounds_padding_fraction"])
    )
    geo = config["geodesic"]
    archive = GeodesicGridArchive(
        bins=tuple(algo["grid_bins"]),
        bd_min=low,
        bd_max=high,
        genome_size=MLPPolicyGenome.parameter_count_for(obs_dim, int(algo["hidden_dim"])),
        graph_k=int(geo["graph_k"]),
        assignment_k=int(geo["assignment_k"]),
        max_graph_points=int(geo["max_graph_points"]),
        refresh_interval=int(geo["refresh_interval"]),
        selection_novelty_neighbors=int(geo["selection_novelty_neighbors"]),
        selection_novelty_floor=float(geo["selection_novelty_floor"]),
    )
    archive.add_visited_latents(latents[:bootstrap])
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    recovered = {}
    for row in rows[: max(bootstrap, max(requested))]:
        evaluation = int(row["evaluation"])
        if evaluation <= bootstrap:
            genome = MLPPolicyGenome.random(
                obs_dim, int(algo["hidden_dim"]), rng, scale=float(algo["init_scale"])
            )
        else:
            genome = archive.sample_elite(rng, obs_dim, int(algo["hidden_dim"])).mutate(
                rng,
                sigma=float(algo["mutation_sigma"]),
                mutation_probability=float(algo["mutation_probability"]),
            )
        if evaluation in requested:
            recovered[evaluation] = genome
        saved_outcome = SimpleNamespace(
            **{
                key: float(row[key])
                for key in (
                    "fitness",
                    "steps",
                    "food_collected",
                    "wall_collisions",
                    "hazard_contacts",
                )
            }
        )
        inserted, _, cell, _ = archive.add(
            genome,
            saved_outcome,
            latents[evaluation - 1],
            evaluation=evaluation if evaluation > bootstrap else None,
            record_visited=evaluation > bootstrap,
        )
        expected_cell = (int(row["cell_x"]), int(row["cell_y"]))
        if cell != expected_cell or (
            evaluation > bootstrap and inserted != (row["inserted"] == "True")
        ):
            raise ValueError(f"Recorded geodesic history differs at evaluation {evaluation}.")
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--evaluations", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contribution = (args.seed_dir / "phase4_summary.json").exists()
    summary_file = "phase4_summary.json" if contribution else "phase3_summary.json"
    summary = json.loads((args.seed_dir / summary_file).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(summary["config"].replace("\\", "/")).read_text())
    env_config = load_env_config(config["experiment"]["env_config"])
    rows = read_csv(args.seed_dir / "evaluations.csv")
    current = ForageMaze2D(env_config)
    recover = restore_geodesic_genomes if contribution else restore_genomes
    genomes = recover(rows, config, current.observation_dim, set(args.evaluations))
    output = []
    for timing, env in [
        ("corrected", current),
        ("historical", HistoricalPhase1ReplayMaze(env_config)),
    ]:
        for evaluation in sorted(genomes):
            saved = rows[evaluation - 1]
            rollout = env.rollout(
                genomes[evaluation].to_policy(),
                seed=int(summary["seed"]) + (evaluation - 1) * 1009,
                record=True,
            )
            sequence = trajectory_to_sequence(
                rollout.trajectory,
                config["trajectory"]["sequence_length"],
                config["trajectory"]["features"],
            )
            observed = trajectory_diagnostics(sequence)
            observed.update(
                {
                    key: getattr(rollout, key)
                    for key in (
                        "fitness",
                        "steps",
                        "food_collected",
                        "wall_collisions",
                        "hazard_contacts",
                    )
                }
            )
            errors = {
                key + "_difference": float(observed[key]) - float(saved[key])
                for key in (
                    "fitness",
                    "steps",
                    "food_collected",
                    "wall_collisions",
                    "hazard_contacts",
                    "path_length",
                    "displacement",
                    "loop_score",
                    "bbox_area",
                )
            }
            output.append(
                {
                    "seed_directory": str(args.seed_dir),
                    "evaluation": evaluation,
                    "respawn_timing": timing,
                    "max_absolute_difference": max(map(abs, errors.values())),
                    **errors,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, output)
    for row in output:
        print(row, flush=True)


if __name__ == "__main__":
    main()
