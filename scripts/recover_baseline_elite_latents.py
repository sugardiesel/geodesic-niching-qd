"""Resolve historical elite identities by replaying only ambiguous saved policies."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.policies import MLPPolicyGenome
from algorithms.trajectory_autoencoder import (
    encode_sequences,
    load_autoencoder_checkpoint,
    trajectory_to_sequence,
)
from envs.forage_maze import load_env_config
from scripts.analyze_common_behavior_space import DATASETS, discrete_key, read_csv, write_csv
from scripts.correct_phase1_distance_diagnostic import HistoricalPhase1ReplayMaze
from scripts.run_phase3_aurora_euclidean import trajectory_diagnostics


def restore_genomes(rows, config, obs_dim, requested):
    """Reconstruct genomes from saved history, without new fitness evaluations."""
    algo = config["algorithm"]
    bootstrap = int(algo["bootstrap_evaluations"])
    retrain = min(algo["retrain_evaluations"])
    if not requested or max(requested) > retrain:
        raise ValueError("Recovery requires nonempty evaluation indices before the first retrain.")
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    hidden = int(algo["hidden_dim"])
    occupied = {}
    recovered = {}
    for row in rows[: max(bootstrap, max(requested))]:
        evaluation = int(row["evaluation"])
        if evaluation <= bootstrap:
            genome = MLPPolicyGenome.random(obs_dim, hidden, rng, scale=float(algo["init_scale"]))
        else:
            cells = sorted(occupied)
            parent = occupied[cells[int(rng.integers(0, len(cells)))]][1]
            genome = parent.mutate(
                rng,
                sigma=float(algo["mutation_sigma"]),
                mutation_probability=float(algo["mutation_probability"]),
            )
        if evaluation in requested:
            recovered[evaluation] = genome
        cell = (int(row["cell_x"]), int(row["cell_y"]))
        fitness = float(row["fitness"])
        inserted = cell not in occupied or fitness > occupied[cell][0]
        if evaluation > bootstrap and inserted != (row["inserted"] == "True"):
            raise ValueError(f"Saved archive history disagrees at evaluation {evaluation}.")
        if inserted:
            occupied[cell] = (fitness, genome)
    return recovered


def recover_seed(seed_dir: Path) -> int:
    summary = json.loads((seed_dir / "phase3_summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(summary["config"].replace("\\", "/")).read_text(encoding="utf-8"))
    rows = read_csv(seed_dir / "evaluations.csv")
    retrain = max(summary["retrain_evaluations"])
    updated = {
        (row["cell_x"], row["cell_y"])
        for row in rows
        if int(row["evaluation"]) > retrain and row["inserted"] == "True"
    }
    requested = set()
    for elite in read_csv(seed_dir / "archive_cells.csv"):
        if (elite["cell_x"], elite["cell_y"]) in updated:
            continue
        candidates = [
            row
            for row in rows
            if int(row["evaluation"]) <= retrain
            and discrete_key(row) == discrete_key(elite)
            and abs(float(row["fitness"]) - float(elite["fitness"])) <= 1e-9
        ]
        if len(candidates) > 1:
            requested.update(int(row["evaluation"]) for row in candidates)
    if not requested:
        return 0
    # Baseline B predates the one-step food-respawn correction.
    env = HistoricalPhase1ReplayMaze(load_env_config(config["experiment"]["env_config"]))
    genomes = restore_genomes(rows, config, env.observation_dim, requested)
    checkpoint = seed_dir / "autoencoder_final.pt"
    training, metadata = load_autoencoder_checkpoint(checkpoint, device="cpu")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = []
    for evaluation in sorted(requested):
        saved = rows[evaluation - 1]
        rollout = env.rollout(
            genomes[evaluation].to_policy(),
            seed=int(summary["seed"]) + (evaluation - 1) * 1009,
            record=True,
        )
        sequence = trajectory_to_sequence(
            rollout.trajectory, metadata["sequence_length"], metadata["feature_names"]
        )
        diagnostics = trajectory_diagnostics(sequence)
        errors = [
            abs(float(getattr(rollout, key)) - float(saved[key]))
            for key in ("fitness", "steps", "food_collected", "wall_collisions", "hazard_contacts")
        ]
        errors.extend(
            abs(float(diagnostics[key]) - float(saved[key]))
            for key in ("path_length", "displacement", "loop_score", "bbox_area")
        )
        if max(errors) > 1e-8:
            raise ValueError(
                f"{seed_dir}, evaluation {evaluation}: replay differs by {max(errors)}."
            )
        code = encode_sequences(
            training.model, training.normalizer, sequence[None], training.device
        )[0]
        output.append(
            {
                "evaluation": evaluation,
                "latent_0": float(code[0]),
                "latent_1": float(code[1]),
                "fitness": rollout.fitness,
                "max_replay_error": max(errors),
                "respawn_timing": "historical_pre_audit",
                "checkpoint_sha256": checkpoint_hash,
            }
        )
    write_csv(seed_dir / "recovered_pre_retrain_latents.csv", output)
    print(
        f"{seed_dir}: verified {len(output)} existing policies; no search or training.", flush=True
    )
    return len(output)


def main() -> None:
    total = 0
    for dataset in DATASETS:
        for seed_dir in sorted(dataset["baseline_dir"].glob("seed_*")):
            total += recover_seed(seed_dir)
    print(f"Total replay evaluations: {total}")


if __name__ == "__main__":
    main()
