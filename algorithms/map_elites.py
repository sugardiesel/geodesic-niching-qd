"""From-scratch hand-crafted MAP-Elites implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithms.archive_metrics import (
    occupancy_entropy,
    pairwise_distance_metrics,
    raw_qd_score,
    shifted_qd_score,
)
from algorithms.policies import MLPPolicyGenome
from envs.forage_maze import ForageMaze2D, RolloutResult


@dataclass
class EvaluationRecord:
    evaluation: int
    source: str
    fitness: float
    descriptor: np.ndarray
    cell: tuple[int, int]
    inserted: bool
    replaced: bool
    steps: int
    food_collected: int
    wall_collisions: int
    hazard_contacts: int


class GridArchive:
    """MAP-Elites archive indexed by a 2D behavior descriptor."""

    def __init__(
        self,
        bins: tuple[int, int],
        bd_min: np.ndarray,
        bd_max: np.ndarray,
        genome_size: int,
        qd_score_fitness_floor: float = -5.0,
        pairwise_geodesic_k: int = 10,
        clip_descriptors: bool = False,
    ):
        self.bins = (int(bins[0]), int(bins[1]))
        self.bd_min = np.asarray(bd_min, dtype=np.float64)
        self.bd_max = np.asarray(bd_max, dtype=np.float64)
        self.genome_size = int(genome_size)
        self.qd_score_fitness_floor = float(qd_score_fitness_floor)
        self.pairwise_geodesic_k = int(pairwise_geodesic_k)
        self.clip_descriptors = bool(clip_descriptors)

        shape = self.bins
        self.fitness = np.full(shape, -np.inf, dtype=np.float64)
        self.occupied = np.zeros(shape, dtype=bool)
        self.genomes = np.zeros((*shape, self.genome_size), dtype=np.float64)
        self.descriptors = np.zeros((*shape, 2), dtype=np.float64)
        self.steps = np.zeros(shape, dtype=np.int32)
        self.food_collected = np.zeros(shape, dtype=np.int32)
        self.wall_collisions = np.zeros(shape, dtype=np.int32)
        self.hazard_contacts = np.zeros(shape, dtype=np.int32)

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.occupied))

    @property
    def capacity(self) -> int:
        return int(self.bins[0] * self.bins[1])

    def cell_for(self, descriptor: np.ndarray) -> tuple[int, int] | None:
        descriptor = np.asarray(descriptor, dtype=np.float64)
        if self.clip_descriptors:
            descriptor = np.clip(descriptor, self.bd_min, self.bd_max)
        elif np.any(descriptor < self.bd_min) or np.any(descriptor > self.bd_max):
            return None
        scaled = (descriptor - self.bd_min) / (self.bd_max - self.bd_min)
        indices = np.floor(scaled * np.asarray(self.bins, dtype=np.float64)).astype(int)
        indices = np.clip(indices, 0, np.asarray(self.bins) - 1)
        return int(indices[0]), int(indices[1])

    def add(
        self,
        genome: MLPPolicyGenome,
        rollout: RolloutResult,
        descriptor: np.ndarray | None = None,
    ) -> tuple[bool, bool, tuple[int, int] | None]:
        descriptor = (
            rollout.final_position
            if descriptor is None
            else np.asarray(descriptor, dtype=np.float64)
        )
        stored_descriptor = (
            np.clip(descriptor, self.bd_min, self.bd_max) if self.clip_descriptors else descriptor
        )
        cell = self.cell_for(descriptor)
        if cell is None:
            return False, False, None
        old_occupied = bool(self.occupied[cell])
        if old_occupied and rollout.fitness <= float(self.fitness[cell]):
            return False, False, cell
        self.occupied[cell] = True
        self.fitness[cell] = rollout.fitness
        self.genomes[cell] = genome.weights
        self.descriptors[cell] = stored_descriptor
        self.steps[cell] = rollout.steps
        self.food_collected[cell] = rollout.food_collected
        self.wall_collisions[cell] = rollout.wall_collisions
        self.hazard_contacts[cell] = rollout.hazard_contacts
        return True, old_occupied, cell

    def sample_elite(
        self,
        rng: np.random.Generator,
        obs_dim: int,
        hidden_dim: int,
    ) -> MLPPolicyGenome:
        occupied_cells = np.argwhere(self.occupied)
        if len(occupied_cells) == 0:
            raise RuntimeError("Cannot sample from an empty archive.")
        cell = occupied_cells[int(rng.integers(0, len(occupied_cells)))]
        weights = self.genomes[int(cell[0]), int(cell[1])].copy()
        return MLPPolicyGenome(weights=weights, obs_dim=obs_dim, hidden_dim=hidden_dim)

    def metrics(self, include_pairwise: bool = True) -> dict[str, float | int]:
        occupied_fitness = self.fitness[self.occupied]
        coverage = self.occupied_count / self.capacity
        if len(occupied_fitness) == 0:
            return {
                "filled_cells": 0,
                "coverage": 0.0,
                "qd_score": 0.0,
                "raw_qd_score": 0.0,
                "qd_score_fitness_floor": self.qd_score_fitness_floor,
                "max_fitness": float("nan"),
                "mean_fitness": float("nan"),
                "occupancy_entropy": 0.0,
                "mean_pairwise_descriptor_euclidean": float("nan"),
                "mean_pairwise_descriptor_knn_geodesic": float("nan"),
                "pairwise_geodesic_k": self.pairwise_geodesic_k,
                "pairwise_geodesic_finite_fraction": 0.0,
                "pairwise_geodesic_disconnected_pairs": 0,
            }
        if include_pairwise:
            descriptor_points = self.descriptors[self.occupied]
            pairwise = pairwise_distance_metrics(descriptor_points, self.pairwise_geodesic_k)
            mean_pairwise_euclidean = pairwise.mean_euclidean
            mean_pairwise_geodesic = pairwise.mean_knn_geodesic
            pairwise_k = pairwise.knn_k
            finite_fraction = pairwise.finite_geodesic_fraction
            disconnected_pairs = pairwise.disconnected_pair_count
        else:
            mean_pairwise_euclidean = float("nan")
            mean_pairwise_geodesic = float("nan")
            pairwise_k = self.pairwise_geodesic_k
            finite_fraction = float("nan")
            disconnected_pairs = -1
        return {
            "filled_cells": self.occupied_count,
            "coverage": coverage,
            "qd_score": shifted_qd_score(occupied_fitness, self.qd_score_fitness_floor),
            "raw_qd_score": raw_qd_score(occupied_fitness),
            "qd_score_fitness_floor": self.qd_score_fitness_floor,
            "max_fitness": float(np.max(occupied_fitness)),
            "mean_fitness": float(np.mean(occupied_fitness)),
            "occupancy_entropy": occupancy_entropy(self.occupied_count, self.capacity),
            "mean_pairwise_descriptor_euclidean": mean_pairwise_euclidean,
            "mean_pairwise_descriptor_knn_geodesic": mean_pairwise_geodesic,
            "pairwise_geodesic_k": pairwise_k,
            "pairwise_geodesic_finite_fraction": finite_fraction,
            "pairwise_geodesic_disconnected_pairs": disconnected_pairs,
        }

    def cell_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ix in range(self.bins[0]):
            for iy in range(self.bins[1]):
                if not self.occupied[ix, iy]:
                    continue
                rows.append(
                    {
                        "cell_x": ix,
                        "cell_y": iy,
                        "fitness": float(self.fitness[ix, iy]),
                        "descriptor_x": float(self.descriptors[ix, iy, 0]),
                        "descriptor_y": float(self.descriptors[ix, iy, 1]),
                        "steps": int(self.steps[ix, iy]),
                        "food_collected": int(self.food_collected[ix, iy]),
                        "wall_collisions": int(self.wall_collisions[ix, iy]),
                        "hazard_contacts": int(self.hazard_contacts[ix, iy]),
                    }
                )
        return rows


def run_map_elites(
    env: ForageMaze2D,
    config: dict[str, Any],
) -> tuple[GridArchive, list[EvaluationRecord], list[dict[str, float | int]]]:
    algo = config["algorithm"]
    seed = int(config["experiment"]["seed"])
    rng = np.random.default_rng(seed)

    obs_dim = env.observation_dim
    hidden_dim = int(algo["hidden_dim"])
    genome_size = MLPPolicyGenome.parameter_count_for(obs_dim, hidden_dim)
    metrics_cfg = config.get("metrics", {})
    archive = GridArchive(
        bins=(int(algo["grid_bins"][0]), int(algo["grid_bins"][1])),
        bd_min=env.world_min,
        bd_max=env.world_max,
        genome_size=genome_size,
        qd_score_fitness_floor=float(metrics_cfg.get("qd_score_fitness_floor", -5.0)),
        pairwise_geodesic_k=int(metrics_cfg.get("pairwise_geodesic_k", 10)),
    )

    total_evaluations = int(algo["total_evaluations"])
    initial_random = int(algo["initial_random"])
    mutation_sigma = float(algo["mutation_sigma"])
    mutation_probability = float(algo["mutation_probability"])
    log_interval = int(algo["log_interval"])
    pairwise_log_interval = int(metrics_cfg.get("pairwise_log_interval", 1000))

    records: list[EvaluationRecord] = []
    metrics_log: list[dict[str, float | int]] = []

    for evaluation in range(total_evaluations):
        if evaluation < initial_random or archive.occupied_count == 0:
            genome = MLPPolicyGenome.random(
                obs_dim, hidden_dim, rng, scale=float(algo["init_scale"])
            )
            source = "random_init"
        else:
            parent = archive.sample_elite(rng, obs_dim=obs_dim, hidden_dim=hidden_dim)
            genome = parent.mutate(
                rng, sigma=mutation_sigma, mutation_probability=mutation_probability
            )
            source = "mutated_elite"

        rollout_seed = seed + evaluation * 1009
        rollout = env.rollout(
            genome.to_policy(), seed=rollout_seed, name=f"phase2_eval_{evaluation}", record=False
        )
        inserted, replaced, cell = archive.add(genome, rollout)
        if cell is None:
            cell = (-1, -1)
        records.append(
            EvaluationRecord(
                evaluation=evaluation + 1,
                source=source,
                fitness=rollout.fitness,
                descriptor=rollout.final_position.copy(),
                cell=cell,
                inserted=inserted,
                replaced=replaced,
                steps=rollout.steps,
                food_collected=rollout.food_collected,
                wall_collisions=rollout.wall_collisions,
                hazard_contacts=rollout.hazard_contacts,
            )
        )

        if (
            (evaluation + 1) % log_interval == 0
            or evaluation == 0
            or evaluation + 1 == total_evaluations
        ):
            is_final = evaluation + 1 == total_evaluations
            include_pairwise = is_final or (
                pairwise_log_interval > 0 and (evaluation + 1) % pairwise_log_interval == 0
            )
            metrics = archive.metrics(include_pairwise=include_pairwise)
            metrics_log.append({"evaluation": evaluation + 1, **metrics})

    return archive, records, metrics_log


def archive_entropy(archive: GridArchive) -> float:
    """Deprecated alias for coverage-derived occupancy entropy."""

    return occupancy_entropy(archive.occupied_count, archive.capacity)
