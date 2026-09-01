"""Geodesic niching archive for learned 2D behavior descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from algorithms.archive_metrics import (
    occupancy_entropy,
    pairwise_distance_metrics,
    raw_qd_score,
    shifted_qd_score,
)
from algorithms.knn_graph import (
    build_endpoint_manifold_graph,
    endpoint_shortest_path_distances,
)
from algorithms.policies import MLPPolicyGenome
from envs.forage_maze import RolloutResult


def compose_real_routing_support(
    elite_points: np.ndarray,
    rolling_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable, deduplicated union with elite points taking precedence."""

    points: list[np.ndarray] = []
    kinds: list[str] = []
    seen: set[bytes] = set()
    for kind, collection in [
        ("archive_elite", np.asarray(elite_points, dtype=np.float64)),
        ("rolling_buffer", np.asarray(rolling_points, dtype=np.float64)),
    ]:
        for point in collection:
            stored = np.asarray(point, dtype=np.float64)
            key = stored.tobytes()
            if key in seen:
                continue
            seen.add(key)
            points.append(stored)
            kinds.append(kind)
    if not points:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=object)
    return np.asarray(points, dtype=np.float64), np.asarray(kinds, dtype=object)


@dataclass(frozen=True)
class GeodesicGraphStats:
    support_points: int
    visited_points: int
    reservoir_points: int
    rolling_buffer_points: int
    archive_elite_points: int
    centroid_count: int
    knn_k: int
    refresh_count: int
    finite_centroid_fraction: float
    connected_components: int
    largest_component_fraction: float
    penalized_centroid_distances: int
    cumulative_penalized_centroid_distances: int
    refreshes_with_penalty: int
    penalty_distance: float
    reservoir_strategy: str
    reservoir_seed: int


class GeodesicGridArchive:
    """Grid archive whose niche assignment uses k-NN graph geodesic distance.

    The archive still stores a 25x25 grid of latent-space centroids for direct comparison with the
    Euclidean learned-BD baseline. A candidate is assigned to the centroid with the shortest
    graph-geodesic distance. The graph is rebuilt periodically over a rolling buffer of recent
    candidate latents and all grid centroids. The optional elite-support mode also retains current
    archive-elite latents while they occupy a niche, so successful regions do not disappear merely
    because they age out of the rolling buffer. Centroids are assignment endpoints only: their
    incident edges must be selected by the combined-set k-NN rule, and shortest paths cannot use
    the regular centroid grid as an unvisited routing lattice.
    """

    def __init__(
        self,
        bins: tuple[int, int],
        bd_min: np.ndarray,
        bd_max: np.ndarray,
        genome_size: int,
        qd_score_fitness_floor: float = -5.0,
        pairwise_geodesic_k: int = 10,
        graph_k: int = 10,
        assignment_k: int = 10,
        max_graph_points: int = 5000,
        include_archive_elites_in_support: bool = False,
        reservoir_seed: int = 0,
        refresh_interval: int = 250,
        selection_novelty_neighbors: int = 10,
        selection_novelty_floor: float = 0.05,
    ):
        self.bins = (int(bins[0]), int(bins[1]))
        self.bd_min = np.asarray(bd_min, dtype=np.float64)
        self.bd_max = np.asarray(bd_max, dtype=np.float64)
        self.genome_size = int(genome_size)
        self.qd_score_fitness_floor = float(qd_score_fitness_floor)
        self.pairwise_geodesic_k = int(pairwise_geodesic_k)
        self.graph_k = int(graph_k)
        self.assignment_k = int(assignment_k)
        self.max_graph_points = int(max_graph_points)
        self.include_archive_elites_in_support = bool(include_archive_elites_in_support)
        self.reservoir_seed = int(reservoir_seed)
        self.refresh_interval = int(refresh_interval)
        self.selection_novelty_neighbors = int(selection_novelty_neighbors)
        self.selection_novelty_floor = float(selection_novelty_floor)
        self.reservoir_strategy = (
            "archive_elites_plus_rolling_recent_buffer"
            if self.include_archive_elites_in_support
            else "rolling_recent_buffer"
        )

        shape = self.bins
        self.fitness = np.full(shape, -np.inf, dtype=np.float64)
        self.occupied = np.zeros(shape, dtype=bool)
        self.genomes = np.zeros((*shape, self.genome_size), dtype=np.float64)
        self.descriptors = np.zeros((*shape, 2), dtype=np.float64)
        self.steps = np.zeros(shape, dtype=np.int32)
        self.food_collected = np.zeros(shape, dtype=np.int32)
        self.wall_collisions = np.zeros(shape, dtype=np.int32)
        self.hazard_contacts = np.zeros(shape, dtype=np.int32)
        self.novelty_scores = np.ones(shape, dtype=np.float64)

        self.centroids = self._make_centroids()
        self.visited_latents: list[np.ndarray] = []
        self._rolling_latents: list[np.ndarray] = []
        self._support_points = self.centroids.copy()
        self._support_is_centroid = np.ones(len(self.centroids), dtype=bool)
        self._support_kinds = np.full(len(self.centroids), "centroid", dtype=object)
        self._tree: cKDTree | None = None
        self._routing_support_indices = np.empty(0, dtype=np.int64)
        self._centroid_distances = np.zeros(
            (len(self.centroids), len(self.centroids)), dtype=np.float64
        )
        self._raw_finite_centroid_fraction = 1.0
        self._penalized_centroid_distances = 0
        self._cumulative_penalized_centroid_distances = 0
        self._refreshes_with_penalty = 0
        self._penalty_distance = 0.0
        self._connected_components = 1
        self._largest_component_fraction = 1.0
        self._refresh_count = 0
        self._last_refresh_visited_count = 0
        self.refresh_graph(force=True)

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.occupied))

    @property
    def capacity(self) -> int:
        return int(self.bins[0] * self.bins[1])

    def add_visited_latents(self, latents: np.ndarray) -> None:
        for latent in np.asarray(latents, dtype=np.float64):
            self._record_visited_latent(latent)
        self.refresh_graph(force=True)

    def add(
        self,
        genome: MLPPolicyGenome,
        rollout: RolloutResult,
        descriptor: np.ndarray,
        evaluation: int | None = None,
        record_visited: bool = True,
    ) -> tuple[bool, bool, tuple[int, int] | None, float]:
        descriptor = np.asarray(descriptor, dtype=np.float64)
        if record_visited:
            self._record_visited_latent(descriptor)
        if self._should_refresh(evaluation):
            self.refresh_graph(force=True)
        cell, geodesic_distance = self.cell_for(descriptor)
        if cell is None:
            return False, False, None, float("nan")
        old_occupied = bool(self.occupied[cell])
        if old_occupied and rollout.fitness <= float(self.fitness[cell]):
            return False, False, cell, geodesic_distance
        self.occupied[cell] = True
        self.fitness[cell] = rollout.fitness
        self.genomes[cell] = genome.weights
        self.descriptors[cell] = descriptor
        self.steps[cell] = rollout.steps
        self.food_collected[cell] = rollout.food_collected
        self.wall_collisions[cell] = rollout.wall_collisions
        self.hazard_contacts[cell] = rollout.hazard_contacts
        self._update_novelty_scores()
        return True, old_occupied, cell, geodesic_distance

    def cell_for(self, descriptor: np.ndarray) -> tuple[tuple[int, int] | None, float]:
        descriptor = np.asarray(descriptor, dtype=np.float64)
        centroid_index, geodesic_distance = self.closest_centroid(descriptor)
        if centroid_index < 0:
            return None, float("nan")
        ix = centroid_index // self.bins[1]
        iy = centroid_index % self.bins[1]
        return (int(ix), int(iy)), geodesic_distance

    def closest_centroid(self, descriptor: np.ndarray) -> tuple[int, float]:
        if self._tree is None or len(self._routing_support_indices) == 0:
            euclidean = np.linalg.norm(self.centroids - descriptor[None, :], axis=1)
            index = int(np.argmin(euclidean))
            return index, float(euclidean[index])
        query_k = min(max(1, self.assignment_k), len(self._routing_support_indices))
        distances, indices = self._tree.query(descriptor, k=query_k)
        distances = np.atleast_1d(distances).astype(np.float64)
        tree_indices = np.atleast_1d(indices).astype(np.int64)
        indices = self._routing_support_indices[tree_indices]
        candidate_distances = self._centroid_distances[:, indices] + distances[None, :]
        geodesic_to_centroids = np.min(candidate_distances, axis=1)
        if not np.any(np.isfinite(geodesic_to_centroids)):
            euclidean = np.linalg.norm(self.centroids - descriptor[None, :], axis=1)
            index = int(np.argmin(euclidean))
            return index, float(euclidean[index])
        index = int(np.nanargmin(geodesic_to_centroids))
        return index, float(geodesic_to_centroids[index])

    def sample_elite(
        self,
        rng: np.random.Generator,
        obs_dim: int,
        hidden_dim: int,
    ) -> MLPPolicyGenome:
        occupied_cells = np.argwhere(self.occupied)
        if len(occupied_cells) == 0:
            raise RuntimeError("Cannot sample from an empty archive.")
        weights = np.asarray(
            [self.novelty_scores[int(cell[0]), int(cell[1])] for cell in occupied_cells]
        )
        if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(occupied_cells), dtype=np.float64)
        probabilities = weights / float(np.sum(weights))
        chosen = occupied_cells[int(rng.choice(len(occupied_cells), p=probabilities))]
        genome_weights = self.genomes[int(chosen[0]), int(chosen[1])].copy()
        return MLPPolicyGenome(weights=genome_weights, obs_dim=obs_dim, hidden_dim=hidden_dim)

    def refresh_graph(self, force: bool = False) -> None:
        if not force:
            return
        routing_points, routing_kinds = self._graph_routing_points()
        if len(routing_points):
            points = np.vstack([routing_points, self.centroids])
            is_centroid = np.concatenate(
                [
                    np.zeros(len(routing_points), dtype=bool),
                    np.ones(len(self.centroids), dtype=bool),
                ]
            )
            support_kinds = np.concatenate(
                [routing_kinds, np.full(len(self.centroids), "centroid", dtype=object)]
            )
        else:
            points = self.centroids.copy()
            is_centroid = np.ones(len(self.centroids), dtype=bool)
            support_kinds = np.full(len(self.centroids), "centroid", dtype=object)

        self._support_points = points
        self._support_is_centroid = is_centroid
        self._support_kinds = support_kinds
        self._routing_support_indices = np.flatnonzero(~is_centroid)
        self._tree = (
            cKDTree(points[self._routing_support_indices])
            if len(self._routing_support_indices)
            else None
        )
        graph = self._build_sparse_knn_graph(points)
        routing_graph = graph[self._routing_support_indices][:, self._routing_support_indices]
        if len(self._routing_support_indices):
            component_count, component_labels = connected_components(routing_graph, directed=False)
            component_sizes = np.bincount(component_labels)
            self._connected_components = int(component_count)
            self._largest_component_fraction = float(
                np.max(component_sizes) / len(self._routing_support_indices)
            )
        else:
            self._connected_components = 0
            self._largest_component_fraction = 0.0
        raw_distances = endpoint_shortest_path_distances(graph, is_centroid)
        self._centroid_distances = self._apply_disconnection_penalty(raw_distances, points)
        self._refresh_count += 1
        self._last_refresh_visited_count = len(self.visited_latents)
        self._update_novelty_scores()

    def graph_stats(self) -> GeodesicGraphStats:
        return GeodesicGraphStats(
            support_points=len(self._support_points),
            visited_points=len(self.visited_latents),
            reservoir_points=len(self._rolling_latents),
            rolling_buffer_points=len(self._rolling_latents),
            archive_elite_points=int(np.count_nonzero(self._support_kinds == "archive_elite")),
            centroid_count=len(self.centroids),
            knn_k=min(self.graph_k, max(0, len(self._support_points) - 1)),
            refresh_count=self._refresh_count,
            finite_centroid_fraction=self._raw_finite_centroid_fraction,
            connected_components=self._connected_components,
            largest_component_fraction=self._largest_component_fraction,
            penalized_centroid_distances=self._penalized_centroid_distances,
            cumulative_penalized_centroid_distances=self._cumulative_penalized_centroid_distances,
            refreshes_with_penalty=self._refreshes_with_penalty,
            penalty_distance=self._penalty_distance,
            reservoir_strategy=self.reservoir_strategy,
            reservoir_seed=self.reservoir_seed,
        )

    def metrics(self, include_pairwise: bool = True) -> dict[str, float | int]:
        occupied_fitness = self.fitness[self.occupied]
        coverage = self.occupied_count / self.capacity
        graph_stats = self.graph_stats()
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
                **self._graph_metric_dict(graph_stats),
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
            **self._graph_metric_dict(graph_stats),
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
                        "centroid_x": float(self.centroids[ix * self.bins[1] + iy, 0]),
                        "centroid_y": float(self.centroids[ix * self.bins[1] + iy, 1]),
                        "novelty_score": float(self.novelty_scores[ix, iy]),
                        "steps": int(self.steps[ix, iy]),
                        "food_collected": int(self.food_collected[ix, iy]),
                        "wall_collisions": int(self.wall_collisions[ix, iy]),
                        "hazard_contacts": int(self.hazard_contacts[ix, iy]),
                    }
                )
        return rows

    def visited_latent_rows(self) -> list[dict[str, Any]]:
        return [
            {"index": idx, "latent_0": float(point[0]), "latent_1": float(point[1])}
            for idx, point in enumerate(self.visited_latents)
        ]

    def support_point_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, point in enumerate(self._support_points):
            rows.append(
                {
                    "index": idx,
                    "kind": str(self._support_kinds[idx]),
                    "latent_0": float(point[0]),
                    "latent_1": float(point[1]),
                }
            )
        return rows

    def _make_centroids(self) -> np.ndarray:
        xs = np.linspace(
            self.bd_min[0],
            self.bd_max[0],
            self.bins[0],
            endpoint=False,
            dtype=np.float64,
        )
        ys = np.linspace(
            self.bd_min[1],
            self.bd_max[1],
            self.bins[1],
            endpoint=False,
            dtype=np.float64,
        )
        step = (self.bd_max - self.bd_min) / np.asarray(self.bins, dtype=np.float64)
        xs = xs + step[0] * 0.5
        ys = ys + step[1] * 0.5
        return np.asarray([[x, y] for x in xs for y in ys], dtype=np.float64)

    def _should_refresh(self, evaluation: int | None) -> bool:
        if self.refresh_interval <= 0:
            return False
        if evaluation is not None and evaluation % self.refresh_interval == 0:
            return True
        return len(self.visited_latents) - self._last_refresh_visited_count >= self.refresh_interval

    def _graph_routing_points(self) -> tuple[np.ndarray, np.ndarray]:
        elite_points = (
            np.asarray(self.descriptors[self.occupied], dtype=np.float64)
            if self.include_archive_elites_in_support
            else np.empty((0, 2), dtype=np.float64)
        )
        rolling_points = (
            np.asarray(self._rolling_latents, dtype=np.float64)
            if self._rolling_latents
            else np.empty((0, 2), dtype=np.float64)
        )

        return compose_real_routing_support(elite_points, rolling_points)

    def _record_visited_latent(self, latent: np.ndarray) -> None:
        stored = np.asarray(latent, dtype=np.float64).copy()
        self.visited_latents.append(stored)
        if self.max_graph_points <= 0:
            return
        self._rolling_latents.append(stored)
        if len(self._rolling_latents) > self.max_graph_points:
            del self._rolling_latents[: len(self._rolling_latents) - self.max_graph_points]

    def _build_sparse_knn_graph(self, points: np.ndarray) -> csr_matrix:
        return build_endpoint_manifold_graph(points, self._support_is_centroid, self.graph_k)

    def _apply_disconnection_penalty(
        self, distances: np.ndarray, support_points: np.ndarray
    ) -> np.ndarray:
        finite = np.isfinite(distances)
        self._raw_finite_centroid_fraction = float(np.mean(finite)) if distances.size else 0.0
        self._penalized_centroid_distances = int(np.count_nonzero(~finite))
        if self._penalized_centroid_distances and len(self._routing_support_indices):
            self._cumulative_penalized_centroid_distances += self._penalized_centroid_distances
            self._refreshes_with_penalty += 1
        if self._penalized_centroid_distances == 0:
            self._penalty_distance = 0.0
            return distances

        positive_finite = distances[finite & (distances > 0.0)]
        finite_scale = float(np.max(positive_finite)) if len(positive_finite) else 0.0
        if len(support_points) >= 2:
            span = support_points[:, None, :] - support_points[None, :, :]
            support_scale = float(np.max(np.linalg.norm(span, axis=2)))
        else:
            support_scale = 0.5
        penalty = max(finite_scale, support_scale, 0.5) * 2.0
        repaired = distances.copy()
        repaired[~finite] = penalty
        self._penalty_distance = penalty
        return repaired

    def _update_novelty_scores(self) -> None:
        self.novelty_scores.fill(1.0)
        occupied = np.argwhere(self.occupied)
        if len(occupied) < 2:
            return
        centroid_indices = np.asarray(
            [int(cell[0]) * self.bins[1] + int(cell[1]) for cell in occupied]
        )
        centroid_columns = self._support_is_centroid.nonzero()[0]
        if len(centroid_columns) != len(self.centroids):
            return
        distances = self._centroid_distances[centroid_indices][
            :, centroid_columns[centroid_indices]
        ]
        distances = np.asarray(distances, dtype=np.float64)
        novelty_values = np.ones(len(occupied), dtype=np.float64)
        neighbors = min(max(1, self.selection_novelty_neighbors), len(occupied) - 1)
        for idx in range(len(occupied)):
            row = distances[idx].copy()
            row[idx] = np.inf
            finite = row[np.isfinite(row)]
            if len(finite):
                nearest = np.sort(finite)[:neighbors]
                novelty_values[idx] = max(self.selection_novelty_floor, float(np.mean(nearest)))
        max_value = float(np.max(novelty_values))
        if max_value > 0.0:
            novelty_values = self.selection_novelty_floor + novelty_values / max_value
        for cell, novelty in zip(occupied, novelty_values, strict=True):
            self.novelty_scores[int(cell[0]), int(cell[1])] = float(novelty)

    @staticmethod
    def _graph_metric_dict(stats: GeodesicGraphStats) -> dict[str, float | int]:
        return {
            "geodesic_graph_support_points": stats.support_points,
            "geodesic_graph_visited_points": stats.visited_points,
            "geodesic_graph_reservoir_points": stats.reservoir_points,
            "geodesic_graph_rolling_buffer_points": stats.rolling_buffer_points,
            "geodesic_graph_archive_elite_points": stats.archive_elite_points,
            "geodesic_graph_centroids": stats.centroid_count,
            "geodesic_graph_k": stats.knn_k,
            "geodesic_graph_refresh_count": stats.refresh_count,
            "geodesic_graph_finite_centroid_fraction": stats.finite_centroid_fraction,
            "geodesic_graph_connected_components": stats.connected_components,
            "geodesic_graph_largest_component_fraction": stats.largest_component_fraction,
            "geodesic_graph_penalized_centroid_distances": stats.penalized_centroid_distances,
            "geodesic_graph_cumulative_penalized_centroid_distances": (
                stats.cumulative_penalized_centroid_distances
            ),
            "geodesic_graph_refreshes_with_penalty": stats.refreshes_with_penalty,
            "geodesic_graph_penalty_distance": stats.penalty_distance,
            "geodesic_graph_reservoir_strategy": stats.reservoir_strategy,
            "geodesic_graph_reservoir_seed": stats.reservoir_seed,
        }
