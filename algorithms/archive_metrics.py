"""Shared archive metrics for all QD conditions."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseDistanceMetrics:
    mean_euclidean: float
    mean_knn_geodesic: float
    knn_k: int
    finite_geodesic_fraction: float
    disconnected_pair_count: int


def raw_qd_score(fitnesses: np.ndarray) -> float:
    if fitnesses.size == 0:
        return 0.0
    return float(np.sum(fitnesses))


def shifted_qd_score(fitnesses: np.ndarray, fitness_floor: float) -> float:
    if fitnesses.size == 0:
        return 0.0
    shifted = np.maximum(fitnesses - fitness_floor, 0.0)
    return float(np.sum(shifted))


def occupancy_entropy(filled_cells: int, capacity: int) -> float:
    """Coverage-derived normalized occupied-cell entropy."""

    if filled_cells <= 1 or capacity <= 1:
        return 0.0
    return math.log(filled_cells) / math.log(capacity)


def pairwise_distance_metrics(points: np.ndarray, knn_k: int) -> PairwiseDistanceMetrics:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return PairwiseDistanceMetrics(
            mean_euclidean=float("nan"),
            mean_knn_geodesic=float("nan"),
            knn_k=int(knn_k),
            finite_geodesic_fraction=0.0,
            disconnected_pair_count=0,
        )

    distances = _pairwise_euclidean_matrix(points)
    tri = np.triu_indices(len(points), k=1)
    mean_euclidean = float(np.mean(distances[tri]))

    graph = _knn_graph_from_distances(distances, knn_k)
    geodesic = _all_pairs_dijkstra(graph)
    finite = np.isfinite(geodesic[tri])
    disconnected = int(np.count_nonzero(~finite))
    mean_geodesic = float(np.mean(geodesic[tri][finite])) if np.any(finite) else float("nan")
    return PairwiseDistanceMetrics(
        mean_euclidean=mean_euclidean,
        mean_knn_geodesic=mean_geodesic,
        knn_k=min(int(knn_k), len(points) - 1),
        finite_geodesic_fraction=float(np.mean(finite)),
        disconnected_pair_count=disconnected,
    )


def _pairwise_euclidean_matrix(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def _knn_graph_from_distances(distances: np.ndarray, knn_k: int) -> list[list[tuple[int, float]]]:
    n = distances.shape[0]
    k = min(max(1, int(knn_k)), n - 1)
    graph: list[dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        order = np.argsort(distances[i])
        neighbors = [j for j in order if j != i][:k]
        for j in neighbors:
            weight = float(distances[i, j])
            old_ij = graph[i].get(j, math.inf)
            old_ji = graph[j].get(i, math.inf)
            graph[i][j] = min(old_ij, weight)
            graph[j][i] = min(old_ji, weight)
    return [[(j, weight) for j, weight in row.items()] for row in graph]


def _all_pairs_dijkstra(graph: list[list[tuple[int, float]]]) -> np.ndarray:
    n = len(graph)
    distances = np.full((n, n), np.inf, dtype=np.float64)
    for source in range(n):
        distances[source] = _dijkstra(graph, source)
    return distances


def _dijkstra(graph: list[list[tuple[int, float]]], source: int) -> np.ndarray:
    distances = np.full(len(graph), np.inf, dtype=np.float64)
    distances[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > float(distances[node]):
            continue
        for neighbor, weight in graph[node]:
            candidate = current_distance + weight
            if candidate < float(distances[neighbor]):
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances
