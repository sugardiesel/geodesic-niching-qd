"""Shared sparse k-NN graph construction for latent-manifold distances."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def build_undirected_knn_graph(points: np.ndarray, graph_k: int) -> csr_matrix:
    """Return the union-symmetrized k-NN graph with one Euclidean weight per edge."""

    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    if point_count <= 1:
        return csr_matrix((point_count, point_count), dtype=np.float64)

    k = min(max(1, int(graph_k)), point_count - 1)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    edge_weights: dict[tuple[int, int], float] = {}
    for source in range(point_count):
        for distance, target in zip(distances[source, 1:], indices[source, 1:], strict=False):
            target = int(target)
            distance = float(distance)
            if source == target or not math.isfinite(distance):
                continue
            edge = (min(source, target), max(source, target))
            true_distance = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
            edge_weights[edge] = min(edge_weights.get(edge, math.inf), true_distance)

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for (left, right), weight in edge_weights.items():
        rows.extend((left, right))
        columns.extend((right, left))
        weights.extend((weight, weight))
    return csr_matrix(
        (weights, (rows, columns)),
        shape=(point_count, point_count),
        dtype=np.float64,
    )


def build_endpoint_manifold_graph(
    points: np.ndarray,
    endpoint_mask: np.ndarray,
    graph_k: int,
) -> csr_matrix:
    """Build combined-set k-NN edges while preventing endpoints from forming a routing lattice.

    Every retained edge first has to be selected by the same k-NN rule on the full combined
    point set. Endpoint-to-endpoint edges are then removed because archive centroids are query
    destinations, not observations of the visited manifold.
    """

    points = np.asarray(points, dtype=np.float64)
    endpoint_mask = np.asarray(endpoint_mask, dtype=bool)
    if endpoint_mask.shape != (len(points),):
        raise ValueError("endpoint_mask must contain one Boolean value per point.")

    combined_graph = build_undirected_knn_graph(points, graph_k)
    upper = triu(combined_graph, k=1, format="coo")
    keep = ~(endpoint_mask[upper.row] & endpoint_mask[upper.col])
    left = upper.row[keep]
    right = upper.col[keep]
    weight = upper.data[keep]
    rows = np.concatenate((left, right))
    columns = np.concatenate((right, left))
    weights = np.concatenate((weight, weight))
    return csr_matrix(
        (weights, (rows, columns)),
        shape=combined_graph.shape,
        dtype=np.float64,
    )


def endpoint_shortest_path_distances(
    graph: csr_matrix,
    endpoint_mask: np.ndarray,
) -> np.ndarray:
    """Return endpoint-to-support distances with endpoints forbidden as transit nodes."""

    endpoint_mask = np.asarray(endpoint_mask, dtype=bool)
    if endpoint_mask.shape != (graph.shape[0],):
        raise ValueError("endpoint_mask must match the graph size.")

    route_indices = np.flatnonzero(~endpoint_mask)
    endpoint_indices = np.flatnonzero(endpoint_mask)
    endpoint_count = len(endpoint_indices)
    result = np.full((endpoint_count, graph.shape[0]), np.inf, dtype=np.float64)
    if endpoint_count == 0:
        return result
    result[np.arange(endpoint_count), endpoint_indices] = 0.0
    if len(route_indices) == 0:
        return result

    routing_graph = graph[route_indices][:, route_indices].tocsr()
    attachments = graph[endpoint_indices][:, route_indices].tocsr()
    routing_count = len(route_indices)

    routing_coo = routing_graph.tocoo()
    attachment_coo = attachments.tocoo()
    source_graph = csr_matrix(
        (
            np.concatenate((routing_coo.data, attachment_coo.data)),
            (
                np.concatenate((routing_coo.row, routing_count + attachment_coo.row)),
                np.concatenate((routing_coo.col, attachment_coo.col)),
            ),
        ),
        shape=(routing_count + endpoint_count, routing_count + endpoint_count),
        dtype=np.float64,
    )
    source_rows = routing_count + np.arange(endpoint_count)
    source_distances = dijkstra(
        source_graph,
        directed=True,
        indices=source_rows,
        return_predecessors=False,
    )
    endpoint_to_route = np.asarray(source_distances[:, :routing_count], dtype=np.float64)
    result[:, route_indices] = endpoint_to_route

    for target_position, target_index in enumerate(endpoint_indices):
        start = attachments.indptr[target_position]
        stop = attachments.indptr[target_position + 1]
        neighbors = attachments.indices[start:stop]
        weights = attachments.data[start:stop]
        if len(neighbors):
            result[:, target_index] = np.min(
                endpoint_to_route[:, neighbors] + weights[None, :],
                axis=1,
            )
        result[target_position, target_index] = 0.0
    return result


def endpoint_safe_all_pairs_distances(
    graph: csr_matrix,
    endpoint_mask: np.ndarray,
) -> np.ndarray:
    """Return all-pairs distances where endpoint nodes may only start or finish paths."""

    endpoint_mask = np.asarray(endpoint_mask, dtype=bool)
    route_indices = np.flatnonzero(~endpoint_mask)
    endpoint_indices = np.flatnonzero(endpoint_mask)
    distances = np.full(graph.shape, np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)

    if len(route_indices):
        routing_graph = graph[route_indices][:, route_indices].tocsr()
        route_distances = dijkstra(routing_graph, directed=False, return_predecessors=False)
        distances[np.ix_(route_indices, route_indices)] = route_distances
    if len(endpoint_indices):
        endpoint_distances = endpoint_shortest_path_distances(graph, endpoint_mask)
        distances[endpoint_indices, :] = endpoint_distances
        distances[:, endpoint_indices] = endpoint_distances.T
    return distances
