import unittest

import numpy as np

from algorithms.geodesic_archive import GeodesicGridArchive
from algorithms.policies import MLPPolicyGenome
from envs.forage_maze import RolloutResult

A_CORRECT_CENTROID_INDEX = 0
A_TRAP_CENTROID_INDEX = 1
B_TRAP_CENTROID_INDEX = 2
B_CORRECT_CENTROID_INDEX = 3


def build_horseshoe_assignment_archive() -> GeodesicGridArchive:
    """Build two tiny latent-space horseshoes where Euclidean nearest-centroid is misleading."""
    archive = GeodesicGridArchive(
        bins=(4, 1),
        bd_min=np.array([-1.6, -1.0]),
        bd_max=np.array([3.2, 1.0]),
        genome_size=1,
        graph_k=5,
        assignment_k=5,
        pairwise_geodesic_k=5,
        max_graph_points=1000,
        refresh_interval=0,
    )

    a_query_cluster = _cluster(np.array([0.0, 0.0]))
    a_trap_cluster = _cluster(np.array([0.2, 0.0]))
    b_trap_cluster = _cluster(np.array([1.4, 0.0]))
    b_query_cluster = _cluster(np.array([1.6, 0.0]))
    a_chain_to_correct_centroid = _line(np.array([-0.1, 0.0]), np.array([-0.9, 0.0]), 9)
    b_chain_to_correct_centroid = _line(np.array([1.7, 0.0]), np.array([2.5, 0.0]), 9)
    a_long_chain_around_gap = np.vstack(
        [
            _line(np.array([0.0, 0.12]), np.array([0.0, 1.5]), 10),
            _line(np.array([-0.25, 1.5]), np.array([-2.5, 1.5]), 12),
            _line(np.array([-2.5, 1.25]), np.array([-2.5, -1.5]), 14),
            _line(np.array([-2.25, -1.5]), np.array([0.2, -1.5]), 13),
            _line(np.array([0.2, -1.25]), np.array([0.2, -0.12]), 9),
        ]
    )
    b_long_chain_around_gap = np.vstack(
        [
            _line(np.array([1.6, 0.12]), np.array([1.6, 1.5]), 10),
            _line(np.array([1.85, 1.5]), np.array([4.0, 1.5]), 12),
            _line(np.array([4.0, 1.25]), np.array([4.0, -1.5]), 14),
            _line(np.array([3.75, -1.5]), np.array([1.4, -1.5]), 13),
            _line(np.array([1.4, -1.25]), np.array([1.4, -0.12]), 9),
        ]
    )
    bridge_between_synthetic_horseshoes = _line(np.array([0.4, -1.5]), np.array([1.2, -1.5]), 5)

    archive.add_visited_latents(
        np.vstack(
            [
                a_query_cluster,
                a_trap_cluster,
                b_trap_cluster,
                b_query_cluster,
                a_chain_to_correct_centroid,
                b_chain_to_correct_centroid,
                a_long_chain_around_gap,
                b_long_chain_around_gap,
                bridge_between_synthetic_horseshoes,
            ]
        )
    )
    return archive


def _cluster(center: np.ndarray, radius: float = 0.035, count: int = 12) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.array(
        [[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)] for t in angles]
    )


def _line(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, count)
    return np.array([start + (end - start) * weight for weight in weights])


def query_points() -> dict[str, np.ndarray]:
    return {
        "cluster_a": np.array([0.0, 0.0]),
        "cluster_b": np.array([1.6, 0.0]),
    }


def euclidean_centroid_distances(archive: GeodesicGridArchive, query: np.ndarray) -> np.ndarray:
    return np.linalg.norm(archive.centroids - query[None, :], axis=1)


def geodesic_centroid_distances_used_by_assignment(
    archive: GeodesicGridArchive,
    query: np.ndarray,
) -> np.ndarray:
    if archive._tree is None:
        return euclidean_centroid_distances(archive, query)
    query_k = min(max(1, archive.assignment_k), len(archive._routing_support_indices))
    distances, indices = archive._tree.query(query, k=query_k)
    distances = np.atleast_1d(distances).astype(np.float64)
    tree_indices = np.atleast_1d(indices).astype(np.int64)
    indices = archive._routing_support_indices[tree_indices]
    candidate_distances = archive._centroid_distances[:, indices] + distances[None, :]
    return np.min(candidate_distances, axis=1)


def assignment_report_rows() -> list[dict[str, float | int | str]]:
    archive = build_horseshoe_assignment_archive()
    rows: list[dict[str, float | int | str]] = []
    for name, query in query_points().items():
        euclidean = euclidean_centroid_distances(archive, query)
        geodesic = geodesic_centroid_distances_used_by_assignment(archive, query)
        geodesic_index, geodesic_distance = archive.closest_centroid(query)
        rows.append(
            {
                "query": name,
                "euclidean_to_correct": float(euclidean[correct_index_for_query(name)]),
                "euclidean_to_trap": float(euclidean[trap_index_for_query(name)]),
                "euclidean_selected_index": int(np.argmin(euclidean)),
                "geodesic_to_correct": float(geodesic[correct_index_for_query(name)]),
                "geodesic_to_trap": float(geodesic[trap_index_for_query(name)]),
                "geodesic_selected_index": int(geodesic_index),
                "geodesic_selected_distance": float(geodesic_distance),
            }
        )
    return rows


def correct_index_for_query(query_name: str) -> int:
    if query_name == "cluster_a":
        return A_CORRECT_CENTROID_INDEX
    if query_name == "cluster_b":
        return B_CORRECT_CENTROID_INDEX
    raise ValueError(f"Unknown query: {query_name}")


def trap_index_for_query(query_name: str) -> int:
    if query_name == "cluster_a":
        return A_TRAP_CENTROID_INDEX
    if query_name == "cluster_b":
        return B_TRAP_CENTROID_INDEX
    raise ValueError(f"Unknown query: {query_name}")


class GeodesicAssignmentGroundTruthTests(unittest.TestCase):
    def test_cluster_a_geodesic_assignment_chooses_reachable_centroid_when_euclidean_is_misleading(
        self,
    ):
        archive = build_horseshoe_assignment_archive()
        query = query_points()["cluster_a"]

        stats = archive.graph_stats()
        self.assertEqual(stats.connected_components, 1)
        self.assertEqual(stats.knn_k, 5)
        self.assertEqual(archive.assignment_k, 5)
        self.assertEqual(stats.penalized_centroid_distances, 0)

        euclidean = euclidean_centroid_distances(archive, query)
        euclidean_index = int(np.argmin(euclidean))
        geodesic = geodesic_centroid_distances_used_by_assignment(archive, query)
        geodesic_index, geodesic_distance = archive.closest_centroid(query)
        geodesic_cell, cell_distance = archive.cell_for(query)

        self.assertEqual(euclidean_index, A_TRAP_CENTROID_INDEX)
        self.assertLess(euclidean[A_TRAP_CENTROID_INDEX], euclidean[A_CORRECT_CENTROID_INDEX])
        self.assertEqual(geodesic_index, A_CORRECT_CENTROID_INDEX)
        self.assertNotEqual(geodesic_index, euclidean_index)
        self.assertAlmostEqual(geodesic[A_CORRECT_CENTROID_INDEX], 1.0, places=9)
        self.assertAlmostEqual(geodesic[A_TRAP_CENTROID_INDEX], 10.503446426819, places=9)
        self.assertLess(geodesic[A_CORRECT_CENTROID_INDEX], geodesic[A_TRAP_CENTROID_INDEX])
        self.assertGreater(geodesic[A_TRAP_CENTROID_INDEX], 2.0)
        self.assertEqual(geodesic_cell, (0, 0))
        self.assertAlmostEqual(cell_distance, geodesic_distance)

    def test_cluster_b_geodesic_assignment_chooses_reachable_centroid_when_euclidean_is_misleading(
        self,
    ):
        archive = build_horseshoe_assignment_archive()
        query = query_points()["cluster_b"]

        euclidean = euclidean_centroid_distances(archive, query)
        euclidean_index = int(np.argmin(euclidean))
        geodesic = geodesic_centroid_distances_used_by_assignment(archive, query)
        geodesic_index, geodesic_distance = archive.closest_centroid(query)
        geodesic_cell, cell_distance = archive.cell_for(query)

        self.assertEqual(euclidean_index, B_TRAP_CENTROID_INDEX)
        self.assertLess(euclidean[B_TRAP_CENTROID_INDEX], euclidean[B_CORRECT_CENTROID_INDEX])
        self.assertEqual(geodesic_index, B_CORRECT_CENTROID_INDEX)
        self.assertNotEqual(geodesic_index, euclidean_index)
        self.assertAlmostEqual(geodesic[B_CORRECT_CENTROID_INDEX], 1.0, places=9)
        self.assertAlmostEqual(geodesic[B_TRAP_CENTROID_INDEX], 10.325314289378, places=9)
        self.assertLess(geodesic[B_CORRECT_CENTROID_INDEX], geodesic[B_TRAP_CENTROID_INDEX])
        self.assertGreater(geodesic[B_TRAP_CENTROID_INDEX], 2.0)
        self.assertEqual(geodesic_cell, (3, 0))
        self.assertAlmostEqual(cell_distance, geodesic_distance)

    def test_archive_add_uses_same_geodesic_assignment_path_for_both_adversarial_queries(self):
        archive = build_horseshoe_assignment_archive()
        cases = [
            ("cluster_a", (0, 0)),
            ("cluster_b", (3, 0)),
        ]
        for case_idx, (query_name, expected_cell) in enumerate(cases):
            query = query_points()[query_name]
            expected_index, expected_distance = archive.closest_centroid(query)
            genome = MLPPolicyGenome(weights=np.array([float(case_idx)]), obs_dim=0, hidden_dim=0)
            rollout = RolloutResult(
                name=query_name,
                fitness=1.0 + case_idx,
                steps=1,
                food_collected=0,
                hazard_contacts=0,
                wall_collisions=0,
                final_position=np.zeros(2, dtype=np.float64),
                trajectory=np.zeros((1, 7), dtype=np.float32),
                termination="synthetic",
            )

            inserted, replaced, cell, distance = archive.add(
                genome,
                rollout,
                descriptor=query,
                evaluation=None,
                record_visited=False,
            )

            self.assertTrue(inserted)
            self.assertFalse(replaced)
            self.assertEqual(cell, expected_cell)
            self.assertEqual(expected_index, expected_cell[0])
            self.assertAlmostEqual(distance, expected_distance)


if __name__ == "__main__":
    unittest.main()
