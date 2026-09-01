import unittest

import numpy as np
from scipy.sparse.csgraph import dijkstra

from algorithms.geodesic_archive import GeodesicGridArchive
from algorithms.knn_graph import (
    build_endpoint_manifold_graph,
    build_undirected_knn_graph,
    endpoint_safe_all_pairs_distances,
)
from algorithms.policies import MLPPolicyGenome
from envs.forage_maze import RolloutResult


class GeodesicArchiveTests(unittest.TestCase):
    def test_mutual_knn_edge_has_one_true_euclidean_weight(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)

        graph = build_undirected_knn_graph(points, graph_k=1)
        distances = dijkstra(graph, directed=False)

        self.assertAlmostEqual(float(graph[0, 1]), 1.0)
        self.assertAlmostEqual(float(graph[1, 0]), 1.0)
        self.assertAlmostEqual(float(distances[0, 2]), 2.0)

    def test_centroids_are_endpoints_not_shortest_path_transit_nodes(self):
        points = np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.1, 0.0],
            ],
            dtype=np.float64,
        )
        endpoint_mask = np.array([False, False, True, True])

        graph = build_endpoint_manifold_graph(points, endpoint_mask, graph_k=2)
        distances = endpoint_safe_all_pairs_distances(graph, endpoint_mask)

        self.assertEqual(float(graph[2, 3]), 0.0)
        self.assertTrue(np.isinf(distances[0, 1]))

    def test_add_assigns_to_a_cell_and_tracks_graph_stats(self):
        archive = GeodesicGridArchive(
            bins=(4, 4),
            bd_min=np.array([-1.0, -1.0]),
            bd_max=np.array([1.0, 1.0]),
            genome_size=3,
            max_graph_points=20,
            refresh_interval=2,
        )
        genome = MLPPolicyGenome(weights=np.array([0.1, -0.2, 0.3]), obs_dim=1, hidden_dim=1)
        rollout = RolloutResult(
            name="test",
            fitness=2.5,
            steps=10,
            food_collected=1,
            hazard_contacts=0,
            wall_collisions=0,
            final_position=np.array([0.0, 0.0]),
            trajectory=np.zeros((2, 7), dtype=np.float32),
            termination="test",
        )

        inserted, replaced, cell, distance = archive.add(
            genome,
            rollout,
            descriptor=np.array([0.2, -0.1]),
            evaluation=1,
        )

        self.assertTrue(inserted)
        self.assertFalse(replaced)
        self.assertIsNotNone(cell)
        self.assertTrue(np.isfinite(distance))
        self.assertEqual(archive.occupied_count, 1)
        self.assertGreaterEqual(archive.graph_stats().support_points, archive.capacity)
        self.assertEqual(archive.graph_stats().reservoir_strategy, "rolling_recent_buffer")

    def test_current_elite_remains_in_support_after_aging_out_of_recent_buffer(self):
        archive = GeodesicGridArchive(
            bins=(2, 2),
            bd_min=np.array([-1.0, -1.0]),
            bd_max=np.array([1.0, 1.0]),
            genome_size=3,
            graph_k=2,
            max_graph_points=2,
            include_archive_elites_in_support=True,
        )
        old_elite = np.array([-0.75, -0.75], dtype=np.float64)
        new_elite = np.array([-0.5, -0.6], dtype=np.float64)
        archive.occupied[0, 0] = True
        archive.descriptors[0, 0] = old_elite
        archive.add_visited_latents(
            np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]], dtype=np.float64)
        )

        support = archive.support_point_rows()
        elite_rows = [row for row in support if row["kind"] == "archive_elite"]
        self.assertEqual(len(elite_rows), 1)
        np.testing.assert_allclose(
            [elite_rows[0]["latent_0"], elite_rows[0]["latent_1"]], old_elite
        )
        self.assertEqual(archive.graph_stats().archive_elite_points, 1)

        archive.descriptors[0, 0] = new_elite
        archive.refresh_graph(force=True)
        elite_rows = [row for row in archive.support_point_rows() if row["kind"] == "archive_elite"]
        self.assertEqual(len(elite_rows), 1)
        np.testing.assert_allclose(
            [elite_rows[0]["latent_0"], elite_rows[0]["latent_1"]], new_elite
        )

    def test_disconnected_graph_uses_finite_penalty(self):
        archive = GeodesicGridArchive(
            bins=(2, 2),
            bd_min=np.array([-1.0, -1.0]),
            bd_max=np.array([1.0, 1.0]),
            genome_size=3,
            graph_k=1,
            assignment_k=1,
            max_graph_points=0,
        )

        repaired = archive._apply_disconnection_penalty(
            np.array([[0.0, np.inf], [np.inf, 0.0]]),
            np.array([[0.0, 0.0], [2.0, 0.0]]),
        )

        self.assertTrue(np.all(np.isfinite(repaired)))
        self.assertLess(archive._raw_finite_centroid_fraction, 1.0)
        self.assertGreater(archive._penalized_centroid_distances, 0)
        self.assertGreater(archive._penalty_distance, 0.0)

        cell, distance = archive.cell_for(np.array([0.8, 0.8]))

        self.assertIsNotNone(cell)
        self.assertTrue(np.isfinite(distance))

    def test_disconnected_support_clusters_keep_assignment_and_novelty_finite(self):
        archive = GeodesicGridArchive(
            bins=(2, 1),
            bd_min=np.array([-100.0, -1.0]),
            bd_max=np.array([100.0, 1.0]),
            genome_size=3,
            graph_k=1,
            assignment_k=1,
            max_graph_points=10,
        )
        archive.add_visited_latents(
            np.array(
                [
                    [-50.1, 0.0],
                    [50.1, 0.0],
                ],
                dtype=np.float64,
            )
        )

        stats = archive.graph_stats()
        self.assertEqual(stats.connected_components, 2)
        self.assertLess(stats.finite_centroid_fraction, 1.0)
        self.assertGreater(stats.penalized_centroid_distances, 0)
        self.assertGreater(stats.penalty_distance, 90.0)

        cell, distance = archive.cell_for(np.array([50.1, 0.0]))
        self.assertEqual(cell, (1, 0))
        self.assertTrue(np.isfinite(distance))
        self.assertGreater(distance, 0.0)

        archive.occupied[0, 0] = True
        archive.occupied[1, 0] = True
        archive._update_novelty_scores()
        occupied_novelty = archive.novelty_scores[archive.occupied]

        self.assertTrue(np.all(np.isfinite(occupied_novelty)))
        self.assertTrue(np.all(occupied_novelty > 0.0))


if __name__ == "__main__":
    unittest.main()
