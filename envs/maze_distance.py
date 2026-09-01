"""Grid shortest-path distances through a ForageMaze2D map."""

from __future__ import annotations

import heapq
import math

import numpy as np

from envs.forage_maze import ForageMaze2D


class GridShortestPath:
    """Dijkstra distances over a discretized free-space grid."""

    def __init__(self, env: ForageMaze2D, grid_size: int = 140, connectivity: int = 8):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")
        self.env = env
        self.grid_size = int(grid_size)
        self.connectivity = int(connectivity)
        self.xs = np.linspace(env.world_min[0], env.world_max[0], self.grid_size)
        self.ys = np.linspace(env.world_min[1], env.world_max[1], self.grid_size)
        self.free = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                self.free[iy, ix] = env.position_is_free(
                    np.asarray([x, y]), radius=env.agent_radius
                )
        self._neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            self._neighbor_offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def point_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        ix = int(np.argmin(np.abs(self.xs - float(point[0]))))
        iy = int(np.argmin(np.abs(self.ys - float(point[1]))))
        if self.free[iy, ix]:
            return iy, ix
        best: tuple[int, int] | None = None
        best_dist = math.inf
        for radius in range(1, self.grid_size):
            y_min = max(0, iy - radius)
            y_max = min(self.grid_size - 1, iy + radius)
            x_min = max(0, ix - radius)
            x_max = min(self.grid_size - 1, ix + radius)
            for cy in range(y_min, y_max + 1):
                for cx in range(x_min, x_max + 1):
                    if not self.free[cy, cx]:
                        continue
                    distance = (cy - iy) * (cy - iy) + (cx - ix) * (cx - ix)
                    if distance < best_dist:
                        best = (cy, cx)
                        best_dist = distance
            if best is not None:
                return best
        raise ValueError("No free grid cell found.")

    def cell_to_point(self, cell: tuple[int, int]) -> np.ndarray:
        iy, ix = cell
        return np.asarray([self.xs[ix], self.ys[iy]], dtype=np.float64)

    def distance(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        source = self.point_to_cell(point_a)
        target = self.point_to_cell(point_b)
        distances = self.distances_from_cell(source)
        return float(distances[target])

    def distances_from_cell(self, source: tuple[int, int]) -> np.ndarray:
        if source not in self._cache:
            self._cache[source] = self._dijkstra(source)
        return self._cache[source]

    def shortest_path(self, point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        source = self.point_to_cell(point_a)
        target = self.point_to_cell(point_b)
        distances = self.distances_from_cell(source)
        if not np.isfinite(distances[target]):
            return np.zeros((0, 2), dtype=np.float64)

        path = [target]
        current = target
        while current != source:
            best_neighbor = None
            best_value = math.inf
            for neighbor, step_cost in self._neighbors(current):
                value = float(distances[neighbor]) + step_cost
                if value < best_value:
                    best_value = value
                    best_neighbor = neighbor
            if best_neighbor is None:
                break
            current = best_neighbor
            path.append(current)
            if len(path) > self.grid_size * self.grid_size:
                break
        path.reverse()
        return np.asarray([self.cell_to_point(cell) for cell in path], dtype=np.float64)

    def _dijkstra(self, source: tuple[int, int]) -> np.ndarray:
        distances = np.full((self.grid_size, self.grid_size), np.inf, dtype=np.float64)
        distances[source] = 0.0
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
        while heap:
            current_distance, cell = heapq.heappop(heap)
            if current_distance > float(distances[cell]):
                continue
            for neighbor, step_cost in self._neighbors(cell):
                candidate = current_distance + step_cost
                if candidate < float(distances[neighbor]):
                    distances[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
        return distances

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        iy, ix = cell
        neighbors: list[tuple[tuple[int, int], float]] = []
        dx_world = float(self.xs[1] - self.xs[0]) if self.grid_size > 1 else 1.0
        dy_world = float(self.ys[1] - self.ys[0]) if self.grid_size > 1 else 1.0
        for dy, dx in self._neighbor_offsets:
            ny = iy + dy
            nx = ix + dx
            if ny < 0 or ny >= self.grid_size or nx < 0 or nx >= self.grid_size:
                continue
            if not self.free[ny, nx]:
                continue
            if dx != 0 and dy != 0:
                if not self.free[iy, nx] or not self.free[ny, ix]:
                    continue
            step_cost = math.hypot(dx * dx_world, dy * dy_world)
            neighbors.append(((ny, nx), step_cost))
        return neighbors


def pairwise_euclidean_and_geodesic(
    points: np.ndarray,
    shortest_path: GridShortestPath,
) -> tuple[np.ndarray, np.ndarray]:
    cells = [shortest_path.point_to_cell(point) for point in points]
    euclidean: list[float] = []
    geodesic: list[float] = []
    for i, source in enumerate(cells):
        distances = shortest_path.distances_from_cell(source)
        for j in range(i + 1, len(cells)):
            euclidean.append(float(np.linalg.norm(points[i] - points[j])))
            geodesic.append(float(distances[cells[j]]))
    return np.asarray(euclidean, dtype=np.float64), np.asarray(geodesic, dtype=np.float64)
