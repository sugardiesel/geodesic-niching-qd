"""ForageMaze2D: a small original artificial-life environment.

The simulation depends only on NumPy; plotting and representation-learning components remain
outside the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Array = np.ndarray
FOOD_RESPAWN_TIMINGS = ("corrected", "historical")


@dataclass(frozen=True)
class RectWall:
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class Food:
    name: str
    position: Array
    radius: float
    reward: float
    energy_gain: float
    respawn_steps: int
    active: bool = True
    respawn_timer: int = 0


@dataclass(frozen=True)
class Hazard:
    name: str
    position: Array
    radius: float
    penalty: float
    energy_drain: float


@dataclass
class AgentState:
    position: Array
    velocity: Array
    heading: float
    energy: float
    step_count: int


@dataclass
class RolloutResult:
    name: str
    fitness: float
    steps: int
    food_collected: int
    hazard_contacts: int
    wall_collisions: int
    final_position: Array
    trajectory: Array
    termination: str


def load_env_config(path: str | Path) -> dict[str, Any]:
    """Load an environment YAML config."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping at the top level.")
    return data


def _as_array(values: list[float] | tuple[float, float]) -> Array:
    return np.asarray(values, dtype=np.float64)


def _angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _circle_rect_intersects(center: Array, radius: float, wall: RectWall) -> bool:
    closest_x = min(max(float(center[0]), wall.x_min), wall.x_max)
    closest_y = min(max(float(center[1]), wall.y_min), wall.y_max)
    dx = float(center[0]) - closest_x
    dy = float(center[1]) - closest_y
    return dx * dx + dy * dy <= radius * radius


class ForageMaze2D:
    """Continuous 2D foraging/survival task with rectangular walls and circular zones."""

    def __init__(self, config: dict[str, Any], *, food_respawn_timing: str = "corrected"):
        if food_respawn_timing not in FOOD_RESPAWN_TIMINGS:
            raise ValueError(f"Unknown food-respawn timing: {food_respawn_timing!r}")
        self.food_respawn_timing = food_respawn_timing
        self.config = config
        self.episode_cfg = config["episode"]
        self.agent_cfg = config["agent"]
        self.fitness_cfg = config["fitness"]
        self.observation_cfg = config["observation"]
        self.map_cfg = config["map"]

        self.world_min = _as_array(self.map_cfg["world_min"])
        self.world_max = _as_array(self.map_cfg["world_max"])
        self.max_steps = int(self.episode_cfg["max_steps"])
        self.dt = float(self.episode_cfg["dt"])
        self.agent_radius = float(self.agent_cfg["radius"])

        self.walls = [
            RectWall(
                name=str(wall["name"]),
                x_min=float(wall["x_min"]),
                y_min=float(wall["y_min"]),
                x_max=float(wall["x_max"]),
                y_max=float(wall["y_max"]),
            )
            for wall in self.map_cfg["walls"]
        ]
        self.base_foods = [
            Food(
                name=str(food["name"]),
                position=_as_array(food["position"]),
                radius=float(food["radius"]),
                reward=float(food["reward"]),
                energy_gain=float(food["energy_gain"]),
                respawn_steps=int(food["respawn_steps"]),
            )
            for food in self.map_cfg["foods"]
        ]
        self.hazards = [
            Hazard(
                name=str(hazard["name"]),
                position=_as_array(hazard["position"]),
                radius=float(hazard["radius"]),
                penalty=float(hazard["penalty"]),
                energy_drain=float(hazard["energy_drain"]),
            )
            for hazard in self.map_cfg["hazards"]
        ]
        self.ray_count = int(self.observation_cfg["ray_count"])
        self.ray_range = float(self.observation_cfg["ray_range"])
        self.ray_steps = int(self.observation_cfg["ray_steps"])
        self.observation_dim = self.ray_count + 10

        self.rng = np.random.default_rng(0)
        self.state: AgentState | None = None
        self.foods: list[Food] = []
        self.total_reward = 0.0
        self.food_collected = 0
        self.hazard_contacts = 0
        self.wall_collisions = 0
        self.visited_cells: set[tuple[int, int]] = set()
        self.termination = "not_started"

    @classmethod
    def from_config_path(
        cls, path: str | Path, *, food_respawn_timing: str = "corrected"
    ) -> ForageMaze2D:
        return cls(load_env_config(path), food_respawn_timing=food_respawn_timing)

    def reset(self, seed: int | None = None) -> Array:
        self.rng = np.random.default_rng(seed)
        start = _as_array(self.agent_cfg["start_position"]).copy()
        jitter = float(self.agent_cfg.get("start_jitter", 0.0))
        if jitter > 0.0:
            start += self.rng.uniform(-jitter, jitter, size=2)
        if not self.position_is_free(start):
            raise ValueError(f"Configured start position is blocked: {start.tolist()}")

        self.state = AgentState(
            position=start,
            velocity=np.zeros(2, dtype=np.float64),
            heading=float(self.agent_cfg["start_heading"]),
            energy=float(self.agent_cfg["initial_energy"]),
            step_count=0,
        )
        self.foods = [
            Food(
                name=food.name,
                position=food.position.copy(),
                radius=food.radius,
                reward=food.reward,
                energy_gain=food.energy_gain,
                respawn_steps=food.respawn_steps,
            )
            for food in self.base_foods
        ]
        self.total_reward = 0.0
        self.food_collected = 0
        self.hazard_contacts = 0
        self.wall_collisions = 0
        self.visited_cells = set()
        self.termination = "running"
        self._mark_explored(start)
        return self.observe()

    def position_is_free(self, position: Array, radius: float | None = None) -> bool:
        check_radius = self.agent_radius if radius is None else radius
        x = float(position[0])
        y = float(position[1])
        if x < self.world_min[0] + check_radius or x > self.world_max[0] - check_radius:
            return False
        if y < self.world_min[1] + check_radius or y > self.world_max[1] - check_radius:
            return False
        return not any(_circle_rect_intersects(position, check_radius, wall) for wall in self.walls)

    def step(
        self, action: Array | list[float] | tuple[float, float, float]
    ) -> tuple[Array, float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        action_arr = np.asarray(action, dtype=np.float64)
        if action_arr.shape != (3,):
            raise ValueError(f"Expected action shape (3,), got {action_arr.shape}.")
        action_arr = np.nan_to_num(action_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        action_arr = np.clip(action_arr, -1.0, 1.0)

        state = self.state
        previous_position = state.position.copy()
        state.heading = _angle_wrap(
            state.heading + float(action_arr[0]) * float(self.agent_cfg["turn_rate"]) * self.dt
        )
        forward = np.asarray([math.cos(state.heading), math.sin(state.heading)], dtype=np.float64)
        thrust = max(0.0, float(action_arr[1]))
        brake = max(0.0, float(action_arr[2]))
        state.velocity += forward * thrust * float(self.agent_cfg["thrust_accel"]) * self.dt
        state.velocity *= float(self.agent_cfg["linear_drag"])
        if brake > 0.0:
            state.velocity *= 1.0 - min(0.95, brake * float(self.agent_cfg["brake_drag"]))
        speed = float(np.linalg.norm(state.velocity))
        max_speed = float(self.agent_cfg["max_speed"])
        if speed > max_speed:
            state.velocity *= max_speed / speed
            speed = max_speed

        attempted_position = state.position + state.velocity * self.dt
        collision = self._move_with_axis_slide(attempted_position)
        if collision:
            self.wall_collisions += 1
            state.velocity *= 0.25

        reward = float(self.fitness_cfg["alive_reward"])
        if collision:
            reward -= float(self.fitness_cfg["wall_collision_penalty"])

        state.energy -= float(self.fitness_cfg["energy_decay"])
        state.energy -= float(self.fitness_cfg["movement_energy_cost"]) * (speed / max_speed)
        reward += self._food_reward()
        reward += self._hazard_penalty()
        reward += self._exploration_reward()
        self._update_food_respawns()

        state.step_count += 1
        done = False
        if state.energy <= 0.0:
            done = True
            self.termination = "energy_depleted"
        elif state.step_count >= self.max_steps:
            done = True
            self.termination = "max_steps"

        self.total_reward += reward
        info = {
            "position": state.position.copy(),
            "previous_position": previous_position,
            "energy": state.energy,
            "food_collected": self.food_collected,
            "hazard_contacts": self.hazard_contacts,
            "wall_collisions": self.wall_collisions,
            "termination": self.termination,
        }
        return self.observe(), reward, done, info

    def observe(self) -> Array:
        if self.state is None:
            raise RuntimeError("Call reset() before observe().")
        state = self.state
        rays = self._ray_observation()
        food = self._nearest_food_sensor()
        hazard = self._nearest_hazard_sensor()
        speed_scale = max(float(self.agent_cfg["max_speed"]), 1e-9)
        velocity_body = self._world_to_body(state.velocity) / speed_scale
        energy = np.asarray([state.energy / float(self.agent_cfg["max_energy"])], dtype=np.float64)
        time_fraction = np.asarray([state.step_count / max(1, self.max_steps)], dtype=np.float64)
        return np.concatenate([rays, food, hazard, velocity_body, energy, time_fraction])

    def state_view(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call reset() before state_view().")
        active_food_positions = [food.position.copy() for food in self.foods if food.active]
        return {
            "position": self.state.position.copy(),
            "velocity": self.state.velocity.copy(),
            "heading": self.state.heading,
            "energy": self.state.energy,
            "step_count": self.state.step_count,
            "active_food_positions": active_food_positions,
            "hazards": self.hazards,
            "world_min": self.world_min.copy(),
            "world_max": self.world_max.copy(),
        }

    def rollout(
        self,
        policy: Any,
        seed: int,
        name: str | None = None,
        record: bool = True,
    ) -> RolloutResult:
        obs = self.reset(seed=seed)
        rows: list[list[float]] = []
        done = False
        while not done:
            view = self.state_view()
            action = policy.act(obs, view)
            obs, _, done, _ = self.step(action)
            if record and self.state is not None:
                rows.append(
                    [
                        float(self.state.position[0]),
                        float(self.state.position[1]),
                        float(self.state.velocity[0]),
                        float(self.state.velocity[1]),
                        float(self.state.energy),
                        float(self.food_collected),
                        float(self.hazard_contacts),
                    ]
                )
        trajectory = (
            np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 7), dtype=np.float64)
        )
        if self.state is None:
            raise RuntimeError("Rollout ended with missing state.")
        return RolloutResult(
            name=name or policy.__class__.__name__,
            fitness=float(self.total_reward),
            steps=int(self.state.step_count),
            food_collected=int(self.food_collected),
            hazard_contacts=int(self.hazard_contacts),
            wall_collisions=int(self.wall_collisions),
            final_position=self.state.position.copy(),
            trajectory=trajectory,
            termination=self.termination,
        )

    def _move_with_axis_slide(self, attempted_position: Array) -> bool:
        if self.state is None:
            raise RuntimeError("Call reset() before moving.")
        state = self.state
        if self.position_is_free(attempted_position):
            state.position = attempted_position
            return False

        collision = True
        x_only = np.asarray([attempted_position[0], state.position[1]], dtype=np.float64)
        y_only = np.asarray([state.position[0], attempted_position[1]], dtype=np.float64)
        if self.position_is_free(x_only):
            state.position = x_only
            state.velocity[1] = 0.0
        elif self.position_is_free(y_only):
            state.position = y_only
            state.velocity[0] = 0.0
        else:
            state.velocity[:] = 0.0
        return collision

    def _food_reward(self) -> float:
        if self.state is None:
            raise RuntimeError("Call reset() before reward computation.")
        reward = 0.0
        for food in self.foods:
            if not food.active:
                continue
            distance = float(np.linalg.norm(self.state.position - food.position))
            if distance <= self.agent_radius + food.radius:
                food.active = False
                food.respawn_timer = food.respawn_steps + int(
                    self.food_respawn_timing == "corrected"
                )
                self.food_collected += 1
                reward += food.reward * float(self.fitness_cfg["food_reward_scale"])
                self.state.energy = min(
                    float(self.agent_cfg["max_energy"]),
                    self.state.energy + food.energy_gain,
                )
        return reward

    def _hazard_penalty(self) -> float:
        if self.state is None:
            raise RuntimeError("Call reset() before hazard computation.")
        penalty = 0.0
        for hazard in self.hazards:
            distance = float(np.linalg.norm(self.state.position - hazard.position))
            if distance <= self.agent_radius + hazard.radius:
                self.hazard_contacts += 1
                self.state.energy -= hazard.energy_drain
                penalty -= hazard.penalty * float(self.fitness_cfg["hazard_penalty_scale"])
        return penalty

    def _exploration_reward(self) -> float:
        if self.state is None:
            raise RuntimeError("Call reset() before exploration computation.")
        return self._mark_explored(self.state.position) * float(
            self.fitness_cfg["exploration_reward"]
        )

    def _mark_explored(self, position: Array) -> int:
        cell_size = float(self.fitness_cfg["exploration_cell_size"])
        key = (
            int(math.floor((float(position[0]) - float(self.world_min[0])) / cell_size)),
            int(math.floor((float(position[1]) - float(self.world_min[1])) / cell_size)),
        )
        if key in self.visited_cells:
            return 0
        self.visited_cells.add(key)
        return 1

    def _update_food_respawns(self) -> None:
        for food in self.foods:
            if food.active or food.respawn_steps <= 0:
                continue
            food.respawn_timer -= 1
            if food.respawn_timer <= 0:
                food.active = True

    def _ray_observation(self) -> Array:
        if self.state is None:
            raise RuntimeError("Call reset() before ray observation.")
        values = np.ones(self.ray_count, dtype=np.float64)
        for idx in range(self.ray_count):
            angle = self.state.heading + (2.0 * math.pi * idx / self.ray_count)
            direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            distance = self._ray_distance_to_boundary(direction)
            for wall in self.walls:
                distance = min(distance, self._ray_distance_to_wall(direction, wall))
            values[idx] = min(1.0, distance / self.ray_range)
        return values

    def _ray_distance_to_boundary(self, direction: Array) -> float:
        if self.state is None:
            raise RuntimeError("Call reset() before ray casting.")
        distances: list[float] = []
        low = self.world_min + self.agent_radius
        high = self.world_max - self.agent_radius
        origin = self.state.position
        if direction[0] > 1e-12:
            distances.append((float(high[0]) - float(origin[0])) / float(direction[0]))
        elif direction[0] < -1e-12:
            distances.append((float(low[0]) - float(origin[0])) / float(direction[0]))
        if direction[1] > 1e-12:
            distances.append((float(high[1]) - float(origin[1])) / float(direction[1]))
        elif direction[1] < -1e-12:
            distances.append((float(low[1]) - float(origin[1])) / float(direction[1]))
        positive = [distance for distance in distances if distance >= 0.0]
        return min(positive) if positive else self.ray_range

    def _ray_distance_to_wall(self, direction: Array, wall: RectWall) -> float:
        if self.state is None:
            raise RuntimeError("Call reset() before ray casting.")
        radius = self.agent_radius
        bounds = (
            wall.x_min - radius,
            wall.y_min - radius,
            wall.x_max + radius,
            wall.y_max + radius,
        )
        return _ray_aabb_distance(self.state.position, direction, bounds, self.ray_range)

    def _nearest_food_sensor(self) -> Array:
        if self.state is None:
            raise RuntimeError("Call reset() before food sensor.")
        active = [food for food in self.foods if food.active]
        if not active:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        nearest = min(
            active, key=lambda food: float(np.linalg.norm(food.position - self.state.position))
        )
        return self._relative_sensor(nearest.position)

    def _nearest_hazard_sensor(self) -> Array:
        if self.state is None:
            raise RuntimeError("Call reset() before hazard sensor.")
        if not self.hazards:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        nearest = min(
            self.hazards,
            key=lambda hazard: float(np.linalg.norm(hazard.position - self.state.position)),
        )
        return self._relative_sensor(nearest.position)

    def _relative_sensor(self, target: Array) -> Array:
        if self.state is None:
            raise RuntimeError("Call reset() before relative sensor.")
        delta = target - self.state.position
        distance = float(np.linalg.norm(delta))
        world_diag = float(np.linalg.norm(self.world_max - self.world_min))
        if distance <= 1e-12:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        body_delta = self._world_to_body(delta / distance)
        return np.asarray(
            [min(1.0, distance / world_diag), body_delta[0], body_delta[1]], dtype=np.float64
        )

    def _world_to_body(self, vector: Array) -> Array:
        if self.state is None:
            raise RuntimeError("Call reset() before frame transform.")
        c = math.cos(-self.state.heading)
        s = math.sin(-self.state.heading)
        return np.asarray(
            [
                c * float(vector[0]) - s * float(vector[1]),
                s * float(vector[0]) + c * float(vector[1]),
            ],
            dtype=np.float64,
        )


def _ray_aabb_distance(
    origin: Array,
    direction: Array,
    bounds: tuple[float, float, float, float],
    max_distance: float,
) -> float:
    x_min, y_min, x_max, y_max = bounds
    t_min = 0.0
    t_max = max_distance
    for axis, low, high in ((0, x_min, x_max), (1, y_min, y_max)):
        axis_origin = float(origin[axis])
        axis_direction = float(direction[axis])
        if abs(axis_direction) < 1e-12:
            if axis_origin < low or axis_origin > high:
                return max_distance
            continue
        t1 = (low - axis_origin) / axis_direction
        t2 = (high - axis_origin) / axis_direction
        near = min(t1, t2)
        far = max(t1, t2)
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return max_distance
    if t_min < 0.0:
        return max_distance
    return t_min
