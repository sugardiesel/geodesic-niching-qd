"""Small policies used for Phase 1 validation and later QD experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class RandomMLPPolicy:
    """Tiny tanh MLP policy, kept under 1k parameters for MAP-Elites later."""

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    @classmethod
    def from_rng(
        cls,
        obs_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
        scale: float = 0.7,
    ) -> RandomMLPPolicy:
        w1 = rng.normal(0.0, scale / math.sqrt(obs_dim), size=(hidden_dim, obs_dim))
        b1 = rng.normal(0.0, scale, size=(hidden_dim,))
        w2 = rng.normal(0.0, scale / math.sqrt(hidden_dim), size=(3, hidden_dim))
        b2 = rng.normal(0.0, scale, size=(3,))
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    @property
    def parameter_count(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    def act(self, obs: np.ndarray, state: dict[str, Any] | None = None) -> np.ndarray:
        hidden = np.tanh(self.w1 @ obs + self.b1)
        return np.tanh(self.w2 @ hidden + self.b2)


@dataclass(frozen=True)
class MLPPolicyGenome:
    """Flat genotype for the small MLP policy used by MAP-Elites."""

    weights: np.ndarray
    obs_dim: int
    hidden_dim: int

    @classmethod
    def random(
        cls,
        obs_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
        scale: float = 0.7,
    ) -> MLPPolicyGenome:
        policy = RandomMLPPolicy.from_rng(obs_dim, hidden_dim, rng, scale=scale)
        return cls.from_policy(policy)

    @classmethod
    def from_policy(cls, policy: RandomMLPPolicy) -> MLPPolicyGenome:
        weights = np.concatenate(
            [
                policy.w1.ravel(),
                policy.b1.ravel(),
                policy.w2.ravel(),
                policy.b2.ravel(),
            ]
        )
        return cls(
            weights=weights.astype(np.float64),
            obs_dim=policy.w1.shape[1],
            hidden_dim=policy.w1.shape[0],
        )

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size)

    @staticmethod
    def parameter_count_for(obs_dim: int, hidden_dim: int) -> int:
        return hidden_dim * obs_dim + hidden_dim + 3 * hidden_dim + 3

    def mutate(
        self,
        rng: np.random.Generator,
        sigma: float,
        mutation_probability: float,
    ) -> MLPPolicyGenome:
        mask = rng.random(self.weights.shape) < mutation_probability
        if not np.any(mask):
            mask[rng.integers(0, self.weights.size)] = True
        noise = rng.normal(0.0, sigma, size=self.weights.shape)
        child = self.weights.copy()
        child[mask] += noise[mask]
        child = np.clip(child, -5.0, 5.0)
        return MLPPolicyGenome(weights=child, obs_dim=self.obs_dim, hidden_dim=self.hidden_dim)

    def to_policy(self) -> RandomMLPPolicy:
        obs_dim = self.obs_dim
        hidden_dim = self.hidden_dim
        cursor = 0
        w1_size = hidden_dim * obs_dim
        w1 = self.weights[cursor : cursor + w1_size].reshape(hidden_dim, obs_dim)
        cursor += w1_size
        b1 = self.weights[cursor : cursor + hidden_dim]
        cursor += hidden_dim
        w2_size = 3 * hidden_dim
        w2 = self.weights[cursor : cursor + w2_size].reshape(3, hidden_dim)
        cursor += w2_size
        b2 = self.weights[cursor : cursor + 3]
        return RandomMLPPolicy(w1=w1, b1=b1, w2=w2, b2=b2)


class RandomActionPolicy:
    """Temporally correlated random action policy for smoke-test rollouts."""

    def __init__(self, seed: int, persistence: float = 0.88):
        self.rng = np.random.default_rng(seed)
        self.persistence = persistence
        self.action = self.rng.uniform(-0.6, 0.8, size=3)

    def act(self, obs: np.ndarray, state: dict[str, Any] | None = None) -> np.ndarray:
        noise = self.rng.normal(0.0, 0.35, size=3)
        self.action = self.persistence * self.action + (1.0 - self.persistence) * noise
        self.action[1] = max(-0.2, self.action[1])
        return np.clip(self.action, -1.0, 1.0)


class StayStillPolicy:
    def act(self, obs: np.ndarray, state: dict[str, Any] | None = None) -> np.ndarray:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


class SeekNearestFoodPolicy:
    """Privileged hand-coded policy used only for environment validation."""

    def act(self, obs: np.ndarray, state: dict[str, Any] | None = None) -> np.ndarray:
        if state is None or not state["active_food_positions"]:
            return np.asarray([0.0, 0.0, 0.5], dtype=np.float64)
        position = state["position"]
        target = min(
            state["active_food_positions"],
            key=lambda food_pos: float(np.linalg.norm(food_pos - position)),
        )
        return _steer_toward(position, float(state["heading"]), target)


class WaypointPolicy:
    """Privileged waypoint follower for validating deliberately different strategies."""

    def __init__(self, waypoints: list[tuple[float, float]], tolerance: float = 0.09):
        self.waypoints = [np.asarray(point, dtype=np.float64) for point in waypoints]
        self.tolerance = tolerance
        self.index = 0

    def act(self, obs: np.ndarray, state: dict[str, Any] | None = None) -> np.ndarray:
        if state is None or not self.waypoints:
            return np.asarray([0.0, 0.0, 0.5], dtype=np.float64)
        position = state["position"]
        while self.index < len(self.waypoints) - 1:
            if float(np.linalg.norm(self.waypoints[self.index] - position)) > self.tolerance:
                break
            self.index += 1
        return _steer_toward(position, float(state["heading"]), self.waypoints[self.index])


def _steer_toward(position: np.ndarray, heading: float, target: np.ndarray) -> np.ndarray:
    delta = target - position
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return np.asarray([0.0, 0.0, 0.8], dtype=np.float64)
    desired = math.atan2(float(delta[1]), float(delta[0]))
    error = angle_wrap(desired - heading)
    turn = float(np.clip(error / 0.75, -1.0, 1.0))
    thrust = float(np.clip(distance * 1.7, 0.18, 0.92))
    if abs(error) > 1.15:
        thrust = 0.0
    elif abs(error) > 0.55:
        thrust *= 0.45
    brake = float(np.clip((0.16 - distance) / 0.16, 0.0, 0.8))
    return np.asarray([turn, thrust, brake], dtype=np.float64)
