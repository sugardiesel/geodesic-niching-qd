"""Matplotlib plots for Phase 1 validation and report diagnostics."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envs.forage_maze import ForageMaze2D, RolloutResult

PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def save_rollout_plot(
    env: ForageMaze2D,
    rollouts: Iterable[RolloutResult],
    output_stem: str | Path,
    title: str,
    extra_paths: dict[str, np.ndarray] | None = None,
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    draw_environment(ax, env)

    if extra_paths:
        for label, coords in extra_paths.items():
            if len(coords) == 0:
                continue
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="#111827",
                linestyle="--",
                linewidth=2.0,
                alpha=0.75,
                label=f"{label} shortest path",
            )

    for idx, rollout in enumerate(rollouts):
        color = PALETTE[idx % len(PALETTE)]
        if len(rollout.trajectory) > 0:
            ax.plot(
                rollout.trajectory[:, 0],
                rollout.trajectory[:, 1],
                color=color,
                linewidth=1.8,
                alpha=0.85,
                label=(f"{rollout.name}: fit {rollout.fitness:.1f}, food {rollout.food_collected}"),
            )
            ax.scatter(
                [rollout.final_position[0]],
                [rollout.final_position[1]],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    return _save_figure(fig, stem)


def save_detour_map_plot(
    env: ForageMaze2D,
    point_a: np.ndarray,
    point_b: np.ndarray,
    shortest_path: np.ndarray,
    output_stem: str | Path,
    euclidean_distance: float,
    geodesic_distance: float,
) -> list[str]:
    """Save a compact map showing one Euclidean shortcut and its traversable path."""
    if shortest_path.ndim != 2 or shortest_path.shape[1] != 2 or len(shortest_path) < 2:
        raise ValueError("shortest_path must contain at least two 2D grid points.")

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.45, 3.35), constrained_layout=True)
    draw_environment(ax, env)

    ax.plot(
        shortest_path[:, 0],
        shortest_path[:, 1],
        color=PALETTE[0],
        linewidth=1.8,
        solid_capstyle="round",
        zorder=4,
        label=f"Shortest path: {geodesic_distance:.3f}",
    )
    ax.plot(
        [float(point_a[0]), float(point_b[0])],
        [float(point_a[1]), float(point_b[1])],
        color=PALETTE[1],
        linestyle=(0, (3.5, 2.2)),
        linewidth=1.5,
        zorder=5,
        label=f"Euclidean: {euclidean_distance:.3f}",
    )
    ax.scatter(
        [float(point_a[0]), float(point_b[0])],
        [float(point_a[1]), float(point_b[1])],
        s=28,
        color=[PALETTE[4], PALETTE[3]],
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    for label, point, offset in [
        ("A", point_a, (-8, -1)),
        ("B", point_b, (5, -1)),
    ]:
        ax.annotate(
            label,
            xy=(float(point[0]), float(point[1])),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            va="center",
            ha="center",
            zorder=7,
        )

    ax.set_xlabel("x position", fontsize=8)
    ax.set_ylabel("y position", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(
        loc="upper right",
        fontsize=6.8,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.92,
        borderpad=0.35,
        handlelength=2.3,
    )
    return _save_figure(fig, stem, tight=True)


def save_distance_scatter(
    euclidean: np.ndarray,
    geodesic: np.ndarray,
    output_stem: str | Path,
    title: str,
    correlation: float,
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(euclidean) & np.isfinite(geodesic)
    x = euclidean[finite]
    y = geodesic[finite]

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.scatter(x, y, s=13, alpha=0.38, color="#2563eb", linewidths=0)
    limit = max(float(np.percentile(x, 99.0)), float(np.percentile(y, 99.0))) * 1.05
    ax.plot([0.0, limit], [0.0, limit], color="#64748b", linestyle="--", linewidth=1.2)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title)
    ax.set_xlabel("Euclidean final-position distance")
    ax.set_ylabel("Maze shortest-path distance")
    ax.text(
        0.03,
        0.96,
        f"overall Pearson r = {correlation:.3f}\npairs = {len(x)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "#cbd5e1",
        },
    )
    return _save_figure(fig, stem)


def save_region_distance_scatter(
    euclidean: np.ndarray,
    geodesic: np.ndarray,
    region_mask: np.ndarray,
    output_stem: str | Path,
    title: str,
    overall_r: float,
    horseshoe_r: float,
    open_r: float,
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(euclidean) & np.isfinite(geodesic)
    x = euclidean[finite]
    y = geodesic[finite]
    mask = region_mask[finite]

    fig, ax = plt.subplots(figsize=(6.6, 5.3), constrained_layout=True)
    ax.scatter(
        x[~mask],
        y[~mask],
        s=13,
        alpha=0.35,
        color="#64748b",
        linewidths=0,
        label=f"open field pairs (r={open_r:.3f}, n={np.count_nonzero(~mask)})",
    )
    ax.scatter(
        x[mask],
        y[mask],
        s=18,
        alpha=0.68,
        color="#dc2626",
        marker="^",
        linewidths=0,
        label=f"horseshoe-region paths (r={horseshoe_r:.3f}, n={np.count_nonzero(mask)})",
    )
    limit = max(float(np.percentile(x, 99.0)), float(np.percentile(y, 99.0))) * 1.05
    ax.plot([0.0, limit], [0.0, limit], color="#0f172a", linestyle="--", linewidth=1.1)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title)
    ax.set_xlabel("Euclidean final-position distance")
    ax.set_ylabel("Maze shortest-path distance")
    ax.text(
        0.03,
        0.96,
        f"overall r = {overall_r:.3f}\nred triangles: shortest path touches horseshoe region",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#cbd5e1",
        },
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    return _save_figure(fig, stem)


def draw_environment(ax: plt.Axes, env: ForageMaze2D) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(env.world_min[0]), float(env.world_max[0]))
    ax.set_ylim(float(env.world_min[1]), float(env.world_max[1]))
    ax.set_facecolor("#f8fafc")
    for wall in env.walls:
        ax.add_patch(
            plt.Rectangle(
                (wall.x_min, wall.y_min),
                wall.x_max - wall.x_min,
                wall.y_max - wall.y_min,
                facecolor="#334155",
                edgecolor="#0f172a",
                linewidth=0.5,
            )
        )
    for hazard in env.hazards:
        ax.add_patch(
            plt.Circle(
                hazard.position,
                hazard.radius,
                facecolor="#ef4444",
                edgecolor="#b91c1c",
                alpha=0.28,
                linewidth=1.0,
            )
        )
    for food in env.base_foods:
        ax.add_patch(
            plt.Circle(
                food.position,
                food.radius,
                facecolor="#22c55e",
                edgecolor="#15803d",
                alpha=0.42,
                linewidth=1.0,
            )
        )
    ax.grid(color="#e2e8f0", linewidth=0.5, alpha=0.8)


def _save_figure(fig: plt.Figure, stem: Path, *, tight: bool = False) -> list[str]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    save_options = {"bbox_inches": "tight", "pad_inches": 0.01} if tight else {}
    for path in paths:
        fig.savefig(path, dpi=220, **save_options)
    plt.close(fig)
    return [str(path) for path in paths]
