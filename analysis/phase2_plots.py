"""Matplotlib plots for Phase 2 hand-crafted MAP-Elites."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algorithms.map_elites import GridArchive
from envs.forage_maze import ForageMaze2D


def save_archive_heatmap(
    archive: GridArchive,
    env: ForageMaze2D,
    output_stem: str | Path,
    title: str,
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    data = archive.fitness.copy().T
    data[~archive.occupied.T] = np.nan

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    image = ax.imshow(
        data,
        origin="lower",
        extent=[env.world_min[0], env.world_max[0], env.world_min[1], env.world_max[1]],
        cmap="viridis",
        aspect="equal",
    )
    for wall in env.walls:
        ax.add_patch(
            plt.Rectangle(
                (wall.x_min, wall.y_min),
                wall.x_max - wall.x_min,
                wall.y_max - wall.y_min,
                facecolor="none",
                edgecolor="#f8fafc",
                linewidth=1.4,
            )
        )
    ax.set_title(title)
    ax.set_xlabel("final x behavior descriptor")
    ax.set_ylabel("final y behavior descriptor")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("elite fitness")
    return _save_figure(fig, stem)


def save_metric_curves(
    metrics: list[dict[str, float | int]],
    output_stem: str | Path,
    title: str = "Phase 2 hand-crafted MAP-Elites progress",
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    evaluations = np.asarray([row["evaluation"] for row in metrics], dtype=np.float64)
    qd_score = np.asarray([row["qd_score"] for row in metrics], dtype=np.float64)
    coverage = np.asarray([row["coverage"] for row in metrics], dtype=np.float64)
    max_fitness = np.asarray([row["max_fitness"] for row in metrics], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(evaluations, qd_score, color="#2563eb", linewidth=2.0)
    axes[0].set_ylabel("shifted QD-score")
    axes[0].grid(alpha=0.25)

    axes[1].plot(evaluations, coverage * 100.0, color="#16a34a", linewidth=2.0)
    axes[1].set_ylabel("coverage (%)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(evaluations, max_fitness, color="#dc2626", linewidth=2.0)
    axes[2].set_ylabel("max fitness")
    axes[2].set_xlabel("evaluations")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title)
    return _save_figure(fig, stem)


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]
