"""Plots for Phase 3 AURORA-style learned behavior descriptor baseline."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algorithms.map_elites import GridArchive


def save_autoencoder_loss_plot(
    train_losses: list[float],
    validation_losses: list[float],
    output_stem: str | Path,
    ylabel: str = "MSE reconstruction loss",
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(epochs, train_losses, label="train", color="#2563eb", linewidth=2.0)
    ax.plot(epochs, validation_losses, label="validation", color="#dc2626", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title("Phase 3 trajectory autoencoder loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, stem)


def save_reconstruction_examples(
    real: np.ndarray,
    reconstructed: np.ndarray,
    output_stem: str | Path,
    feature_names: list[str],
    labels: list[str] | None = None,
    max_examples: int = 4,
    xy_limits: tuple[float, float] = (-1.05, 1.05),
    feature_limits: tuple[float, float] = (-1.1, 1.25),
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_examples, len(real))
    fig, axes = plt.subplots(count, 2, figsize=(8.2, 2.4 * count), constrained_layout=True)
    if count == 1:
        axes = np.asarray([axes])
    for idx in range(count):
        label = labels[idx] if labels and idx < len(labels) else f"trajectory {idx + 1}"
        axes[idx, 0].plot(real[idx, :, 0], real[idx, :, 1], color="#2563eb", label="real")
        axes[idx, 0].plot(
            reconstructed[idx, :, 0],
            reconstructed[idx, :, 1],
            color="#dc2626",
            linestyle="--",
            label="reconstructed",
        )
        axes[idx, 0].set_title(f"{label}: x/y path", fontsize=10)
        axes[idx, 0].set_aspect("equal", adjustable="box")
        axes[idx, 0].set_xlim(*xy_limits)
        axes[idx, 0].set_ylim(*xy_limits)
        axes[idx, 0].grid(alpha=0.25)
        axes[idx, 0].legend(frameon=False, fontsize=8)

        for feature_idx, name in enumerate(feature_names[: min(5, len(feature_names))]):
            axes[idx, 1].plot(real[idx, :, feature_idx], linewidth=1.6, label=f"{name} real")
            axes[idx, 1].plot(
                reconstructed[idx, :, feature_idx],
                linewidth=1.2,
                linestyle="--",
                label=f"{name} recon",
            )
        axes[idx, 1].set_title(f"{label}: sequence features", fontsize=10)
        axes[idx, 1].set_xlim(0, real.shape[1] - 1)
        axes[idx, 1].set_ylim(*feature_limits)
        axes[idx, 1].grid(alpha=0.25)
    return _save_figure(fig, stem)


def save_latent_archive_heatmap(
    archive: GridArchive,
    output_stem: str | Path,
    title: str,
) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    data = archive.fitness.copy().T
    data[~archive.occupied.T] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 5.5), constrained_layout=True)
    image = ax.imshow(
        data,
        origin="lower",
        extent=[archive.bd_min[0], archive.bd_max[0], archive.bd_min[1], archive.bd_max[1]],
        cmap="viridis",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("learned latent dimension 1")
    ax.set_ylabel("learned latent dimension 2")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("elite fitness")
    return _save_figure(fig, stem)


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return [str(path) for path in paths]
