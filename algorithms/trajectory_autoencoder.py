"""Trajectory sequence autoencoder for AURORA-style learned descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLUMNS = {
    "x": 0,
    "y": 1,
    "vx": 2,
    "vy": 3,
    "energy": 4,
    "food_collected": 5,
    "hazard_contacts": 6,
}


class TrajectoryAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


@dataclass
class AutoencoderTrainingResult:
    model: TrajectoryAutoencoder
    normalizer: SequenceNormalizer
    train_losses: list[float]
    validation_losses: list[float]
    device: str
    normalization_audit: dict[str, Any]


@dataclass(frozen=True)
class SequenceNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def identity(cls, sequences: np.ndarray) -> SequenceNormalizer:
        flat_dim = flatten_sequences(sequences).shape[1]
        return cls(
            mean=np.zeros(flat_dim, dtype=np.float32),
            std=np.ones(flat_dim, dtype=np.float32),
        )

    @classmethod
    def fit(cls, sequences: np.ndarray, eps: float = 1e-6) -> SequenceNormalizer:
        flat = flatten_sequences(sequences)
        mean = np.mean(flat, axis=0).astype(np.float32)
        std = np.std(flat, axis=0).astype(np.float32)
        std = np.maximum(std, eps)
        return cls(mean=mean, std=std)

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        flat = flatten_sequences(sequences)
        return ((flat - self.mean) / self.std).astype(np.float32)

    def inverse_transform(
        self, flat: np.ndarray, sequence_shape: tuple[int, int, int]
    ) -> np.ndarray:
        restored = flat * self.std + self.mean
        return restored.reshape(sequence_shape).astype(np.float32)


def trajectory_to_sequence(
    trajectory: np.ndarray,
    sequence_length: int,
    feature_names: list[str],
) -> np.ndarray:
    columns = [FEATURE_COLUMNS[name] for name in feature_names]
    if trajectory.size == 0:
        return np.zeros((sequence_length, len(columns)), dtype=np.float32)
    selected = trajectory[:, columns].astype(np.float32)
    source_x = np.linspace(0.0, 1.0, len(selected), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    sequence = np.zeros((sequence_length, len(columns)), dtype=np.float32)
    for idx in range(len(columns)):
        sequence[:, idx] = np.interp(target_x, source_x, selected[:, idx])
    return sequence


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    return sequences.reshape(sequences.shape[0], -1).astype(np.float32)


def train_autoencoder(
    sequences: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> AutoencoderTrainingResult:
    torch.manual_seed(seed)
    device = _select_device(str(config.get("device", "auto")))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(sequences))
    validation_fraction = float(config.get("validation_fraction", 0.15))
    validation_count = max(1, int(len(sequences) * validation_fraction))
    validation_idx = indices[:validation_count]
    train_idx = indices[validation_count:]
    normalizer = (
        SequenceNormalizer.fit(sequences)
        if bool(config.get("normalize", True))
        else SequenceNormalizer.identity(sequences)
    )
    train_only_normalizer = (
        SequenceNormalizer.fit(sequences[train_idx])
        if bool(config.get("normalize", True))
        else SequenceNormalizer.identity(sequences[train_idx])
    )
    normalization_audit = compare_normalizers(
        used=normalizer,
        train_only=train_only_normalizer,
        sequence_shape=sequences.shape[1:],
        training_count=len(train_idx),
        validation_count=len(validation_idx),
    )
    flat = normalizer.transform(sequences)
    sequence_length = sequences.shape[1]
    feature_count = sequences.shape[2]

    train_tensor = torch.from_numpy(flat[train_idx])
    validation_tensor = torch.from_numpy(flat[validation_idx]).to(device)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    model = TrajectoryAutoencoder(input_dim=flat.shape[1], latent_dim=int(config["latent_dim"])).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = nn.MSELoss()
    derivative_loss_weight = float(config.get("derivative_loss_weight", 0.0))

    train_losses: list[float] = []
    validation_losses: list[float] = []
    epochs = int(config["epochs"])
    for _ in range(epochs):
        model.train()
        running = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            if derivative_loss_weight > 0.0:
                batch_seq = batch.reshape(batch.shape[0], sequence_length, feature_count)
                recon_seq = reconstruction.reshape(batch.shape[0], sequence_length, feature_count)
                derivative_loss = criterion(
                    recon_seq[:, 1:, :] - recon_seq[:, :-1, :],
                    batch_seq[:, 1:, :] - batch_seq[:, :-1, :],
                )
                loss = loss + derivative_loss_weight * derivative_loss
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch)
            count += len(batch)
        train_losses.append(running / max(1, count))

        model.eval()
        with torch.no_grad():
            reconstruction, _ = model(validation_tensor)
            validation_loss = criterion(reconstruction, validation_tensor)
        validation_losses.append(float(validation_loss.item()))

    return AutoencoderTrainingResult(
        model=model,
        normalizer=normalizer,
        train_losses=train_losses,
        validation_losses=validation_losses,
        device=device,
        normalization_audit=normalization_audit,
    )


def compare_normalizers(
    used: SequenceNormalizer,
    train_only: SequenceNormalizer,
    sequence_shape: tuple[int, int],
    training_count: int,
    validation_count: int,
) -> dict[str, Any]:
    """Quantify all-data normalizer leakage against a training-split-only fit."""

    denominator = np.maximum(train_only.std.astype(np.float64), 1e-12)
    mean_delta = np.abs(used.mean.astype(np.float64) - train_only.mean.astype(np.float64))
    std_delta = np.abs(used.std.astype(np.float64) - train_only.std.astype(np.float64))
    standardized_mean_delta = mean_delta / denominator
    relative_std_delta = std_delta / denominator
    sequence_length, feature_count = sequence_shape

    per_feature = []
    for feature_index in range(feature_count):
        feature_slice = slice(feature_index, sequence_length * feature_count, feature_count)
        feature_mean_delta = standardized_mean_delta[feature_slice]
        feature_std_delta = relative_std_delta[feature_slice]
        per_feature.append(
            {
                "feature_index": feature_index,
                "mean_standardized_mean_delta": float(np.mean(feature_mean_delta)),
                "max_standardized_mean_delta": float(np.max(feature_mean_delta)),
                "mean_relative_std_delta": float(np.mean(feature_std_delta)),
                "max_relative_std_delta": float(np.max(feature_std_delta)),
            }
        )

    return {
        "used_fit_scope": "all_sequences_including_validation",
        "comparison_fit_scope": "training_split_only",
        "training_count": int(training_count),
        "validation_count": int(validation_count),
        "dimension_count": int(len(used.mean)),
        "mean_standardized_mean_delta": float(np.mean(standardized_mean_delta)),
        "p95_standardized_mean_delta": float(np.quantile(standardized_mean_delta, 0.95)),
        "max_standardized_mean_delta": float(np.max(standardized_mean_delta)),
        "mean_relative_std_delta": float(np.mean(relative_std_delta)),
        "p95_relative_std_delta": float(np.quantile(relative_std_delta, 0.95)),
        "max_relative_std_delta": float(np.max(relative_std_delta)),
        "used_mean_l2": float(np.linalg.norm(used.mean)),
        "train_only_mean_l2": float(np.linalg.norm(train_only.mean)),
        "used_std_l2": float(np.linalg.norm(used.std)),
        "train_only_std_l2": float(np.linalg.norm(train_only.std)),
        "per_feature": per_feature,
    }


def encode_sequences(
    model: TrajectoryAutoencoder,
    normalizer: SequenceNormalizer,
    sequences: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    flat = normalizer.transform(sequences)
    model.eval()
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            batch = torch.from_numpy(flat[start : start + batch_size]).to(device)
            latent = model.encoder(batch)
            latents.append(latent.cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float64)


def reconstruct_sequences(
    model: TrajectoryAutoencoder,
    normalizer: SequenceNormalizer,
    sequences: np.ndarray,
    device: str,
) -> np.ndarray:
    shape = sequences.shape
    flat = normalizer.transform(sequences)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(flat), 512):
            batch = torch.from_numpy(flat[start : start + 512]).to(device)
            reconstruction, _ = model(batch)
            outputs.append(reconstruction.cpu().numpy())
    return normalizer.inverse_transform(np.concatenate(outputs, axis=0), shape)


def save_autoencoder_checkpoint(
    training: AutoencoderTrainingResult,
    path: str | Path,
    sequence_length: int,
    feature_names: list[str],
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": training.model.state_dict(),
        "normalizer_mean": training.normalizer.mean,
        "normalizer_std": training.normalizer.std,
        "input_dim": int(training.normalizer.mean.shape[0]),
        "latent_dim": int(config["latent_dim"]),
        "sequence_length": int(sequence_length),
        "feature_names": list(feature_names),
        "autoencoder_config": dict(config),
        "train_losses": list(training.train_losses),
        "validation_losses": list(training.validation_losses),
        "normalization_audit": dict(training.normalization_audit),
    }
    torch.save(checkpoint, checkpoint_path)


def load_autoencoder_checkpoint(
    path: str | Path,
    device: str = "auto",
) -> tuple[AutoencoderTrainingResult, dict[str, Any]]:
    selected_device = _select_device(device)
    try:
        checkpoint = torch.load(Path(path), map_location=selected_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=selected_device)
    model = TrajectoryAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = SequenceNormalizer(
        mean=np.asarray(checkpoint["normalizer_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["normalizer_std"], dtype=np.float32),
    )
    training = AutoencoderTrainingResult(
        model=model,
        normalizer=normalizer,
        train_losses=[float(value) for value in checkpoint.get("train_losses", [])],
        validation_losses=[float(value) for value in checkpoint.get("validation_losses", [])],
        device=selected_device,
        normalization_audit=dict(checkpoint.get("normalization_audit", {})),
    )
    metadata = {
        "sequence_length": int(checkpoint["sequence_length"]),
        "feature_names": list(checkpoint["feature_names"]),
        "autoencoder_config": dict(checkpoint.get("autoencoder_config", {})),
        "latent_dim": int(checkpoint["latent_dim"]),
        "input_dim": int(checkpoint["input_dim"]),
    }
    return training, metadata


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested
