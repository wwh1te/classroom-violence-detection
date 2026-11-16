
"""Fit and apply feature normalization using training data only."""

from __future__ import annotations

import numpy as np
import torch

from src.features.aggregation import (
    aggregate_frame_features,
    aggregate_for_mlp,
)


def compute_normalization_stats(loader, model_type: str):
    """Calculate mean and standard deviation from training samples only."""

    if model_type not in {"mlp", "lstm"}:
        raise ValueError(f"Invalid model type: {model_type}")

    feature_batches = []

    for batch in loader:
        if model_type == "mlp":
            features, _ = batch
            feature_batches.append(features.numpy())

        else:
            features, lengths, _ = batch

            for index, length in enumerate(lengths.tolist()):
                valid_frames = features[index, :length]
                feature_batches.append(valid_frames.numpy())

    if not feature_batches:
        raise ValueError("Training loader contains no samples")

    all_features = np.concatenate(feature_batches, axis=0)

    mean = all_features.mean(axis=0).astype(np.float32)
    std = all_features.std(axis=0).astype(np.float32)

    # Avoid division by zero for constant features.
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    return {
        "mean": mean,
        "std": std,
    }


def normalize_features(features, stats):
    """Apply previously calculated normalization statistics."""

    mean = torch.as_tensor(stats["mean"], dtype=features.dtype)
    std = torch.as_tensor(stats["std"], dtype=features.dtype)

    if features.shape[-1] != mean.numel():
        raise ValueError("Feature dimension does not match normalization stats")

    return (features - mean) / std


def normalize_mlp_batch(features, stats):
    """Normalize a batch of MLP inputs."""

    return normalize_features(features, stats)


def normalize_lstm_batch(features, lengths, stats):
    """Normalize valid LSTM frames while keeping padding at zero."""

    normalized = torch.zeros_like(features)

    for index, length in enumerate(lengths.tolist()):
        valid_frames = features[index, :length]

        normalized[index, :length] = normalize_features(
            valid_frames,
            stats,
        )

    return normalized



