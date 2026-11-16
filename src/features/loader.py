"""Load aggregated feature datasets for MLP and LSTM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.aggregation import (
    aggregate_frame_features,
    aggregate_for_mlp,
    aggregate_for_lstm,
)

FEATURE_ROOT = Path("outputs/features")
VALID_SPLITS = {"train", "val"}
VALID_MODELS = {"mlp", "lstm"}


class VideoFeatureDataset(Dataset):
    """Load video features for classification."""

    def __init__(self, split: str, model_type: str):
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        if model_type not in VALID_MODELS:
            raise ValueError(f"Invalid model type: {model_type}")

        self.split = split
        self.model_type = model_type

        index_path = FEATURE_ROOT / split / "index.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(index_path)

        with index_path.open("r", encoding="utf-8") as file:
            self.records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if not self.records:
            raise ValueError(f"No samples found in {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        feature_path = Path(record["output_path"])

        with np.load(feature_path, allow_pickle=False) as data:
            frame_features = aggregate_frame_features(
                data["features"],
                data["person_mask"],
                data["motion_features"],
                data["motion_mask"],
            )

            if self.model_type == "mlp":
                sample_features = aggregate_for_mlp(frame_features)
            else:
                sample_features = aggregate_for_lstm(frame_features)

            label = int(data["label"])

        x = torch.from_numpy(sample_features.copy())
        y = torch.tensor(label, dtype=torch.long)

        return x, y



def collate_mlp_batch(batch):
    """Combine fixed-size MLP samples into a batch."""

    features, labels = zip(*batch)

    x = torch.stack(features, dim=0)
    y = torch.stack(labels, dim=0)

    return x, y


def collate_lstm_batch(batch):
    """Pad variable-length sequences for LSTM."""

    from torch.nn.utils.rnn import pad_sequence

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.long,
    )

    x = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    y = torch.stack(labels, dim=0)

    return x, lengths, y



def create_data_loader(
    split: str,
    model_type: str,
    batch_size: int = 16,
    shuffle: bool | None = None,
    num_workers: int = 0,
):
    """Create a PyTorch DataLoader for MLP or LSTM."""

    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    if shuffle is None:
        shuffle = split == "train"

    dataset = VideoFeatureDataset(
        split=split,
        model_type=model_type,
    )

    collate_function = (
        collate_mlp_batch
        if model_type == "mlp"
        else collate_lstm_batch
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_function,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )



