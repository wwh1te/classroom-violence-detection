"""Aggregate per-person pose and motion features into per-frame features."""

from __future__ import annotations

import numpy as np


POSE_FEATURES = 89
MOTION_FEATURES = 34
MAX_TRACKS = 8

AGGREGATED_FEATURES = (
    POSE_FEATURES * 2
    + MOTION_FEATURES * 2
)


def validate_feature_arrays(
    features: np.ndarray,
    person_mask: np.ndarray,
    motion_features: np.ndarray,
    motion_mask: np.ndarray,
) -> None:
    """Validate the shapes and values of extracted feature arrays."""

    if features.ndim != 3:
        raise ValueError("features must have shape (T, 8, 89)")

    frames, tracks, feature_count = features.shape

    if tracks != MAX_TRACKS or feature_count != POSE_FEATURES:
        raise ValueError(
            f"Invalid features shape: {features.shape}"
        )

    if person_mask.shape != (frames, MAX_TRACKS):
        raise ValueError("Invalid person_mask shape")

    if motion_features.shape != (
        frames,
        MAX_TRACKS,
        MOTION_FEATURES,
    ):
        raise ValueError("Invalid motion_features shape")

    if motion_mask.shape != (frames, MAX_TRACKS):
        raise ValueError("Invalid motion_mask shape")

    arrays = [
        features,
        person_mask,
        motion_features,
        motion_mask,
    ]

    for array in arrays:
        if not np.isfinite(array).all():
            raise ValueError("Feature arrays contain NaN or infinity")

    if not np.isin(person_mask, [0, 1]).all():
        raise ValueError("person_mask must contain only 0 and 1")

    if not np.isin(motion_mask, [0, 1]).all():
        raise ValueError("motion_mask must contain only 0 and 1")

    if np.any(motion_mask > person_mask):
        raise ValueError("motion_mask cannot exceed person_mask")


def aggregate_pose_features(
    features: np.ndarray,
    person_mask: np.ndarray,
) -> np.ndarray:
    """Aggregate pose features across people in each frame."""

    valid = person_mask.astype(bool)

    mean_features = np.zeros(
        (features.shape[0], POSE_FEATURES),
        dtype=np.float32,
    )

    max_features = np.zeros_like(mean_features)

    for frame_index in range(features.shape[0]):
        active_people = features[frame_index, valid[frame_index]]

        if active_people.shape[0] == 0:
            continue

        mean_features[frame_index] = active_people.mean(axis=0)
        max_features[frame_index] = active_people.max(axis=0)

    return np.concatenate(
        [mean_features, max_features],
        axis=1,
    )



def aggregate_motion_features(
    motion_features: np.ndarray,
    motion_mask: np.ndarray,
) -> np.ndarray:
    """Aggregate valid motion features across people in each frame."""

    valid = motion_mask.astype(bool)

    mean_features = np.zeros(
        (motion_features.shape[0], MOTION_FEATURES),
        dtype=np.float32,
    )

    max_features = np.zeros_like(mean_features)

    for frame_index in range(motion_features.shape[0]):
        active_motion = motion_features[frame_index, valid[frame_index]]

        if active_motion.shape[0] == 0:
            continue

        mean_features[frame_index] = active_motion.mean(axis=0)
        max_features[frame_index] = active_motion.max(axis=0)

    return np.concatenate(
        [mean_features, max_features],
        axis=1,
    )


def aggregate_frame_features(
    features: np.ndarray,
    person_mask: np.ndarray,
    motion_features: np.ndarray,
    motion_mask: np.ndarray,
) -> np.ndarray:
    """Create one 246-dimensional feature vector per frame."""

    validate_feature_arrays(
        features,
        person_mask,
        motion_features,
        motion_mask,
    )

    pose_aggregated = aggregate_pose_features(
        features,
        person_mask,
    )

    motion_aggregated = aggregate_motion_features(
        motion_features,
        motion_mask,
    )

    frame_features = np.concatenate(
        [pose_aggregated, motion_aggregated],
        axis=1,
    )

    return frame_features.astype(np.float32)



def aggregate_for_mlp(
    frame_features: np.ndarray,
) -> np.ndarray:
    """Convert frame sequences into fixed-size MLP inputs."""

    if frame_features.ndim != 2:
        raise ValueError("frame_features must have shape (T, 246)")

    if frame_features.shape[1] != AGGREGATED_FEATURES:
        raise ValueError(
            f"Expected {AGGREGATED_FEATURES} features per frame"
        )

    if frame_features.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty sequence")

    if not np.isfinite(frame_features).all():
        raise ValueError("frame_features contain NaN or infinity")

    mean_features = frame_features.mean(axis=0)
    std_features = frame_features.std(axis=0)

    mlp_features = np.concatenate(
        [mean_features, std_features],
        axis=0,
    )

    return mlp_features.astype(np.float32)



def aggregate_for_lstm(
    frame_features: np.ndarray,
) -> np.ndarray:
    """Prepare a temporal feature sequence for LSTM input."""

    if frame_features.ndim != 2:
        raise ValueError("frame_features must have shape (T, 246)")

    if frame_features.shape[1] != AGGREGATED_FEATURES:
        raise ValueError(
            f"Expected {AGGREGATED_FEATURES} features per frame"
        )

    if frame_features.shape[0] == 0:
        raise ValueError("Cannot use an empty sequence")

    if not np.isfinite(frame_features).all():
        raise ValueError("frame_features contain NaN or infinity")

    return frame_features.astype(np.float32, copy=True)



