
"""Extract fixed-layout pose features from a single video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.selection import select_tracks


KEYPOINT_COUNT = 17
KEYPOINT_FEATURES = 89
MAX_TRACKS_DEFAULT = 8

# 17 keypoints:
# 34 frame-normalized coordinates
# 34 bounding-box-normalized coordinates
# 17 keypoint confidence values
# 4 bounding-box coordinates


def _safe_float(value: Any) -> float:
    """Convert a value to a finite float; use zero if invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(number):
        return 0.0

    return number


def _person_to_feature(person: dict[str, Any]) -> np.ndarray:
    """Convert one person's pose in one frame into 89 features."""

    keypoints = person.get("keypoints", [])

    if not isinstance(keypoints, list):
        keypoints = []

    frame_xy = []
    bbox_xy = []
    confidences = []

    for index in range(KEYPOINT_COUNT):
        if index < len(keypoints):
            keypoint = keypoints[index]
        else:
            keypoint = {}

        if not isinstance(keypoint, dict):
            keypoint = {}

        # Coordinates normalized relative to the whole frame.
        frame_xy.extend([
            _safe_float(keypoint.get("x_frame_norm")),
            _safe_float(keypoint.get("y_frame_norm")),
        ])

        # Coordinates normalized relative to the person's bounding box.
        bbox_xy.extend([
            _safe_float(keypoint.get("x_bbox_norm")),
            _safe_float(keypoint.get("y_bbox_norm")),
        ])

        confidences.append(
            _safe_float(keypoint.get("confidence"))
        )

    # Bounding box normalized relative to the whole frame.
    bbox = person.get("bbox_xyxy_frame_norm", [])

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        bbox = [0.0, 0.0, 0.0, 0.0]

    bbox_features = [_safe_float(value) for value in bbox]

    features = (
        frame_xy
        + bbox_xy
        + confidences
        + bbox_features
    )

    result = np.asarray(features, dtype=np.float32)

    if result.shape != (KEYPOINT_FEATURES,):
        raise ValueError(
            f"Expected {KEYPOINT_FEATURES} features, "
            f"got {result.shape}"
        )

    return result


def _load_pose_frames(poses_path: str | Path) -> list[dict[str, Any]]:
    """Load pose frames in their original temporal order."""

    poses_path = Path(poses_path)

    if not poses_path.is_file():
        raise FileNotFoundError(
            f"Pose file not found: {poses_path}"
        )

    frames = []

    with poses_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            frame_index = frame.get("frame_index")

            if not isinstance(frame_index, int):
                raise ValueError(
                    f"Invalid frame_index at line {line_number}"
                )

            people = frame.get("people", [])

            if not isinstance(people, list):
                raise ValueError(
                    f"Invalid people list at line {line_number}"
                )

            frames.append(frame)

    if not frames:
        raise ValueError(
            f"No valid frames found: {poses_path}"
        )

    indices = [frame["frame_index"] for frame in frames]

    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(
            "Frame indices must be strictly increasing."
        )

    return frames



def extract_video_features(
    poses_path: str | Path,
    max_tracks: int = MAX_TRACKS_DEFAULT,
) -> dict[str, Any]:
    """Extract temporal pose and motion features for one video."""

    if max_tracks < 1:
        raise ValueError("max_tracks must be at least 1")

    frames = _load_pose_frames(poses_path)

    selection = select_tracks(
        poses_path,
        max_tracks=max_tracks,
    )

    selected_track_ids = selection["selected_track_ids"]
    frame_count = len(frames)

    # Fixed layout: time x people x pose features.
    features = np.zeros(
        (frame_count, max_tracks, KEYPOINT_FEATURES),
        dtype=np.float32,
    )

    person_mask = np.zeros(
        (frame_count, max_tracks),
        dtype=np.float32,
    )

    # Motion: 34 keypoint coordinate changes per person.
    motion_features = np.zeros(
        (frame_count, max_tracks, 34),
        dtype=np.float32,
    )

    motion_mask = np.zeros(
        (frame_count, max_tracks),
        dtype=np.float32,
    )

    # Track IDs are used only for matching detections to stable slots.
    track_to_slot = {
        track_id: slot
        for slot, track_id in enumerate(selected_track_ids)
    }

    for frame_position, frame in enumerate(frames):
        for person in frame["people"]:
            track_id = person.get("track_id")

            if track_id not in track_to_slot:
                continue

            slot = track_to_slot[track_id]

            features[frame_position, slot] = (
                _person_to_feature(person)
            )

            person_mask[frame_position, slot] = 1.0

    # Calculate motion only across consecutive frames
    # where the same tracked person is present.
    for frame_position in range(1, frame_count):
        current_frame = frames[frame_position]
        previous_frame = frames[frame_position - 1]

        # Do not calculate motion across missing frame indices.
        if (
            current_frame["frame_index"]
            != previous_frame["frame_index"] + 1
        ):
            continue

        delta_time = (
            current_frame["timestamp_s"]
            - previous_frame["timestamp_s"]
        )

        if delta_time <= 0:
            continue

        for slot in range(len(selected_track_ids)):
            if (
                person_mask[frame_position, slot] != 1.0
                or person_mask[frame_position - 1, slot] != 1.0
            ):
                continue

            motion_features[frame_position, slot] = (
                _calculate_keypoint_motion(
                    features[frame_position, slot],
                    features[frame_position - 1, slot],
                    delta_time,
                )
            )

            motion_mask[frame_position, slot] = 1.0

    return {
        "features": features,
        "person_mask": person_mask,
        "motion_features": motion_features,
        "motion_mask": motion_mask,
        "selected_track_ids": selected_track_ids,
        "frame_count": frame_count,
        "track_count": len(selected_track_ids),
        "selection": selection,
    }



def _calculate_keypoint_motion(
    current_features: np.ndarray,
    previous_features: np.ndarray,
    delta_time: float,
) -> np.ndarray:
    """Calculate frame-normalized keypoint displacement per second."""

    if delta_time <= 0:
        raise ValueError("delta_time must be positive")

    # First 34 values are frame-normalized keypoint coordinates.
    current_xy = current_features[:34].reshape(KEYPOINT_COUNT, 2)
    previous_xy = previous_features[:34].reshape(KEYPOINT_COUNT, 2)

    motion = (current_xy - previous_xy) / delta_time

    return motion.astype(np.float32).reshape(34)
