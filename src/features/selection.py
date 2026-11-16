
"""Select the most useful person tracks from a pose-extracted video."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def select_tracks(
    poses_path: str | Path,
    max_tracks: int = 8,
) -> dict[str, Any]:
    """Select up to max_tracks using duration, visibility and pose confidence.

    Track IDs are only used to associate detections within this video.
    They are never treated as numeric model features.
    """
    poses_path = Path(poses_path)

    if not poses_path.is_file():
        raise FileNotFoundError(f"Pose file not found: {poses_path}")

    if max_tracks < 1:
        raise ValueError("max_tracks must be at least 1")

    tracks: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "frames": set(),
            "detection_count": 0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
            "box_confidence_sum": 0.0,
            "box_confidence_count": 0,
        }
    )

    total_frames = 0
    total_detections = 0

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
                    f"Missing or invalid frame_index at line {line_number}"
                )

            total_frames += 1

            people = frame.get("people", [])
            if not isinstance(people, list):
                raise ValueError(
                    f"'people' must be a list at line {line_number}"
                )

            for person in people:
                total_detections += 1
                track_id = person.get("track_id")
                if track_id is None:
                    continue

                stats = tracks[track_id]
                stats["frames"].add(frame_index)
                stats["detection_count"] += 1

                keypoints = person.get("keypoints", [])
                for keypoint in keypoints:
                    confidence = keypoint.get("confidence")
                    if isinstance(confidence, (int, float)):
                        if 0.0 <= confidence <= 1.0:
                            stats["confidence_sum"] += float(confidence)
                            stats["confidence_count"] += 1

                box_confidence = person.get("box_confidence")
                if isinstance(box_confidence, (int, float)):
                    if 0.0 <= box_confidence <= 1.0:
                        stats["box_confidence_sum"] += float(box_confidence)
                        stats["box_confidence_count"] += 1

    if total_frames == 0:
        raise ValueError(f"No frames found in pose file: {poses_path}")

    candidates = []

    for track_id, stats in tracks.items():
        duration = len(stats["frames"])

        if stats["confidence_count"]:
            pose_confidence = (
                stats["confidence_sum"] / stats["confidence_count"]
            )
        else:
            pose_confidence = 0.0

        if stats["box_confidence_count"]:
            box_confidence = (
                stats["box_confidence_sum"]
                / stats["box_confidence_count"]
            )
        else:
            box_confidence = 0.0

        visibility = duration / total_frames

        # Equal weighting keeps one criterion from dominating the score.
        score = (
            0.40 * visibility
            + 0.40 * pose_confidence
            + 0.20 * box_confidence
        )

        candidates.append(
            {
                "track_id": track_id,
                "duration_frames": duration,
                "visibility": visibility,
                "pose_confidence": pose_confidence,
                "box_confidence": box_confidence,
                "selection_score": score,
            }
        )

    # Deterministic tie-breaking makes repeated runs reproducible.
    candidates.sort(
        key=lambda item: (
            -item["selection_score"],
            -item["duration_frames"],
            item["track_id"],
        )
    )

    selected = candidates[:max_tracks]
    selected_track_ids = {
    item["track_id"] for item in selected
    }

    selected_detections = sum(
        tracks[track_id]["detection_count"]
        for track_id in selected_track_ids
    )

    excluded_detections = total_detections - selected_detections

    detection_coverage = (
        selected_detections / total_detections
        if total_detections > 0
        else 0.0
    )

    return {
        "poses_path": str(poses_path),
        "total_frames": total_frames,
        "total_detections": total_detections,
        "selected_detections": selected_detections,
        "excluded_detections": excluded_detections,
        "detection_coverage": detection_coverage,
        "total_tracks": len(candidates),
        "max_tracks": max_tracks,
        "selected_track_ids": [
            item["track_id"] for item in selected
        ],
        "selected_tracks": selected,
        "all_tracks": candidates,
    }
