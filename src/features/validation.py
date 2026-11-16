
"""Validate pose and motion feature extraction across videos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features.extraction import extract_video_features
from src.features.dataset import FEATURE_ROOT, LABEL_TO_ID

MANIFEST_PATH = Path("data/metadata/local/manifest.jsonl")
POSE_ROOT = Path("outputs/pose")



def load_manifest() -> dict[str, dict]:
    """Load train and validation samples from the manifest."""

    samples = {}

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["split"] not in {"train", "val"}:
                continue

            sample_id = record["sample_id"]

            if sample_id in samples:
                raise ValueError(
                    f"Duplicate sample_id in manifest: {sample_id}"
                )

            samples[sample_id] = record

    return samples



def find_latest_pose_outputs() -> dict[str, Path]:
    """Find the latest pose summary for each sample."""

    latest = {}

    for summary_path in POSE_ROOT.glob("*/*/summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        sample_id = summary.get("sample_id")

        if not sample_id:
            continue

        if sample_id not in latest:
            latest[sample_id] = summary_path
            continue

        current_time = summary_path.stat().st_mtime
        previous_time = latest[sample_id].stat().st_mtime

        if current_time > previous_time:
            latest[sample_id] = summary_path

    return latest



def validate_one_video(
    sample_id: str,
    summary_path: Path,
    expected_label: str,
) -> dict:
    """Validate extracted pose and motion features for one video."""

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("sample_id") != sample_id:
        raise ValueError(f"Sample ID mismatch: {sample_id}")

    if summary.get("source_label") != expected_label:
        raise ValueError(f"Label mismatch: {sample_id}")

    poses_path = summary_path.parent / summary["outputs"]["poses"]

    result = extract_video_features(poses_path)

    features = result["features"]
    person_mask = result["person_mask"]
    motion_features = result["motion_features"]
    motion_mask = result["motion_mask"]

    expected_frames = summary["frames"]

    if result["frame_count"] != expected_frames:
        raise ValueError(f"Frame count mismatch: {sample_id}")

    if features.shape != (expected_frames, 8, 89):
        raise ValueError(f"Pose feature shape error: {sample_id}")

    if motion_features.shape != (expected_frames, 8, 34):
        raise ValueError(f"Motion feature shape error: {sample_id}")

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite pose features: {sample_id}")

    if not np.isfinite(motion_features).all():
        raise ValueError(f"Non-finite motion features: {sample_id}")

    if not np.isin(person_mask, [0.0, 1.0]).all():
        raise ValueError(f"Invalid person mask: {sample_id}")

    if not np.isin(motion_mask, [0.0, 1.0]).all():
        raise ValueError(f"Invalid motion mask: {sample_id}")

    if motion_mask[0].sum() != 0:
        raise ValueError(f"First frame has invalid motion: {sample_id}")

    if np.any(motion_mask > person_mask):
        raise ValueError(f"Motion exists without a person: {sample_id}")

    return {
        "sample_id": sample_id,
        "label": expected_label,
        "split": result["selection"].get("split"),
        "frames": expected_frames,
        "tracks": result["track_count"],
        "person_slots": int(person_mask.sum()),
        "motion_slots": int(motion_mask.sum()),
    }



def run_validation() -> None:
    """Validate all available train and validation pose outputs."""

    samples = load_manifest()
    pose_outputs = find_latest_pose_outputs()

    print(f"Manifest samples: {len(samples)}")
    print(f"Pose outputs found: {len(pose_outputs)}")

    results = []
    errors = []
    missing = []

    for sample_id, record in samples.items():
        summary_path = pose_outputs.get(sample_id)

        if summary_path is None:
            missing.append(sample_id)
            continue

        try:
            result = validate_one_video(
                sample_id=sample_id,
                summary_path=summary_path,
                expected_label=record["label"],
            )

            result["split"] = record["split"]
            results.append(result)

        except Exception as error:
            errors.append({
                "sample_id": sample_id,
                "error": str(error),
            })

    print("\n===== VALIDATION SUMMARY =====")
    print(f"Expected videos: {len(samples)}")
    print(f"Validated videos: {len(results)}")
    print(f"Missing pose outputs: {len(missing)}")
    print(f"Errors: {len(errors)}")

    print("\n===== SPLIT COUNTS =====")

    for split in ("train", "val"):
        split_results = [
            item for item in results
            if item["split"] == split
        ]

        print(f"{split}: {len(split_results)}")

    print("\n===== LABEL COUNTS =====")

    for label in ("normal", "violence"):
        count = sum(
            item["label"] == label
            for item in results
        )

        print(f"{label}: {count}")

    print("\n===== FEATURE TOTALS =====")
    print(
        "Person slots:",
        sum(item["person_slots"] for item in results),
    )
    print(
        "Motion slots:",
        sum(item["motion_slots"] for item in results),
    )

    if missing:
        print("\nMissing sample IDs:", missing)

    if errors:
        print("\nValidation errors:")
        for item in errors:
            print(item)


if __name__ == "__main__":
    run_validation()


def validate_feature_split(split):
    if split not in {"train", "val"}:
        raise ValueError(f"Invalid split: {split}")

    split_dir = FEATURE_ROOT / split
    index_path = split_dir / "index.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(index_path)

    with index_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    errors = []
    seen_ids = set()
    label_counts = {label: 0 for label in LABEL_TO_ID}

    for record in records:
        sample_id = record["sample_id"]

        if sample_id in seen_ids:
            errors.append(f"Duplicate sample ID: {sample_id}")
            continue

        seen_ids.add(sample_id)

        feature_path = split_dir / f"{sample_id}.npz"

        if not feature_path.exists():
            errors.append(f"Missing file: {feature_path}")
            continue

        try:
            with np.load(feature_path, allow_pickle=False) as data:
                features = data["features"]
                person_mask = data["person_mask"]
                motion = data["motion_features"]
                motion_mask = data["motion_mask"]
                label = int(data["label"])
                saved_id = str(data["sample_id"].item())
                saved_split = str(data["split"].item())

                frames = features.shape[0]

                if features.shape != (frames, 8, 89):
                    errors.append(f"{sample_id}: invalid feature shape")

                if person_mask.shape != (frames, 8):
                    errors.append(f"{sample_id}: invalid person mask shape")

                if motion.shape != (frames, 8, 34):
                    errors.append(f"{sample_id}: invalid motion shape")

                if motion_mask.shape != (frames, 8):
                    errors.append(f"{sample_id}: invalid motion mask shape")

                for name, array in [
                    ("features", features),
                    ("motion", motion),
                    ("person_mask", person_mask),
                    ("motion_mask", motion_mask),
                ]:
                    if not np.isfinite(array).all():
                        errors.append(f"{sample_id}: non-finite values in {name}")

                if not np.isin(person_mask, [0, 1]).all():
                    errors.append(f"{sample_id}: invalid person mask values")

                if not np.isin(motion_mask, [0, 1]).all():
                    errors.append(f"{sample_id}: invalid motion mask values")

                if np.any(motion_mask > person_mask):
                    errors.append(f"{sample_id}: motion mask exceeds person mask")

                if frames > 0 and np.any(motion_mask[0] != 0):
                    errors.append(f"{sample_id}: first frame has motion mask")

                if saved_id != sample_id:
                    errors.append(f"{sample_id}: sample ID mismatch")

                if saved_split != split:
                    errors.append(f"{sample_id}: split mismatch")

                expected_label = LABEL_TO_ID[record["label"]]

                if label != expected_label:
                    errors.append(f"{sample_id}: label mismatch")

                label_counts[record["label"]] += 1

        except Exception as error:
            errors.append(f"{sample_id}: {error}")

    actual_files = list(split_dir.glob("*.npz"))

    if len(actual_files) != len(records):
        errors.append(
            f"File count mismatch: {len(actual_files)} files, "
            f"{len(records)} index records"
        )

    print(f"\n===== {split.upper()} FEATURE VALIDATION =====")
    print("Index records:", len(records))
    print("NPZ files:", len(actual_files))
    print("Label counts:", label_counts)
    print("Errors:", len(errors))

    if errors:
        for error in errors:
            print(error)
    else:
        print("All feature files passed validation.")


