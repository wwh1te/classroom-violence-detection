
"""Build and save pose-motion feature datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.extraction import extract_video_features


MANIFEST_PATH = Path("data/metadata/local/manifest.jsonl")
POSE_ROOT = Path("outputs/pose")
FEATURE_ROOT = Path("outputs/features")

VALID_SPLITS = {"train", "val", "test"}
LABEL_TO_ID = {
    "normal": 0,
    "violence": 1,
}



def load_manifest(split: str) -> list[dict[str, Any]]:
    """Load samples belonging to one dataset split."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split}")

    samples = []

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["split"] != split:
                continue

            label = record["label"]

            if label not in LABEL_TO_ID:
                raise ValueError(
                    f"Unknown label {label}: {record['sample_id']}"
                )

            samples.append(record)

    return samples



def find_pose_outputs() -> dict[str, Path]:
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



def save_video_features(
    record: dict[str, Any],
    summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract and save features for one video."""

    sample_id = record["sample_id"]
    label = record["label"]
    split = record["split"]

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("sample_id") != sample_id:
        raise ValueError(f"Sample ID mismatch: {sample_id}")

    if summary.get("source_label") != label:
        raise ValueError(f"Label mismatch: {sample_id}")

    poses_path = summary_path.parent / summary["outputs"]["poses"]

    result = extract_video_features(poses_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sample_id}.npz"

    np.savez_compressed(
        output_path,
        features=result["features"],
        person_mask=result["person_mask"],
        motion_features=result["motion_features"],
        motion_mask=result["motion_mask"],
        label=np.asarray(LABEL_TO_ID[label], dtype=np.int64),
        sample_id=np.asarray(sample_id),
        split=np.asarray(split),
    )

    return {
        "sample_id": sample_id,
        "label": label,
        "split": split,
        "frames": result["frame_count"],
        "tracks": result["track_count"],
        "output_path": str(output_path),
    }



def build_feature_split(split: str) -> list[dict[str, Any]]:
    """Build and save features for one dataset split."""

    samples = load_manifest(split)
    pose_outputs = find_pose_outputs()

    output_dir = FEATURE_ROOT / split

    if output_dir.exists() and any(output_dir.glob("*.npz")):
        raise FileExistsError(
            f"Feature files already exist in {output_dir}. "
            "Use a new output directory or inspect existing files first."
        )

    missing = [
        record["sample_id"]
        for record in samples
        if record["sample_id"] not in pose_outputs
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing pose outputs for {len(missing)} samples: {missing[:10]}"
        )

    results = []

    for index, record in enumerate(samples, start=1):
        sample_id = record["sample_id"]

        result = save_video_features(
            record=record,
            summary_path=pose_outputs[sample_id],
            output_dir=output_dir,
        )

        results.append(result)

        print(
            f"[{index}/{len(samples)}] "
            f"{sample_id} saved"
        )

    index_path = output_dir / "index.jsonl"

    with index_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print(f"\nSplit: {split}")
    print(f"Samples saved: {len(results)}")
    print(f"Output directory: {output_dir}")
    print(f"Index: {index_path}")

    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build pose-motion feature dataset."
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        required=True,
        help="Dataset split to process.",
    )

    args = parser.parse_args()

    build_feature_split(args.split)
