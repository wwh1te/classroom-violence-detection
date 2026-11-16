
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from src.evaluation import find_checkpoint, load_model
from src.features.extraction import extract_video_features
from src.features.aggregation import (
    aggregate_frame_features,
    aggregate_for_mlp,
    aggregate_for_lstm,
)
from src.features.normalization import (
    normalize_mlp_batch,
    normalize_lstm_batch,
)
from src.pose.pipeline import process_video
from src.data.pipeline import _normalize_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a video as Normal or Violence and raise an alert."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "lstm"],
        default="lstm",
        help="Violence classifier to use.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a specific classifier checkpoint.",
    )
    parser.add_argument(
        "--pose-config",
        default="configs/pose.json",
        help="Path to YOLO pose configuration.",
    )
    parser.add_argument(
        "--pose-model",
        default=None,
        help="Optional YOLO pose model override.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory for inference outputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Violence alert probability threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, for example cpu or 0.",
    )

    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()

def create_temporal_windows(
    sequence,
    window_size=64,
    stride=32,
):
    """Create overlapping temporal windows and include the final frames."""

    if sequence.ndim != 2:
        raise ValueError("sequence must have shape (T, features)")

    total_frames = sequence.shape[0]

    if total_frames == 0:
        raise ValueError("Cannot create windows from an empty sequence")

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if total_frames <= window_size:
        return [sequence]

    windows = []

    starts = list(range(0, total_frames - window_size + 1, stride))

    final_start = total_frames - window_size

    if starts[-1] != final_start:
        starts.append(final_start)

    for start in starts:
        end = start + window_size
        windows.append(sequence[start:end])

    return windows

def predict_lstm_window(
    classifier,
    sequence,
    normalization,
    device,
):
    """Predict Normal and Violence probabilities for one window."""

    x = torch.from_numpy(sequence).float().unsqueeze(0)

    lengths = torch.tensor(
        [sequence.shape[0]],
        dtype=torch.long,
    )

    x = normalize_lstm_batch(x, lengths, normalization)
    x = x.to(device)

    with torch.inference_mode():
        logits = classifier(x, lengths)
        probabilities = torch.softmax(logits, dim=1)[0]

    return {
        "normal_probability": float(probabilities[0].item()),
        "violence_probability": float(probabilities[1].item()),
    }

def evaluate_window_alerts(
    window_predictions,
    threshold=0.5,
    consecutive_windows=2,
):
    """Confirm violence only after consecutive positive windows."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if consecutive_windows <= 0:
        raise ValueError("consecutive_windows must be positive")

    results = []
    consecutive_count = 0
    violence_alert = False

    for index, prediction in enumerate(window_predictions):
        probability = prediction["violence_probability"]
        detected = probability >= threshold

        if detected:
            consecutive_count += 1
        else:
            consecutive_count = 0

        confirmed = consecutive_count >= consecutive_windows

        if confirmed:
            violence_alert = True

        results.append({
            "window_index": index + 1,
            "violence_probability": probability,
            "detected": detected,
            "consecutive_count": consecutive_count,
            "confirmed": confirmed,
        })

    return {
        "violence_alert": violence_alert,
        "window_results": results,
        "threshold": threshold,
        "consecutive_windows": consecutive_windows,
    }



def build_suspicious_segments(
    window_predictions,
    fps,
    threshold=0.5,
):
    """Merge overlapping positive windows into suspicious time segments."""

    if fps <= 0:
        raise ValueError("fps must be positive")

    positive_windows = sorted(
        [
            p for p in window_predictions
            if p["violence_probability"] >= threshold
        ],
        key=lambda p: p["start_frame"],
    )

    segments = []

    for prediction in positive_windows:
        start_frame = prediction["start_frame"]
        end_frame = prediction["end_frame"]
        probability = prediction["violence_probability"]

        if (
            not segments
            or start_frame > segments[-1]["end_frame"] + 1
        ):
            segments.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "max_violence_probability": probability,
            })
        else:
            segments[-1]["end_frame"] = max(
                segments[-1]["end_frame"],
                end_frame,
            )

            segments[-1]["max_violence_probability"] = max(
                segments[-1]["max_violence_probability"],
                probability,
            )

    for segment in segments:
        start_seconds = segment["start_frame"] / fps
        end_seconds = (segment["end_frame"] + 1) / fps

        segment["start_seconds"] = round(start_seconds, 3)
        segment["end_seconds"] = round(end_seconds, 3)

    return segments


def main():
    args = parse_args()

    source_path = Path(args.source).expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Input video does not exist: {source_path}"
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    config_path = resolve_path(args.pose_config)

    with config_path.open("r", encoding="utf-8") as file:
        pose_config = json.load(file)

    pose_model_path = args.pose_model or pose_config["model"]

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint_path = find_checkpoint(args.model)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint does not exist: {checkpoint_path}"
        )

    if args.device == "cpu" or not torch.cuda.is_available():
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda")

    print("=" * 60)
    print("CLASSROOM VIOLENCE DETECTION")
    print("=" * 60)
    print(f"Video: {source_path}")
    print(f"Classifier: {args.model.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {torch_device}")
    print(f"Alert threshold: {args.threshold:.2f}")

    # Load the trained violence classifier.
    print("\n[1/4] Loading classifier...")

    classifier, normalization, checkpoint = load_model(
        args.model,
        checkpoint_path,
        torch_device,
    )

    label_to_id = checkpoint["label_to_id"]
    id_to_label = {
        int(class_id): label
        for label, class_id in label_to_id.items()
    }

    # Load YOLO pose model.
    print("\n[2/4] Loading YOLO pose model...")

    pose_model = YOLO(pose_model_path)

    # Create a unique output directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_root = resolve_path(args.output_dir)
    run_dir = output_root / f"{source_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Normalize input video to match the training preprocessing.
    print("\n[3/5] Normalizing input video...")

    normalized_path = run_dir / "normalized_input.mp4"

    normalized_info = _normalize_video(
        source_path,
        normalized_path,
        {
            "width": 640,
            "height": 360,
            "fps": 15,
        },
    )

    print(
        f"Normalized video: {normalized_info['video']['frames']} frames, "
        f"{normalized_info['video']['fps']} FPS, "
        f"{normalized_info['video']['width']}x"
        f"{normalized_info['video']['height']}"
    )

    # Extract poses and tracking information.
    # Fall baseline is explicitly disabled.
    print("\n[4/5] Processing video and extracting poses...")

    pose_summary = process_video(
        source=normalized_path,
        run_dir=run_dir,
        model=pose_model,
        model_path=str(pose_model_path),
        confidence=pose_config["confidence"],
        image_size=pose_config["image_size"],
        tracker=pose_config["tracker"],
        keypoint_confidence=pose_config["keypoint_confidence"],
        minimum_box_aspect_ratio=pose_config["fall_box_aspect_ratio"],
        minimum_torso_tilt_degrees=pose_config["fall_torso_tilt_degrees"],
        confirm_frames=pose_config["fall_confirm_frames"],
        fall_baseline_enabled=False,
        device=args.device,
        sample_id=None,
        label=None,
    )

    poses_path = run_dir / "poses.jsonl"

    if not poses_path.is_file():
        raise RuntimeError(
            f"Pose extraction did not produce poses.jsonl. "
            f"Check the output directory: {run_dir}"
        )

    # Convert pose data into the exact feature format used during training.
    extracted = extract_video_features(poses_path)

    frame_features = aggregate_frame_features(
        extracted["features"],
        extracted["person_mask"],
        extracted["motion_features"],
        extracted["motion_mask"],
    )

    # Run the selected classifier.
    
    # Run the selected classifier.
    print("\n[5/5] Running violence classification...")

    window_predictions = []

    if args.model == "mlp":
        features = aggregate_for_mlp(frame_features)

        x = torch.from_numpy(features).float().unsqueeze(0)
        x = normalize_mlp_batch(x, normalization)
        x = x.to(torch_device)

        with torch.inference_mode():
            logits = classifier(x)
            probabilities = torch.softmax(logits, dim=1)[0]

        normal_probability = float(probabilities[0].item())
        violence_probability = float(probabilities[1].item())

        predicted_id = int(torch.argmax(probabilities).item())
        predicted_label = id_to_label[predicted_id]

        violence_alert = violence_probability >= args.threshold

        alert_result = {
            "violence_alert": violence_alert,
            "window_results": [],
            "threshold": args.threshold,
            "consecutive_windows": 1,
        }

    else:
        sequence = aggregate_for_lstm(frame_features)

        windows = create_temporal_windows(
            sequence,
            window_size=64,
            stride=32,
        )

        print(f"Total frames: {len(sequence)}")
        print(f"Temporal windows: {len(windows)}")

        window_starts = list(
            range(0, len(sequence) - 64 + 1, 32)
        )

        final_start = len(sequence) - 64

        if len(sequence) > 64 and window_starts[-1] != final_start:
            window_starts.append(final_start)

        if len(sequence) <= 64:
            window_starts = [0]

        for index, window in enumerate(windows, start=1):
            prediction = predict_lstm_window(
                classifier,
                window,
                normalization,
                torch_device,
            )

            prediction["window_index"] = index
            prediction["start_frame"] = window_starts[index - 1]
            prediction["end_frame"] = (
                prediction["start_frame"] + len(window) - 1
            )

            window_predictions.append(prediction)

            print(
                f"Window {index}: "
                f"Violence={prediction['violence_probability']:.2%}"
            )

        required_consecutive = min(2, len(window_predictions))

        alert_result = evaluate_window_alerts(
            window_predictions,
            threshold=args.threshold,
            consecutive_windows=required_consecutive,
        )

        violence_alert = alert_result["violence_alert"]

        # Use average probabilities for the overall video prediction.
        normal_probability = sum(
            p["normal_probability"] for p in window_predictions
        ) / len(window_predictions)

        violence_probability = sum(
            p["violence_probability"] for p in window_predictions
        ) / len(window_predictions)

        predicted_label = (
            "violence"
            if violence_probability >= args.threshold
            else "normal"
        )

        predicted_id = label_to_id[predicted_label]

    alert_result["window_predictions"] = window_predictions
    suspicious_segments = build_suspicious_segments(
        window_predictions,
        fps=pose_summary["fps"],
        threshold=args.threshold,
    )


    # Save inference result.
    result = {
        "source": str(source_path),
        "classifier": args.model,
        "checkpoint": str(checkpoint_path),
        "window_alert": alert_result,
        "predicted_label": predicted_label,
        "normal_probability": normal_probability,
        "violence_probability": violence_probability,
        "alert_threshold": args.threshold,
        "violence_alert": violence_alert,
        "suspicious_segments": suspicious_segments,
        "pose_summary": pose_summary,
    }

    result_path = run_dir / "inference_result.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False, default=str)

    # Display results.
    print("\n" + "=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)

    print(f"Prediction: {predicted_label.upper()}")
    print(f"Normal probability:   {normal_probability:.2%}")
    print(f"Violence probability: {violence_probability:.2%}")
    print(f"Alert threshold:      {args.threshold:.2%}")

    if violence_alert:
        print("\n" + "!" * 60)
        print("!!! VIOLENCE ALERT !!!")
        print("Potential violence detected in this video.")
        print("!" * 60)
    else:
        print("\nNo violence alert.")

    print(f"\nResults saved to: {run_dir}")
    print(f"Result JSON: {result_path}")
    print(f"Pose summary: {run_dir / 'summary.json'}")
    print(f"Annotated video: {run_dir / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
