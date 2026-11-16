
"""Evaluate trained MLP and LSTM classifiers on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.features.loader import create_data_loader
from src.features.normalization import (
    normalize_lstm_batch,
    normalize_mlp_batch,
)
from src.models import MLPClassifier, LSTMClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
FEATURE_ROOT = PROJECT_ROOT / "outputs" / "features"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "evaluation"

LABEL_NAMES = {
    0: "normal",
    1: "violence",
}


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict:
    """Calculate classification metrics without external dependencies."""

    confusion = np.zeros((2, 2), dtype=np.int64)

    for actual, predicted in zip(labels, predictions):
        confusion[actual, predicted] += 1

    per_class = {}

    for class_id, class_name in LABEL_NAMES.items():
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[class_id, :].sum()),
        }

    accuracy = (
        float(np.trace(confusion) / confusion.sum())
        if confusion.sum() else 0.0
    )

    macro_f1 = float(
        np.mean([values["f1"] for values in per_class.values()])
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def find_checkpoint(model_type: str) -> Path:
    """Find the latest checkpoint directory for a model."""

    model_root = CHECKPOINT_ROOT / model_type

    candidates = sorted(
        model_root.glob("*/best.pt"),
        key=lambda path: path.parent.name,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No best.pt checkpoint found in {model_root}"
        )

    return candidates[0]


def load_model(
    model_type: str,
    checkpoint_path: Path,
    device: torch.device,
):
    """Load a model and its saved normalization statistics."""

    # weights_only=False is required because these trusted checkpoints
    # contain NumPy normalization arrays. Only load checkpoints created
    # by this project's own training pipeline.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["model_type"] != model_type:
        raise ValueError("Checkpoint model type does not match.")

    if checkpoint["label_to_id"] != {"normal": 0, "violence": 1}:
        raise ValueError("Checkpoint label mapping does not match.")

    if model_type == "mlp":
        model = MLPClassifier(input_size=492, num_classes=2)
    else:
        model = LSTMClassifier(
            input_size=246,
            hidden_size=64,
            num_layers=2,
            num_classes=2,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    stats = {
        "mean": torch.as_tensor(
            checkpoint["normalization"]["mean"],
            dtype=torch.float32,
        ),
        "std": torch.as_tensor(
            checkpoint["normalization"]["std"],
            dtype=torch.float32,
        ),
    }

    return model, stats, checkpoint


def evaluate_model(
    model_type: str,
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate one model on validation data."""

    model, stats, checkpoint = load_model(
        model_type,
        checkpoint_path,
        device,
    )

    loader = create_data_loader(
        split="val",
        model_type=model_type,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch in loader:
            if model_type == "mlp":
                features, labels = batch
                features = normalize_mlp_batch(
                    features,
                    stats,
                ).to(device)

                logits = model(features)

            else:
                features, lengths, labels = batch
                features = normalize_lstm_batch(
                    features,
                    lengths,
                    stats,
                ).to(device)

                logits = model(features, lengths)

            loss = criterion(logits, labels.to(device))
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            batch_size_actual = labels.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = total_loss / total_samples
    metrics["model_type"] = model_type
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics["best_val_macro_f1_during_training"] = (
        checkpoint["best_val_macro_f1"]
    )
    metrics["sample_count"] = total_samples

    return {
        "metrics": metrics,
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


def save_predictions(
    model_type: str,
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    output_path: Path,
) -> None:
    """Save per-video predictions in validation index order."""

    index_path = FEATURE_ROOT / "val" / "index.jsonl"

    with index_path.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(records) != len(labels):
        raise ValueError(
            "Validation index size does not match prediction count."
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "actual",
                "predicted",
                "correct",
                "normal_probability",
                "violence_probability",
            ],
        )
        writer.writeheader()

        for record, actual, predicted, probability in zip(
            records,
            labels,
            predictions,
            probabilities,
        ):
            writer.writerow({
                "sample_id": record["sample_id"],
                "actual": LABEL_NAMES[actual],
                "predicted": LABEL_NAMES[predicted],
                "correct": actual == predicted,
                "normal_probability": probability[0],
                "violence_probability": probability[1],
            })


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLP or LSTM on validation data."
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "lstm", "both"],
        default="both",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Split: validation")
    print("Test split will not be accessed.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    model_types = (
        ["mlp", "lstm"]
        if args.model == "both"
        else [args.model]
    )

    summary = {}

    for model_type in model_types:
        checkpoint_path = find_checkpoint(model_type)

        print(f"\nEvaluating {model_type.upper()}...")
        print(f"Checkpoint: {checkpoint_path}")

        result = evaluate_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            device=device,
        )

        metrics = result["metrics"]
        summary[model_type] = metrics

        prediction_path = (
            output_dir / f"{model_type}_predictions.csv"
        )

        save_predictions(
            model_type=model_type,
            labels=result["labels"],
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            output_path=prediction_path,
        )

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Loss: {metrics['loss']:.4f}")

        print("\nPer-class metrics:")
        for name, values in metrics["per_class"].items():
            print(
                f"  {name}: "
                f"Precision={values['precision']:.4f}, "
                f"Recall={values['recall']:.4f}, "
                f"F1={values['f1']:.4f}, "
                f"Support={values['support']}"
            )

        print("\nConfusion matrix:")
        print("Rows = actual, columns = predicted")
        print("             Pred Normal  Pred Violence")
        matrix = metrics["confusion_matrix"]
        print(f"Actual Normal     {matrix[0][0]:>5}          {matrix[0][1]:>5}")
        print(f"Actual Violence   {matrix[1][0]:>5}          {matrix[1][1]:>5}")

        print(f"\nPredictions saved: {prediction_path}")

    summary_path = output_dir / "metrics.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nEvaluation completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
