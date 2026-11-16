
"""Train MLP or LSTM classifiers and save model checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.features.loader import create_data_loader
from src.features.normalization import (
    compute_normalization_stats,
    normalize_lstm_batch,
    normalize_mlp_batch,
)
from src.models import MLPClassifier, LSTMClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_ROOT = Path("outputs/checkpoints")

DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 8
DEFAULT_LEARNING_RATE = 0.001
BATCH_SIZE = 16


def calculate_metrics(labels, predictions):
    """Calculate accuracy and macro F1 for two classes."""

    confusion = np.zeros((2, 2), dtype=np.int64)

    for true_label, predicted_label in zip(labels, predictions):
        confusion[int(true_label), int(predicted_label)] += 1

    class_f1 = []

    for class_id in range(2):
        true_positive = confusion[class_id, class_id]
        false_positive = confusion[:, class_id].sum() - true_positive
        false_negative = confusion[class_id, :].sum() - true_positive

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        class_f1.append(float(f1))

    accuracy = float(np.trace(confusion) / confusion.sum())

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(class_f1)),
        "class_f1": class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def calculate_class_weights(loader):
    """Calculate balanced class weights from training labels."""

    counts = torch.zeros(2, dtype=torch.float32)

    for batch in loader:
        labels = batch[-1]
        counts += torch.bincount(labels, minlength=2).float()

    if torch.any(counts == 0):
        raise ValueError("Training data is missing a class")

    weights = counts.sum() / (2.0 * counts)

    return weights.to(DEVICE)


def run_epoch(
    model,
    loader,
    criterion,
    model_type,
    stats,
    optimizer=None,
):
    """Run one training or validation epoch."""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for batch in loader:
        if model_type == "mlp":
            features, labels = batch
            features = normalize_mlp_batch(features, stats).to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = model(features)
                loss = criterion(logits, labels)

                if training:
                    loss.backward()
                    optimizer.step()

        else:
            features, lengths, labels = batch
            features = normalize_lstm_batch(
                features,
                lengths,
                stats,
            ).to(DEVICE)

            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = model(features, lengths)
                loss = criterion(logits, labels)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )
                    optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = calculate_metrics(all_labels, all_predictions)

    metrics["loss"] = total_loss / total_samples

    return metrics


def train_model(
    model_type: str,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
):
    """Train one model and save its best checkpoint."""

    if model_type not in {"mlp", "lstm"}:
        raise ValueError("model_type must be mlp or lstm")

    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")

    print(f"Device: {DEVICE}")
    print(f"Model: {model_type.upper()}")

    # Calculate normalization statistics using training data only.
    stats_loader = create_data_loader(
        "train",
        model_type,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    stats = compute_normalization_stats(
        stats_loader,
        model_type,
    )

    # Create independent training and validation loaders.
    train_loader = create_data_loader(
        "train",
        model_type,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = create_data_loader(
        "val",
        model_type,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_weights = calculate_class_weights(stats_loader)

    if model_type == "mlp":
        model = MLPClassifier()
    else:
        model = LSTMClassifier()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LEARNING_RATE,
        weight_decay=1e-4,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CHECKPOINT_ROOT / model_type / run_id

    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model_type": model_type,
        "epochs": epochs,
        "patience": patience,
        "batch_size": BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": 1e-4,
        "class_weights": class_weights.cpu().tolist(),
        "device": str(DEVICE),
        "normalization": "training_data_only",
    }

    with (output_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            model_type,
            stats,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            model_type,
            stats,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_macro_f1": best_macro_f1,
                "normalization": stats,
                "config": config,
                "label_to_id": {
                    "normal": 0,
                    "violence": 1,
                },
            }

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )

        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            output_dir / "last.pt",
        )

        with (output_dir / "history.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("\n===== TRAINING COMPLETE =====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP or LSTM classifier."
    )

    parser.add_argument(
        "--model",
        choices=["mlp", "lstm"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        patience=args.patience,
    )
