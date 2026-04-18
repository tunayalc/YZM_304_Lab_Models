from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import compute_classification_metrics


@dataclass
class ExperimentResult:
    name: str
    history: List[Dict[str, float]]
    best_epoch: int
    model: nn.Module
    metrics: Dict[str, object]
    predictions: np.ndarray
    targets: np.ndarray


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions_buffer: List[np.ndarray] = []
    targets_buffer: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            predictions_buffer.append(predictions.cpu().numpy())
            targets_buffer.append(labels.cpu().numpy())

    y_pred = np.concatenate(predictions_buffer) if predictions_buffer else np.array([], dtype=np.int64)
    y_true = np.concatenate(targets_buffer) if targets_buffer else np.array([], dtype=np.int64)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["predictions"] = y_pred
    metrics["targets"] = y_true
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> ExperimentResult:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_epoch = 1
    best_val_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    model_name = getattr(model, "experiment_name", model.__class__.__name__.lower())

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = evaluate_model(model, val_loader, criterion, device)

        print(
            f"[{model_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f}"
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_stats["loss"]),
                "train_accuracy": float(train_stats["accuracy"]),
                "val_loss": float(val_stats["loss"]),
                "val_accuracy": float(val_stats["accuracy"]),
            }
        )

        if float(val_stats["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_stats["accuracy"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_stats = evaluate_model(model, test_loader, criterion, device)
    predictions = np.asarray(test_stats.pop("predictions"))
    targets = np.asarray(test_stats.pop("targets"))

    return ExperimentResult(
        name=model_name,
        history=history,
        best_epoch=best_epoch,
        model=model,
        metrics=test_stats,
        predictions=predictions,
        targets=targets,
    )


def extract_feature_arrays(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "extract_features"):
        raise AttributeError("Model must define extract_features for hybrid feature extraction.")

    model.eval()
    feature_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.extract_features(images)
            feature_batches.append(features.cpu().numpy())
            label_batches.append(labels.numpy())

    return np.concatenate(feature_batches), np.concatenate(label_batches)


def run_hybrid_random_forest(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, object]:
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    metrics = compute_classification_metrics(test_labels, predictions)

    return {
        "classifier": classifier,
        "predictions": predictions,
        "targets": test_labels,
        "metrics": metrics,
    }
