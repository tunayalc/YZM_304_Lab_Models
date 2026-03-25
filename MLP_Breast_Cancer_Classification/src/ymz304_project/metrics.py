from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    _ = y_prob

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": _specificity_score(y_true, y_pred),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def generalization_diagnosis(train_accuracy: float, val_accuracy: float) -> str:
    gap = train_accuracy - val_accuracy
    if train_accuracy < 0.9 and val_accuracy < 0.9:
        return "high_bias"
    if gap > 0.05:
        return "high_variance"
    return "balanced"


def _specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = confusion_matrix(y_true, y_pred)
    if matrix.shape != (2, 2):
        return 0.0
    tn, fp, _, _ = matrix.ravel()
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    return float(tn / denominator)
