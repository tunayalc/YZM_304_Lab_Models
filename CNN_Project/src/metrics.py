from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, object]:
    y_true_array = np.asarray(list(y_true))
    y_pred_array = np.asarray(list(y_pred))

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": float(precision_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred_array).tolist(),
    }


def history_to_frame(history: List[Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(history)
