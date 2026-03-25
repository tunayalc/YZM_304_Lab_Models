from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(slots=True)
class PreparedDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    scaler: Any
    scaler_type: str


def load_and_prepare_dataset(
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    scaler_type: str = "standard",
) -> PreparedDataset:
    raw_dataset = load_breast_cancer()
    X = raw_dataset.data.astype(np.float64)
    y = raw_dataset.target.astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=adjusted_val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return PreparedDataset(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(raw_dataset.feature_names),
        class_names=list(raw_dataset.target_names),
        scaler=scaler,
        scaler_type=scaler_type,
    )

