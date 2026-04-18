from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class DatasetBundle:
    features: np.ndarray
    targets: np.ndarray
    feature_names: list[str]
    target_names: list[str]


@dataclass(slots=True)
class DataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_wine_dataset() -> DatasetBundle:
    raw = load_wine()
    return DatasetBundle(
        features=raw.data.astype(np.float64),
        targets=raw.target.astype(np.int64),
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
    )


def split_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    random_state: int,
    validation_size: float = 0.2,
    test_size: float = 0.2,
) -> DataSplits:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be smaller than 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    adjusted_validation_size = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=adjusted_validation_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return DataSplits(
        X_train=X_train.astype(np.float64),
        X_val=X_val.astype(np.float64),
        X_test=X_test.astype(np.float64),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
    )


def preprocess_splits(splits: DataSplits, *, strategy: str) -> DataSplits:
    strategy = strategy.lower()
    if strategy == "none":
        return DataSplits(
            X_train=splits.X_train.copy(),
            X_val=splits.X_val.copy(),
            X_test=splits.X_test.copy(),
            y_train=splits.y_train.copy(),
            y_val=splits.y_val.copy(),
            y_test=splits.y_test.copy(),
        )

    if strategy == "standardize":
        mean = splits.X_train.mean(axis=0)
        std = splits.X_train.std(axis=0)
        std = np.where(std == 0, 1.0, std)

        def transform(values: np.ndarray) -> np.ndarray:
            return (values - mean) / std

    elif strategy == "minmax":
        minimum = splits.X_train.min(axis=0)
        maximum = splits.X_train.max(axis=0)
        scale = np.where((maximum - minimum) == 0, 1.0, maximum - minimum)

        def transform(values: np.ndarray) -> np.ndarray:
            return (values - minimum) / scale

    else:
        raise ValueError(f"Unsupported preprocessing strategy: {strategy}")

    return DataSplits(
        X_train=transform(splits.X_train),
        X_val=transform(splits.X_val),
        X_test=transform(splits.X_test),
        y_train=splits.y_train.copy(),
        y_val=splits.y_val.copy(),
        y_test=splits.y_test.copy(),
    )
