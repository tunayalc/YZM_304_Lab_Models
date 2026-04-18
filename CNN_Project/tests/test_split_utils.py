import numpy as np

from src.data import stratified_train_val_split


def test_stratified_train_val_split_preserves_index_count() -> None:
    labels = np.array([0] * 12 + [1] * 12 + [2] * 12)
    train_indices, val_indices = stratified_train_val_split(labels, val_size=9, seed=42)

    assert len(train_indices) + len(val_indices) == len(labels)
    assert len(set(train_indices).intersection(set(val_indices))) == 0
