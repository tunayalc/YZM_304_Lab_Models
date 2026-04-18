import numpy as np

from src.data import (
    load_wine_dataset,
    split_dataset,
    preprocess_splits,
)


def test_split_dataset_has_expected_shapes_and_class_coverage():
    dataset = load_wine_dataset()
    splits = split_dataset(
        dataset.features,
        dataset.targets,
        random_state=42,
        validation_size=0.2,
        test_size=0.2,
    )

    assert splits.X_train.shape[1] == dataset.features.shape[1]
    assert splits.X_val.shape[1] == dataset.features.shape[1]
    assert splits.X_test.shape[1] == dataset.features.shape[1]

    observed = np.unique(np.concatenate([splits.y_train, splits.y_val, splits.y_test]))
    assert np.array_equal(observed, np.unique(dataset.targets))


def test_standardization_centers_training_data():
    dataset = load_wine_dataset()
    splits = split_dataset(dataset.features, dataset.targets, random_state=7)
    processed = preprocess_splits(splits, strategy="standardize")

    train_means = processed.X_train.mean(axis=0)
    assert np.allclose(train_means, np.zeros_like(train_means), atol=1e-7)


def test_none_preprocessing_returns_unchanged_values():
    dataset = load_wine_dataset()
    splits = split_dataset(dataset.features, dataset.targets, random_state=9)

    processed = preprocess_splits(splits, strategy="none")

    assert np.array_equal(processed.X_train, splits.X_train)
    assert np.array_equal(processed.y_test, splits.y_test)
