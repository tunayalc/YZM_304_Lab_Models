import numpy as np

from ymz304_project.data import load_and_prepare_dataset
from ymz304_project.metrics import evaluate_binary_classifier


def test_load_and_prepare_dataset_returns_scaled_train_val_test_splits() -> None:
    dataset = load_and_prepare_dataset(random_state=7)

    total_rows = dataset.X_train.shape[0] + dataset.X_val.shape[0] + dataset.X_test.shape[0]
    assert total_rows == 569
    assert dataset.X_train.shape[1] == dataset.X_val.shape[1] == dataset.X_test.shape[1]
    assert dataset.class_names == ["malignant", "benign"]
    np.testing.assert_allclose(dataset.X_train.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(dataset.X_train.std(axis=0), 1.0, atol=1e-5)


def test_evaluate_binary_classifier_returns_core_metrics_and_confusion_matrix() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.8, 0.4, 0.2])

    metrics = evaluate_binary_classifier(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 2 / 3
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]

