from src.metrics import compute_classification_metrics


def test_compute_classification_metrics_contains_expected_keys() -> None:
    metrics = compute_classification_metrics([0, 1, 1, 2], [0, 1, 0, 2])
    expected_keys = {"accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"}
    assert expected_keys.issubset(metrics.keys())
