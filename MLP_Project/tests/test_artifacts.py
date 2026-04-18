import json
from pathlib import Path

import numpy as np

from src.data import load_wine_dataset, preprocess_splits, split_dataset
from src.models.numpy_mlp import NumpyMLPClassifier
from src.run_experiments import export_data_artifacts, export_supporting_artifacts


def test_export_data_artifacts_creates_split_manifest_and_weights(tmp_path: Path):
    dataset = load_wine_dataset()
    splits = preprocess_splits(
        split_dataset(dataset.features, dataset.targets, random_state=42),
        strategy="standardize",
    )
    initializer = NumpyMLPClassifier(
        input_dim=splits.X_train.shape[1],
        hidden_layers=(16,),
        output_dim=len(dataset.target_names),
        learning_rate=0.01,
        epochs=2,
        batch_size=8,
        seed=42,
    )

    export_data_artifacts(
        root_dir=tmp_path,
        dataset=dataset,
        splits=splits,
        experiment_name="standardized_baseline",
        initial_parameters=initializer.get_parameters_copy(),
        random_state=42,
        preprocessing="standardize",
    )

    manifest_path = tmp_path / "data" / "splits" / "split_manifest.json"
    weights_npz_path = tmp_path / "data" / "weights" / "standardized_baseline.npz"
    weights_json_path = tmp_path / "data" / "weights" / "standardized_baseline.json"

    assert manifest_path.exists()
    assert weights_npz_path.exists()
    assert weights_json_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["random_state"] == 42
    assert manifest["preprocessing"] == "standardize"
    assert manifest["counts"]["train"] == int(splits.X_train.shape[0])
    assert manifest["class_distribution"]["train"] == {str(i): int((splits.y_train == i).sum()) for i in range(3)}

    saved_arrays = np.load(weights_npz_path)
    assert saved_arrays["W1"].shape == (13, 16)
    assert saved_arrays["b2"].shape == (1, 3)


def test_export_supporting_artifacts_creates_dataset_summary_and_figures(tmp_path: Path):
    dataset = load_wine_dataset()
    splits = preprocess_splits(
        split_dataset(dataset.features, dataset.targets, random_state=42),
        strategy="standardize",
    )

    custom_histories = {
        "standardized_baseline": {
            "history": {
                "train_accuracy": [0.80, 0.95],
                "val_accuracy": [0.75, 0.90],
            },
            "metrics": {
                "accuracy": 0.90,
                "precision_macro": 0.91,
                "recall_macro": 0.89,
                "f1_macro": 0.90,
                "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 1, 8]],
            },
            "spec": {
                "name": "standardized_baseline",
                "preprocessing": "standardize",
                "hidden_layers": [16],
                "learning_rate": 0.01,
                "epochs": 250,
                "batch_size": 16,
                "l2_lambda": 0.0,
                "description": "test payload",
            },
        }
    }

    export_supporting_artifacts(
        root_dir=tmp_path,
        dataset=dataset,
        splits=splits,
        custom_histories=custom_histories,
    )

    summary_path = tmp_path / "artifacts" / "reports" / "data_split_summary.json"
    detailed_csv_path = tmp_path / "artifacts" / "reports" / "custom_experiment_metrics_detailed.csv"
    class_plot_path = tmp_path / "artifacts" / "plots" / "class_distribution.png"
    confusion_path = tmp_path / "artifacts" / "plots" / "confusion_matrix_standardized_baseline.png"

    assert summary_path.exists()
    assert detailed_csv_path.exists()
    assert class_plot_path.exists()
    assert confusion_path.exists()
