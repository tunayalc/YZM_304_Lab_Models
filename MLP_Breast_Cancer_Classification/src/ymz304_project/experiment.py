from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ymz304_project.data import PreparedDataset, load_and_prepare_dataset
from ymz304_project.initialization import ParameterBundle, create_parameter_bundle
from ymz304_project.metrics import evaluate_binary_classifier, generalization_diagnosis
from ymz304_project.numpy_mlp import NumpyMLPClassifier
from ymz304_project.reporting import (
    ensure_output_dir,
    plot_confusion_matrix,
    plot_learning_curve,
    save_comparison_csv,
    save_json,
)
from ymz304_project.sklearn_model import ControlledMLPClassifier
from ymz304_project.torch_model import TorchMLPClassifier


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    l2_lambda: float
    random_state: int
    batch_size: int = 0  # 0 = full-batch (train set boyutu kullanılır)


def run_experiments(
    output_dir: str | Path,
    numpy_epochs: int = 250,
    sklearn_epochs: int = 250,
    torch_epochs: int = 250,
) -> dict[str, Any]:
    output_path = ensure_output_dir(output_dir)

    # --- İki farklı ön işleme ile veri setleri ---
    dataset_standard = load_and_prepare_dataset(random_state=42, scaler_type="standard")
    dataset_minmax = load_and_prepare_dataset(random_state=42, scaler_type="minmax")

    specs = [
        ModelSpec(
            name="baseline_single_hidden",
            hidden_layers=(16,),
            learning_rate=0.05,
            l2_lambda=0.0,
            random_state=17,
        ),
        ModelSpec(
            name="regularized_deep",
            hidden_layers=(32, 16),
            learning_rate=0.03,
            l2_lambda=0.001,
            random_state=29,
        ),
        ModelSpec(
            name="wider_single_hidden",
            hidden_layers=(64,),
            learning_rate=0.05,
            l2_lambda=0.0,
            random_state=17,
            batch_size=64,
        ),
        ModelSpec(
            name="deep_three_layer",
            hidden_layers=(32, 16, 8),
            learning_rate=0.03,
            l2_lambda=0.0001,
            random_state=29,
        ),
    ]

    eda_summary = _build_eda_summary(dataset_standard)
    results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    # --- StandardScaler ile tüm modeller ---
    for spec in specs:
        _run_all_frameworks(
            spec=spec,
            dataset=dataset_standard,
            scaler_label="standard",
            numpy_epochs=numpy_epochs,
            sklearn_epochs=sklearn_epochs,
            torch_epochs=torch_epochs,
            results=results,
            comparison_rows=comparison_rows,
            output_path=output_path,
        )

    # --- MinMaxScaler ile baseline model ---
    _run_all_frameworks(
        spec=specs[0],
        dataset=dataset_minmax,
        scaler_label="minmax",
        numpy_epochs=numpy_epochs,
        sklearn_epochs=sklearn_epochs,
        torch_epochs=torch_epochs,
        results=results,
        comparison_rows=comparison_rows,
        output_path=output_path,
    )

    best_model = sorted(
        results,
        key=lambda item: (-item["val_metrics"]["accuracy"], item["steps"]),
    )[0]

    summary = {
        "dataset": eda_summary,
        "selection_rule": "Highest validation accuracy, then lowest training steps.",
        "best_model": {
            "framework": best_model["framework"],
            "model_name": best_model["model_name"],
            "scaler": best_model["scaler"],
            "hidden_layers": list(best_model["hidden_layers"]),
            "steps": best_model["steps"],
            "val_accuracy": best_model["val_metrics"]["accuracy"],
            "test_accuracy": best_model["test_metrics"]["accuracy"],
        },
        "results": results,
    }

    save_json(eda_summary, output_path / "eda_summary.json")
    save_json(summary, output_path / "summary.json")
    save_comparison_csv(comparison_rows, output_path / "model_comparison.csv")

    return summary


def _run_all_frameworks(
    *,
    spec: ModelSpec,
    dataset: PreparedDataset,
    scaler_label: str,
    numpy_epochs: int,
    sklearn_epochs: int,
    torch_epochs: int,
    results: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    bundle = create_parameter_bundle(
        input_dim=dataset.X_train.shape[1],
        hidden_layers=spec.hidden_layers,
        output_dim=1,
        random_state=spec.random_state,
    )

    framework_runs = [
        _run_numpy_model(spec, dataset, bundle, numpy_epochs, scaler_label),
        _run_sklearn_model(spec, dataset, bundle, sklearn_epochs, scaler_label),
        _run_torch_model(spec, dataset, bundle, torch_epochs, scaler_label),
    ]

    for run in framework_runs:
        results.append(run)
        comparison_rows.append(
            {
                "model_name": run["model_name"],
                "framework": run["framework"],
                "scaler": run["scaler"],
                "hidden_layers": "-".join(str(width) for width in run["hidden_layers"]),
                "batch_size": run["batch_size"],
                "steps": run["steps"],
                "train_accuracy": run["train_metrics"]["accuracy"],
                "val_accuracy": run["val_metrics"]["accuracy"],
                "test_accuracy": run["test_metrics"]["accuracy"],
                "precision": run["test_metrics"]["precision"],
                "recall": run["test_metrics"]["recall"],
                "f1": run["test_metrics"]["f1"],
                "roc_auc": run["test_metrics"]["roc_auc"],
                "generalization": run["generalization"],
            }
        )

        suffix = f"{run['framework']}_{run['model_name']}_{scaler_label}"
        plot_confusion_matrix(
            run["test_metrics"]["confusion_matrix"],
            output_path / f"confusion_matrix_{suffix}.png",
            title=f"{run['framework']} - {run['model_name']} ({scaler_label})",
        )
        plot_learning_curve(
            run["history"]["train_loss"],
            output_path / f"learning_curve_{suffix}.png",
            title=f"{run['framework']} - {run['model_name']} ({scaler_label})",
        )


def _resolve_batch_size(spec: ModelSpec, train_size: int) -> int:
    return spec.batch_size if spec.batch_size > 0 else train_size


def _run_numpy_model(
    spec: ModelSpec,
    dataset: PreparedDataset,
    bundle: ParameterBundle,
    epochs: int,
    scaler_label: str,
) -> dict[str, Any]:
    batch = _resolve_batch_size(spec, dataset.X_train.shape[0])
    model = NumpyMLPClassifier(
        input_dim=dataset.X_train.shape[1],
        hidden_layers=spec.hidden_layers,
        learning_rate=spec.learning_rate,
        epochs=epochs,
        batch_size=batch,
        random_state=spec.random_state,
        l2_lambda=spec.l2_lambda,
        initial_parameters=bundle,
    )
    history = model.fit(dataset.X_train, dataset.y_train)
    return _package_run_result(
        framework="numpy",
        model_name=spec.name,
        hidden_layers=spec.hidden_layers,
        batch_size=batch,
        history=history,
        predict_fn=model.predict,
        predict_proba_fn=model.predict_proba,
        dataset=dataset,
        scaler_label=scaler_label,
    )


def _run_sklearn_model(
    spec: ModelSpec,
    dataset: PreparedDataset,
    bundle: ParameterBundle,
    epochs: int,
    scaler_label: str,
) -> dict[str, Any]:
    batch = _resolve_batch_size(spec, dataset.X_train.shape[0])
    model = ControlledMLPClassifier(
        hidden_layer_sizes=spec.hidden_layers,
        activation="logistic",
        solver="sgd",
        alpha=spec.l2_lambda,
        batch_size=batch,
        learning_rate="constant",
        learning_rate_init=spec.learning_rate,
        max_iter=epochs,
        shuffle=False,
        random_state=spec.random_state,
        tol=0.0,
        momentum=0.0,
        nesterovs_momentum=False,
        n_iter_no_change=epochs + 1,
        provided_weights=bundle.weights,
        provided_biases=bundle.biases,
    )
    model.fit(dataset.X_train, dataset.y_train)
    history = {"train_loss": [float(value) for value in model.loss_curve_]}
    return _package_run_result(
        framework="sklearn",
        model_name=spec.name,
        hidden_layers=spec.hidden_layers,
        batch_size=batch,
        history=history,
        predict_fn=model.predict,
        predict_proba_fn=lambda features: model.predict_proba(features)[:, 1],
        dataset=dataset,
        scaler_label=scaler_label,
    )


def _run_torch_model(
    spec: ModelSpec,
    dataset: PreparedDataset,
    bundle: ParameterBundle,
    epochs: int,
    scaler_label: str,
) -> dict[str, Any]:
    batch = _resolve_batch_size(spec, dataset.X_train.shape[0])
    model = TorchMLPClassifier(
        input_dim=dataset.X_train.shape[1],
        hidden_layers=spec.hidden_layers,
        learning_rate=spec.learning_rate,
        epochs=epochs,
        batch_size=batch,
        random_state=spec.random_state,
        l2_lambda=spec.l2_lambda,
        initial_parameters=bundle,
    )
    history = model.fit(dataset.X_train, dataset.y_train)
    return _package_run_result(
        framework="torch",
        model_name=spec.name,
        hidden_layers=spec.hidden_layers,
        batch_size=batch,
        history=history,
        predict_fn=model.predict,
        predict_proba_fn=model.predict_proba,
        dataset=dataset,
        scaler_label=scaler_label,
    )


def _package_run_result(
    framework: str,
    model_name: str,
    hidden_layers: tuple[int, ...],
    batch_size: int,
    history: dict[str, list[float]],
    predict_fn: Any,
    predict_proba_fn: Any,
    dataset: PreparedDataset,
    scaler_label: str,
) -> dict[str, Any]:
    train_pred = np.asarray(predict_fn(dataset.X_train))
    train_prob = np.asarray(predict_proba_fn(dataset.X_train))
    val_pred = np.asarray(predict_fn(dataset.X_val))
    val_prob = np.asarray(predict_proba_fn(dataset.X_val))
    test_pred = np.asarray(predict_fn(dataset.X_test))
    test_prob = np.asarray(predict_proba_fn(dataset.X_test))

    train_metrics = evaluate_binary_classifier(dataset.y_train, train_pred, train_prob)
    val_metrics = evaluate_binary_classifier(dataset.y_val, val_pred, val_prob)
    test_metrics = evaluate_binary_classifier(dataset.y_test, test_pred, test_prob)

    return {
        "framework": framework,
        "model_name": model_name,
        "scaler": scaler_label,
        "hidden_layers": list(hidden_layers),
        "batch_size": batch_size,
        "steps": len(history["train_loss"]),
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "generalization": generalization_diagnosis(
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
        ),
    }


def _build_eda_summary(dataset: PreparedDataset) -> dict[str, Any]:
    combined_targets = np.concatenate([dataset.y_train, dataset.y_val, dataset.y_test])
    unique_labels, counts = np.unique(combined_targets, return_counts=True)
    class_distribution = {
        dataset.class_names[int(label)]: int(count)
        for label, count in zip(unique_labels, counts, strict=True)
    }

    return {
        "samples": int(combined_targets.shape[0]),
        "feature_count": int(dataset.X_train.shape[1]),
        "train_samples": int(dataset.X_train.shape[0]),
        "val_samples": int(dataset.X_val.shape[0]),
        "test_samples": int(dataset.X_test.shape[0]),
        "class_distribution": class_distribution,
        "missing_values": 0,
        "standardization": "StandardScaler fit on train split only",
    }
