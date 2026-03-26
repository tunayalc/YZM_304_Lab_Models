from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_wine_dataset, preprocess_splits, split_dataset
from src.metrics import classification_metrics
from src.models.numpy_mlp import NumpyMLPClassifier
from src.models.sklearn_mlp import SklearnMLPClassifier
from src.models.torch_mlp import TorchMLPClassifier


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DATA_DIR = ROOT / "data"


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    preprocessing: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    epochs: int
    batch_size: int
    l2_lambda: float
    description: str


CUSTOM_EXPERIMENTS = [
    ExperimentSpec(
        name="raw_baseline",
        preprocessing="none",
        hidden_layers=(16,),
        learning_rate=0.01,
        epochs=250,
        batch_size=16,
        l2_lambda=0.0,
        description="Ham veri ile tek gizli katmanli temel model.",
    ),
    ExperimentSpec(
        name="standardized_baseline",
        preprocessing="standardize",
        hidden_layers=(16,),
        learning_rate=0.01,
        epochs=250,
        batch_size=16,
        l2_lambda=0.0,
        description="Standartlastirilmis veri ile temel model.",
    ),
    ExperimentSpec(
        name="standardized_deeper_l2",
        preprocessing="standardize",
        hidden_layers=(32, 16),
        learning_rate=0.01,
        epochs=300,
        batch_size=16,
        l2_lambda=0.001,
        description="Standartlastirma, ikinci gizli katman ve L2 regülarizasyonu.",
    ),
]


def ensure_directories() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "weights").mkdir(parents=True, exist_ok=True)


def export_data_artifacts(
    *,
    root_dir: Path,
    dataset: Any,
    splits: Any,
    experiment_name: str,
    initial_parameters: list[tuple[Any, Any]],
    random_state: int,
    preprocessing: str,
) -> None:
    data_dir = root_dir / "data"
    splits_dir = data_dir / "splits"
    weights_dir = data_dir / "weights"
    splits_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": "wine",
        "random_state": random_state,
        "preprocessing": preprocessing,
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "counts": {
            "train": int(splits.X_train.shape[0]),
            "validation": int(splits.X_val.shape[0]),
            "test": int(splits.X_test.shape[0]),
        },
        "class_distribution": {
            "train": {str(label): int((splits.y_train == label).sum()) for label in sorted(set(splits.y_train.tolist()))},
            "validation": {str(label): int((splits.y_val == label).sum()) for label in sorted(set(splits.y_val.tolist()))},
            "test": {str(label): int((splits.y_test == label).sum()) for label in sorted(set(splits.y_test.tolist()))},
        },
    }
    with (splits_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    arrays: dict[str, Any] = {}
    shapes: list[dict[str, Any]] = []
    for layer_index, (weights, bias) in enumerate(initial_parameters, start=1):
        arrays[f"W{layer_index}"] = weights
        arrays[f"b{layer_index}"] = bias
        shapes.append(
            {
                "layer": layer_index,
                "weights_shape": list(weights.shape),
                "bias_shape": list(bias.shape),
            }
        )

    npz_path = weights_dir / f"{experiment_name}.npz"
    json_path = weights_dir / f"{experiment_name}.json"
    np.savez(npz_path, **arrays)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_name": experiment_name,
                "preprocessing": preprocessing,
                "shapes": shapes,
            },
            handle,
            indent=2,
        )


def export_supporting_artifacts(
    *,
    root_dir: Path,
    dataset: Any,
    splits: Any,
    custom_histories: dict[str, dict[str, Any]],
) -> None:
    artifacts_dir = root_dir / "artifacts"
    plots_dir = artifacts_dir / "plots"
    reports_dir = artifacts_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "dataset_name": "wine",
        "total_samples": int(dataset.features.shape[0]),
        "feature_count": int(dataset.features.shape[1]),
        "target_names": list(dataset.target_names),
        "split_counts": {
            "train": int(splits.X_train.shape[0]),
            "validation": int(splits.X_val.shape[0]),
            "test": int(splits.X_test.shape[0]),
        },
        "split_class_distribution": {
            "train": {str(label): int((splits.y_train == label).sum()) for label in sorted(set(splits.y_train.tolist()))},
            "validation": {str(label): int((splits.y_val == label).sum()) for label in sorted(set(splits.y_val.tolist()))},
            "test": {str(label): int((splits.y_test == label).sum()) for label in sorted(set(splits.y_test.tolist()))},
        },
    }
    with (reports_dir / "data_split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2)

    detailed_rows: list[dict[str, Any]] = []
    for experiment_name, payload in custom_histories.items():
        spec = payload["spec"]
        metrics = payload["metrics"]
        history = payload["history"]
        detailed_rows.append(
            {
                "name": experiment_name,
                "preprocessing": spec["preprocessing"],
                "hidden_layers": "-".join(str(value) for value in spec["hidden_layers"]),
                "learning_rate": spec["learning_rate"],
                "epochs": spec["epochs"],
                "batch_size": spec["batch_size"],
                "l2_lambda": spec["l2_lambda"],
                "final_train_accuracy": history["train_accuracy"][-1],
                "final_val_accuracy": history["val_accuracy"][-1],
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )

        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=list(dataset.target_names),
            columns=list(dataset.target_names),
        )
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{experiment_name} confusion matrix")
        plt.xlabel("Tahmin")
        plt.ylabel("Gercek")
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{experiment_name}.png", dpi=200)
        plt.close()

    pd.DataFrame(detailed_rows).to_csv(
        reports_dir / "custom_experiment_metrics_detailed.csv",
        index=False,
    )

    class_distribution = (
        pd.Series(dataset.targets)
        .value_counts()
        .sort_index()
        .rename(index={index: name for index, name in enumerate(dataset.target_names)})
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title("Wine veri seti sinif dagilimi")
    plt.xlabel("Sinif")
    plt.ylabel("Ornek sayisi")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png", dpi=200)
    plt.close()


def run_custom_experiments() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], Any, Any]:
    dataset = load_wine_dataset()
    base_splits = split_dataset(dataset.features, dataset.targets, random_state=42)
    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}

    for spec in CUSTOM_EXPERIMENTS:
        processed = preprocess_splits(base_splits, strategy=spec.preprocessing)
        model = NumpyMLPClassifier(
            input_dim=processed.X_train.shape[1],
            hidden_layers=spec.hidden_layers,
            output_dim=len(dataset.target_names),
            learning_rate=spec.learning_rate,
            epochs=spec.epochs,
            batch_size=spec.batch_size,
            seed=42,
            l2_lambda=spec.l2_lambda,
            shuffle=False,
        )
        history = model.fit(processed.X_train, processed.y_train, processed.X_val, processed.y_val)
        predictions = model.predict(processed.X_test)
        metrics = classification_metrics(processed.y_test, predictions)

        best_val_accuracy = max(history["val_accuracy"])
        best_epoch = history["val_accuracy"].index(best_val_accuracy) + 1

        rows.append(
            {
                "name": spec.name,
                "preprocessing": spec.preprocessing,
                "hidden_layers": "-".join(str(value) for value in spec.hidden_layers),
                "learning_rate": spec.learning_rate,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "l2_lambda": spec.l2_lambda,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "final_train_accuracy": history["train_accuracy"][-1],
                "final_val_accuracy": history["val_accuracy"][-1],
                "test_accuracy": metrics["accuracy"],
                "test_f1_macro": metrics["f1_macro"],
                "n_steps": model.training_steps_,
                "description": spec.description,
            }
        )
        histories[spec.name] = {
            "history": history,
            "metrics": metrics,
            "spec": asdict(spec),
        }

    results = pd.DataFrame(rows).sort_values(
        by=["best_val_accuracy", "n_steps"],
        ascending=[False, True],
    )
    best_row = results.iloc[0]
    best_spec = next(spec for spec in CUSTOM_EXPERIMENTS if spec.name == best_row["name"])
    selected_splits = preprocess_splits(base_splits, strategy=best_spec.preprocessing)
    return results, histories, best_spec, selected_splits


def run_library_comparison(spec: ExperimentSpec, selected_splits: Any) -> pd.DataFrame:
    output_dim = len(set(selected_splits.y_train.tolist()))
    common_kwargs = dict(
        input_dim=selected_splits.X_train.shape[1],
        hidden_layers=spec.hidden_layers,
        output_dim=output_dim,
        learning_rate=spec.learning_rate,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        seed=42,
        l2_lambda=spec.l2_lambda,
        shuffle=False,
    )

    initializer = NumpyMLPClassifier(**common_kwargs)
    initial_parameters = initializer.get_parameters_copy()

    library_rows: list[dict[str, Any]] = []
    model_builders = {
        "NumPy": NumpyMLPClassifier,
        "PyTorch": TorchMLPClassifier,
        "Scikit-learn": SklearnMLPClassifier,
    }

    for library_name, builder in model_builders.items():
        model = builder(**common_kwargs)
        fit_kwargs = {}
        if library_name != "NumPy":
            fit_kwargs["initial_parameters"] = initial_parameters
        else:
            model.set_parameters(initial_parameters)

        history = model.fit(
            selected_splits.X_train,
            selected_splits.y_train,
            selected_splits.X_val,
            selected_splits.y_val,
            **fit_kwargs,
        )
        predictions = model.predict(selected_splits.X_test)
        metrics = classification_metrics(selected_splits.y_test, predictions)
        library_rows.append(
            {
                "library": library_name,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "n_steps": model.training_steps_,
                "confusion_matrix": metrics["confusion_matrix"],
                "train_accuracy": history["train_accuracy"][-1],
                "val_accuracy": history["val_accuracy"][-1],
            }
        )
    return pd.DataFrame(library_rows)


def save_reports(
    custom_results: pd.DataFrame,
    custom_histories: dict[str, dict[str, Any]],
    library_results: pd.DataFrame,
    selected_spec: ExperimentSpec,
) -> None:
    custom_results.to_csv(REPORTS_DIR / "custom_experiments.csv", index=False)
    library_results.to_csv(REPORTS_DIR / "library_comparison.csv", index=False)

    summary = {
        "selected_model": asdict(selected_spec),
        "custom_experiments": custom_results.to_dict(orient="records"),
        "libraries": library_results.to_dict(orient="records"),
        "histories": custom_histories,
    }
    with (REPORTS_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def plot_custom_results(custom_results: pd.DataFrame, custom_histories: dict[str, dict[str, Any]]) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=custom_results, x="name", y="best_val_accuracy", hue="preprocessing")
    plt.title("Custom NumPy Modelleri - En Iyi Validation Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "custom_validation_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    for name, payload in custom_histories.items():
        plt.plot(payload["history"]["train_accuracy"], label=f"{name} train")
        plt.plot(payload["history"]["val_accuracy"], linestyle="--", label=f"{name} val")
    plt.title("Custom NumPy Modelleri - Accuracy Egrileri")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "custom_training_curves.png", dpi=200)
    plt.close()


def plot_library_results(library_results: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=library_results, x="library", y="accuracy")
    plt.title("Kutuphane Karsilastirmasi - Test Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Kutuphane")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "library_accuracy_comparison.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, len(library_results), figsize=(15, 4))
    if len(library_results) == 1:
        axes = [axes]
    for axis, (_, row) in zip(axes, library_results.iterrows()):
        matrix = pd.DataFrame(row["confusion_matrix"])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(row["library"])
        axis.set_xlabel("Tahmin")
        axis.set_ylabel("Gercek")
    fig.suptitle("Kutuphanelere Gore Karmaşıklık Matrisleri")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "library_confusion_matrices.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    dataset = load_wine_dataset()
    custom_results, custom_histories, selected_spec, selected_splits = run_custom_experiments()
    library_results = run_library_comparison(selected_spec, selected_splits)
    initializer = NumpyMLPClassifier(
        input_dim=selected_splits.X_train.shape[1],
        hidden_layers=selected_spec.hidden_layers,
        output_dim=len(dataset.target_names),
        learning_rate=selected_spec.learning_rate,
        epochs=selected_spec.epochs,
        batch_size=selected_spec.batch_size,
        seed=42,
        l2_lambda=selected_spec.l2_lambda,
        shuffle=False,
    )
    export_data_artifacts(
        root_dir=ROOT,
        dataset=dataset,
        splits=selected_splits,
        experiment_name=selected_spec.name,
        initial_parameters=initializer.get_parameters_copy(),
        random_state=42,
        preprocessing=selected_spec.preprocessing,
    )
    export_supporting_artifacts(
        root_dir=ROOT,
        dataset=dataset,
        splits=selected_splits,
        custom_histories=custom_histories,
    )
    save_reports(custom_results, custom_histories, library_results, selected_spec)
    plot_custom_results(custom_results, custom_histories)
    plot_library_results(library_results)

    print("Secilen model:", selected_spec.name)
    print(custom_results.to_string(index=False))
    print(library_results.to_string(index=False))


if __name__ == "__main__":
    main()
