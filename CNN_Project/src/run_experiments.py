from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DATA_DIR, FEATURES_DIR, PLOTS_DIR, REPORTS_DIR, SPLITS_DIR, TrainingConfig
from src.data import build_cifar10_loaders, build_cifar10_split_bundle, set_seed, write_split_manifest
from src.metrics import history_to_frame
from src.models import HybridFeatureCNN, LeNetBaselineCNN, LeNetImprovedCNN, ResNet18ReferenceCNN
from src.reporting import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_training_curves,
    save_csv,
    save_json,
)
from src.training import extract_feature_arrays, fit_model, resolve_device, run_hybrid_random_forest


def main() -> None:
    seed = 42
    set_seed(seed)

    split_bundle = build_cifar10_split_bundle(root=DATA_DIR, seed=seed, val_size=10_000)
    write_split_manifest(SPLITS_DIR / "split_manifest.json", split_bundle.split_manifest)
    plot_class_distribution(
        split_bundle.split_manifest["class_distribution"]["train"],
        PLOTS_DIR / "train_class_distribution.png",
    )

    device = resolve_device()
    standard_batch_size = 128
    reference_batch_size = 64

    standard_loaders = build_cifar10_loaders(
        root=DATA_DIR,
        split_bundle=split_bundle,
        batch_size=standard_batch_size,
        image_size=32,
    )
    reference_loaders = build_cifar10_loaders(
        root=DATA_DIR,
        split_bundle=split_bundle,
        batch_size=reference_batch_size,
        image_size=64,
    )

    experiments = [
        {
            "model": LeNetBaselineCNN(),
            "config": TrainingConfig(
                name="model_1_lenet_baseline",
                epochs=15,
                learning_rate=1e-3,
                batch_size=standard_batch_size,
                weight_decay=0.0,
                image_size=32,
            ),
            "loaders": standard_loaders,
            "title": "Model 1 - LeNet benzeri temel CNN",
        },
        {
            "model": LeNetImprovedCNN(),
            "config": TrainingConfig(
                name="model_2_lenet_improved",
                epochs=18,
                learning_rate=1e-3,
                batch_size=standard_batch_size,
                weight_decay=1e-4,
                image_size=32,
            ),
            "loaders": standard_loaders,
            "title": "Model 2 - BatchNorm ve Dropout ile iyilestirilmis CNN",
        },
        {
            "model": ResNet18ReferenceCNN(),
            "config": TrainingConfig(
                name="model_3_resnet18_reference",
                epochs=6,
                learning_rate=5e-4,
                batch_size=reference_batch_size,
                weight_decay=1e-4,
                image_size=64,
            ),
            "loaders": reference_loaders,
            "title": "Model 3 - Hazir ResNet18 mimarisi",
        },
        {
            "model": HybridFeatureCNN(),
            "config": TrainingConfig(
                name="model_5_full_cnn_for_hybrid_comparison",
                epochs=20,
                learning_rate=1e-3,
                batch_size=standard_batch_size,
                weight_decay=1e-4,
                image_size=32,
            ),
            "loaders": standard_loaders,
            "title": "Model 5 - Hibrit karsilastirma icin tam CNN",
        },
    ]

    result_rows = []
    histories = {}
    trained_models = {}
    best_epochs = {}

    for spec in experiments:
        print(f"Running {spec['config'].name} on device={device} ...")
        result = fit_model(
            model=spec["model"],
            train_loader=spec["loaders"].train_loader,
            val_loader=spec["loaders"].val_loader,
            test_loader=spec["loaders"].test_loader,
            epochs=spec["config"].epochs,
            learning_rate=spec["config"].learning_rate,
            weight_decay=spec["config"].weight_decay,
            device=device,
        )

        trained_models[result.name] = result.model
        histories[result.name] = result.history
        best_epochs[result.name] = result.best_epoch

        history_frame = history_to_frame(result.history)
        save_csv(REPORTS_DIR / f"{result.name}_history.csv", history_frame)

        plot_confusion_matrix(
            matrix=result.metrics["confusion_matrix"],
            class_names=split_bundle.class_names,
            title=spec["title"],
            output_path=PLOTS_DIR / f"confusion_matrix_{result.name}.png",
        )

        result_rows.append(
            {
                "model": result.name,
                "type": "cnn",
                "best_epoch": result.best_epoch,
                "image_size": spec["config"].image_size,
                "accuracy": result.metrics["accuracy"],
                "precision_macro": result.metrics["precision_macro"],
                "recall_macro": result.metrics["recall_macro"],
                "f1_macro": result.metrics["f1_macro"],
                "loss": result.metrics["loss"],
            }
        )

    comparison_model = trained_models["model_5_full_cnn_for_hybrid_comparison"]
    train_features, train_labels = extract_feature_arrays(comparison_model, standard_loaders.train_eval_loader, device)
    test_features, test_labels = extract_feature_arrays(comparison_model, standard_loaders.test_loader, device)

    np.save(FEATURES_DIR / "train_features.npy", train_features)
    np.save(FEATURES_DIR / "train_labels.npy", train_labels)
    np.save(FEATURES_DIR / "test_features.npy", test_features)
    np.save(FEATURES_DIR / "test_labels.npy", test_labels)

    print(f"Hybrid train_features shape: {train_features.shape}")
    print(f"Hybrid train_labels shape: {train_labels.shape}")
    print(f"Hybrid test_features shape: {test_features.shape}")
    print(f"Hybrid test_labels shape: {test_labels.shape}")

    hybrid_result = run_hybrid_random_forest(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
    )

    plot_confusion_matrix(
        matrix=hybrid_result["metrics"]["confusion_matrix"],
        class_names=split_bundle.class_names,
        title="Model 4 - Hibrit RandomForest",
        output_path=PLOTS_DIR / "confusion_matrix_model_4_hybrid_random_forest.png",
    )

    result_rows.append(
        {
            "model": "model_4_hybrid_random_forest",
            "type": "hybrid_ml",
            "best_epoch": None,
            "image_size": 32,
            "accuracy": hybrid_result["metrics"]["accuracy"],
            "precision_macro": hybrid_result["metrics"]["precision_macro"],
            "recall_macro": hybrid_result["metrics"]["recall_macro"],
            "f1_macro": hybrid_result["metrics"]["f1_macro"],
            "loss": None,
            "feature_source_model": "model_5_full_cnn_for_hybrid_comparison",
        }
    )

    results_frame = pd.DataFrame(result_rows)
    save_csv(REPORTS_DIR / "experiment_metrics.csv", results_frame)
    plot_metric_comparison(result_rows, PLOTS_DIR / "model_metric_comparison.png")
    plot_training_curves(histories, PLOTS_DIR / "training_curves_cnn_models.png")

    feature_summary = {
        "train_features_shape": list(train_features.shape),
        "train_labels_shape": list(train_labels.shape),
        "test_features_shape": list(test_features.shape),
        "test_labels_shape": list(test_labels.shape),
        "feature_source_model": "model_5_full_cnn_for_hybrid_comparison",
        "hybrid_classifier": "RandomForestClassifier",
    }
    save_json(REPORTS_DIR / "hybrid_feature_summary.json", feature_summary)

    summary = {
        "dataset": split_bundle.split_manifest["dataset"],
        "device": str(device),
        "best_epochs": best_epochs,
        "split_manifest": split_bundle.split_manifest,
        "results": json.loads(results_frame.to_json(orient="records")),
        "hybrid_feature_summary": feature_summary,
    }
    save_json(REPORTS_DIR / "summary.json", summary)


if __name__ == "__main__":
    main()
