from __future__ import annotations

import argparse
from pathlib import Path

from .classical import train_classical_model
from .config import RESULTS_DIR
from .data import download_imdb_dataset, imdb_available, load_imdb_train_test, load_sample_data
from .metrics import (
    evaluate_predictions,
    majority_class_predictions,
    plot_confusion_matrix,
    print_table,
    write_results_csv,
)
from .neural import TORCH_AVAILABLE, train_neural_model
from .preprocess import PREPROCESS_PROFILES, clean_many


CLASSICAL_MODELS = {"tfidf_lr", "tfidf_nb", "tfidf_svm"}
NEURAL_MODELS = {"embedding_dense", "textcnn", "bilstm", "cnn_bilstm"}
ALL_MODELS = ["majority", "tfidf_lr", "tfidf_nb", "tfidf_svm", "embedding_dense", "textcnn", "bilstm", "cnn_bilstm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMDb sentiment analysis experiments")
    parser.add_argument("--download", action="store_true", help="Download and extract Stanford IMDb dataset if missing")
    parser.add_argument("--sample", action="store_true", help="Use tiny built-in sample data for a fast smoke test")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit IMDb examples per class for faster runs")
    parser.add_argument("--models", nargs="+", default=["majority", "tfidf_lr", "tfidf_nb"])
    parser.add_argument("--preprocess", nargs="+", default=["basic"])
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown_models = [model for model in args.models if model not in ALL_MODELS]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Valid models: {ALL_MODELS}")

    unknown_profiles = [profile for profile in args.preprocess if profile not in PREPROCESS_PROFILES]
    if unknown_profiles:
        raise ValueError(f"Unknown preprocess profiles: {unknown_profiles}. Valid profiles: {list(PREPROCESS_PROFILES)}")


def load_data(args: argparse.Namespace) -> tuple[list[str], list[int], list[str], list[int]]:
    if args.sample:
        print("Using built-in sample data.")
        return load_sample_data()

    if args.download:
        download_imdb_dataset()
    if not imdb_available():
        raise FileNotFoundError("IMDb dataset not found. Re-run with --download or use --sample.")
    return load_imdb_train_test(limit_per_class=args.limit_per_class)


def confusion_path(output_dir: Path, model_name: str, preprocess_name: str) -> Path:
    return output_dir / f"confusion_{model_name}_{preprocess_name}.png"


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train_raw, y_train, x_test_raw, y_test = load_data(args)
    print(f"Train size: {len(x_train_raw)} | Test size: {len(x_test_raw)}")

    rows: list[dict[str, float | str | int]] = []

    for preprocess_name in args.preprocess:
        config = PREPROCESS_PROFILES[preprocess_name]
        print(f"\nPreprocess profile: {preprocess_name}")
        x_train = clean_many(x_train_raw, config)
        x_test = clean_many(x_test_raw, config)

        for model_name in args.models:
            print(f"Running model: {model_name}")

            if model_name == "majority":
                predictions = majority_class_predictions(y_train, len(y_test))
                metrics = evaluate_predictions(y_test, predictions, model_name, preprocess_name, train_seconds=0.0)

            elif model_name in CLASSICAL_MODELS:
                metrics, predictions = train_classical_model(
                    model_name,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    preprocess_name,
                    max_features=args.max_features,
                    ngram_max=args.ngram_max,
                )

            elif model_name in NEURAL_MODELS:
                if not TORCH_AVAILABLE:
                    print(f"Skipping {model_name}: PyTorch is not installed.")
                    continue
                metrics, predictions = train_neural_model(
                    model_name,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    preprocess_name,
                    max_vocab=args.max_vocab,
                    max_len=args.max_len,
                    embedding_dim=args.embedding_dim,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            rows.append(metrics)
            plot_confusion_matrix(
                y_test,
                predictions,
                title=f"{model_name} / {preprocess_name}",
                output_path=confusion_path(args.output_dir, model_name, preprocess_name),
            )

    results_path = args.output_dir / "experiment_results.csv"
    write_results_csv(rows, results_path)
    print("\nResults")
    print_table(rows)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
