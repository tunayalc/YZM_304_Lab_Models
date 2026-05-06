from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import RESULTS_DIR
from .data import download_imdb_dataset, imdb_available, load_imdb_train_test, load_sample_data
from .preprocess import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple IMDb dataset EDA charts")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        x_train, y_train, x_test, y_test = load_sample_data()
    else:
        if args.download:
            download_imdb_dataset()
        if not imdb_available():
            raise FileNotFoundError("IMDb dataset not found. Re-run with --download or use --sample.")
        x_train, y_train, x_test, y_test = load_imdb_train_test(limit_per_class=args.limit_per_class)

    rows = []
    for split, texts, labels in [("train", x_train, y_train), ("test", x_test, y_test)]:
        for text, label in zip(texts, labels):
            rows.append({"split": split, "label": "positive" if label == 1 else "negative", "tokens": len(tokenize(text))})
    frame = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=frame, x="split", hue="label")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(args.output_dir / "eda_class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=frame, x="tokens", hue="label", bins=40, element="step", stat="density", common_norm=False)
    plt.title("Review length distribution")
    plt.xlabel("Token count")
    plt.tight_layout()
    plt.savefig(args.output_dir / "eda_review_lengths.png", dpi=180)
    plt.close()

    print(frame.groupby(["split", "label"])["tokens"].describe().round(2))


if __name__ == "__main__":
    main()
