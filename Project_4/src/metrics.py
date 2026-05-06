from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def now() -> float:
    return time.perf_counter()


def elapsed(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 3)


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    model_name: str,
    preprocess_name: str,
    train_seconds: float,
) -> dict[str, float | str | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "model": model_name,
        "preprocess": preprocess_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "train_seconds": train_seconds,
        "n_test": len(y_true),
    }


def write_results_csv(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def print_table(rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        print("No results.")
        return
    headers = ["model", "preprocess", "accuracy", "precision", "recall", "f1", "train_seconds"]
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    line = " | ".join(header.ljust(widths[header]) for header in headers)
    print(line)
    print("-" * len(line))
    for row in sorted(rows, key=lambda item: float(item["f1"]), reverse=True):
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def majority_class_predictions(y_train: list[int], n_test: int) -> list[int]:
    values, counts = np.unique(y_train, return_counts=True)
    majority = int(values[np.argmax(counts)])
    return [majority] * n_test
