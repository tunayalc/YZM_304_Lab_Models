from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def ensure_output_dir(path_like: str | Path) -> Path:
    output_dir = Path(path_like)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    serializable = json.dumps(data, indent=2, ensure_ascii=False)
    Path(output_path).write_text(serializable, encoding="utf-8")


def save_comparison_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)


def plot_confusion_matrix(matrix: list[list[int]], output_path: str | Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_learning_curve(losses: list[float], output_path: str | Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, len(losses) + 1), losses, linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

