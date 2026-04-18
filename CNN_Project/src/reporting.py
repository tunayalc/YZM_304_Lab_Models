from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    frame.to_csv(path, index=False)


def plot_class_distribution(class_distribution: Dict[str, int], output_path: Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(class_distribution.keys())
    counts = list(class_distribution.values())
    sns.barplot(x=names, y=counts, hue=names, legend=False, ax=ax, palette="crest")
    ax.set_title("CIFAR-10 Eğitim Sınıf Dağılımı")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Örnek Sayısı")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_curves(histories: Dict[str, List[Dict[str, float]]], output_path: Path) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for name, history in histories.items():
        frame = pd.DataFrame(history)
        axes[0].plot(frame["epoch"], frame["train_loss"], label=f"{name} train")
        axes[0].plot(frame["epoch"], frame["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(frame["epoch"], frame["train_accuracy"], label=f"{name} train")
        axes[1].plot(frame["epoch"], frame["val_accuracy"], linestyle="--", label=f"{name} val")

    axes[0].set_title("Eğitim ve Doğrulama Kayıp Eğrileri")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Eğitim ve Doğrulama Doğruluk Eğrileri")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_comparison(results: Iterable[Dict[str, object]], output_path: Path) -> None:
    ensure_parent(output_path)
    frame = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=frame, x="model", y="accuracy", ax=ax, color="#4C78A8")
    sns.scatterplot(data=frame, x="model", y="f1_macro", ax=ax, color="#F58518", s=90)
    ax.set_title("Beş Model İçin Accuracy ve F1 Macro Karşılaştırması")
    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
