from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-ready result visualizations")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_DIR / "experiment_results.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def normalize_model(name: str) -> str:
    return {
        "majority": "Majority",
        "tfidf_lr": "TF-IDF + LR",
        "tfidf_svm": "TF-IDF + SVM",
        "tfidf_nb": "TF-IDF + NB",
        "embedding_dense": "Embedding + Dense",
        "textcnn": "TextCNN",
        "bilstm": "BiLSTM",
        "cnn_bilstm": "CNN + BiLSTM",
    }.get(name, name)


def normalize_preprocess(name: str) -> str:
    return {
        "basic": "Basic",
        "punctuation_removed": "Punctuation removed",
        "stopwords_removed": "Stopwords removed",
        "no_lowercase": "No lowercase",
    }.get(name, name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results_csv)
    frame["model_label"] = frame["model"].map(normalize_model)
    frame["preprocess_label"] = frame["preprocess"].map(normalize_preprocess)
    frame["f1_pct"] = frame["f1"] * 100
    frame["accuracy_pct"] = frame["accuracy"] * 100

    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    basic = frame[frame["preprocess"] == "basic"].sort_values("f1", ascending=False)
    plt.figure(figsize=(9, 4.8))
    colors = ["#C45A35" if idx == 0 else "#1F8A86" if row.model.startswith("tfidf") else "#D9A441" for idx, row in enumerate(basic.itertuples())]
    ax = sns.barplot(data=basic, y="model_label", x="f1_pct", palette=colors, hue="model_label", legend=False)
    ax.set_xlim(0, 95)
    ax.set_xlabel("F1-score (%)")
    ax.set_ylabel("")
    ax.set_title("Basic preprocessing: model F1 leaderboard")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "viz_model_f1_leaderboard.png", dpi=220)
    plt.close()

    pivot = frame.pivot_table(index="model_label", columns="preprocess_label", values="f1_pct", aggfunc="mean")
    order = basic["model_label"].tolist()
    pivot = pivot.reindex(order)
    plt.figure(figsize=(9, 5.2))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.8, cbar_kws={"label": "F1-score (%)"})
    plt.title("F1-score by preprocessing profile and model")
    plt.xlabel("Preprocessing")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(args.output_dir / "viz_preprocess_model_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.5, 4.8))
    ax = sns.scatterplot(
        data=frame[frame["model"] != "majority"],
        x="train_seconds",
        y="f1_pct",
        hue="model_label",
        style="preprocess_label",
        s=95,
        palette="Set2",
    )
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Accuracy-speed tradeoff across experiments")
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    plt.tight_layout()
    plt.savefig(args.output_dir / "viz_time_vs_f1.png", dpi=220)
    plt.close()

    best = frame.sort_values("f1", ascending=False).head(8).copy()
    best["experiment"] = best["model_label"] + "\n" + best["preprocess_label"]
    plt.figure(figsize=(9, 4.8))
    ax = sns.barplot(data=best, y="experiment", x="f1_pct", color="#C45A35")
    ax.set_xlim(0, 95)
    ax.set_xlabel("F1-score (%)")
    ax.set_ylabel("")
    ax.set_title("Top experiments by F1-score")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "viz_top_experiments.png", dpi=220)
    plt.close()

    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
