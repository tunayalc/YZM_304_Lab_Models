from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class SplitBundle:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    class_names: List[str]
    normalization_mean: Tuple[float, ...]
    normalization_std: Tuple[float, ...]
    split_manifest: Dict[str, object]


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_train_val_split(
    labels: Sequence[int],
    val_size: int = 10_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels)
    indices = np.arange(len(label_array))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_indices, val_indices = next(splitter.split(indices, label_array))
    return train_indices, val_indices


def _class_distribution(labels: Iterable[int], class_names: Sequence[str]) -> Dict[str, int]:
    counts = Counter(labels)
    return {class_name: int(counts.get(index, 0)) for index, class_name in enumerate(class_names)}


def compute_channel_stats(dataset: Dataset, batch_size: int = 256) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    channel_sum: torch.Tensor | None = None
    channel_squared_sum: torch.Tensor | None = None
    sample_count = 0

    for images, _ in loader:
        if channel_sum is None or channel_squared_sum is None:
            channel_sum = torch.zeros(images.size(1), dtype=torch.float64)
            channel_squared_sum = torch.zeros(images.size(1), dtype=torch.float64)

        channel_sum += images.sum(dim=(0, 2, 3), dtype=torch.float64)
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3), dtype=torch.float64)
        sample_count += images.size(0) * images.size(2) * images.size(3)

    if channel_sum is None or channel_squared_sum is None:
        raise ValueError("Channel statistics could not be computed from an empty dataset.")

    mean = channel_sum / sample_count
    variance = torch.clamp((channel_squared_sum / sample_count) - (mean ** 2), min=1e-12)
    std = torch.sqrt(variance)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def write_split_manifest(path: Path, split_manifest: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_mnist_split_bundle(
    root: Path,
    seed: int = 42,
    val_size: int = 10_000,
    image_size: int = 32,
) -> SplitBundle:
    stats_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    raw_train_dataset = datasets.MNIST(root=str(root), train=True, download=True, transform=stats_transform)
    raw_test_dataset = datasets.MNIST(root=str(root), train=False, download=True, transform=stats_transform)

    train_targets = raw_train_dataset.targets.tolist()
    test_targets = raw_test_dataset.targets.tolist()
    train_indices, val_indices = stratified_train_val_split(train_targets, val_size=val_size, seed=seed)
    train_subset_for_stats = Subset(raw_train_dataset, train_indices.tolist())
    mean, std = compute_channel_stats(train_subset_for_stats)

    class_names = [str(index) for index in range(10)]
    test_indices = np.arange(len(raw_test_dataset))
    split_manifest = {
        "dataset": "MNIST",
        "seed": seed,
        "image_shape": [1, image_size, image_size],
        "num_classes": len(class_names),
        "splits": {
            "train_size": int(len(train_indices)),
            "validation_size": int(len(val_indices)),
            "test_size": int(len(test_indices)),
        },
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "class_distribution": {
            "train": _class_distribution(np.asarray(train_targets)[train_indices], class_names),
            "validation": _class_distribution(np.asarray(train_targets)[val_indices], class_names),
            "test": _class_distribution(test_targets, class_names),
        },
    }

    return SplitBundle(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        class_names=class_names,
        normalization_mean=mean,
        normalization_std=std,
        split_manifest=split_manifest,
    )


def _expand_norm_stats(stats: Tuple[float, ...], channels: int) -> Tuple[float, ...]:
    if len(stats) == channels:
        return stats
    if len(stats) == 1 and channels > 1:
        return tuple(stats[0] for _ in range(channels))
    raise ValueError(f"Normalization stats length {len(stats)} is incompatible with channels={channels}.")


def _build_mnist_transform(
    image_size: int,
    channels: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> transforms.Compose:
    steps: List[object] = [transforms.Resize((image_size, image_size))]
    if channels == 3:
        steps.append(transforms.Grayscale(num_output_channels=3))
    elif channels != 1:
        raise ValueError("MNIST loader only supports channels=1 or channels=3.")

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_expand_norm_stats(mean, channels),
                std=_expand_norm_stats(std, channels),
            ),
        ]
    )
    return transforms.Compose(steps)


def build_mnist_loaders(
    root: Path,
    split_bundle: SplitBundle,
    batch_size: int,
    image_size: int = 32,
    channels: int = 1,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> LoaderBundle:
    train_transform = _build_mnist_transform(
        image_size=image_size,
        channels=channels,
        mean=split_bundle.normalization_mean,
        std=split_bundle.normalization_std,
    )
    eval_transform = _build_mnist_transform(
        image_size=image_size,
        channels=channels,
        mean=split_bundle.normalization_mean,
        std=split_bundle.normalization_std,
    )

    train_dataset = datasets.MNIST(root=str(root), train=True, download=False, transform=train_transform)
    train_eval_dataset = datasets.MNIST(root=str(root), train=True, download=False, transform=eval_transform)
    val_dataset = datasets.MNIST(root=str(root), train=True, download=False, transform=eval_transform)
    test_dataset = datasets.MNIST(root=str(root), train=False, download=False, transform=eval_transform)

    train_subset = Subset(train_dataset, split_bundle.train_indices.tolist())
    train_eval_subset = Subset(train_eval_dataset, split_bundle.train_indices.tolist())
    val_subset = Subset(val_dataset, split_bundle.val_indices.tolist())
    test_subset = Subset(test_dataset, split_bundle.test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    train_eval_loader = DataLoader(train_eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return LoaderBundle(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
