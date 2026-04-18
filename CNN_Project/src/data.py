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
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
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
    return {class_names[index]: int(counts.get(index, 0)) for index in range(len(class_names))}


def compute_channel_stats(dataset: Dataset, batch_size: int = 256) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    sample_count = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        sample_count += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / sample_count
    variance = torch.clamp((channel_squared_sum / sample_count) - (mean ** 2), min=1e-12)
    std = torch.sqrt(variance)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def write_split_manifest(path: Path, split_manifest: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_cifar10_split_bundle(root: Path, seed: int = 42, val_size: int = 10_000) -> SplitBundle:
    tensor_transform = transforms.ToTensor()
    raw_train_dataset = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tensor_transform)
    raw_test_dataset = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tensor_transform)

    train_indices, val_indices = stratified_train_val_split(raw_train_dataset.targets, val_size=val_size, seed=seed)
    train_subset_for_stats = Subset(raw_train_dataset, train_indices.tolist())
    mean, std = compute_channel_stats(train_subset_for_stats)

    class_names = list(raw_train_dataset.classes)
    test_indices = np.arange(len(raw_test_dataset))
    split_manifest = {
        "dataset": "CIFAR-10",
        "seed": seed,
        "image_shape": [3, 32, 32],
        "num_classes": len(class_names),
        "splits": {
            "train_size": int(len(train_indices)),
            "validation_size": int(len(val_indices)),
            "test_size": int(len(test_indices)),
        },
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "class_distribution": {
            "train": _class_distribution(np.asarray(raw_train_dataset.targets)[train_indices], class_names),
            "validation": _class_distribution(np.asarray(raw_train_dataset.targets)[val_indices], class_names),
            "test": _class_distribution(raw_test_dataset.targets, class_names),
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


def build_cifar10_loaders(
    root: Path,
    split_bundle: SplitBundle,
    batch_size: int,
    image_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> LoaderBundle:
    transform_steps = []
    if image_size != 32:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=split_bundle.normalization_mean, std=split_bundle.normalization_std),
        ]
    )
    transform = transforms.Compose(transform_steps)

    train_dataset = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
    val_dataset = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(root), train=False, download=False, transform=transform)

    train_subset = Subset(train_dataset, split_bundle.train_indices.tolist())
    val_subset = Subset(val_dataset, split_bundle.val_indices.tolist())
    test_subset = Subset(test_dataset, split_bundle.test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    train_eval_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return LoaderBundle(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
