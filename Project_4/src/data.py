from __future__ import annotations

import random
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from .config import IMDB_ARCHIVE, IMDB_EXTRACTED_DIR, IMDB_URL, RANDOM_SEED, RAW_DATA_DIR


class DownloadProgress(tqdm):
    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int | None = None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download_imdb_dataset(force: bool = False) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if IMDB_EXTRACTED_DIR.exists() and not force:
        return IMDB_EXTRACTED_DIR

    if not IMDB_ARCHIVE.exists() or force:
        print(f"Downloading IMDb dataset from {IMDB_URL}")
        with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=IMDB_ARCHIVE.name) as progress:
            urllib.request.urlretrieve(IMDB_URL, IMDB_ARCHIVE, reporthook=progress.update_to)

    print(f"Extracting {IMDB_ARCHIVE}")
    with tarfile.open(IMDB_ARCHIVE, "r:gz") as tar:
        _safe_extract(tar, RAW_DATA_DIR)
    return IMDB_EXTRACTED_DIR


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe path in archive: {member.name}") from exc
    tar.extractall(destination)


def imdb_available() -> bool:
    return (IMDB_EXTRACTED_DIR / "train" / "pos").exists() and (IMDB_EXTRACTED_DIR / "test" / "neg").exists()


def load_imdb_split(split: str, limit_per_class: int | None = None) -> tuple[list[str], list[int]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if not imdb_available():
        raise FileNotFoundError("IMDb data is missing. Run with --download first.")

    texts: list[str] = []
    labels: list[int] = []
    for label_name, label in [("neg", 0), ("pos", 1)]:
        files = sorted((IMDB_EXTRACTED_DIR / split / label_name).glob("*.txt"))
        if limit_per_class is not None:
            files = files[:limit_per_class]
        for file_path in files:
            texts.append(file_path.read_text(encoding="utf-8", errors="replace"))
            labels.append(label)
    return texts, labels


def shuffled(texts: list[str], labels: list[int], seed: int = RANDOM_SEED) -> tuple[list[str], list[int]]:
    pairs = list(zip(texts, labels))
    random.Random(seed).shuffle(pairs)
    shuffled_texts, shuffled_labels = zip(*pairs)
    return list(shuffled_texts), list(shuffled_labels)


def load_imdb_train_test(limit_per_class: int | None = None) -> tuple[list[str], list[int], list[str], list[int]]:
    train_texts, y_train = load_imdb_split("train", limit_per_class)
    test_texts, y_test = load_imdb_split("test", limit_per_class)
    train_texts, y_train = shuffled(train_texts, y_train, RANDOM_SEED)
    test_texts, y_test = shuffled(test_texts, y_test, RANDOM_SEED + 1)
    return train_texts, y_train, test_texts, y_test


def load_sample_data() -> tuple[list[str], list[int], list[str], list[int]]:
    positive = [
        "A wonderful film with touching performances and a beautiful ending.",
        "I loved the story, the acting was excellent and the soundtrack was memorable.",
        "This movie is surprisingly smart, funny, and emotionally satisfying.",
        "Brilliant direction and great characters made this a joy to watch.",
        "The pacing is strong and the final act is genuinely uplifting.",
        "It is not perfect, but it is charming and very enjoyable.",
        "A fantastic cast turns a simple plot into something special.",
        "The film has heart, humor, and several unforgettable scenes.",
    ]
    negative = [
        "A boring film with weak acting and a painfully predictable ending.",
        "I hated the story, the dialogue was awful and the soundtrack was annoying.",
        "This movie is messy, dull, and emotionally empty.",
        "Poor direction and flat characters made this hard to finish.",
        "The pacing is terrible and the final act is completely disappointing.",
        "It is not good, not clever, and not worth watching.",
        "A bad script wastes a talented cast.",
        "The film has no energy, no humor, and several forgettable scenes.",
    ]
    train_texts = positive[:6] + negative[:6]
    y_train = [1] * 6 + [0] * 6
    test_texts = positive[6:] + negative[6:]
    y_test = [1] * 2 + [0] * 2

    train_texts, y_train = shuffled(train_texts, y_train, RANDOM_SEED)
    test_texts, y_test = shuffled(test_texts, y_test, RANDOM_SEED + 1)
    return train_texts, y_train, test_texts, y_test
