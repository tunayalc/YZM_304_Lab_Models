from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .metrics import elapsed, evaluate_predictions, now
from .preprocess import tokenize

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    TORCH_AVAILABLE = False


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    pad_id: int = 0
    unk_id: int = 1

    @classmethod
    def build(cls, texts: list[str], max_vocab: int = 20000, min_freq: int = 2) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, count in counter.most_common(max_vocab - 2):
            if count < min_freq:
                continue
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids


if TORCH_AVAILABLE:

    class TextDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int):
            self.encoded = [vocab.encode(text, max_len) for text in texts]
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int):
            return (
                torch.tensor(self.encoded[index], dtype=torch.long),
                torch.tensor(self.labels[index], dtype=torch.float32),
            )


    class EmbeddingDenseClassifier(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(embedding_dim, 1)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids)
            mask = (input_ids != 0).unsqueeze(-1)
            summed = (embedded * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths
            return self.classifier(self.dropout(pooled)).squeeze(1)


    class TextCNNClassifier(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            num_filters: int = 96,
            kernel_sizes: tuple[int, ...] = (3, 4, 5),
            dropout: float = 0.4,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(embedding_dim, num_filters, kernel_size=size) for size in kernel_sizes]
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            features = []
            for conv in self.convs:
                activation = torch.relu(conv(embedded))
                pooled = torch.max(activation, dim=2).values
                features.append(pooled)
            combined = torch.cat(features, dim=1)
            return self.classifier(self.dropout(combined)).squeeze(1)


    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int = 96, dropout: float = 0.4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, 1)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids)
            _, (hidden, _) = self.lstm(embedded)
            final = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(self.dropout(final)).squeeze(1)


    class CNNBiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int = 80, dropout: float = 0.4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, 1)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            convolved = torch.relu(self.conv(embedded)).transpose(1, 2)
            _, (hidden, _) = self.lstm(convolved)
            final = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(self.dropout(final)).squeeze(1)


def _make_model(model_name: str, vocab_size: int, embedding_dim: int):
    model_builders: dict[str, Callable[[], nn.Module]] = {
        "embedding_dense": lambda: EmbeddingDenseClassifier(vocab_size, embedding_dim),
        "textcnn": lambda: TextCNNClassifier(vocab_size, embedding_dim),
        "bilstm": lambda: BiLSTMClassifier(vocab_size, embedding_dim),
        "cnn_bilstm": lambda: CNNBiLSTMClassifier(vocab_size, embedding_dim),
    }
    if model_name not in model_builders:
        raise ValueError(f"Unsupported neural model: {model_name}")
    return model_builders[model_name]()


def train_neural_model(
    model_name: str,
    x_train: list[str],
    y_train: list[int],
    x_test: list[str],
    y_test: list[int],
    preprocess_name: str,
    max_vocab: int = 20000,
    max_len: int = 300,
    embedding_dim: int = 128,
    batch_size: int = 64,
    epochs: int = 4,
    learning_rate: float = 1e-3,
) -> tuple[dict[str, float | str | int], list[int]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Run: pip install -r requirements-neural.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.build(x_train, max_vocab=max_vocab, min_freq=2 if len(x_train) > 100 else 1)
    train_dataset = TextDataset(x_train, y_train, vocab, max_len)
    test_dataset = TextDataset(x_test, y_test, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = _make_model(model_name, len(vocab.token_to_id), embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    start = now()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        mean_loss = total_loss / max(1, len(train_loader))
        print(f"{model_name} epoch {epoch + 1}/{epochs} - loss={mean_loss:.4f}")
    train_seconds = elapsed(start)

    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for input_ids, _ in test_loader:
            logits = model(input_ids.to(device))
            probs = torch.sigmoid(logits)
            predictions.extend((probs >= 0.5).long().cpu().tolist())

    metrics = evaluate_predictions(y_test, predictions, model_name, preprocess_name, train_seconds)
    return metrics, predictions
