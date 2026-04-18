from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _TorchMLPNetwork(nn.Module):
    def __init__(self, layer_dims: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(fan_in, fan_out) for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs
        for layer_index, layer in enumerate(self.layers):
            activations = layer(activations)
            if layer_index < len(self.layers) - 1:
                activations = torch.relu(activations)
        return activations


class TorchMLPClassifier:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        output_dim: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        seed: int,
        l2_lambda: float = 0.0,
        shuffle: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.l2_lambda = l2_lambda
        self.shuffle = shuffle

        self.layer_dims = [input_dim, *hidden_layers, output_dim]
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.device = torch.device("cpu")
        self.network = _TorchMLPNetwork(self.layer_dims).to(self.device)
        self.training_steps_ = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        initial_parameters: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, list[float]]:
        if initial_parameters is not None:
            self.set_parameters(initial_parameters)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._optimizer_param_groups(), lr=self.learning_rate)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for _ in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self._rng.shuffle(indices)

            self.network.train()
            for start in range(0, X_train.shape[0], self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                inputs = X_train_tensor[batch_indices]
                targets = y_train_tensor[batch_indices]

                optimizer.zero_grad()
                logits = self.network(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                self.training_steps_ += 1

            train_logits = self.network(X_train_tensor)
            train_loss = criterion(train_logits, y_train_tensor).item()
            history["train_loss"].append(float(train_loss))
            history["train_accuracy"].append(
                float((train_logits.argmax(dim=1) == y_train_tensor).float().mean().item())
            )

            if X_val is not None and y_val is not None:
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)
                val_logits = self.network(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                history["val_loss"].append(float(val_loss))
                history["val_accuracy"].append(
                    float((val_logits.argmax(dim=1) == y_val_tensor).float().mean().item())
                )

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.tensor(X, dtype=torch.float32, device=self.device))
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def set_parameters(self, parameters: list[tuple[np.ndarray, np.ndarray]]) -> None:
        with torch.no_grad():
            for layer, (weights, bias) in zip(self.network.layers, parameters):
                layer.weight.copy_(torch.tensor(weights.T, dtype=torch.float32))
                layer.bias.copy_(torch.tensor(bias.reshape(-1), dtype=torch.float32))

    def _optimizer_param_groups(self) -> list[dict[str, object]]:
        decay_params = []
        no_decay_params = []
        for name, parameter in self.network.named_parameters():
            if name.endswith("weight"):
                decay_params.append(parameter)
            else:
                no_decay_params.append(parameter)
        return [
            {"params": decay_params, "weight_decay": self.l2_lambda},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
