from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ymz304_project.initialization import ParameterBundle, create_parameter_bundle


class _BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Sigmoid())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(slots=True)
class TorchMLPClassifier:
    input_dim: int
    hidden_layers: tuple[int, ...]
    learning_rate: float = 0.1
    epochs: int = 200
    batch_size: int = 32
    random_state: int = 42
    l2_lambda: float = 0.0
    initial_parameters: ParameterBundle | None = None
    model: _BinaryMLP = field(init=False)
    loss_fn: nn.BCELoss = field(init=False, repr=False)
    optimizer: torch.optim.Optimizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        torch.manual_seed(self.random_state)
        self.model = _BinaryMLP(input_dim=self.input_dim, hidden_layers=self.hidden_layers)

        parameter_bundle = self.initial_parameters or create_parameter_bundle(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            output_dim=1,
            random_state=self.random_state,
        )
        self._load_initial_parameters(parameter_bundle)

        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        history: dict[str, list[float]] = {"train_loss": []}

        batch_size = max(1, min(self.batch_size, X_tensor.shape[0]))
        generator = torch.Generator().manual_seed(self.random_state)

        for _epoch in range(self.epochs):
            indices = torch.randperm(X_tensor.shape[0], generator=generator)
            for start in range(0, X_tensor.shape[0], batch_size):
                batch_indices = indices[start : start + batch_size]
                logits = self.model(X_tensor[batch_indices])
                loss = self.loss_fn(logits, y_tensor[batch_indices])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                full_loss = self.loss_fn(self.model(X_tensor), y_tensor)
                history["train_loss"].append(float(full_loss.item()))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probabilities = self.model(torch.tensor(X, dtype=torch.float32)).squeeze(1)
        return probabilities.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def _load_initial_parameters(self, bundle: ParameterBundle) -> None:
        linear_layers = [module for module in self.model.network if isinstance(module, nn.Linear)]
        for layer, weight, bias in zip(linear_layers, bundle.weights, bundle.biases, strict=True):
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(weight.T, dtype=torch.float32))
                layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
