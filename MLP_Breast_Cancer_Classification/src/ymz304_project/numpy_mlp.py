from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ymz304_project.initialization import ParameterBundle, create_parameter_bundle


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(slots=True)
class NumpyMLPClassifier:
    input_dim: int
    hidden_layers: tuple[int, ...]
    learning_rate: float = 0.1
    epochs: int = 200
    batch_size: int = 32
    random_state: int = 42
    l2_lambda: float = 0.0
    initial_parameters: ParameterBundle | None = None
    weights: list[np.ndarray] = field(init=False)
    biases: list[np.ndarray] = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bundle = self.initial_parameters or create_parameter_bundle(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            output_dim=1,
            random_state=self.random_state,
        )
        self.weights = [weight.copy() for weight in bundle.weights]
        self.biases = [bias.copy() for bias in bundle.biases]
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=np.float64)
        y_column = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        history: dict[str, list[float]] = {"train_loss": []}

        batch_size = max(1, min(self.batch_size, X.shape[0]))
        indices = np.arange(X.shape[0])

        for _epoch in range(self.epochs):
            self._rng.shuffle(indices)
            for start in range(0, X.shape[0], batch_size):
                batch_indices = indices[start : start + batch_size]
                self._train_batch(X[batch_indices], y_column[batch_indices])

            train_prob = self.predict_proba(X).reshape(-1, 1)
            history["train_loss"].append(float(self._binary_cross_entropy(y_column, train_prob)))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations = np.asarray(X, dtype=np.float64)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            activations = _sigmoid(activations @ weight + bias)
        output = _sigmoid(activations @ self.weights[-1] + self.biases[-1])
        return output.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def _train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> None:
        activations = [X_batch]
        zs: list[np.ndarray] = []

        current = X_batch
        for weight, bias in zip(self.weights, self.biases, strict=True):
            z_values = current @ weight + bias
            zs.append(z_values)
            current = _sigmoid(z_values)
            activations.append(current)

        delta = activations[-1] - y_batch
        gradients_w = [np.zeros_like(weight) for weight in self.weights]
        gradients_b = [np.zeros_like(bias) for bias in self.biases]

        for layer_index in reversed(range(len(self.weights))):
            gradients_w[layer_index] = activations[layer_index].T @ delta / X_batch.shape[0]
            gradients_b[layer_index] = delta.mean(axis=0)

            if self.l2_lambda > 0.0:
                gradients_w[layer_index] += (
                    self.l2_lambda / X_batch.shape[0]
                ) * self.weights[layer_index]

            if layer_index > 0:
                prev_activation = activations[layer_index]
                delta = (
                    delta @ self.weights[layer_index].T
                ) * prev_activation * (1.0 - prev_activation)

        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= self.learning_rate * gradients_w[layer_index]
            self.biases[layer_index] -= self.learning_rate * gradients_b[layer_index]

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        clipped = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
        return float(-(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)).mean())
