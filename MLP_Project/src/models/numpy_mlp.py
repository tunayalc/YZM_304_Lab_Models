from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LayerCache:
    a_prev: np.ndarray
    z: np.ndarray


class NumpyMLPClassifier:
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
        shuffle: bool = True,
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

        self._rng = np.random.default_rng(seed)
        self.layer_dims = [input_dim, *hidden_layers, output_dim]
        self.parameters = self._initialize_parameters()
        self.training_steps_ = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
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

            for start in range(0, X_train.shape[0], self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                probabilities, caches = self._forward(X_batch)
                gradients = self._backward(X_batch, y_batch, probabilities, caches)
                self._apply_gradients(gradients)
                self.training_steps_ += 1

            train_probabilities, _ = self._forward(X_train)
            train_loss = self._compute_loss(train_probabilities, y_train)
            history["train_loss"].append(float(train_loss))
            history["train_accuracy"].append(float(self._accuracy(train_probabilities, y_train)))

            if X_val is not None and y_val is not None:
                val_probabilities, _ = self._forward(X_val)
                val_loss = self._compute_loss(val_probabilities, y_val)
                history["val_loss"].append(float(val_loss))
                history["val_accuracy"].append(float(self._accuracy(val_probabilities, y_val)))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities, _ = self._forward(X)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return probabilities.argmax(axis=1)

    def get_parameters_copy(self) -> list[tuple[np.ndarray, np.ndarray]]:
        copied: list[tuple[np.ndarray, np.ndarray]] = []
        for weights, bias in self.parameters:
            copied.append((weights.copy(), bias.copy()))
        return copied

    def set_parameters(self, parameters: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.parameters = [(weights.copy(), bias.copy()) for weights, bias in parameters]

    def _initialize_parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        parameters: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            limit = np.sqrt(2.0 / fan_in)
            weights = self._rng.normal(loc=0.0, scale=limit, size=(fan_in, fan_out))
            bias = np.zeros((1, fan_out), dtype=np.float64)
            parameters.append((weights.astype(np.float64), bias))
        return parameters

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[LayerCache]]:
        activations = X.astype(np.float64)
        caches: list[LayerCache] = []

        for layer_index, (weights, bias) in enumerate(self.parameters):
            z = activations @ weights + bias
            caches.append(LayerCache(a_prev=activations, z=z))
            if layer_index == len(self.parameters) - 1:
                activations = self._softmax(z)
            else:
                activations = self._relu(z)
        return activations, caches

    def _backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        caches: list[LayerCache],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        del X
        gradients: list[tuple[np.ndarray, np.ndarray]] = []
        one_hot = self._one_hot(y_true)
        dZ = (probabilities - one_hot) / y_true.shape[0]

        for layer_index in reversed(range(len(self.parameters))):
            weights, _ = self.parameters[layer_index]
            cache = caches[layer_index]

            dW = cache.a_prev.T @ dZ + (self.l2_lambda / y_true.shape[0]) * weights
            db = np.sum(dZ, axis=0, keepdims=True)
            gradients.append((dW, db))

            if layer_index > 0:
                previous_cache = caches[layer_index - 1]
                dA = dZ @ weights.T
                dZ = dA * self._relu_derivative(previous_cache.z)

        gradients.reverse()
        return gradients

    def _apply_gradients(self, gradients: list[tuple[np.ndarray, np.ndarray]]) -> None:
        updated: list[tuple[np.ndarray, np.ndarray]] = []
        for (weights, bias), (dW, db) in zip(self.parameters, gradients):
            updated.append(
                (
                    weights - self.learning_rate * dW,
                    bias - self.learning_rate * db,
                )
            )
        self.parameters = updated

    def _compute_loss(self, probabilities: np.ndarray, y_true: np.ndarray) -> float:
        one_hot = self._one_hot(y_true)
        clipped = np.clip(probabilities, 1e-12, 1.0)
        data_loss = -np.sum(one_hot * np.log(clipped)) / y_true.shape[0]
        regularization = 0.0
        if self.l2_lambda > 0:
            regularization = 0.5 * self.l2_lambda * sum(
                np.sum(weights * weights) for weights, _ in self.parameters
            ) / y_true.shape[0]
        return float(data_loss + regularization)

    def _accuracy(self, probabilities: np.ndarray, y_true: np.ndarray) -> float:
        predictions = probabilities.argmax(axis=1)
        return float(np.mean(predictions == y_true))

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        encoded = np.zeros((y.shape[0], self.output_dim), dtype=np.float64)
        encoded[np.arange(y.shape[0]), y] = 1.0
        return encoded

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values)

    @staticmethod
    def _relu_derivative(values: np.ndarray) -> np.ndarray:
        return (values > 0).astype(np.float64)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
