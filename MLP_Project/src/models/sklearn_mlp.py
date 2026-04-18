from __future__ import annotations

import math

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer


class SklearnMLPClassifier:
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
        self.training_steps_ = 0

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="sgd",
            alpha=l2_lambda,
            batch_size=batch_size,
            learning_rate="constant",
            learning_rate_init=learning_rate,
            max_iter=1,
            shuffle=shuffle,
            random_state=seed,
            warm_start=True,
            momentum=0.0,
            nesterovs_momentum=False,
            early_stopping=False,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        initial_parameters: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, list[float]]:
        self._initialize_model(X_train, y_train, initial_parameters)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for _ in range(self.epochs):
            self.model.partial_fit(X_train, y_train)
            self.training_steps_ += math.ceil(X_train.shape[0] / self.batch_size)

            train_probabilities = self.model.predict_proba(X_train)
            history["train_loss"].append(float(self._cross_entropy(train_probabilities, y_train)))
            history["train_accuracy"].append(float(self.model.score(X_train, y_train)))

            if X_val is not None and y_val is not None:
                val_probabilities = self.model.predict_proba(X_val)
                history["val_loss"].append(float(self._cross_entropy(val_probabilities, y_val)))
                history["val_accuracy"].append(float(self.model.score(X_val, y_val)))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _initialize_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        initial_parameters: list[tuple[np.ndarray, np.ndarray]] | None,
    ) -> None:
        classes = np.unique(y_train)
        self.model.partial_fit(X_train, y_train, classes=classes)

        if initial_parameters is not None:
            self.model.coefs_ = [weights.copy() for weights, _ in initial_parameters]
            self.model.intercepts_ = [bias.reshape(-1).copy() for _, bias in initial_parameters]

        self.model._optimizer = SGDOptimizer(
            self.model.coefs_ + self.model.intercepts_,
            learning_rate_init=self.learning_rate,
            lr_schedule="constant",
            momentum=0.0,
            nesterov=False,
            power_t=0.5,
        )
        self.model.n_iter_ = 0
        self.model.t_ = 0
        self.model.loss_curve_ = []
        self.model.best_loss_ = np.inf

    @staticmethod
    def _cross_entropy(probabilities: np.ndarray, y_true: np.ndarray) -> float:
        clipped = np.clip(probabilities, 1e-12, 1.0)
        return float(-np.log(clipped[np.arange(y_true.shape[0]), y_true]).mean())
