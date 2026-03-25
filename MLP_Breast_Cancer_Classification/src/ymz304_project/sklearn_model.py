from __future__ import annotations

from numbers import Real

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils._param_validation import Interval


class ControlledMLPClassifier(MLPClassifier):
    _parameter_constraints = {
        **MLPClassifier._parameter_constraints,
        "learning_rate_init": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = "relu",
        *,
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: int | str = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: int | None = None,
        tol: float = 0.0001,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
        provided_weights: list[np.ndarray] | None = None,
        provided_biases: list[np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.provided_weights = provided_weights
        self.provided_biases = provided_biases

    def _initialize(self, y: np.ndarray, layer_units: list[int], dtype: np.dtype) -> None:
        super()._initialize(y, layer_units, dtype)

        if self.provided_weights is None or self.provided_biases is None:
            return

        if len(self.provided_weights) != len(layer_units) - 1:
            msg = "Provided weight list does not match layer structure."
            raise ValueError(msg)

        self.coefs_ = [np.asarray(weight, dtype=dtype).copy() for weight in self.provided_weights]
        self.intercepts_ = [np.asarray(bias, dtype=dtype).copy() for bias in self.provided_biases]
        self._best_coefs = [coef.copy() for coef in self.coefs_]
        self._best_intercepts = [bias.copy() for bias in self.intercepts_]
