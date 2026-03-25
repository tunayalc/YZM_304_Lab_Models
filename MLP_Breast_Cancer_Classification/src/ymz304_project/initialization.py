from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ParameterBundle:
    weights: list[np.ndarray]
    biases: list[np.ndarray]


def create_parameter_bundle(
    input_dim: int,
    hidden_layers: tuple[int, ...],
    output_dim: int,
    random_state: int,
) -> ParameterBundle:
    rng = np.random.default_rng(random_state)
    layer_dims = [input_dim, *hidden_layers, output_dim]

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:], strict=True):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64))
        biases.append(np.zeros(fan_out, dtype=np.float64))

    return ParameterBundle(weights=weights, biases=biases)

