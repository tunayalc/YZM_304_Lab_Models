import numpy as np

from src.models.sklearn_mlp import SklearnMLPClassifier
from src.models.torch_mlp import TorchMLPClassifier


def _toy_multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 4))
    y = np.array([0, 1, 2] * 10)
    return X, y


def test_torch_wrapper_fit_and_predict():
    X, y = _toy_multiclass_data()
    model = TorchMLPClassifier(
        input_dim=4,
        hidden_layers=(6,),
        output_dim=3,
        learning_rate=0.05,
        epochs=2,
        batch_size=6,
        seed=123,
    )

    history = model.fit(X[:24], y[:24], X[24:], y[24:])
    predictions = model.predict(X[24:])

    assert "train_accuracy" in history
    assert predictions.shape == y[24:].shape


def test_sklearn_wrapper_fit_and_predict():
    X, y = _toy_multiclass_data()
    model = SklearnMLPClassifier(
        input_dim=4,
        hidden_layers=(6,),
        output_dim=3,
        learning_rate=0.05,
        epochs=2,
        batch_size=6,
        seed=321,
    )

    history = model.fit(X[:24], y[:24], X[24:], y[24:])
    predictions = model.predict(X[24:])

    assert "val_loss" in history
    assert predictions.shape == y[24:].shape
