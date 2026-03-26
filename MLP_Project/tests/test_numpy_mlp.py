import numpy as np

from src.models.numpy_mlp import NumpyMLPClassifier


def test_predict_proba_rows_sum_to_one():
    rng = np.random.default_rng(0)
    model = NumpyMLPClassifier(
        input_dim=4,
        hidden_layers=(6,),
        output_dim=3,
        learning_rate=0.05,
        epochs=2,
        batch_size=4,
        seed=123,
    )
    sample = rng.normal(size=(5, 4))

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), np.ones(5), atol=1e-7)


def test_fit_returns_history_and_predictions():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 5))
    y = np.array([0, 1, 2] * 8)
    X_train, X_val = X[:18], X[18:]
    y_train, y_val = y[:18], y[18:]

    model = NumpyMLPClassifier(
        input_dim=5,
        hidden_layers=(8,),
        output_dim=3,
        learning_rate=0.05,
        epochs=3,
        batch_size=6,
        seed=99,
    )

    history = model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_val)

    assert "train_loss" in history
    assert "val_accuracy" in history
    assert predictions.shape == y_val.shape
