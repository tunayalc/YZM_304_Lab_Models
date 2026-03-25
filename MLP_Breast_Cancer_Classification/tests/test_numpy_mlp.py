import numpy as np

from ymz304_project.numpy_mlp import NumpyMLPClassifier


def test_numpy_mlp_classifier_learns_simple_separable_problem() -> None:
    rng = np.random.default_rng(5)
    class_zero = rng.normal(loc=-1.0, scale=0.25, size=(32, 2))
    class_one = rng.normal(loc=1.0, scale=0.25, size=(32, 2))
    X = np.vstack([class_zero, class_one])
    y = np.concatenate([np.zeros(32, dtype=int), np.ones(32, dtype=int)])

    model = NumpyMLPClassifier(
        input_dim=2,
        hidden_layers=(4,),
        learning_rate=0.3,
        epochs=400,
        batch_size=64,
        random_state=5,
        l2_lambda=0.0,
    )

    history = model.fit(X, y)
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()

    assert history["train_loss"][0] > history["train_loss"][-1]
    assert accuracy >= 0.98
