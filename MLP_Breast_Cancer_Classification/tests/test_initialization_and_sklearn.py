import numpy as np

from ymz304_project.initialization import create_parameter_bundle
from ymz304_project.sklearn_model import ControlledMLPClassifier


def test_create_parameter_bundle_is_deterministic() -> None:
    bundle_a = create_parameter_bundle(
        input_dim=2,
        hidden_layers=(3,),
        output_dim=1,
        random_state=11,
    )
    bundle_b = create_parameter_bundle(
        input_dim=2,
        hidden_layers=(3,),
        output_dim=1,
        random_state=11,
    )

    for left, right in zip(bundle_a.weights, bundle_b.weights, strict=True):
        np.testing.assert_allclose(left, right)

    for left, right in zip(bundle_a.biases, bundle_b.biases, strict=True):
        np.testing.assert_allclose(left, right)


def test_controlled_mlp_classifier_uses_provided_initial_weights_when_learning_rate_is_zero(
) -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    bundle = create_parameter_bundle(input_dim=2, hidden_layers=(3,), output_dim=1, random_state=13)

    model = ControlledMLPClassifier(
        hidden_layer_sizes=(3,),
        activation="logistic",
        solver="sgd",
        learning_rate_init=0.0,
        batch_size=4,
        shuffle=False,
        momentum=0.0,
        nesterovs_momentum=False,
        max_iter=1,
        random_state=13,
        provided_weights=bundle.weights,
        provided_biases=bundle.biases,
    )

    model.fit(X, y)

    np.testing.assert_allclose(model.coefs_[0], bundle.weights[0])
    np.testing.assert_allclose(model.intercepts_[0], bundle.biases[0])
    np.testing.assert_allclose(model.coefs_[1], bundle.weights[1])
    np.testing.assert_allclose(model.intercepts_[1], bundle.biases[1])
