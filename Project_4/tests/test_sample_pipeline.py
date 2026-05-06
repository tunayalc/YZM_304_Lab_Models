from src.classical import train_classical_model
from src.data import load_sample_data
from src.preprocess import PREPROCESS_PROFILES, clean_many


def test_sample_pipeline_runs_tfidf_lr():
    x_train, y_train, x_test, y_test = load_sample_data()
    config = PREPROCESS_PROFILES["basic"]
    x_train = clean_many(x_train, config)
    x_test = clean_many(x_test, config)

    metrics, predictions = train_classical_model("tfidf_lr", x_train, y_train, x_test, y_test, "basic")

    assert len(predictions) == len(y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
