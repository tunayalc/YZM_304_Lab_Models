from __future__ import annotations

from .metrics import elapsed, evaluate_predictions, now


def train_classical_model(
    model_name: str,
    x_train: list[str],
    y_train: list[int],
    x_test: list[str],
    y_test: list[int],
    preprocess_name: str,
    max_features: int = 20000,
    ngram_max: int = 2,
) -> tuple[dict[str, float | str | int], list[int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    estimators = {
        "tfidf_lr": LogisticRegression(max_iter=1000, C=2.0, solver="liblinear", random_state=42),
        "tfidf_nb": MultinomialNB(alpha=0.3),
        "tfidf_svm": LinearSVC(C=1.0, random_state=42),
    }
    if model_name not in estimators:
        raise ValueError(f"Unsupported classical model: {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    min_df=2 if len(x_train) > 100 else 1,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", estimators[model_name]),
        ]
    )

    start = now()
    pipeline.fit(x_train, y_train)
    train_seconds = elapsed(start)
    predictions = pipeline.predict(x_test).astype(int).tolist()
    metrics = evaluate_predictions(y_test, predictions, model_name, preprocess_name, train_seconds)
    return metrics, predictions
