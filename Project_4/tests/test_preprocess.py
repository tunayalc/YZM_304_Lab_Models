from src.preprocess import PREPROCESS_PROFILES, clean_text, tokenize


def test_html_lowercase_and_stopword_config_keeps_negation():
    text = "This movie was <br /> NOT good, but the acting was fine!"
    cleaned = clean_text(text, PREPROCESS_PROFILES["stopwords_removed"])

    assert "<br" not in cleaned
    assert "not" in cleaned
    assert "good" in cleaned
    assert "acting" in cleaned
    assert "the" not in tokenize(cleaned)


def test_basic_profile_keeps_punctuation_context():
    text = "I didn't love it."
    cleaned = clean_text(text, PREPROCESS_PROFILES["basic"])

    assert "did not" in cleaned
