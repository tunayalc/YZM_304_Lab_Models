import html
import re
import string
from dataclasses import dataclass

from .config import SAFE_STOPWORDS


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
PUNCT_TABLE = str.maketrans({char: " " for char in string.punctuation if char != "'"})


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    lowercase: bool = True
    remove_html: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False


PREPROCESS_PROFILES = {
    "basic": PreprocessConfig(name="basic", lowercase=True, remove_html=True),
    "no_lowercase": PreprocessConfig(name="no_lowercase", lowercase=False, remove_html=True),
    "punctuation_removed": PreprocessConfig(
        name="punctuation_removed",
        lowercase=True,
        remove_html=True,
        remove_punctuation=True,
    ),
    "stopwords_removed": PreprocessConfig(
        name="stopwords_removed",
        lowercase=True,
        remove_html=True,
        remove_punctuation=True,
        remove_stopwords=True,
    ),
}


def normalize_contractions(text: str) -> str:
    replacements = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def clean_text(text: str, config: PreprocessConfig) -> str:
    text = html.unescape(text)
    if config.remove_html:
        text = HTML_TAG_RE.sub(" ", text)
    if config.lowercase:
        text = text.lower()
    text = normalize_contractions(text)
    if config.remove_punctuation:
        text = text.translate(PUNCT_TABLE)
    text = WHITESPACE_RE.sub(" ", text).strip()
    if config.remove_stopwords:
        tokens = tokenize(text)
        tokens = [token for token in tokens if token not in SAFE_STOPWORDS]
        text = " ".join(tokens)
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def clean_many(texts: list[str], config: PreprocessConfig) -> list[str]:
    return [clean_text(text, config) for text in texts]
