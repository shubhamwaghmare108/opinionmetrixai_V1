"""Data loading and classical NLP preprocessing utilities."""

from __future__ import annotations

import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from logging_config import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
logger = get_logger(__name__)

_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
_RESOURCES_READY = False


def _ensure_nltk_resources() -> None:
    """Ensure NLP data exists only when tokenization is actually requested."""
    global _RESOURCES_READY
    if _RESOURCES_READY:
        return
    for package, resource in _RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", package)
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Unable to download NLTK resource: {package}")
    _RESOURCES_READY = True


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
EMOJI_RE = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U00002600-\U000026FF]+", flags=re.UNICODE)
NUMBER_RE = re.compile(r"\d+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def load_data(path: str | Path, text_col: str = "review", label_col: str = "sentiment") -> pd.DataFrame:
    """Load, validate, and de-duplicate a review dataset."""
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)
    for column in (text_col, label_col):
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)


def clean_text(text: str) -> str:
    """Normalize text for the classical NLP pipeline."""
    value = str(text).lower()
    value = HTML_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    value = EMOJI_RE.sub(" ", value)
    value = NUMBER_RE.sub(" ", value)
    return re.sub(r"\s+", " ", value.translate(PUNCT_TABLE)).strip()


def tokenize_and_normalize(text: str, use_lemmatization: bool = True) -> list[str]:
    """Tokenize text, downloading NLTK data lazily on first use."""
    _ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    cleaned = clean_text(text)
    tokens = [token for token in word_tokenize(cleaned) if token not in stop_words and len(token) > 1]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review", use_lemmatization: bool = True) -> pd.DataFrame:
    """Add token and cleaned-text columns without mutating the input frame."""
    result = df.copy()
    result["tokens"] = result[text_col].apply(lambda value: tokenize_and_normalize(value, use_lemmatization))
    result["clean_review"] = result["tokens"].apply(" ".join)
    return result


if __name__ == "__main__":
    preprocess_dataframe(pd.read_csv(DATA_PATH))
