"""Data loading, preprocessing, and splitting utilities."""

import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from src.config import DATA_CSV, LABEL2ID, SEED, TEST_SIZE, VAL_RATIO

# Ensure NLTK resources are available
for resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "omw-1.4" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ── Loading ────────────────────────────────────────────────────────────

def load_dataframe(path=None):
    """Load the raw FinancialPhraseBank CSV and return a DataFrame with
    columns: headline, sentiment, label."""
    path = path or DATA_CSV
    df = pd.read_csv(path, encoding="latin-1", header=None, names=["sentiment", "headline"])
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df


# ── Text cleaning ──────────────────────────────────────────────────────

def clean_text(text: str, lemmatize: bool = True) -> str:
    """Lower-case, strip non-alpha, remove stopwords, optionally lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in _STOPWORDS and len(t) > 2]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ── Splitting ──────────────────────────────────────────────────────────

def split_data(df, test_size=TEST_SIZE, val_ratio=VAL_RATIO, seed=SEED):
    """Stratified 70/10/20 split → (train_df, val_df, test_df)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=seed, stratify=train_df["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_splits():
    """Convenience: load → split → return (train, val, test) DataFrames."""
    df = load_dataframe()
    return split_data(df)


# ── Numpy helpers ──────────────────────────────────────────────────────

def arrays_for_sklearn(train_df, val_df, test_df, text_col="clean"):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) as arrays."""
    return (
        train_df[text_col].values, train_df["label"].values,
        val_df[text_col].values, val_df["label"].values,
        test_df[text_col].values, test_df["label"].values,
    )
