"""
features.py — Feature Extraction Module
=========================================
Wraps scikit-learn's TfidfVectorizer with project-specific defaults.
Provides fit/transform helpers and persistence utilities.

Usage:
    from src.features import build_tfidf_vectorizer, extract_features
"""

from typing import Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Vectorizer Factory
# ---------------------------------------------------------------------------
def build_tfidf_vectorizer(
    max_features: int = 10_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with sensible defaults for emotion detection.

    Parameters
    ----------
    max_features : int, default 10000
        Maximum vocabulary size.  Keeps only the top-N features by
        term-frequency across the corpus — controls dimensionality.
    ngram_range : tuple of (int, int), default (1, 2)
        Extract unigrams AND bigrams so the model can capture short
        phrases like "not happy" or "so sad".
    min_df : int, default 2
        Ignore terms that appear in fewer than ``min_df`` documents.
        Filters out ultra-rare typos / noise.
    max_df : float, default 0.95
        Ignore terms that appear in more than 95 % of documents.
        Acts as a corpus-level stopword filter.
    sublinear_tf : bool, default True
        Apply sublinear TF scaling (1 + log(tf)).  Dampens the effect
        of very high raw term frequencies.

    Returns
    -------
    TfidfVectorizer
        An *unfitted* vectorizer ready for ``.fit()`` or ``.fit_transform()``.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",  # tokens >= 2 chars
    )


# ---------------------------------------------------------------------------
# Fit + Transform Helpers
# ---------------------------------------------------------------------------
def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    texts: np.ndarray,
) -> csr_matrix:
    """
    Fit the vectorizer on training texts and return the TF-IDF matrix.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        An unfitted vectorizer instance.
    texts : array-like of str
        Training corpus (already preprocessed).

    Returns
    -------
    csr_matrix
        Sparse TF-IDF feature matrix (n_samples × n_features).
    """
    X = vectorizer.fit_transform(texts)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[features] Vocabulary size after fit: {vocab_size:,} terms")
    print(f"[features] TF-IDF matrix shape: {X.shape}")
    return X


def transform_texts(
    vectorizer: TfidfVectorizer,
    texts: np.ndarray,
) -> csr_matrix:
    """
    Transform new texts using an already-fitted vectorizer.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        A *fitted* vectorizer.
    texts : array-like of str
        New texts to vectorize.

    Returns
    -------
    csr_matrix
        Sparse TF-IDF feature matrix.
    """
    return vectorizer.transform(texts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Serialize a fitted vectorizer to disk."""
    joblib.dump(vectorizer, path)
    print(f"[features] Vectorizer saved → {path}")


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load a previously saved vectorizer."""
    vectorizer = joblib.load(path)
    print(f"[features] Vectorizer loaded ← {path}")
    return vectorizer
