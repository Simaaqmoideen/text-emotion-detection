"""
preprocess.py — Text Preprocessing Pipeline
=============================================
Provides a robust, reusable text cleaning pipeline for NLP tasks.
Handles lowercasing, URL/punctuation removal, tokenization,
stopword removal, and lemmatization.

Usage:
    from src.preprocess import preprocess_text, preprocess_dataframe
    clean = preprocess_text("I'm SO happy today!!! 😊 https://t.co/abc")
    # Returns: "happy today"
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

# ---------------------------------------------------------------------------
# NLTK Resource Bootstrap
# ---------------------------------------------------------------------------
# Download required NLTK data packages (runs once, then cached).

NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def ensure_nltk_resources() -> None:
    """Download NLTK data packages if they are not already present."""
    for pkg in NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


# Run on import so downstream code never hits a missing-resource error.
ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Module-level singletons (initialized once for performance)
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Pre-compiled regex patterns for speed
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core Preprocessing Function
# ---------------------------------------------------------------------------
def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_token_length: int = 2,
) -> str:
    """
    Clean and normalize a raw text string for NLP modeling.

    Pipeline steps (in order):
        1. Lowercase the entire string
        2. Remove URLs (http/https/www)
        3. Remove HTML tags
        4. Remove @mentions
        5. Remove hashtag symbols (keep the word)
        6. Remove emojis and unicode pictographs
        7. Remove punctuation and special characters
        8. Collapse multiple whitespace into single spaces
        9. Tokenize into word-level tokens
       10. Remove English stopwords (optional)
       11. Lemmatize each token (optional)
       12. Filter out very short tokens (< min_token_length)

    Parameters
    ----------
    text : str
        Raw input text (tweet, review, sentence, etc.).
    remove_stopwords : bool, default True
        Whether to filter out NLTK English stopwords.
    lemmatize : bool, default True
        Whether to apply WordNet lemmatization.
    min_token_length : int, default 2
        Discard tokens shorter than this length.

    Returns
    -------
    str
        A single cleaned string with tokens joined by spaces.

    Examples
    --------
    >>> preprocess_text("I'm SO happy today!!! 😊 https://t.co/abc")
    'happy today'
    >>> preprocess_text("The cats were running quickly", lemmatize=True)
    'cat running quickly'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2-6: Remove noise with pre-compiled patterns
    text = URL_PATTERN.sub("", text)
    text = HTML_TAG_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)

    # Step 7: Remove punctuation and special characters (keep letters + spaces)
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)

    # Step 8: Collapse whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Step 9: Tokenize
    tokens: List[str] = word_tokenize(text)

    # Step 10: Stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # Step 11: Lemmatization
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t, pos="v") for t in tokens]

    # Step 12: Filter short tokens
    tokens = [t for t in tokens if len(t) >= min_token_length]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Batch / DataFrame Helper
# ---------------------------------------------------------------------------
def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    output_column: str = "clean_text",
    **kwargs,
) -> pd.DataFrame:
    """
    Apply ``preprocess_text`` to an entire DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw text.
    text_column : str
        Name of the column with raw text.
    output_column : str
        Name of the new column for cleaned text.
    **kwargs
        Forwarded to ``preprocess_text`` (e.g., ``remove_stopwords=False``).

    Returns
    -------
    pd.DataFrame
        The same dataframe with the new ``output_column`` appended.
    """
    df = df.copy()
    df[output_column] = df[text_column].apply(
        lambda x: preprocess_text(x, **kwargs)
    )
    # Drop rows where cleaning produced empty strings
    initial_len = len(df)
    df = df[df[output_column].str.strip().astype(bool)].reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[preprocess] Dropped {dropped} empty rows after cleaning.")
    return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "I'm SO happy today!!! 😊 https://t.co/abc",
        "This is absolutely terrible... #angry @someone",
        "Wow, I can't believe it happened! <b>Surprising</b>",
        "Just another normal day, nothing special.",
        "😢😢😢 I feel so sad and lonely right now...",
    ]
    print("=" * 60)
    print("PREPROCESSING PIPELINE — DEMO")
    print("=" * 60)
    for s in samples:
        print(f"  RAW : {s}")
        print(f"  CLEAN: {preprocess_text(s)}")
        print()
