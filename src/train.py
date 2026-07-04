"""
train.py — Model Training & Evaluation
========================================
Trains multi-class emotion classifiers (Logistic Regression and
Multinomial Naive Bayes), evaluates them with comprehensive metrics,
and persists the best model to disk.

Usage:
    python -m src.train              # (called from main.py)
    from src.train import train_and_evaluate
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.features import (
    build_tfidf_vectorizer,
    fit_vectorizer,
    save_vectorizer,
    transform_texts,
)
from src.visualize import (
    plot_class_distribution,
    plot_class_metrics,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
def _get_models() -> Dict[str, object]:
    """
    Return a dictionary of candidate classifiers to evaluate.

    Logistic Regression is our primary baseline — it works well with
    sparse TF-IDF features and supports probability estimates via
    softmax (multi_class='multinomial').

    Multinomial Naive Bayes is a fast alternative that performs
    surprisingly well on text classification tasks.
    """
    return {
        # Primary model: LR with C=1.0 as specified — better semantic separation
        "LogisticRegression": LogisticRegression(
            C=1.0,                       # Balanced regularization
            solver="lbfgs",              # Robust solver for multinomial
            class_weight="balanced",     # Handle class imbalances
            max_iter=1000,               # Ensure convergence
            random_state=42,
        ),
        "MultinomialNB": MultinomialNB(
            alpha=0.1,                   # Laplace smoothing parameter
        ),
    }


# ---------------------------------------------------------------------------
# Training + Evaluation Pipeline
# ---------------------------------------------------------------------------
def train_and_evaluate(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    label_column: str = "emotion",
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: str = "outputs",
    max_features: int = 10_000,
    run_cross_val: bool = True,
) -> Tuple[object, object, Dict]:
    """
    End-to-end training pipeline: vectorize → split → train → evaluate.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with ``text_column`` and ``label_column``.
    text_column : str
        Column containing cleaned text.
    label_column : str
        Column containing emotion labels.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.
    output_dir : str
        Directory for saving models, plots, and reports.
    max_features : int
        Maximum vocabulary size for TF-IDF.
    run_cross_val : bool
        Whether to run 5-fold stratified cross-validation.

    Returns
    -------
    best_model : fitted sklearn estimator
        The best-performing model.
    vectorizer : fitted TfidfVectorizer
        The fitted vectorizer (needed for inference).
    results : dict
        Evaluation metrics for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    texts = df[text_column].values
    labels = df[label_column].values

    # STRICT: Always use sorted order so class indices are deterministic
    # and perfectly aligned between training and inference.
    CANONICAL_CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    class_names = sorted(np.unique(labels).tolist())
    assert class_names == CANONICAL_CLASSES, (
        f"Label mismatch! Found {class_names}, expected {CANONICAL_CLASSES}. "
        "Check your dataset labels."
    )

    print("=" * 65)
    print("  EMOTION DETECTION — MODEL TRAINING")
    print("=" * 65)
    print(f"  Total samples : {len(texts):,}")
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  [VERIFIED] Class order is canonical and deterministic.")
    print(f"  Test split    : {test_size:.0%}")
    print(f"  Max features  : {max_features:,}")
    print("=" * 65)

    # ---- 1. Plot class distribution ----
    plot_class_distribution(
        labels,
        title="Training Data — Emotion Distribution",
        save_path=os.path.join(output_dir, "plots", "class_distribution.png"),
    )

    # ---- 2. Train / Test Split (stratified) ----
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,              # Preserve class proportions
    )
    print(f"\n[train] Split → Train: {len(X_train_txt):,}  |  Test: {len(X_test_txt):,}")

    # ---- 3. TF-IDF Vectorization ----
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    X_train = fit_vectorizer(vectorizer, X_train_txt)
    X_test = transform_texts(vectorizer, X_test_txt)

    # ---- 4. Train & Evaluate Each Model ----
    models = _get_models()
    results: Dict[str, Dict] = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""

    for name, model in models.items():
        print(f"\n{'─' * 55}")
        print(f"  Training: {name}")
        print(f"{'─' * 55}")

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # STRICT: Verify model.classes_ aligns exactly with our canonical order
        if hasattr(model, 'classes_'):
            assert list(model.classes_) == class_names, (
                f"CRITICAL: model.classes_ {list(model.classes_)} does not "
                f"match expected {class_names}. Label encoding is misaligned!"
            )
            print(f"  [VERIFIED] model.classes_ = {list(model.classes_)}")

        # Predictions
        y_pred = model.predict(X_test)

        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report_dict = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0,
        )
        report_str = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0,
        )

        print(f"  Accuracy        : {acc:.4f}")
        print(f"  F1 (macro)      : {f1_macro:.4f}")
        print(f"  F1 (weighted)   : {f1_weighted:.4f}")
        print(f"  Training time   : {train_time:.2f}s")
        print(f"\n{report_str}")

        # Optional: Cross-validation
        cv_scores = None
        if run_cross_val:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1,
            )
            print(f"  5-Fold CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Store results
        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report_dict,
            "classification_report_str": report_str,
            "cv_scores": cv_scores,
            "train_time_sec": train_time,
        }

        # Confusion matrix plot
        plot_confusion_matrix(
            y_test, y_pred, labels=class_names,
            title=f"Confusion Matrix — {name}",
            save_path=os.path.join(output_dir, "plots", f"confusion_matrix_{name}.png"),
        )

        # Per-class F1 bar chart
        plot_class_metrics(
            report_dict, metric="f1-score",
            title=f"Per-Class F1-Score — {name}",
            save_path=os.path.join(output_dir, "plots", f"f1_per_class_{name}.png"),
        )

        # Track the best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model
            best_name = name

    # ---- 5. Save Best Model & Vectorizer ----
    print(f"\n{'=' * 65}")
    print(f"  ✓ Best Model: {best_name}  (F1 macro = {best_f1:.4f})")
    print(f"{'=' * 65}")

    model_path = os.path.join(output_dir, "best_model.joblib")
    vec_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    joblib.dump(best_model, model_path)
    print(f"[train] Model saved     → {model_path}")
    save_vectorizer(vectorizer, vec_path)

    # Save the class names for inference
    meta_path = os.path.join(output_dir, "model_meta.joblib")
    joblib.dump({"class_names": class_names, "best_model_name": best_name}, meta_path)
    print(f"[train] Metadata saved  → {meta_path}")

    return best_model, vectorizer, results
