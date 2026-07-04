"""
predict.py — Inference Pipeline
=================================
Provides a standalone ``predict_emotion()`` function that takes raw text,
runs it through the full preprocessing + vectorization pipeline, and
returns the predicted emotion label with confidence scores.

Usage:
    from src.predict import EmotionPredictor
    predictor = EmotionPredictor.load("outputs")
    result = predictor.predict("I am so happy today!")
    print(result)
    # {'label': 'happy', 'confidence': 0.87, 'probabilities': {...}}
"""

import os
from typing import Dict, List, Optional, Union

import joblib
import numpy as np

from src.preprocess import preprocess_text
from src.features import load_vectorizer


# ---------------------------------------------------------------------------
# Predictor Class
# ---------------------------------------------------------------------------
class EmotionPredictor:
    """
    Encapsulates the trained model, vectorizer, and preprocessing logic
    into a single inference object.

    Attributes
    ----------
    model : sklearn estimator
        The trained classifier.
    vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer.
    class_names : list of str
        Ordered list of emotion class labels.
    model_name : str
        Name of the model architecture (e.g., 'LogisticRegression').
    """

    def __init__(self, model, vectorizer, class_names: List[str], model_name: str = ""):
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Factory: Load from disk
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, output_dir: str = "outputs") -> "EmotionPredictor":
        """
        Load a trained predictor from saved artifacts.

        Parameters
        ----------
        output_dir : str
            Directory containing ``best_model.joblib``,
            ``tfidf_vectorizer.joblib``, and ``model_meta.joblib``.

        Returns
        -------
        EmotionPredictor
            Ready-to-use predictor instance.
        """
        model = joblib.load(os.path.join(output_dir, "best_model.joblib"))
        vectorizer = load_vectorizer(os.path.join(output_dir, "tfidf_vectorizer.joblib"))
        meta = joblib.load(os.path.join(output_dir, "model_meta.joblib"))

        print(f"[predict] Loaded model: {meta['best_model_name']}")
        print(f"[predict] Classes: {meta['class_names']}")

        return cls(
            model=model,
            vectorizer=vectorizer,
            class_names=meta["class_names"],
            model_name=meta["best_model_name"],
        )

    # ------------------------------------------------------------------
    # Single-text prediction
    # ------------------------------------------------------------------
    def predict(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict the emotion of a single raw text input.
        """
        # Step 1: Preprocess
        clean = preprocess_text(text)
        
        # DEBUG: Verify preprocessing output as requested
        print(f"Debug Cleaned Text: '{clean}'")

        if not clean.strip():
            return {
                "input": text,
                "clean_text": "",
                "label": "Neutral / Inconclusive",
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.class_names},
            }

        # Step 2: Vectorize using the pre-fitted vectorizer instance
        # IMPORTANT: We use .transform(), NEVER .fit_transform() here!
        X = self.vectorizer.transform([clean])

        # Fallback Check: Is the vector entirely zeros? (No vocabulary match)
        if X.nnz == 0:
            return {
                "input": text,
                "clean_text": clean,
                "label": "Neutral / Inconclusive (OOV)",
                "confidence": 1.0 / len(self.class_names),
                "probabilities": {c: round(1.0/len(self.class_names), 4) for c in self.class_names},
            }

        # Step 3: Predict label
        label = self.model.predict(X)[0]

        # Step 4: Get probability distribution
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)[0]
            exp_d = np.exp(decision - np.max(decision))
            proba = exp_d / exp_d.sum()
        else:
            proba = np.zeros(len(self.class_names))

        prob_dict = {
            cls_name: round(float(p), 4)
            for cls_name, p in zip(self.class_names, proba)
        }
        confidence = round(float(max(proba)), 4)
        
        # Extra fallback for perfectly uniform distributions
        if max(proba) - min(proba) < 0.001 or confidence <= 1.0 / len(self.class_names) + 0.01:
            label = "Neutral / Inconclusive"

        return {
            "input": text,
            "clean_text": clean,
            "label": label,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict emotions for a list of raw text inputs.

        Parameters
        ----------
        texts : list of str
            Raw text strings.

        Returns
        -------
        list of dict
            One result dict per input text (same schema as ``predict``).
        """
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# Convenience function (matches the spec's predict_emotion signature)
# ---------------------------------------------------------------------------
_global_predictor: Optional[EmotionPredictor] = None


def predict_emotion(
    input_text: str,
    output_dir: str = "outputs",
) -> Dict[str, Union[str, float, Dict]]:
    """
    Standalone convenience function: predict the emotion of ``input_text``.

    On the first call, loads the model from ``output_dir``. Subsequent
    calls reuse the cached predictor for performance.

    Parameters
    ----------
    input_text : str
        Raw text to classify.
    output_dir : str
        Path to the directory containing saved model artifacts.

    Returns
    -------
    dict
        Prediction result with keys: input, clean_text, label,
        confidence, probabilities.

    Examples
    --------
    >>> result = predict_emotion("I am so happy today!")
    >>> print(result["label"])
    'happy'
    >>> print(result["confidence"])
    0.87
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = EmotionPredictor.load(output_dir)
    return _global_predictor.predict(input_text)
