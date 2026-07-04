# AI-Powered Emotion Detection from Text

A modular, production-ready machine learning pipeline for multi-class emotion classification from text using traditional NLP techniques.

## Project Structure

```
emotion-detection/
├── main.py                  # End-to-end pipeline runner (entry point)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── report_template.md       # Internship report template
├── src/
│   ├── __init__.py          # Package initialization
│   ├── preprocess.py        # Text cleaning & lemmatization pipeline
│   ├── features.py          # TF-IDF feature extraction & persistence
│   ├── train.py             # Model training, evaluation & comparison
│   ├── predict.py           # Inference pipeline (single & batch)
│   └── visualize.py         # Confusion matrix & metric visualizations
└── outputs/                 # (Generated) Models, plots, reports
    ├── best_model.joblib
    ├── tfidf_vectorizer.joblib
    ├── model_meta.joblib
    └── plots/
        ├── class_distribution.png
        ├── confusion_matrix_LogisticRegression.png
        ├── confusion_matrix_MultinomialNB.png
        ├── f1_per_class_LogisticRegression.png
        └── f1_per_class_MultinomialNB.png
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Synthetic Data (No Downloads Needed)

```bash
python main.py
```

### 3. Run with Your Own Dataset

```bash
python main.py --data path/to/your_dataset.csv --text-col text --label-col emotion
```

### 4. Use the Predictor Programmatically

```python
from src.predict import predict_emotion

result = predict_emotion("I'm so happy today!")
print(result["label"])        # "happy"
print(result["confidence"])   # 0.87
```

## Supported Emotion Classes

| Emotion    | Description                          |
|------------|--------------------------------------|
| Happy      | Joy, excitement, gratitude           |
| Sad        | Grief, loneliness, disappointment    |
| Angry      | Frustration, outrage, irritation     |
| Surprise   | Shock, disbelief, astonishment       |
| Fear       | Anxiety, terror, dread               |
| Neutral    | Factual, unemotional, informational  |

## CLI Options

| Flag              | Default     | Description                          |
|-------------------|-------------|--------------------------------------|
| `--data`          | (synthetic) | Path to CSV dataset                  |
| `--text-col`      | `text`      | Name of the text column              |
| `--label-col`     | `emotion`   | Name of the label column             |
| `--output-dir`    | `outputs`   | Where to save models & plots         |
| `--max-features`  | `10000`     | TF-IDF vocabulary size               |
| `--no-cross-val`  | `False`     | Skip 5-fold cross-validation         |

## Recommended Datasets

To replace the synthetic data with a real-world dataset:

- **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** — 20K tweets labeled with 6 emotions
- **[Kaggle Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)** — Text samples with emotion labels
- **[GoEmotions (Google)](https://github.com/google-research/google-research/tree/master/goemotions)** — 58K Reddit comments, 27 emotion categories

## Tech Stack

- **Python 3.9+**
- **scikit-learn** — TF-IDF, Logistic Regression, Multinomial NB
- **NLTK** — Tokenization, stopwords, lemmatization
- **Matplotlib + Seaborn** — Evaluation visualizations
- **joblib** — Model serialization
