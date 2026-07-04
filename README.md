<div align="center">
  <h1>🧠 AI-Powered Emotion Detection from Text</h1>
  <p><i>An End-to-End NLP Pipeline & Interactive Prototype</i></p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
    <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/NLTK-3.8-154f5b?style=for-the-badge" alt="NLTK" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
    <img src="https://img.shields.io/badge/Status-Internship_Prototype-success?style=for-the-badge" alt="Status" />
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License" />
  </p>
</div>

---

## 📖 Project Overview
Understanding the underlying emotion of text is a critical component of modern Natural Language Processing (NLP). This project provides a robust, end-to-end Machine Learning prototype capable of reading raw sentences and predicting their underlying emotional category across six key dimensions: **Happy**, **Sad**, **Angry**, **Fear**, **Surprise**, and **Neutral**. 

Built as an internship project, it demonstrates best practices in data handling, modular architecture, hyperparameter tuning, and defensive programming for ML pipelines.

---

## ⚡ Feature Architecture & System Workflow

### ✨ Key Features
* 🧹 **Robust Preprocessing Pipeline:** Automated cleaning, URL stripping, custom regex filtering, and NLTK-based lemmatization.
* 📊 **TF-IDF Feature Extraction:** High-performance vectorization capturing bi-gram context and mitigating rare-word drops.
* 🚀 **Modular & Extensible Design:** Clean separation of concerns across data loading, training, evaluation, and inference.
* 💻 **Interactive Real-Time Terminal Mode:** An interactive shell wrapper to test custom sentences against the deployed model in real-time.
* 🛡️ **Defensive OOV Handling:** Built-in safeguards against out-of-vocabulary anomalies and uniform probability distributions.

### 🔄 Text Preprocessing Visual Pipeline
```text
Raw Text ➔ Regex Cleaning ➔ Tokenization ➔ Stopword Removal ➔ Lemmatization ➔ Vectorization
```

---

## ⚙️ Core Technical Specifications

### 🛠️ Tech Stack

| Category | Tool / Library | Purpose in Project |
| :--- | :--- | :--- |
| **Language** | Python | Core programming language for all logic |
| **ML Engine** | Scikit-Learn | Training models (Logistic Regression, Naive Bayes), TF-IDF, Metrics |
| **NLP Engine** | NLTK | Stopword corpus, advanced tokenization, WordNet lemmatization |
| **Data Handling** | Pandas & NumPy | High-performance dataset manipulation and vector math |
| **Visualization** | Matplotlib & Seaborn | Generating confusion matrices and class-metric bar charts |
| **Persistence** | Joblib | Saving/loading fitted models and vocabulary vectorizers |

### 🧠 Model Baseline
The core architecture leverages a **TF-IDF Vectorizer** paired with **Logistic Regression** (configured with `class_weight='balanced'` and optimized `C=10.0`). 
This architecture was selected over heavier deep learning approaches (like Transformers) because it provides a highly efficient, CPU-friendly baseline that is blazingly fast to train, perfectly interpretable, and establishes a strong benchmark metric before scaling up computational resources.

---

## 🚀 Setup, Installation & Usage Instructions

Follow these steps to deploy and test the pipeline on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Pipeline & Interactive UI
Running the main script automatically triggers the full data preprocessing, training, evaluation, and then drops you into a real-time interactive terminal mode to test the model.
```bash
python main.py
```

---

## 🎯 Highlight: Production Bug Fixes & Engineering Maturity

Throughout development, several complex bugs were identified and successfully patched, ensuring production-grade resilience:

> **Bug 1: Out-of-Vocabulary Uniform Probability Glitch**
> * **Issue:** When the model encountered entirely unseen strings (e.g., highly colloquial slang), the vectorizer outputted an empty matrix. This forced the classifier to split probabilities equally (16.7% across 6 classes) and lazily default to the first alphabetical class (ANGRY), generating false positives.
> * **Patch:** Engineered a strict fallback handling mechanism inside `predict.py`. If the incoming matrix is entirely zeros, or if the delta between `max(proba)` and `min(proba)` falls below a 0.001 threshold, the inference engine dynamically intercepts the prediction and routes it to `Neutral / Inconclusive (OOV)`.

> **Bug 2: Inference Vectorizer Scope Isolation**
> * **Issue:** The inference loop was originally instantiating an empty, unfitted vectorizer, causing all live user inputs to return zero-vectors and fail classification.
> * **Patch:** Refactored the architecture to ensure global vectorizer scope. The `TfidfVectorizer` is explicitly fitted **once** during the `train.py` lifecycle, serialized via Joblib, and safely loaded/passed into the `EmotionPredictor` instance. Live inputs exclusively utilize the `.transform()` method to maintain vocabulary alignment.

---

## 📂 Deliverables & Directory Structure

```text
emotion-detection/
├── main.py                     # E2E pipeline orchestrator & interactive demo
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── report_template.md          # Internship report structure
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Regex cleaning, NLTK tokenization/lemmatization
│   ├── features.py             # TF-IDF vectorization and serialization
│   ├── train.py                # Model training, CV, hyperparameter tuning
│   ├── predict.py              # Inference pipeline with OOV safety fallback
│   └── visualize.py            # Confusion matrix & distribution plot generators
└── outputs/
    ├── best_model.joblib       # Serialized trained model
    ├── tfidf_vectorizer.joblib # Serialized vocabulary state
    └── plots/                  # Generated evaluation charts
```
