import joblib
import numpy as np
import pandas as pd

from typing import Dict, Tuple
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "model_a" / "traditional"

# Models are loaded once at import time; all inference calls share these objects.
logistic_model = joblib.load(MODEL_DIR / "logistic_regression.pkl")
svm_model      = joblib.load(MODEL_DIR / "linear_svm.pkl")
vectorizer     = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

# Article is repeated to up-weight passage tokens relative to question/option tokens.
# Empirically this improves TF-IDF cosine alignment for answer verification.
LR_WEIGHT  = 0.6   # Logistic Regression contributes 60% — it provides calibrated probabilities.
SVM_WEIGHT = 0.4   # SVM contributes 40% — it provides a hard margin discriminant score.


def safe_text(value) -> str:
    """Convert missing pandas/NumPy values into strings accepted by TF-IDF."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def build_combined_text(article: str, question: str, option_text: str) -> str:
    # Article is duplicated so TF-IDF weights passage vocabulary more heavily.
    article = safe_text(article)
    question = safe_text(question)
    option_text = safe_text(option_text)
    return f"{article} {article} {question} {option_text}"


def compute_cosine_features(article: str, question: str, option_text: str) -> np.ndarray:
    """Return shape (1, 3): [sim(q,opt), sim(art,opt), sim(art,q)]."""
    article = safe_text(article)
    question = safe_text(question)
    option_text = safe_text(option_text)
    vecs = vectorizer.transform([article, question, option_text])
    article_vec, question_vec, option_vec = vecs[0], vecs[1], vecs[2]

    q_opt   = cosine_similarity(question_vec, option_vec)[0][0]   # question–option overlap
    art_opt = cosine_similarity(article_vec,  option_vec)[0][0]   # article–option overlap (key signal)
    art_q   = cosine_similarity(article_vec,  question_vec)[0][0] # article–question relevance

    return np.array([q_opt, art_opt, art_q]).reshape(1, -1)


def prepare_features(article: str, question: str, option_text: str) -> csr_matrix:
    """Concatenate TF-IDF sparse vector with 3 cosine similarity features."""
    article = safe_text(article)
    question = safe_text(question)
    option_text = safe_text(option_text)
    tfidf_features  = vectorizer.transform([build_combined_text(article, question, option_text)])
    cosine_features = compute_cosine_features(article, question, option_text)
    return hstack([tfidf_features, cosine_features])


def predict_best_answer(
    article: str, question: str, options: Dict[str, str]
) -> Tuple[str, float, Dict[str, float]]:
    """
    Score each option with the LR+SVM ensemble and return the highest-scoring label.

    SVM decision_function output is unbounded; we pass it through a sigmoid so
    both models contribute scores in [0, 1] before the weighted average.
    """
    results: Dict[str, float] = {}
    article = safe_text(article)
    question = safe_text(question)
    options = {str(label): safe_text(option_text) for label, option_text in options.items()}

    for label, option_text in options.items():
        features = prepare_features(article, question, option_text)

        lr_prob   = logistic_model.predict_proba(features)[0][1]
        svm_raw   = svm_model.decision_function(features)[0]
        svm_score = 1 / (1 + np.exp(-svm_raw))   # sigmoid normalisation

        results[label] = (LR_WEIGHT * lr_prob) + (SVM_WEIGHT * svm_score)

    best_answer = max(results, key=results.get)
    confidence  = results[best_answer]
    return best_answer, confidence, results


def main() -> None:
    """Main function to demonstrate the prediction pipeline."""
    article = """
    The Earth revolves around the Sun once every year.
    This movement causes seasons and changes in daylight.
    """

    question = "What does the Earth revolve around?"

    options = {
        "A": "The Moon",
        "B": "Mars",
        "C": "The Sun",
        "D": "Jupiter"
    }

    prediction, confidence, scores = predict_best_answer(
        article,
        question,
        options
    )

    print("\nPrediction Results")
    print("-" * 40)

    for label, score in scores.items():
        print(f"{label}: {score:.4f}")

    print(f"\nPredicted Answer: {prediction}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
