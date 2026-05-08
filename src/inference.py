import joblib
import numpy as np

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "model_a" / "traditional"


# Load trained models
logistic_model = joblib.load(
    MODEL_DIR / "logistic_regression.pkl"
)

svm_model = joblib.load(
    MODEL_DIR / "linear_svm.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


def build_combined_text(article, question, option_text):

    return (
        article + " " +
        article + " " +
        question + " " +
        option_text
    )


def compute_cosine_features(article, question, option_text):

    article_vec = vectorizer.transform([article])

    question_vec = vectorizer.transform([question])

    option_vec = vectorizer.transform([option_text])

    q_opt = cosine_similarity(
        question_vec,
        option_vec
    )[0][0]

    art_opt = cosine_similarity(
        article_vec,
        option_vec
    )[0][0]

    art_q = cosine_similarity(
        article_vec,
        question_vec
    )[0][0]

    return np.array([
        q_opt,
        art_opt,
        art_q
    ]).reshape(1, -1)


def prepare_features(article, question, option_text):

    combined_text = build_combined_text(
        article,
        question,
        option_text
    )

    tfidf_features = vectorizer.transform([combined_text])

    cosine_features = compute_cosine_features(
        article,
        question,
        option_text
    )

    final_features = hstack([
        tfidf_features,
        cosine_features
    ])

    return final_features


def predict_best_answer(article, question, options):

    results = {}

    for label, option_text in options.items():

        features = prepare_features(
            article,
            question,
            option_text
        )

        # Logistic Regression probability
        lr_prob = logistic_model.predict_proba(features)[0][1]

        # SVM prediction
        svm_pred = svm_model.decision_function(features)[0]

        # Simple ensemble score
        svm_score = 1 / (1 + np.exp(-svm_pred))

        final_score = (
            0.6 * lr_prob +
            0.4 * svm_score
        )

        results[label] = final_score

    best_answer = max(results, key=results.get)
    confidence = results[best_answer]

    return best_answer, confidence, results


def main():

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