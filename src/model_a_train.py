import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from scipy.sparse import hstack

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models" / "model_a" / "traditional"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    train_df = pd.read_csv(PROCESSED_DIR / "train_model_a.csv")
    dev_df = pd.read_csv(PROCESSED_DIR / "dev_model_a.csv")

    return train_df, dev_df


def build_combined_text(df):
    """
    Give more importance to article text.

    article + article + question + option
    """

    return (
        df["article"] + " " +
        df["article"] + " " +
        df["question"] + " " +
        df["option_text"]
    )


def create_tfidf_features(train_texts, dev_texts):

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(train_texts)

    X_dev_tfidf = vectorizer.transform(dev_texts)

    return X_train_tfidf, X_dev_tfidf, vectorizer


def compute_cosine_features(df, vectorizer):
    """
    Cosine similarity between:
    question <-> option
    article <-> option
    article <-> question
    """

    question_vectors = vectorizer.transform(df["question"])
    option_vectors = vectorizer.transform(df["option_text"])
    article_vectors = vectorizer.transform(df["article"])

    q_opt = cosine_similarity(
        question_vectors,
        option_vectors
    ).diagonal()

    art_opt = cosine_similarity(
        article_vectors,
        option_vectors
    ).diagonal()

    art_q = cosine_similarity(
        article_vectors,
        question_vectors
    ).diagonal()

    cosine_features = np.vstack([
        q_opt,
        art_opt,
        art_q
    ]).T

    return cosine_features


def evaluate_model(name, model, X_dev, y_dev):

    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)

    f1 = f1_score(y_dev, predictions)

    print(f"\n{name} Results")
    print("-" * 50)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_dev, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_dev, predictions))


def main():

    print("Loading processed data...")

    train_df, dev_df = load_data()

    print("Building combined text...")

    train_texts = build_combined_text(train_df)

    dev_texts = build_combined_text(dev_df)

    print("Creating TF-IDF vectors...")

    X_train_tfidf, X_dev_tfidf, vectorizer = create_tfidf_features(
        train_texts,
        dev_texts
    )

    print("Computing cosine similarity features...")

    train_cosine = compute_cosine_features(
        train_df,
        vectorizer
    )

    dev_cosine = compute_cosine_features(
        dev_df,
        vectorizer
    )

    print("Combining TF-IDF + cosine features...")

    X_train = hstack([
        X_train_tfidf,
        train_cosine
    ])

    X_dev = hstack([
        X_dev_tfidf,
        dev_cosine
    ])

    y_train = train_df["label"]

    y_dev = dev_df["label"]

    print("\nTraining Logistic Regression...")

    logistic_model = LogisticRegression(
        max_iter=1000
    )

    logistic_model.fit(X_train, y_train)

    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_dev,
        y_dev
    )

    print("\nTraining Linear SVM...")

    svm_model = LinearSVC()

    svm_model.fit(X_train, y_train)

    evaluate_model(
        "Linear SVM",
        svm_model,
        X_dev,
        y_dev
    )

    print("\nSaving models...")

    joblib.dump(
        logistic_model,
        MODEL_DIR / "logistic_regression.pkl"
    )

    joblib.dump(
        svm_model,
        MODEL_DIR / "linear_svm.pkl"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    print("\nAll models saved successfully!")


if __name__ == "__main__":
    main()