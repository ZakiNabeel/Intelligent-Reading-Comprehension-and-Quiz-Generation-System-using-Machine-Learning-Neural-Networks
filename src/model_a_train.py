import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models" / "model_a" / "traditional"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train_model_a.csv")
    dev_df = pd.read_csv(PROCESSED_DIR / "dev_model_a.csv")

    return train_df, dev_df


def vectorize_text(train_texts, dev_texts):
    """
    Convert text into TF-IDF vectors.
    """

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english'
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)

    return X_train, X_dev, vectorizer


def evaluate_model(name, model, X_dev, y_dev):
    """
    Print evaluation metrics.
    """

    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    f1 = f1_score(y_dev, predictions)

    print(f"\n{name} Results")
    print("-" * 40)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_dev, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_dev, predictions))


def main():

    print("Loading processed data...")

    train_df, dev_df = load_data()

    X_train_text = train_df["combined_text"]
    y_train = train_df["label"]

    X_dev_text = dev_df["combined_text"]
    y_dev = dev_df["label"]

    print("Vectorizing text with TF-IDF...")

    X_train, X_dev, vectorizer = vectorize_text(
        X_train_text,
        X_dev_text
    )

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