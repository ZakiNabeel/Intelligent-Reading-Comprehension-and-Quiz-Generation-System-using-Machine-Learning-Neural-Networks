import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    silhouette_score
)

from scipy.sparse import hstack

import matplotlib.pyplot as plt
import seaborn as sns

import joblib


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models" / "model_a" / "traditional"

PLOT_DIR = BASE_DIR / "models" / "model_a" / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    train_df = pd.read_csv(
        PROCESSED_DIR / "train_model_a.csv"
    )

    dev_df = pd.read_csv(
        PROCESSED_DIR / "dev_model_a.csv"
    )

    return train_df, dev_df


# =========================================================
# BUILD COMBINED TEXT
# =========================================================

def build_combined_text(df):

    return (
        df["article"] + " " +
        df["article"] + " " +
        df["question"] + " " +
        df["option_text"]
    )


# =========================================================
# TF-IDF FEATURES
# =========================================================

def create_tfidf_features(train_texts, dev_texts):

    vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(
        train_texts
    )

    X_dev_tfidf = vectorizer.transform(
        dev_texts
    )

    return (
        X_train_tfidf,
        X_dev_tfidf,
        vectorizer
    )


# =========================================================
# COSINE FEATURES
# =========================================================

def compute_cosine_features(df, vectorizer):

    question_vectors = vectorizer.transform(
        df["question"]
    )

    option_vectors = vectorizer.transform(
        df["option_text"]
    )

    article_vectors = vectorizer.transform(
        df["article"]
    )

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


# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================

def plot_confusion_matrix(cm, model_name):

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(
        PLOT_DIR / f"{model_name}_confusion_matrix.png"
    )

    plt.close()


# =========================================================
# ROC CURVE
# =========================================================

def plot_roc_curve(y_dev, probs, model_name):

    fpr, tpr, _ = roc_curve(
        y_dev,
        probs
    )

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"{model_name} ROC Curve")

    plt.savefig(
        PLOT_DIR / f"{model_name}_roc_curve.png"
    )

    plt.close()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_names(vectorizer):

    tfidf_features = vectorizer.get_feature_names_out()

    cosine_features = np.array([
        "cosine_question_option",
        "cosine_article_option",
        "cosine_article_question"
    ])

    return np.concatenate([
        tfidf_features,
        cosine_features
    ])


def show_top_features(model, vectorizer, top_n=20):

    feature_names = get_feature_names(vectorizer)

    coefficients = model.coef_[0]

    if coefficients.shape[0] != feature_names.shape[0]:
        usable_count = min(
            coefficients.shape[0],
            feature_names.shape[0]
        )

        print(
            "\nWarning: feature name count does not match coefficient count. "
            f"Using first {usable_count} features."
        )

        feature_names = feature_names[:usable_count]
        coefficients = coefficients[:usable_count]

    top_positive = np.argsort(coefficients)[-top_n:]

    top_negative = np.argsort(coefficients)[:top_n]

    print("\nTop Positive Features:\n")

    for idx in reversed(top_positive):
        print(
            f"{feature_names[idx]} : {coefficients[idx]:.4f}"
        )

    print("\nTop Negative Features:\n")

    for idx in top_negative:
        print(
            f"{feature_names[idx]} : {coefficients[idx]:.4f}"
        )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    name,
    model,
    X_dev,
    y_dev,
    vectorizer
):

    predictions = model.predict(X_dev)

    accuracy = accuracy_score(
        y_dev,
        predictions
    )

    macro_f1 = f1_score(
        y_dev,
        predictions,
        average='macro'
    )

    print(f"\n{name} Results")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    try:

        probs = model.predict_proba(X_dev)[:, 1]

        roc_auc = roc_auc_score(
            y_dev,
            probs
        )

        print(f"ROC-AUC Score: {roc_auc:.4f}")

        plot_roc_curve(
            y_dev,
            probs,
            name
        )

    except:
        print("ROC-AUC not available.")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_dev,
            predictions,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_dev,
        predictions
    )

    print("\nConfusion Matrix:\n")
    print(cm)

    plot_confusion_matrix(
        cm,
        name
    )

    if hasattr(model, "coef_"):
        show_top_features(
            model,
            vectorizer
        )


# =========================================================
# KMEANS
# =========================================================

def run_kmeans(X_train_tfidf):

    print("\nRunning KMeans Clustering...")

    kmeans = KMeans(
        n_clusters=2,
        random_state=42
    )

    cluster_labels = kmeans.fit_predict(
        X_train_tfidf
    )

    sil_score = silhouette_score(
        X_train_tfidf,
        cluster_labels
    )

    print(
        f"Silhouette Score: {sil_score:.4f}"
    )

    return kmeans


# =========================================================
# LABEL PROPAGATION (Semi-Supervised)
# =========================================================

def run_label_propagation(X_train_tfidf, y_train, unlabeled_fraction=0.3, sample_size=3000):
    """
    Semi-supervised Label Propagation.

    Simulates a semi-supervised scenario by masking `unlabeled_fraction` of
    training labels (set to -1).  LabelPropagation then propagates labels from
    the known examples to the unlabeled ones via a k-NN graph.

    Evaluation is reported on the originally-labeled subset only, so we can
    compare fairly against fully-supervised baselines.
    """

    print("\nRunning Label Propagation (Semi-Supervised)...")

    # Work on a manageable sample (dense matrix required)
    n = min(sample_size, len(y_train))
    idx = np.random.RandomState(42).choice(len(y_train), n, replace=False)

    X_dense = X_train_tfidf[idx].toarray()
    y_sample = y_train.values[idx].copy()

    # Mask a fraction of labels to simulate unlabeled data
    n_unlabeled = int(n * unlabeled_fraction)
    rng = np.random.RandomState(0)
    unlabeled_idx = rng.choice(n, n_unlabeled, replace=False)
    y_semi = y_sample.copy()
    y_semi[unlabeled_idx] = -1  # -1 = unlabeled

    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=200)
    lp.fit(X_dense, y_semi)

    labeled_mask = y_semi != -1
    lp_preds = lp.predict(X_dense[labeled_mask])
    true_labels = y_sample[labeled_mask]

    lp_acc = accuracy_score(true_labels, lp_preds)
    lp_f1  = f1_score(true_labels, lp_preds, average="macro", zero_division=0)
    lp_cm  = confusion_matrix(true_labels, lp_preds)

    print(f"Label Propagation — labeled subset size : {labeled_mask.sum()}")
    print(f"Label Propagation — unlabeled subset    : {n_unlabeled}")
    print(f"Label Propagation — Accuracy            : {lp_acc:.4f}")
    print(f"Label Propagation — Macro F1            : {lp_f1:.4f}")
    print(f"Label Propagation — Confusion Matrix:\n{lp_cm}")

    return lp, {"accuracy": lp_acc, "macro_f1": lp_f1, "confusion_matrix": lp_cm}


# =========================================================
# GAUSSIAN MIXTURE MODEL (Unsupervised Clustering)
# =========================================================

def run_gmm(X_train_tfidf, n_components=2, sample_size=3000):
    """
    Gaussian Mixture Model for unsupervised question-answer clustering.

    Uses a dense sample of the TF-IDF matrix.  Reports silhouette score and
    log-likelihood so results can be compared against KMeans clustering.
    """

    print("\nRunning Gaussian Mixture Model (GMM) Clustering...")

    n = min(sample_size, X_train_tfidf.shape[0])
    idx = np.random.RandomState(42).choice(X_train_tfidf.shape[0], n, replace=False)
    X_dense = X_train_tfidf[idx].toarray()

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",   # memory-efficient for high-dim TF-IDF
        max_iter=200,
        random_state=42,
    )
    gmm.fit(X_dense)
    cluster_labels = gmm.predict(X_dense)

    sil = silhouette_score(X_dense, cluster_labels)

    print(f"GMM Converged        : {gmm.converged_}")
    print(f"GMM Silhouette Score : {sil:.4f}")
    print(f"GMM Log-Likelihood   : {gmm.lower_bound_:.4f}")

    cluster_counts = np.bincount(cluster_labels, minlength=n_components)
    for i, cnt in enumerate(cluster_counts):
        print(f"  Cluster {i}: {cnt} samples")

    return gmm, {"silhouette_score": sil, "converged": gmm.converged_}


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

def tune_logistic_regression(X_train, y_train):

    print("\nRunning GridSearchCV...")

    param_grid = {
        'C': [0.1, 1, 5]
    }

    grid = GridSearchCV(
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        ),
        param_grid,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    print("\nBest Parameters:")
    print(grid.best_params_)

    print(
        f"Best CV Score: {grid.best_score_:.4f}"
    )

    return grid.best_estimator_


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading processed data...")

    train_df, dev_df = load_data()

    print("Building combined text...")

    train_texts = build_combined_text(
        train_df
    )

    dev_texts = build_combined_text(
        dev_df
    )

    print("Creating TF-IDF vectors...")

    (
        X_train_tfidf,
        X_dev_tfidf,
        vectorizer
    ) = create_tfidf_features(
        train_texts,
        dev_texts
    )

    print("Computing cosine features...")

    train_cosine = compute_cosine_features(
        train_df,
        vectorizer
    )

    dev_cosine = compute_cosine_features(
        dev_df,
        vectorizer
    )

    print("Combining features...")

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

    # =====================================================
    # TUNED LOGISTIC REGRESSION
    # =====================================================

    logistic_model = tune_logistic_regression(
        X_train,
        y_train
    )

    evaluate_model(
        "Logistic_Regression",
        logistic_model,
        X_dev,
        y_dev,
        vectorizer
    )

    # =====================================================
    # SVM
    # =====================================================

    print("\nTraining Linear SVM...")

    svm_model = LinearSVC(
        class_weight='balanced'
    )

    svm_model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        "Linear_SVM",
        svm_model,
        X_dev,
        y_dev,
        vectorizer
    )

    # =====================================================
    # KMEANS
    # =====================================================

    kmeans = run_kmeans(
        X_train_tfidf
    )

    # =====================================================
    # LABEL PROPAGATION (Semi-Supervised)
    # =====================================================

    lp_model, lp_metrics = run_label_propagation(
        X_train_tfidf,
        y_train
    )

    # =====================================================
    # GAUSSIAN MIXTURE MODEL
    # =====================================================

    gmm_model, gmm_metrics = run_gmm(
        X_train_tfidf
    )

    # =====================================================
    # COMPARISON TABLE
    # =====================================================

    print("\n\nModel Comparison Table")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 70)

    for name, model in [("Logistic Regression", logistic_model), ("Linear SVM", svm_model)]:
        preds = model.predict(X_dev)
        acc   = accuracy_score(y_dev, preds)
        f1    = f1_score(y_dev, preds, average="macro", zero_division=0)
        print(f"{name:<30} {acc:>10.4f} {f1:>10.4f}")

    lp_acc = lp_metrics["accuracy"]
    lp_f1  = lp_metrics["macro_f1"]
    print(f"{'Label Propagation (semi-sup)':<30} {lp_acc:>10.4f} {lp_f1:>10.4f}")

    gmm_sil = gmm_metrics["silhouette_score"]
    print(f"{'GMM Clustering':<30} {'N/A (unsup)':>10} {gmm_sil:>10.4f}  ← silhouette")
    print("-" * 70)

    # =====================================================
    # SAVE MODELS
    # =====================================================

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

    joblib.dump(
        kmeans,
        MODEL_DIR / "kmeans.pkl"
    )

    joblib.dump(
        lp_model,
        MODEL_DIR / "label_propagation.pkl"
    )

    joblib.dump(
        gmm_model,
        MODEL_DIR / "gmm.pkl"
    )

    print("\nAll models saved.")


if __name__ == "__main__":
    main()
