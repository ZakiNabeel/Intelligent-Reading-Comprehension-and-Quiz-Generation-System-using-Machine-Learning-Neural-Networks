"""
evaluate.py — Centralized evaluation utilities for Model A and Model B.

Covers:
  Model A  — Accuracy, Macro F1, Precision, Recall, Exact Match (EM), Confusion Matrix
  Model B  — Distractor: Precision, Recall, F1, Accuracy, Confusion Matrix
           — Hint scorer: R² Score
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    r2_score,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared Utilities
# ═══════════════════════════════════════════════════════════════════════════

def exact_match(y_true, y_pred):
    """Strict character-level exact match rate (case-insensitive strip)."""
    matches = sum(
        str(t).strip().lower() == str(p).strip().lower()
        for t, p in zip(y_true, y_pred)
    )
    return matches / max(len(y_true), 1)


def _print_divider(label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print('=' * 60)


def _tokenize_for_text_metric(text):
    """Tokenize text for BLEU, ROUGE-L, and METEOR style metrics."""
    import re
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _safe_divide(numerator, denominator):
    """Avoid zero-division in text metrics."""
    return numerator / denominator if denominator else 0.0


def compute_bleu(reference, hypothesis):
    """
    Compute sentence BLEU.

    Uses NLTK when available, with a small unigram BLEU fallback so evaluation
    still produces a score without optional packages.
    """
    ref_tokens = _tokenize_for_text_metric(reference)
    hyp_tokens = _tokenize_for_text_metric(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return float(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        overlap = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
        precision = _safe_divide(overlap, len(hyp_tokens))
        brevity_penalty = min(1.0, np.exp(1 - _safe_divide(len(ref_tokens), len(hyp_tokens))))
        return float(brevity_penalty * precision)


def compute_rouge_l(reference, hypothesis):
    """
    Compute ROUGE-L F1 using longest common subsequence.

    This implementation has no external dependency.
    """
    ref_tokens = _tokenize_for_text_metric(reference)
    hyp_tokens = _tokenize_for_text_metric(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    prev = [0] * (len(hyp_tokens) + 1)
    for ref_token in ref_tokens:
        curr = [0]
        for idx, hyp_token in enumerate(hyp_tokens, start=1):
            if ref_token == hyp_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr

    lcs = prev[-1]
    precision = _safe_divide(lcs, len(hyp_tokens))
    recall = _safe_divide(lcs, len(ref_tokens))
    return float(_safe_divide(2 * precision * recall, precision + recall))


def compute_meteor(reference, hypothesis):
    """
    Compute METEOR.

    Uses NLTK METEOR when available. If NLTK or its WordNet resources are not
    installed, falls back to the standard METEOR-style harmonic mean with
    exact-token overlap and fragmentation penalty.
    """
    ref_tokens = _tokenize_for_text_metric(reference)
    hyp_tokens = _tokenize_for_text_metric(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    try:
        from nltk.translate.meteor_score import meteor_score

        return float(meteor_score([ref_tokens], hyp_tokens))
    except Exception:
        ref_counts = Counter(ref_tokens)
        matches = []
        used_ref_positions = set()
        for hyp_index, token in enumerate(hyp_tokens):
            if ref_counts[token] <= 0:
                continue
            for ref_index, ref_token in enumerate(ref_tokens):
                if ref_index not in used_ref_positions and token == ref_token:
                    used_ref_positions.add(ref_index)
                    ref_counts[token] -= 1
                    matches.append((hyp_index, ref_index))
                    break

        match_count = len(matches)
        if match_count == 0:
            return 0.0

        precision = _safe_divide(match_count, len(hyp_tokens))
        recall = _safe_divide(match_count, len(ref_tokens))
        f_mean = _safe_divide(10 * precision * recall, recall + 9 * precision)

        matches.sort()
        chunks = 1
        for idx in range(1, len(matches)):
            prev_hyp, prev_ref = matches[idx - 1]
            curr_hyp, curr_ref = matches[idx]
            if curr_hyp != prev_hyp + 1 or curr_ref != prev_ref + 1:
                chunks += 1

        penalty = 0.5 * (_safe_divide(chunks, match_count) ** 3)
        return float((1 - penalty) * f_mean)


def evaluate_question_text_metrics(reference_questions, generated_questions):
    """
    Evaluate generated questions with BLEU, ROUGE-L, and METEOR.

    Returns a DataFrame with per-question scores plus the generated/reference
    text, so it can be saved or averaged by callers.
    """
    rows = []
    for reference, generated in zip(reference_questions, generated_questions):
        rows.append({
            "reference_question": reference,
            "generated_question": generated,
            "bleu": compute_bleu(reference, generated),
            "rouge_l": compute_rouge_l(reference, generated),
            "meteor": compute_meteor(reference, generated),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Model A — Answer Verification
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model_a(y_true, y_pred, model_name="Model A"):
    """
    Evaluate Model A answer verification predictions.

    Parameters
    ----------
    y_true      : array-like — Ground truth binary labels (1=correct, 0=incorrect).
    y_pred      : array-like — Predicted labels.
    model_name  : str        — Label for printing.

    Returns
    -------
    dict with keys: accuracy, macro_f1, precision, recall, exact_match,
                    confusion_matrix (list), classification_report (str).
    """
    _print_divider(f"{model_name} — Answer Verification")

    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    em        = exact_match(y_true, y_pred)
    cm        = confusion_matrix(y_true, y_pred)
    report    = classification_report(y_true, y_pred, zero_division=0)

    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Exact Match: {em:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "model":                  model_name,
        "accuracy":               acc,
        "macro_f1":               macro_f1,
        "precision":              precision,
        "recall":                 recall,
        "exact_match":            em,
        "confusion_matrix":       cm.tolist(),
        "classification_report":  report,
    }


def compare_model_a_results(results_list):
    """
    Print a comparison table of multiple Model A result dicts.

    Parameters
    ----------
    results_list : list of dicts returned by evaluate_model_a().
    """
    _print_divider("Model A — Comparison Table")
    rows = []
    for r in results_list:
        rows.append({
            "Model":       r["model"],
            "Accuracy":    f"{r['accuracy']:.4f}",
            "Macro F1":    f"{r['macro_f1']:.4f}",
            "Precision":   f"{r['precision']:.4f}",
            "Recall":      f"{r['recall']:.4f}",
            "Exact Match": f"{r['exact_match']:.4f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Model B — Distractor Generation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model_b_distractors(generated_distractors_list, correct_answers_list):
    """
    Evaluate distractor generation quality.

    A distractor is valid (label=1) if it differs from the correct answer
    and does not contain placeholder text such as 'No suitable distractor'.

    Parameters
    ----------
    generated_distractors_list : list of lists — Distractors per sample.
    correct_answers_list       : list of str   — Correct answer per sample.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    total_distractors, valid_distractors.
    """
    _print_divider("Model B — Distractor Evaluation")

    y_true, y_pred = [], []

    for distractors, correct in zip(generated_distractors_list, correct_answers_list):
        correct_clean = str(correct).strip().lower()
        for d in distractors:
            d_clean = str(d).strip().lower()
            is_valid = int(
                d_clean != correct_clean
                and "no suitable" not in d_clean
                and "unavailable" not in d_clean
            )
            y_pred.append(is_valid)
            y_true.append(1)  # All should be valid distractors

    total = len(y_true)
    valid = sum(y_pred)

    accuracy  = valid / max(total, 1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"Distractor Accuracy: {accuracy:.4f}  ({valid}/{total} valid)")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Confusion Matrix (rows=true, cols=pred):\n{cm}")

    return {
        "accuracy":          accuracy,
        "precision":         precision,
        "recall":            recall,
        "f1":                f1,
        "confusion_matrix":  cm.tolist(),
        "total_distractors": total,
        "valid_distractors": valid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model B — Hint Relevance (R² Score)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_hint_r2(predicted_scores, true_scores):
    """
    R² Score for a regression-based hint relevance scorer.

    Parameters
    ----------
    predicted_scores : array-like — Model-predicted relevance scores.
    true_scores      : array-like — Ground-truth relevance labels.

    Returns
    -------
    float — R² score (1.0 = perfect, 0.0 = baseline mean predictor).
    """
    if len(predicted_scores) < 2:
        print("Not enough samples to compute R².")
        return 0.0

    r2 = r2_score(true_scores, predicted_scores)
    print(f"\nHint Scorer R² Score: {r2:.4f}")
    return r2


# ═══════════════════════════════════════════════════════════════════════════
# Session Metrics Helper (used by the Streamlit dashboard)
# ═══════════════════════════════════════════════════════════════════════════

def compute_session_metrics(history):
    """
    Derive live Model A accuracy and distractor quality from session history.

    Parameters
    ----------
    history : list of dicts (st.session_state.history entries).

    Returns
    -------
    dict — user_accuracy, model_a_accuracy, total_attempts, or None if empty.
    """
    if not history:
        return None

    total          = len(history)
    user_correct   = sum(1 for h in history if h.get("is_correct"))
    model_a_match  = sum(
        1 for h in history
        if h.get("model_a_prediction") == h.get("correct_label")
    )

    return {
        "total_attempts":  total,
        "user_accuracy":   user_correct  / total,
        "model_a_accuracy": model_a_match / total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Standalone demo
# ═══════════════════════════════════════════════════════════════════════════

def main():
    BASE_DIR  = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models" / "model_a" / "traditional"
    PROC_DIR  = BASE_DIR / "data" / "processed"

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import joblib
    from inference import predict_best_answer

    dev_path = PROC_DIR / "dev_model_a.csv"
    if not dev_path.exists():
        print("dev_model_a.csv not found — run preprocessing.py first.")
        return

    print("Loading dev data …")
    dev_df = pd.read_csv(dev_path)
    sample = dev_df.sample(min(500, len(dev_df)), random_state=42)

    logistic = joblib.load(MODEL_DIR / "logistic_regression.pkl")
    svm      = joblib.load(MODEL_DIR / "linear_svm.pkl")
    vec      = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

    from scipy.sparse import hstack, csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    def build_X(df):
        texts = df["article"] + " " + df["article"] + " " + df["question"] + " " + df["option_text"]
        tfidf = vec.transform(texts)
        q_opt  = cos_sim(vec.transform(df["question"]),    vec.transform(df["option_text"])).diagonal()
        a_opt  = cos_sim(vec.transform(df["article"]),     vec.transform(df["option_text"])).diagonal()
        a_q    = cos_sim(vec.transform(df["article"]),     vec.transform(df["question"])).diagonal()
        cosine = np.vstack([q_opt, a_opt, a_q]).T
        return hstack([tfidf, cosine])

    X = build_X(sample)
    y = sample["label"].values

    lr_preds  = logistic.predict(X)
    svm_preds = svm.predict(X)

    lr_results  = evaluate_model_a(y, lr_preds,  "Logistic Regression")
    svm_results = evaluate_model_a(y, svm_preds, "Linear SVM")

    compare_model_a_results([lr_results, svm_results])

    # Model B distractor demo
    raw_path = BASE_DIR / "data" / "raw" / "dev.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path).sample(min(50, len(pd.read_csv(raw_path))), random_state=42)
        from model_b_inference import generate_distractors
        distractors_list = []
        correct_list     = []
        for _, row in raw.iterrows():
            correct = row[row["answer"]]
            dists   = generate_distractors(row["article"], correct)
            distractors_list.append(dists)
            correct_list.append(correct)
        evaluate_model_b_distractors(distractors_list, correct_list)


if __name__ == "__main__":
    main()
