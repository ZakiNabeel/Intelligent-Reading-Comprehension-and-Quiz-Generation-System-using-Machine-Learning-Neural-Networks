"""
model_b_train.py
================
Model B: Classical ML Distractor Generator for Reading Comprehension (RACE dataset).

Pipeline
--------
Given a passage, question, and correct answer, this model:
  1. Extracts candidate distractor phrases from the passage (rule-based, frequency-driven).
  2. Engineers features for each candidate using TF-IDF similarity, passage frequency,
     and character-level overlap — all computed without dense all-to-all matrices.
  3. Trains a RandomForestClassifier to rank candidates and outputs the top-3 distractors.

Constraints
-----------
  - No neural networks, transformers, or deep learning (no BERT/spaCy/HuggingFace).
  - Uses only: scikit-learn, pandas, numpy, joblib, re, collections (stdlib).
  - Memory-safe: uses sklearn.metrics.pairwise.paired_cosine_distances (O(n), not O(n²)).

Author  : Senior ML Engineer scaffold — adapt for your RACE lab setup.
Usage   : python model_b_train.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import re
import logging
import warnings
from collections import Counter
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline  # available for future extension

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Constants
# ─────────────────────────────────────────────────────────────────────────────

# Standard English stopwords (no external library required)
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "as", "is", "it", "its",
    "be", "am", "are", "was", "were", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "that", "this", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "than", "too", "very", "just", "with",
    "about", "after", "before", "between", "into", "through", "during",
    "from", "then", "there", "here", "because", "if", "while", "although",
    "however", "also", "any", "even", "over", "under", "again", "once",
}

# Maximum bigram/unigram candidates extracted per passage
DEFAULT_TOP_N = 20

# Model artifact paths
MODEL_SAVE_PATH = "model_b_ranker.joblib"
VECTORIZER_SAVE_PATH = "model_b_tfidf.joblib"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Candidate Extraction (Rule-Based)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_passage(passage: str) -> List[str]:
    """
    Lowercase and tokenize a passage into alphabetic words only.
    Strips punctuation and numeric tokens so frequency counts are clean.

    Parameters
    ----------
    passage : str
        Raw passage text from the RACE dataset.

    Returns
    -------
    List[str]
        Filtered list of lowercase alphabetic tokens.
    """
    # Keep only alphabetic characters and spaces; discard digits/punctuation
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", passage.lower())
    tokens = cleaned.split()
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove standard English stopwords from a token list.

    Parameters
    ----------
    tokens : List[str]
        Raw lowercase token list from tokenize_passage().

    Returns
    -------
    List[str]
        Tokens with stopwords removed; very short tokens (≤ 2 chars) also dropped.
    """
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def extract_bigrams(tokens: List[str]) -> List[str]:
    """
    Generate consecutive bigrams from a token list to capture two-word phrases.
    Example: ["machine", "learning"] → "machine learning"

    Parameters
    ----------
    tokens : List[str]
        Content word tokens (after stopword removal).

    Returns
    -------
    List[str]
        List of space-joined bigram strings.
    """
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def extract_candidates(
    passage: str,
    correct_answer: str,
    top_n: int = DEFAULT_TOP_N
) -> List[Tuple[str, int]]:
    """
    Extract the top-N most frequent unigram and bigram candidates from the passage,
    excluding the correct answer and its sub-tokens.

    Strategy
    --------
    - Tokenize & remove stopwords → content words.
    - Count frequency of unigrams + bigrams.
    - Filter out tokens/phrases that are sub-strings of the correct answer.
    - Return the top_n by frequency as (candidate_text, frequency) tuples.

    Parameters
    ----------
    passage       : str   — Full reading passage.
    correct_answer: str   — The ground-truth answer to exclude from candidates.
    top_n         : int   — Maximum number of candidates to return.

    Returns
    -------
    List[Tuple[str, int]]
        Sorted list of (candidate_phrase, passage_frequency) pairs.
    """
    tokens = tokenize_passage(passage)
    content_words = remove_stopwords(tokens)

    # Build a combined frequency map over unigrams and bigrams
    freq_map: Counter = Counter()
    freq_map.update(content_words)                   # unigrams
    freq_map.update(extract_bigrams(content_words))  # bigrams

    # Normalise the correct answer for exclusion checks
    answer_normalised = correct_answer.strip().lower()
    answer_tokens = set(tokenize_passage(answer_normalised))

    filtered_candidates = []
    for phrase, freq in freq_map.items():
        # Skip if the candidate IS the correct answer or fully contained within it
        if phrase in answer_normalised or answer_normalised in phrase:
            continue
        # Skip single tokens that are sub-components of the answer
        if phrase in answer_tokens:
            continue
        filtered_candidates.append((phrase, freq))

    # Sort by frequency descending, return top_n
    filtered_candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered_candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_tfidf_similarity(
    candidates: List[str],
    correct_answer: str,
    vectorizer: TfidfVectorizer
) -> np.ndarray:
    """
    Compute the cosine similarity between each candidate and the correct answer
    using PAIRED distances — O(n) memory, never builds an n×n matrix.

    Design note
    -----------
    paired_cosine_distances(A, B) returns 1 - cosine_similarity element-wise
    for rows A[i] and B[i]. We therefore tile the correct_answer vector to match
    the number of candidates and compute row-wise distances in one pass.

    Parameters
    ----------
    candidates     : List[str]  — Candidate distractor strings.
    correct_answer : str        — The ground-truth answer text.
    vectorizer     : TfidfVectorizer — Fitted TF-IDF vectorizer.

    Returns
    -------
    np.ndarray shape (n_candidates,)
        Cosine SIMILARITY scores in [0, 1].  Low score → good distractor.
    """
    if not candidates:
        return np.array([])

    # Stack candidates + the answer so they share the same vocabulary axis
    all_texts = candidates + [correct_answer]
    tfidf_matrix = vectorizer.transform(all_texts)  # sparse (n+1) × vocab

    # Split: candidate vectors vs the single answer vector
    candidate_vecs = tfidf_matrix[:-1]       # shape: (n_candidates, vocab)
    answer_vec_row = tfidf_matrix[-1]        # shape: (1, vocab)

    # Tile the answer vector to match candidate count for paired computation.
    # csr_matrix has no .repeat(); use scipy.sparse.vstack for memory-safe stacking.
    from scipy.sparse import vstack as sparse_vstack
    answer_tiled = sparse_vstack([answer_vec_row] * len(candidates))  # (n, vocab)

    # paired_cosine_distances returns DISTANCE (1 - similarity) for each pair
    distances = paired_cosine_distances(candidate_vecs, answer_tiled)  # (n,)
    similarities = 1.0 - distances  # convert to similarity

    return similarities  # shape: (n_candidates,)


def compute_char_overlap(candidate: str, correct_answer: str) -> float:
    """
    Compute a normalised character-level overlap score between a candidate
    and the correct answer using the Jaccard coefficient on character bigrams.

    A low overlap score suggests the candidate looks visually different from
    the answer (desirable for a plausible-but-wrong distractor).

    Parameters
    ----------
    candidate      : str — Candidate distractor text.
    correct_answer : str — Ground-truth answer text.

    Returns
    -------
    float
        Jaccard similarity in [0, 1].  0 = no shared bigrams; 1 = identical.
    """
    def char_bigrams(text: str):
        t = text.lower().replace(" ", "")
        return set(t[i:i+2] for i in range(len(t) - 1))

    cand_bg = char_bigrams(candidate)
    ans_bg  = char_bigrams(correct_answer)

    if not cand_bg and not ans_bg:
        return 1.0  # both empty — treat as identical
    if not cand_bg or not ans_bg:
        return 0.0  # one is empty

    intersection = cand_bg & ans_bg
    union        = cand_bg | ans_bg
    return len(intersection) / len(union)


def build_feature_matrix(
    candidates: List[str],
    passage_freqs: List[int],
    correct_answer: str,
    vectorizer: TfidfVectorizer
) -> pd.DataFrame:
    """
    Construct the 3-feature DataFrame used by the ML ranker.

    Features
    --------
    1. tfidf_similarity   : TF-IDF cosine similarity to correct answer.
                            Ideal range: low (0.0–0.3) — not too close to answer.
    2. passage_frequency  : Raw count of the phrase in the passage.
                            Higher = more plausible / topic-relevant.
    3. char_overlap       : Jaccard similarity on character bigrams vs answer.
                            Low = visually distinct (good distractor property).

    Parameters
    ----------
    candidates     : List[str]        — Extracted candidate phrases.
    passage_freqs  : List[int]        — Frequency of each candidate in the passage.
    correct_answer : str              — Ground-truth answer.
    vectorizer     : TfidfVectorizer  — Fitted vectorizer (must be pre-fitted).

    Returns
    -------
    pd.DataFrame
        Shape (n_candidates, 3) with columns:
        ['tfidf_similarity', 'passage_frequency', 'char_overlap']
    """
    tfidf_sims = compute_tfidf_similarity(candidates, correct_answer, vectorizer)

    char_overlaps = np.array([
        compute_char_overlap(c, correct_answer) for c in candidates
    ])

    feature_df = pd.DataFrame({
        "tfidf_similarity":  tfidf_sims,
        "passage_frequency": passage_freqs,
        "char_overlap":      char_overlaps,
    })

    return feature_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3a — Fit TF-IDF Vectorizer on the training corpus
# ─────────────────────────────────────────────────────────────────────────────

def fit_tfidf_vectorizer(corpus: List[str]) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on a mixed corpus of candidate and answer texts.
    Character n-gram range (2–4) is included alongside word unigrams so that
    morphological variants of words share signal.

    Parameters
    ----------
    corpus : List[str]
        All candidate texts + correct answer texts from the training set.

    Returns
    -------
    TfidfVectorizer
        Fitted vectorizer ready for .transform() calls.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character-level with word-boundary padding
        ngram_range=(2, 4),   # bigrams through 4-grams at char level
        max_features=8000,    # vocabulary cap for memory safety
        sublinear_tf=True,    # apply log(1 + tf) smoothing
        min_df=1,
    )
    vectorizer.fit(corpus)
    logger.info(f"TF-IDF vectorizer fitted | vocab size: {len(vectorizer.vocabulary_)}")
    return vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3b — Build Training Features from a Labelled DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def prepare_training_data(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Process a labelled DataFrame of (candidate, answer, passage) rows into
    feature vectors (X) and binary labels (y) for ML model training.

    Expected DataFrame columns
    --------------------------
    candidate_text          : str   — Extracted candidate phrase.
    correct_answer_text     : str   — Ground-truth answer for that example.
    passage_text            : str   — The reading passage this row comes from.
    is_actual_distractor    : int   — 1 = good distractor, 0 = bad/irrelevant.

    Parameters
    ----------
    df         : pd.DataFrame    — Labelled training data.
    vectorizer : TfidfVectorizer — Pre-fitted vectorizer.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X : Feature DataFrame with columns
            ['tfidf_similarity', 'passage_frequency', 'char_overlap'].
        y : Binary label Series (is_actual_distractor).
    """
    logger.info(f"Preparing training features for {len(df)} rows …")

    required_cols = {
        "candidate_text", "correct_answer_text",
        "passage_text", "is_actual_distractor"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    feature_rows = []

    for _, row in df.iterrows():
        candidate      = str(row["candidate_text"])
        correct_answer = str(row["correct_answer_text"])
        passage        = str(row["passage_text"])

        # Passage frequency: how many times does the candidate appear?
        passage_lower = passage.lower()
        freq = passage_lower.count(candidate.lower())

        # Compute features for this single candidate (list of one element)
        feat_df = build_feature_matrix(
            candidates     = [candidate],
            passage_freqs  = [freq],
            correct_answer = correct_answer,
            vectorizer     = vectorizer,
        )
        feature_rows.append(feat_df.iloc[0])

    X = pd.DataFrame(feature_rows).reset_index(drop=True)
    y = df["is_actual_distractor"].reset_index(drop=True)

    logger.info(f"Feature matrix shape: {X.shape} | Label distribution:\n{y.value_counts()}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3c — Train the ML Ranker
# ─────────────────────────────────────────────────────────────────────────────

def train_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest"
) -> RandomForestClassifier:
    """
    Train a binary classifier to distinguish good distractors from bad ones.

    Model options
    -------------
    "random_forest"  : RandomForestClassifier — handles non-linear feature
                       interactions; preferred when training data is large.
    "logistic"       : LogisticRegression — interpretable, fast; suitable
                       for smaller datasets or when explainability matters.

    Parameters
    ----------
    X          : pd.DataFrame  — Feature matrix from prepare_training_data().
    y          : pd.Series     — Binary labels (1 = good distractor).
    model_type : str           — One of {"random_forest", "logistic"}.

    Returns
    -------
    Fitted classifier with .predict_proba() support.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,        # enough trees for stable probability estimates
            max_depth=8,             # prevent overfitting on small training sets
            min_samples_leaf=3,
            class_weight="balanced", # handle label imbalance gracefully
            random_state=42,
            n_jobs=-1,               # parallelise tree building
        )
    elif model_type == "logistic":
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=42,
            solver="lbfgs",
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'random_forest' or 'logistic'.")

    logger.info(f"Training {model_type} on {len(X_train)} samples …")
    model.fit(X_train, y_train)

    # Validation report
    y_pred = model.predict(X_val)
    logger.info(
        f"\nValidation Report ({len(X_val)} samples):\n"
        + classification_report(y_val, y_pred, target_names=["bad", "good_distractor"])
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Inference: Generate Top-3 Distractors for a New Example
# ─────────────────────────────────────────────────────────────────────────────

def generate_distractors(
    passage: str,
    correct_answer: str,
    model,                      # fitted classifier
    vectorizer: TfidfVectorizer,
    top_n_candidates: int = DEFAULT_TOP_N,
    n_distractors: int = 3
) -> List[str]:
    """
    Full inference pipeline: extract candidates → compute features → rank → return top-3.

    The function guarantees that:
      - None of the returned distractors equals the correct answer.
      - Distractors are de-duplicated.
      - Results are ordered by predicted distractor probability (highest first).

    Parameters
    ----------
    passage          : str  — Reading passage.
    correct_answer   : str  — The correct answer to the question.
    model            : fitted sklearn classifier with predict_proba().
    vectorizer       : TfidfVectorizer — fitted, same instance used during training.
    top_n_candidates : int  — How many candidates to extract before ranking.
    n_distractors    : int  — Number of distractors to return (default 3).

    Returns
    -------
    List[str]
        Up to n_distractors distractor strings, ranked best-first.
        Returns an empty list if no valid candidates are found.
    """
    # 1. Extract candidates from passage
    candidates_with_freq = extract_candidates(passage, correct_answer, top_n=top_n_candidates)

    if not candidates_with_freq:
        logger.warning("No candidates extracted from passage.")
        return []

    candidates = [c for c, _ in candidates_with_freq]
    freqs      = [f for _, f in candidates_with_freq]

    # 2. Build feature matrix
    feat_df = build_feature_matrix(
        candidates     = candidates,
        passage_freqs  = freqs,
        correct_answer = correct_answer,
        vectorizer     = vectorizer,
    )

    # 3. Score candidates with the trained model
    #    proba[:, 1] = probability of being a GOOD distractor
    scores = model.predict_proba(feat_df)[:, 1]

    # 4. Rank by score descending, deduplicate, exclude correct answer
    ranked_indices = np.argsort(scores)[::-1]
    distractors = []
    seen = set()
    answer_lower = correct_answer.strip().lower()

    for idx in ranked_indices:
        candidate = candidates[idx].strip()
        key = candidate.lower()

        # Skip duplicates and anything too close to the correct answer
        if key in seen:
            continue
        if key == answer_lower or answer_lower in key:
            continue

        distractors.append(candidate)
        seen.add(key)

        if len(distractors) >= n_distractors:
            break

    logger.info(f"Generated {len(distractors)} distractors: {distractors}")
    return distractors


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save & Load Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, vectorizer: TfidfVectorizer) -> None:
    """
    Persist the trained classifier and fitted vectorizer to disk using joblib.
    Both must be saved together since the vectorizer defines the feature space.

    Parameters
    ----------
    model      : fitted sklearn classifier.
    vectorizer : fitted TfidfVectorizer.
    """
    joblib.dump(model,      MODEL_SAVE_PATH)
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    logger.info(f"Model saved      → {MODEL_SAVE_PATH}")
    logger.info(f"Vectorizer saved → {VECTORIZER_SAVE_PATH}")


def load_artifacts():
    """
    Load the trained model and vectorizer from disk.

    Returns
    -------
    Tuple[classifier, TfidfVectorizer]
    """
    model      = joblib.load(MODEL_SAVE_PATH)
    vectorizer = joblib.load(VECTORIZER_SAVE_PATH)
    logger.info("Artifacts loaded from disk.")
    return model, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo: Synthetic RACE-style Training + Inference
# ─────────────────────────────────────────────────────────────────────────────

import random

def load_real_race_data(csv_path: str, sample_size: int = 1000) -> pd.DataFrame:
    """
    Loads real RACE data and formats it for Model B training.
    """
    logger.info(f"Loading real RACE data from {csv_path}...")
    df = pd.read_csv(csv_path).dropna()
    
    # Optional: Sample it down so your local PC doesn't crash during testing
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)

    rows = []
    
    for _, row in df.iterrows():
        passage = str(row["article"])
        correct_letter = str(row["answer"]).strip() # "A", "B", "C", or "D"
        
        options = {
            "A": str(row["A"]),
            "B": str(row["B"]),
            "C": str(row["C"]),
            "D": str(row["D"])
        }
        
        correct_answer_text = options.get(correct_letter, "")
        if not correct_answer_text:
            continue
            
        # 1. Add the ACTUAL human-written distractors (Positive Labels)
        for letter, opt_text in options.items():
            if letter != correct_letter:
                rows.append({
                    "candidate_text": opt_text,
                    "correct_answer_text": correct_answer_text,
                    "passage_text": passage,
                    "is_actual_distractor": 1
                })
                
        # 2. Add some BAD distractors (Negative Labels)
        # Grab random frequent words from the passage to act as bad examples
        tokens = tokenize_passage(passage)
        content_words = remove_stopwords(tokens)
        if content_words:
            # Pick a few random words to act as bad distractors
            bad_samples = random.sample(content_words, min(3, len(content_words)))
            for bad_word in bad_samples:
                # Make sure our random bad word isn't accidentally the correct answer
                if bad_word not in correct_answer_text.lower():
                    rows.append({
                        "candidate_text": bad_word,
                        "correct_answer_text": correct_answer_text,
                        "passage_text": passage,
                        "is_actual_distractor": 0
                    })

    formatted_df = pd.DataFrame(rows)
    return formatted_df


def main():
    """
    End-to-end demonstration:
      1. Build (or load) training data.
      2. Fit TF-IDF vectorizer on the corpus.
      3. Prepare feature matrix.
      4. Train RandomForestClassifier.
      5. Save model + vectorizer.
      6. Run inference on a new RACE-style example.
    """
    logger.info("=" * 60)
    logger.info("  Model B — Distractor Generator (Classical ML)")
    logger.info("=" * 60)

    # ── 1. Load training data ──────────────────────────────────────────────
    # Replace build_synthetic_dataframe() with your actual data loading:
    #   df = pd.read_csv("race_candidates_labelled.csv")
    df = load_real_race_data(csv_path="data/raw/train.csv", sample_size=1000)
    logger.info(f"Training data loaded: {len(df)} labelled candidate rows.")

    # ── 2. Fit TF-IDF vectorizer ───────────────────────────────────────────
    # Fit on ALL text that the model will see at inference: candidates + answers
    corpus = (
        df["candidate_text"].tolist()
        + df["correct_answer_text"].tolist()
        + df["passage_text"].tolist()
    )
    vectorizer = fit_tfidf_vectorizer(corpus)

    # ── 3. Engineer feature matrix ─────────────────────────────────────────
    X, y = prepare_training_data(df, vectorizer)

    # ── 4. Train ranker ────────────────────────────────────────────────────
    # Swap model_type="logistic" for a simpler, more interpretable model
    ranker = train_ranker(X, y, model_type="random_forest")

    # ── 5. Save artifacts ──────────────────────────────────────────────────
    save_artifacts(ranker, vectorizer)

    # ── 6. Inference demo ─────────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("  Inference Demo")
    logger.info("─" * 60)

    test_passage = (
        "The industrial revolution began in Britain during the eighteenth century. "
        "Steam engines powered factories and transformed manufacturing. "
        "Cotton production rose dramatically, and workers moved from rural farms "
        "to urban centres. Railways expanded rapidly, connecting distant regions "
        "and accelerating trade. Coal mining became essential for fuel supply."
    )
    test_question      = "What powered factories during the industrial revolution?"
    test_correct_answer = "steam engines"

    logger.info(f"Passage (truncated): {test_passage[:80]}…")
    logger.info(f"Question           : {test_question}")
    logger.info(f"Correct answer     : {test_correct_answer}")

    distractors = generate_distractors(
        passage        = test_passage,
        correct_answer = test_correct_answer,
        model          = ranker,
        vectorizer     = vectorizer,
        top_n_candidates = 20,
        n_distractors  = 3,
    )

    logger.info("\n  ── Generated Distractors ──")
    for i, d in enumerate(distractors, 1):
        logger.info(f"  Option {i}: {d}")

    logger.info("\nDone. Artifacts written to disk — import load_artifacts() for inference.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()