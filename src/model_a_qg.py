"""
model_a_train.py  —  Model A: Template-Based Question Generation with ML Ranking
==================================================================================
Implements the rubric requirement:
  "Template-Based Question Generation with ML Ranking"

Pipeline
--------
  Step 1  extract_candidate_sentence()
          Split passage into sentences; pick the sentence with the highest
          TF-IDF keyword overlap with the correct answer.

  Step 2  apply_question_templates()
          Transform that declarative sentence into Wh-word question candidates
          using pattern-based rules (Who / What / Where / When / How many).

  Step 3  train_question_ranker() + rank_questions()
          Train a RandomForestClassifier on lexical features so the best
          generated question can be selected at inference time.

Constraints
-----------
  No deep learning, transformers, or neural networks.
  Libraries: scikit-learn, pandas, numpy, re (stdlib).
  Sentence splitting: regex-based (no NLTK dependency required).
"""

import re
import logging
import warnings
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Artifact paths ────────────────────────────────────────────────────────
QG_MODEL_PATH      = "models/model_a_qg_ranker.joblib"
QG_VECTORIZER_PATH = "models/model_a_qg_tfidf.joblib"

# ─── Minimal stopword set (no external dependency) ─────────────────────────
_STOPWORDS = {
    "a","an","the","and","but","or","for","nor","so","yet","at","by","in",
    "of","on","to","up","as","is","it","its","be","am","are","was","were",
    "been","have","has","had","do","does","did","will","would","shall",
    "should","may","might","must","can","could","that","this","these",
    "those","i","me","my","we","our","you","your","he","him","his","she",
    "her","they","them","their","what","which","who","whom","when","where",
    "why","how","all","each","every","both","few","more","most","other",
    "some","such","no","not","only","same","than","too","very","just",
    "with","about","after","before","between","into","through","during",
    "from","then","there","here","because","if","while","although",
    "however","also","any","even","over","under","again","once",
}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Sentence Extraction
# ═══════════════════════════════════════════════════════════════════════════

def split_into_sentences(text: str) -> List[str]:
    """
    Regex-based sentence splitter. Splits on '. ', '? ', '! ' followed by
    a capital letter, or on explicit newlines.  Handles common abbreviations
    (Mr., Dr., etc.) by requiring the next token to start with a capital.

    Parameters
    ----------
    text : str — Full passage text.

    Returns
    -------
    List[str] — List of cleaned sentence strings (empty strings removed).
    """
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 10]
    return sentences


def _content_words(text: str) -> List[str]:
    """Lowercase alphabetic tokens with stopwords and short words removed."""
    tokens = re.sub(r"[^a-zA-Z\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def extract_candidate_sentence(
    passage: str,
    correct_answer: str
) -> Tuple[str, float]:
    """
    Find the sentence in the passage with the highest TF-IDF keyword overlap
    with the correct answer. This sentence is the prime candidate for
    template-based question generation.

    Strategy
    --------
    1. Split passage into sentences.
    2. Fit a TF-IDF vectorizer on all sentences + the answer.
    3. Compute cosine similarity between the answer vector and each sentence.
    4. Return the highest-scoring sentence and its score.

    Parameters
    ----------
    passage        : str — Full reading passage.
    correct_answer : str — Ground-truth answer text.

    Returns
    -------
    Tuple[str, float] — (best_sentence, similarity_score)
    """
    sentences = split_into_sentences(passage)
    if not sentences:
        return passage, 0.0

    corpus = sentences + [correct_answer]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)  # (n_sents+1, vocab)

    # Answer vector is the last row; sentence vectors are all but the last
    answer_vec    = tfidf_matrix[-1]          # shape (1, vocab)
    sentence_vecs = tfidf_matrix[:-1]         # shape (n_sents, vocab)

    # cosine_similarity returns (1, n_sents); flatten to 1-D
    scores = cosine_similarity(answer_vec, sentence_vecs).flatten()

    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    logger.info(
        f"Best sentence (score={best_score:.3f}): "
        f"{sentences[best_idx][:80]}…"
    )
    return sentences[best_idx], best_score


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Template-Based Question Generation
# ═══════════════════════════════════════════════════════════════════════════

# Named-entity heuristic cues used to pick the Wh-word template
_WHO_CUES   = {"scientist", "scientists", "researcher", "researchers",
               "author", "authors", "president", "doctor", "dr", "mr",
               "mrs", "professor", "people", "person", "man", "woman",
               "he", "she", "they", "politician", "leader", "team"}

_WHERE_CUES = {"country", "city", "region", "forest", "river", "ocean",
               "continent", "area", "place", "location", "zone",
               "land", "mountain", "village", "island", "sea", "lake"}

_WHEN_CUES  = {"year", "century", "decade", "month", "day", "period",
               "era", "age", "time", "date", "season", "hour"}

_HOWMANY_CUES = {"million", "billion", "thousand", "hundred", "percent",
                 "number", "amount", "total", "count", "species",
                 "kilometre", "kilometer", "metre", "meter", "km"}


def _choose_wh_word(answer: str, sentence: str) -> str:
    """
    Heuristically pick the most appropriate Wh-word for a question template
    based on cue words present in the answer and surrounding sentence.

    Returns one of: "Who", "What", "Where", "When", "How many".
    Defaults to "What" when no strong cue is found.
    """
    combined = (answer + " " + sentence).lower()
    tokens   = set(re.findall(r"[a-z]+", combined))

    if tokens & _WHO_CUES:
        return "Who"
    if tokens & _WHERE_CUES:
        return "Where"
    if tokens & _WHEN_CUES:
        return "When"
    if tokens & _HOWMANY_CUES:
        return "How many"
    return "What"


def _blank_answer_in_sentence(sentence: str, answer: str) -> str:
    """
    Replace the answer span in the sentence with '___' (case-insensitive).
    Falls back to appending '___?' if the answer isn't found verbatim.
    """
    pattern = re.compile(re.escape(answer.strip()), re.IGNORECASE)
    blanked, n_subs = pattern.subn("___", sentence, count=1)
    if n_subs == 0:
        # Answer not verbatim in sentence — append a fill-in blank
        blanked = sentence.rstrip(".!?") + " (___)"
    return blanked


def apply_question_templates(
    sentence: str,
    correct_answer: str
) -> List[str]:
    """
    Transform a declarative sentence into a list of Wh-word question
    candidates using rule-based templates.

    Templates generated
    -------------------
    T1  Wh-word + blanked sentence?
        e.g. "What ___ plays a critical role in regulating the global climate?"
    T2  Wh-word + "does/did" inversion on the subject noun phrase?
        e.g. "What role does the Amazon play in climate regulation?"
    T3  Fill-in-the-blank format (always generated as a fallback):
        e.g. "___ plays a critical role in regulating the global climate."

    Parameters
    ----------
    sentence       : str — Best candidate sentence from Step 1.
    correct_answer : str — The correct answer to be replaced / asked about.

    Returns
    -------
    List[str] — 2–4 generated question strings.
    """
    wh = _choose_wh_word(correct_answer, sentence)
    blanked = _blank_answer_in_sentence(sentence, correct_answer)

    questions = []

    # Template 1: Wh-word + blanked sentence
    q1 = f"{wh} {blanked.lstrip().rstrip('.')}?"
    questions.append(q1)

    # Template 2: Wh-word + role/purpose framing
    # Extract the first verb phrase heuristically (first VBZ/VBD-like token)
    words = sentence.split()
    verb_idx = None
    common_verbs = {"is","are","was","were","plays","played","has","have",
                    "had","contains","contains","covers","covered","argues",
                    "argued","produces","produced","absorbs","absorbed"}
    for idx, w in enumerate(words):
        if w.lower() in common_verbs:
            verb_idx = idx
            break

    if verb_idx and verb_idx > 0:
        subject = " ".join(words[:verb_idx])
        predicate = " ".join(words[verb_idx:]).rstrip(".")
        # Replace answer in predicate
        pred_blanked = _blank_answer_in_sentence(predicate, correct_answer)
        q2 = f"{wh} {pred_blanked} {subject}?".replace("  ", " ")
        questions.append(q2)

    # Template 3: Fill-in-the-blank (always included)
    q3 = blanked if blanked.endswith("?") else blanked.rstrip(".") + "."
    questions.append(q3)

    # Template 4: Passive-style "What is known about X?"
    answer_words = _content_words(correct_answer)
    if answer_words:
        key_phrase = answer_words[0]
        q4 = f"{wh} is the significance of {key_phrase} according to the passage?"
        questions.append(q4)

    # Deduplicate while preserving order
    seen, unique_qs = set(), []
    for q in questions:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_qs.append(q)

    return unique_qs


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — ML Question Ranker
# ═══════════════════════════════════════════════════════════════════════════

def _question_features(
    question: str,
    correct_answer: str,
    passage: str
) -> Dict[str, float]:
    """
    Extract lexical features for a single generated question.

    Features
    --------
    q_length          : Token count of the question.
    starts_with_wh    : 1 if question starts with a Wh-word, else 0.
    has_question_mark : 1 if the string ends with '?', else 0.
    answer_overlap    : Fraction of answer tokens present in the question.
    passage_overlap   : Fraction of question content words found in passage.
    unique_tokens     : Number of unique content words in the question.
    answer_absent     : 1 if correct_answer does NOT appear verbatim (good).
    """
    WH_WORDS = {"what", "who", "where", "when", "why", "how", "which"}

    q_tokens   = re.findall(r"[a-zA-Z]+", question.lower())
    q_content  = [t for t in q_tokens if t not in _STOPWORDS and len(t) > 2]
    ans_tokens = set(_content_words(correct_answer))
    pas_tokens = set(_content_words(passage))

    answer_overlap = (
        len(set(q_content) & ans_tokens) / max(len(ans_tokens), 1)
    )
    passage_overlap = (
        len(set(q_content) & pas_tokens) / max(len(q_content), 1)
    )

    return {
        "q_length":          float(len(q_tokens)),
        "starts_with_wh":    float(q_tokens[0] in WH_WORDS if q_tokens else 0),
        "has_question_mark": float(question.strip().endswith("?")),
        "answer_overlap":    answer_overlap,
        "passage_overlap":   passage_overlap,
        "unique_tokens":     float(len(set(q_content))),
        "answer_absent":     float(
            correct_answer.lower() not in question.lower()
        ),
    }


def build_question_feature_matrix(
    questions: List[str],
    correct_answer: str,
    passage: str
) -> pd.DataFrame:
    """
    Build the feature DataFrame for a list of generated questions.

    Parameters
    ----------
    questions      : List[str] — Generated question candidates.
    correct_answer : str       — Ground-truth answer.
    passage        : str       — Source passage.

    Returns
    -------
    pd.DataFrame shape (n_questions, 7)
    """
    rows = [
        _question_features(q, correct_answer, passage)
        for q in questions
    ]
    return pd.DataFrame(rows)


def train_question_ranker(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer = None
):
    """
    Train a RandomForestClassifier to score generated questions.

    Expected DataFrame columns
    --------------------------
    question_text   : str  — Generated question string.
    correct_answer  : str  — Ground-truth answer.
    passage_text    : str  — Source passage.
    is_good_question: int  — 1 = well-formed/relevant, 0 = poor.

    The model learns to classify good vs. bad questions from the 7 lexical
    features.  At inference time we use predict_proba()[:, 1] as a score.

    Parameters
    ----------
    df         : pd.DataFrame — Labelled training data.
    vectorizer : TfidfVectorizer (optional, unused here but kept for API parity).

    Returns
    -------
    Tuple[RandomForestClassifier, List[str]]
        (fitted model, list of feature column names)
    """
    required = {"question_text", "correct_answer", "passage_text", "is_good_question"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"DataFrame missing columns: {missing}")

    logger.info(f"Building features for {len(df)} training questions …")

    feature_rows = [
        _question_features(
            row["question_text"],
            row["correct_answer"],
            row["passage_text"]
        )
        for _, row in df.iterrows()
    ]

    X = pd.DataFrame(feature_rows)
    y = df["is_good_question"].reset_index(drop=True)
    feature_cols = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    logger.info(
        "\nQuestion Ranker — Validation Report:\n"
        + classification_report(
            y_val, y_pred,
            target_names=["poor_question", "good_question"],
            zero_division=0
        )
    )
    logger.info(
        f"Feature importances:\n"
        + "\n".join(
            f"  {c}: {v:.3f}"
            for c, v in zip(
                feature_cols,
                model.feature_importances_
            )
        )
    )
    return model, feature_cols


def rank_questions(
    questions: List[str],
    correct_answer: str,
    passage: str,
    model: RandomForestClassifier,
    feature_cols: List[str],
    top_k: int = 1
) -> List[Tuple[str, float]]:
    """
    Score generated questions with the trained ranker and return top_k.

    Parameters
    ----------
    questions      : List[str]                — Candidates from Step 2.
    correct_answer : str                       — Ground-truth answer.
    passage        : str                       — Source passage.
    model          : RandomForestClassifier    — Fitted question ranker.
    feature_cols   : List[str]                 — Column order from training.
    top_k          : int                       — Number of best questions.

    Returns
    -------
    List[Tuple[str, float]] — [(question, score), …] sorted best-first.
    """
    feat_df = build_question_feature_matrix(questions, correct_answer, passage)
    feat_df = feat_df[feature_cols]  # enforce column order

    scores = model.predict_proba(feat_df)[:, 1]
    ranked = sorted(
        zip(questions, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════════

def save_qg_model(model, feature_cols: List[str]) -> None:
    """Save the question ranker and feature column list."""
    import os; os.makedirs("models", exist_ok=True)
    joblib.dump((model, feature_cols), QG_MODEL_PATH)
    logger.info(f"QG ranker saved → {QG_MODEL_PATH}")


def load_qg_model():
    """Load question ranker and feature column list from disk."""
    model, feature_cols = joblib.load(QG_MODEL_PATH)
    logger.info("QG ranker loaded.")
    return model, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# Full Inference: passage + answer → best question
# ═══════════════════════════════════════════════════════════════════════════

def generate_best_questions(
    passage: str,
    correct_answer: str,
    model: RandomForestClassifier,
    feature_cols: List[str],
    num_questions: int = 3  # Add this parameter!
) -> List[str]: 
    """
    Returns the top K best questions for the passage.
    """
    best_sentence, _score = extract_candidate_sentence(passage, correct_answer)
    candidates = apply_question_templates(best_sentence, correct_answer)
    
    # Ask the ranker for more than 1 question
    ranked = rank_questions(
        candidates, correct_answer, passage, model, feature_cols, top_k=num_questions
    )
    
    # Extract just the question text from the ranked tuples
    best_qs = [q_text for q_text, score in ranked]
    return best_qs


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic training data + main demo
# ═══════════════════════════════════════════════════════════════════════════

def _build_synthetic_qg_df() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame that mirrors expected labelled schema.
    Replace with: df = pd.read_csv("data/processed/qg_training.csv")
    """
    passage = (
        "The Amazon rainforest covers over five million square kilometres. "
        "It is home to an estimated three million species of plants and animals. "
        "The forest plays a critical role in regulating the global climate by "
        "absorbing vast amounts of carbon dioxide. Scientists argue that "
        "protecting the Amazon is essential for global biodiversity."
    )
    rows = [
        # good questions (is_good_question=1)
        ("What role does the Amazon play in climate regulation?",
         "absorbing carbon dioxide", passage, 1),
        ("Where does the Amazon rainforest cover five million kilometres?",
         "Amazon rainforest", passage, 1),
        ("How many species does the Amazon contain?",
         "three million species", passage, 1),
        ("Who argues that protecting the Amazon is essential?",
         "Scientists", passage, 1),
        # poor questions (is_good_question=0)
        ("absorbing carbon dioxide plays a role?",
         "absorbing carbon dioxide", passage, 0),
        ("five million square kilometres what?",
         "five million square kilometres", passage, 0),
        ("The forest is critical.",
         "carbon dioxide", passage, 0),
        ("is essential for biodiversity Scientists?",
         "Scientists", passage, 0),
    ]
    return pd.DataFrame(rows, columns=[
        "question_text", "correct_answer", "passage_text", "is_good_question"
    ])


def main():
    logger.info("=" * 60)
    logger.info("  Model A — Question Generation Pipeline")
    logger.info("=" * 60)

    # ── Train ─────────────────────────────────────────────────────────────
    df = _build_synthetic_qg_df()
    # Replace with real data:
    # df = pd.read_csv("data/raw/train.csv") and build your labelled QG rows

    model, feature_cols = train_question_ranker(df)
    save_qg_model(model, feature_cols)

    # ── Inference demo ────────────────────────────────────────────────────
    passage = (
        "The Amazon rainforest covers over five million square kilometres. "
        "It is home to an estimated three million species of plants and animals. "
        "The forest plays a critical role in regulating the global climate by "
        "absorbing vast amounts of carbon dioxide. Scientists argue that "
        "protecting the Amazon is essential for global biodiversity."
    )
    answer = "absorbing carbon dioxide"

    logger.info("\n── Step 1: Candidate Sentence Extraction ──")
    best_sent, score = extract_candidate_sentence(passage, answer)
    logger.info(f"  → '{best_sent}'  (sim={score:.3f})")

    logger.info("\n── Step 2: Template Generation ──")
    templates = apply_question_templates(best_sent, answer)
    for i, q in enumerate(templates, 1):
        logger.info(f"  T{i}: {q}")

    logger.info("\n── Step 3: ML Ranking ──")
    best_q = generate_best_questions(passage, answer, model, feature_cols, num_questions=1)[0]
    logger.info(f"\n  ★ Best Question: {best_q}")


if __name__ == "__main__":
    main()
