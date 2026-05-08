"""
model_b_train.py  —  Model B: Rule-Based Hint Extraction
==========================================================
Implements the rubric requirement:
  "Rule-Based Hint Extraction"

Pipeline
--------
  Given a passage and a question, return the top-3 most relevant sentences
  from the passage as graduated hints (most → least relevant).

  Scoring uses a dual-weight overlap heuristic:
    1. Bag-of-important-words overlap (TF-IDF weighted)
    2. Raw content-word Jaccard overlap (frequency baseline)
  Final score = α * tfidf_overlap + (1-α) * jaccard_overlap

Constraints
-----------
  No deep learning, transformers, or neural networks.
  Libraries: scikit-learn, pandas, numpy, re (stdlib).
  Sentence splitting: regex-based (no NLTK required).
"""

import re
import logging
import warnings
from typing import List, Tuple

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Stopword set ──────────────────────────────────────────────────────────
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

# Blend weight: how much to favour TF-IDF overlap vs raw Jaccard overlap
_TFIDF_ALPHA = 0.65


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def split_into_sentences(text: str) -> List[str]:
    """
    Regex-based sentence splitter — no NLTK required.
    Splits on sentence-ending punctuation followed by whitespace + capital.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str] — Non-empty sentences of at least 10 characters.
    """
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _content_words(text: str) -> List[str]:
    """Lowercase alphabetic tokens with stopwords and short tokens removed."""
    tokens = re.sub(r"[^a-zA-Z\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _jaccard_overlap(set_a: set, set_b: set) -> float:
    """
    Jaccard similarity between two sets.
    Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ═══════════════════════════════════════════════════════════════════════════
# TF-IDF Overlap Scorer
# ═══════════════════════════════════════════════════════════════════════════

def _compute_tfidf_overlaps(
    sentences: List[str],
    question: str
) -> np.ndarray:
    """
    Compute cosine similarity between the question vector and each sentence
    vector using a TF-IDF representation fitted on the passage + question.

    Uses cosine_similarity(question_vec, sentence_matrix) which is O(n · vocab)
    — safe for typical passage lengths.

    Parameters
    ----------
    sentences : List[str] — Passage sentences.
    question  : str       — The question text.

    Returns
    -------
    np.ndarray shape (n_sentences,) — Cosine similarity scores in [0, 1].
    """
    corpus = sentences + [question]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)  # (n+1, vocab)

    question_vec   = tfidf_matrix[-1]    # (1, vocab)
    sentence_vecs  = tfidf_matrix[:-1]   # (n, vocab)

    # cosine_similarity returns (1, n); flatten
    scores = cosine_similarity(question_vec, sentence_vecs).flatten()
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Core Hint Extraction Function
# ═══════════════════════════════════════════════════════════════════════════

def extract_hints(
    passage: str,
    question: str,
    n_hints: int = 3,
    tfidf_alpha: float = _TFIDF_ALPHA,
) -> List[Tuple[str, float]]:
    """
    Extract the top-N most question-relevant sentences from a passage to
    serve as graduated hints (Hint 1 = most relevant, Hint N = least relevant).

    Scoring Formula
    ---------------
    For each passage sentence s:
      tfidf_score(s)   = TF-IDF cosine similarity(question, s)
      jaccard_score(s) = Jaccard overlap of content-word sets
      final_score(s)   = α * tfidf_score + (1-α) * jaccard_score

    This dual-signal approach avoids over-relying on TF-IDF for very short
    sentences where the sparse vector may miss lexical matches.

    Parameters
    ----------
    passage     : str   — The full reading passage.
    question    : str   — The question being answered.
    n_hints     : int   — Number of hint sentences to return (default 3).
    tfidf_alpha : float — Weight for TF-IDF component vs. Jaccard (default 0.65).

    Returns
    -------
    List[Tuple[str, float]]
        Ranked list of (sentence, relevance_score) pairs,
        most relevant first.  Length ≤ n_hints.
    """
    sentences = split_into_sentences(passage)

    if not sentences:
        logger.warning("No sentences extracted from passage.")
        return []

    # ── Signal 1: TF-IDF cosine similarity ────────────────────────────────
    tfidf_scores = _compute_tfidf_overlaps(sentences, question)

    # ── Signal 2: Bag-of-important-words Jaccard overlap ──────────────────
    question_words = set(_content_words(question))
    jaccard_scores = np.array([
        _jaccard_overlap(question_words, set(_content_words(s)))
        for s in sentences
    ])

    # ── Blend scores ──────────────────────────────────────────────────────
    combined_scores = (
        tfidf_alpha * tfidf_scores
        + (1.0 - tfidf_alpha) * jaccard_scores
    )

    # ── Rank and select top-N ─────────────────────────────────────────────
    ranked_indices = np.argsort(combined_scores)[::-1]

    results = []
    seen    = set()

    for idx in ranked_indices:
        sentence = sentences[idx].strip()
        key      = sentence.lower()

        # Skip near-duplicate sentences (same first 60 chars)
        dedup_key = key[:60]
        if dedup_key in seen:
            continue

        results.append((sentence, float(combined_scores[idx])))
        seen.add(dedup_key)

        if len(results) >= n_hints:
            break

    logger.info(f"Extracted {len(results)} hints for question: '{question[:60]}…'")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Hint Formatter (human-readable output)
# ═══════════════════════════════════════════════════════════════════════════

def format_hints(
    hints: List[Tuple[str, float]],
    show_scores: bool = False
) -> str:
    """
    Format the extracted hints for display or logging.

    Parameters
    ----------
    hints       : List[Tuple[str, float]] — Output of extract_hints().
    show_scores : bool — If True, append relevance scores for debugging.

    Returns
    -------
    str — Multi-line formatted hint string.
    """
    if not hints:
        return "No hints available."

    lines = ["─── Graduated Hints ───────────────────────────────────"]
    for rank, (sentence, score) in enumerate(hints, 1):
        score_str = f"  [relevance: {score:.3f}]" if show_scores else ""
        lines.append(f"  Hint {rank}{score_str}: {sentence}")
    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Batch Evaluation over a DataFrame
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_hint_coverage(df, n_hints: int = 3) -> dict:
    """
    Measure how often the correct answer text appears in the extracted hints
    across a DataFrame of RACE examples.  This is the primary quality metric
    for rule-based hint extraction (no labelled hint data needed).

    Expected DataFrame columns
    --------------------------
    passage_text        : str — Full passage.
    question            : str — Question text.
    correct_answer_text : str — The correct answer string.

    Parameters
    ----------
    df      : pd.DataFrame
    n_hints : int — Number of hints to extract per example.

    Returns
    -------
    dict — {"coverage_rate": float, "total": int, "covered": int}
    """
    required = {"passage_text", "question", "correct_answer_text"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"DataFrame missing: {missing}")

    total, covered = 0, 0
    for _, row in df.iterrows():
        hints = extract_hints(row["passage_text"], row["question"], n_hints=n_hints)
        hint_text = " ".join(h for h, _ in hints).lower()
        answer_lower = str(row["correct_answer_text"]).lower()

        if answer_lower in hint_text:
            covered += 1
        total += 1

    rate = covered / total if total > 0 else 0.0
    logger.info(
        f"Hint Coverage: {covered}/{total} ({rate:.1%}) — "
        f"answer found in top-{n_hints} hint sentences."
    )
    return {"coverage_rate": rate, "total": total, "covered": covered}


# ═══════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import pandas as pd

    logger.info("=" * 60)
    logger.info("  Model B — Hint Extraction Pipeline")
    logger.info("=" * 60)

    passage = (
        "The Amazon rainforest is the world's largest tropical rainforest, "
        "covering over five million square kilometres. "
        "It is home to an estimated three million species of plants and animals. "
        "The forest plays a critical role in regulating the global climate by "
        "absorbing vast amounts of carbon dioxide. "
        "Deforestation threatens the entire ecosystem and accelerates climate change. "
        "Scientists argue that protecting the Amazon is essential for global biodiversity."
    )

    question = "What role does the Amazon rainforest play in regulating the climate?"

    logger.info(f"\nPassage  : {passage[:80]}…")
    logger.info(f"Question : {question}\n")

    hints = extract_hints(passage, question, n_hints=3)
    print(format_hints(hints, show_scores=True))

    # ── Batch evaluation demo ─────────────────────────────────────────────
    logger.info("\n── Batch Coverage Evaluation ──")
    df = pd.DataFrame([
        {
            "passage_text": passage,
            "question": question,
            "correct_answer_text": "absorbing vast amounts of carbon dioxide",
        },
        {
            "passage_text": passage,
            "question": "How many species live in the Amazon?",
            "correct_answer_text": "three million species",
        },
        {
            "passage_text": passage,
            "question": "What does deforestation accelerate?",
            "correct_answer_text": "climate change",
        },
    ])
    stats = evaluate_hint_coverage(df, n_hints=3)
    logger.info(f"Coverage rate: {stats['coverage_rate']:.1%}")


if __name__ == "__main__":
    main()
