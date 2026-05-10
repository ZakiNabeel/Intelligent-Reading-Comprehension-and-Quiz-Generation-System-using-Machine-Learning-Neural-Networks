import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_B_DIR = BASE_DIR / "models" / "model_b" / "traditional"

try:
    vectorizer = joblib.load(MODEL_B_DIR / "model_b_vectorizer.pkl")
except Exception:
    vectorizer = None


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
    "in", "on", "for", "with", "as", "by", "at", "from", "it", "this",
    "that", "he", "she", "they", "we", "you", "i", "his", "her", "their",
    "but", "or", "so", "because", "if", "then", "than", "about", "into",
    "over", "after", "before", "there", "here", "also", "very", "can",
    "could", "would", "should", "will", "just", "more", "most", "be",
    "been", "being", "which", "who", "what", "when", "where", "why",
    "how", "only", "such", "each", "every", "many", "much",
}

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"
)


def clean_text(text):
    """Lowercase and lightly normalize text for similarity comparison."""
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())

    return text


def split_sentences(article):
    """Split an article into sentences long enough to be useful as evidence."""
    sentences = re.split(r'(?<=[.!?])\s+', str(article))

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


def _dedupe(items):
    """Deduplicate candidate strings while preserving order."""
    seen = set()
    result = []
    for item in items:
        text = str(item).strip(" ,.;:!?")
        key = _normal_key(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _normal_key(text):
    """Normalize candidate keys so 'Earth' and 'The Earth' count as the same."""
    return re.sub(r"^(the|a|an)\s+", "", clean_text(text))


def _content_words(text):
    """Return non-stopword tokens from text."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?", str(text))
    return [
        word.lower()
        for word in words
        if word.lower() not in STOPWORDS and len(word) > 2
    ]


def _capitalized_phrases(article):
    """Extract simple named-entity-like phrases from the original article."""
    phrases = re.findall(r"\b(?:[A-Z][a-z]+\.?[ \t]*){1,4}\b", str(article))
    return [
        phrase.strip()
        for phrase in phrases
        if clean_text(phrase) not in STOPWORDS and len(phrase.strip()) > 2
    ]


def _numeric_phrases(article):
    """Extract dates, years, percentages, and other numeric spans."""
    month_pattern = r"\b(?:" + "|".join(MONTHS) + r")\s+\d{1,2},?\s+\d{4}\b"
    number_pattern = r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|years?|days?|months?|miles?|kilometers?|people|students|species)?\b"
    years = r"\b(?:18|19|20)\d{2}\b"
    return re.findall(month_pattern, article) + re.findall(years, article) + re.findall(number_pattern, article, flags=re.I)


def _noun_like_phrases(article):
    """Build short noun-like phrases from consecutive content words."""
    phrases = []
    for sentence in split_sentences(article):
        words = _content_words(sentence)
        phrases.extend(words)
        for n in (2, 3):
            for idx in range(0, max(len(words) - n + 1, 0)):
                phrase = " ".join(words[idx:idx + n])
                if len(phrase) > 5:
                    phrases.append(phrase)
    return phrases


def _answer_type(text):
    """Classify an answer span so distractors can match its surface type."""
    raw = str(text).strip()
    clean = clean_text(raw)
    if re.search(r"\b(?:18|19|20)\d{2}\b|\d", raw):
        return "number"
    if any(month.lower() in clean for month in MONTHS):
        return "date"
    if re.match(r"^(?:[A-Z][a-z]+\.?\s*){1,4}$", raw):
        return "entity"
    if len(clean.split()) >= 2:
        return "phrase"
    return "word"


def extract_candidate_phrases(article):
    """
    Extract a diverse candidate pool for distractors.

    This combines named entities, dates/numbers, and noun-like phrases. The
    output is still classical/rule based and does not call any neural model.
    """
    candidates = []
    candidates.extend(_capitalized_phrases(article))
    candidates.extend(_numeric_phrases(str(article)))
    candidates.extend(_noun_like_phrases(article))
    return _dedupe(candidates)


def _candidate_quality(candidate, correct_answer, similarity):
    """Score how plausible a distractor is using type, length, and TF-IDF similarity."""
    candidate_clean = clean_text(candidate)
    correct_clean = clean_text(correct_answer)
    cand_words = candidate_clean.split()
    answer_words = correct_clean.split()

    target_similarity = 0.28
    similarity_score = 1.0 - min(abs(float(similarity) - target_similarity) / target_similarity, 1.0)
    length_score = 1.0 - min(abs(len(cand_words) - max(len(answer_words), 1)) / 4, 1.0)
    type_score = 1.0 if _answer_type(candidate) == _answer_type(correct_answer) else 0.35
    overlap = len(set(cand_words) & set(answer_words)) / max(len(set(answer_words)), 1)
    overlap_penalty = 0.45 if overlap > 0.6 else 0.0

    return (0.45 * similarity_score) + (0.25 * type_score) + (0.20 * length_score) - overlap_penalty


def _fallback_vectorizer(corpus):
    """Fit a local TF-IDF vectorizer when the trained Model B vectorizer is unavailable."""
    local_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    local_vectorizer.fit(corpus)
    return local_vectorizer


def generate_distractors(article, correct_answer, top_k=3):
    """Generate ranked distractors from article phrases using TF-IDF similarity."""
    candidates = extract_candidate_phrases(article)
    correct_answer_clean = clean_text(correct_answer)
    correct_key = _normal_key(correct_answer)

    candidates = [
        c for c in candidates
        if clean_text(c) != correct_answer_clean
        and _normal_key(c) != correct_key
        and correct_answer_clean not in clean_text(c)
        and clean_text(c) not in correct_answer_clean
    ]

    if len(candidates) == 0:
        return ["No suitable distractor"] * top_k

    try:
        active_vectorizer = vectorizer or _fallback_vectorizer([correct_answer_clean] + [clean_text(c) for c in candidates])
        answer_vec = active_vectorizer.transform([correct_answer_clean])
        candidate_vecs = active_vectorizer.transform([clean_text(c) for c in candidates])
        similarities = cosine_similarity(answer_vec, candidate_vecs)[0]
    except Exception:
        similarities = [0.15 for _ in candidates]

    scored_candidates = sorted(
        zip(candidates, similarities),
        key=lambda item: _candidate_quality(item[0], correct_answer, item[1]),
        reverse=True,
    )

    distractors = []
    seen = {correct_answer_clean, correct_key}
    for candidate, _score in scored_candidates:
        key = _normal_key(candidate)
        if key not in seen:
            distractors.append(candidate)
            seen.add(key)
        if len(distractors) == top_k:
            break

    while len(distractors) < top_k:
        distractors.append("No suitable distractor")

    return distractors


def generate_hints(article, question, top_k=3):
    """Return general-to-specific hint sentences ranked by TF-IDF similarity."""
    sentences = split_sentences(article)

    if len(sentences) == 0:
        return [
            "Read the passage carefully.",
            "Look for information related to the question.",
            "The answer is directly or indirectly present in the passage.",
        ]

    try:
        active_vectorizer = vectorizer or _fallback_vectorizer([clean_text(question)] + [clean_text(s) for s in sentences])
        question_vec = active_vectorizer.transform([clean_text(question)])
        sentence_vecs = active_vectorizer.transform([clean_text(s) for s in sentences])
        similarities = cosine_similarity(question_vec, sentence_vecs)[0]
    except Exception:
        similarities = [0.0 for _ in sentences]

    ranked = sorted(zip(sentences, similarities), key=lambda x: x[1])
    selected = sorted(ranked[-top_k:], key=lambda x: x[1])
    hints = [s for s, _ in selected]

    while len(hints) < top_k:
        hints.insert(0, "Think about the main idea of the passage.")

    return hints


def test_model_b_inference():
    article = """
    The Earth revolves around the Sun once every year.
    This movement causes seasons and changes in daylight.
    The Moon revolves around the Earth.
    Mars and Jupiter are also planets in the solar system.
    """

    question = "What does the Earth revolve around?"
    correct_answer = "The Sun"

    distractors = generate_distractors(article, correct_answer)
    hints = generate_hints(article, question)

    print("\nDistractors:")
    for distractor in distractors:
        print("-", distractor)

    print("\nHints:")
    for hint in hints:
        print("-", hint)


if __name__ == "__main__":
    test_model_b_inference()
