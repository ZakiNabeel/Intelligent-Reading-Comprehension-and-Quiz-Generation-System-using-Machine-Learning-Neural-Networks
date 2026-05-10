import re
import string
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_B_DIR = BASE_DIR / "models" / "model_b" / "traditional"

# Shared TF-IDF vectorizer fitted on article + option text from the training split.
vectorizer = joblib.load(MODEL_B_DIR / "model_b_vectorizer.pkl")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())

    return text


def split_sentences(article):
    sentences = re.split(r'(?<=[.!?])\s+', str(article))

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


def extract_candidate_phrases(article):
    article = clean_text(article)

    words = article.split()

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
        "in", "on", "for", "with", "as", "by", "at", "from", "it", "this",
        "that", "he", "she", "they", "we", "you", "i", "his", "her", "their",
        "but", "or", "so", "because", "if", "then", "than", "about", "into",
        "over", "after", "before", "there", "here", "also", "very", "can",
        "could", "would", "should", "will", "just", "more", "most"
    }

    candidates = []

    for word in words:
        if word not in stopwords and len(word) > 3:
            candidates.append(word)

    return list(dict.fromkeys(candidates))


def generate_distractors(article, correct_answer, top_k=3):
    candidates = extract_candidate_phrases(article)
    correct_answer_clean = clean_text(correct_answer)

    # Remove the correct answer and any phrase that contains it (or vice versa).
    candidates = [
        c for c in candidates
        if c != correct_answer_clean
        and correct_answer_clean not in c
        and c not in correct_answer_clean
    ]

    if len(candidates) == 0:
        return ["No suitable distractor"] * top_k

    answer_vec     = vectorizer.transform([correct_answer_clean])
    candidate_vecs = vectorizer.transform(candidates)
    similarities   = cosine_similarity(answer_vec, candidate_vecs)[0]

    # Target similarity ~0.25: similar enough to be plausible, different enough to be wrong.
    # Sorting by |sim - 0.25| picks the "sweet spot" candidates first.
    scored_candidates = sorted(
        zip(candidates, similarities),
        key=lambda x: abs(x[1] - 0.25)
    )

    distractors = []

    for candidate, score in scored_candidates:
        if candidate not in distractors:
            distractors.append(candidate)

        if len(distractors) == top_k:
            break

    while len(distractors) < top_k:
        distractors.append("No suitable distractor")

    return distractors


def generate_hints(article, question, top_k=3):
    sentences = split_sentences(article)

    if len(sentences) == 0:
        return [
            "Read the passage carefully.",
            "Look for information related to the question.",
            "The answer is directly or indirectly present in the passage.",
        ]

    question_vec  = vectorizer.transform([clean_text(question)])
    sentence_vecs = vectorizer.transform([clean_text(s) for s in sentences])
    similarities  = cosine_similarity(question_vec, sentence_vecs)[0]

    # Rank sentences by ascending cosine similarity, then take the top_k highest.
    # Returning them in ascending order gives a "general → specific" reveal sequence.
    ranked = sorted(zip(sentences, similarities), key=lambda x: x[1])
    selected = sorted(ranked[-top_k:], key=lambda x: x[1])  # keep ascending order
    hints = [s for s, _ in selected]

    # Pad with a generic hint if the article has fewer than top_k sentences.
    while len(hints) < top_k:
        hints.insert(0, "Think about the main idea of the passage.")

    return hints


def test_model_b_inference():
    article = """
    The Earth revolves around the Sun once every year.
    This movement causes seasons and changes in daylight.
    The Moon revolves around the Earth.
    """

    question = "What does the Earth revolve around?"

    correct_answer = "The Sun"

    distractors = generate_distractors(
        article,
        correct_answer
    )

    hints = generate_hints(
        article,
        question
    )

    print("\nDistractors:")
    for distractor in distractors:
        print("-", distractor)

    print("\nHints:")
    for hint in hints:
        print("-", hint)


if __name__ == "__main__":
    test_model_b_inference()