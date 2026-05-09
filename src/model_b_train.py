import re
import string
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_B_DIR = BASE_DIR / "models" / "model_b" / "traditional"

MODEL_B_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def split_sentences(article):
    sentences = re.split(r'(?<=[.!?])\s+', str(article))
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_candidate_phrases(article):
    article = clean_text(article)

    words = article.split()

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
        "in", "on", "for", "with", "as", "by", "at", "from", "it", "this",
        "that", "he", "she", "they", "we", "you", "i", "his", "her", "their",
        "but", "or", "so", "because", "if", "then", "than", "about"
    }

    candidates = []

    for word in words:
        if word not in stopwords and len(word) > 3:
            candidates.append(word)

    # remove duplicates while keeping order
    return list(dict.fromkeys(candidates))


def generate_distractors(article, correct_answer, vectorizer, top_k=3):
    candidates = extract_candidate_phrases(article)

    correct_answer_clean = clean_text(correct_answer)

    candidates = [
        c for c in candidates
        if c != correct_answer_clean
        and correct_answer_clean not in c
        and c not in correct_answer_clean
    ]

    if len(candidates) == 0:
        return ["Option not found", "Option unavailable", "No distractor"]

    answer_vec = vectorizer.transform([correct_answer_clean])
    candidate_vecs = vectorizer.transform(candidates)

    similarities = cosine_similarity(answer_vec, candidate_vecs)[0]

    scored_candidates = list(zip(candidates, similarities))

    # choose medium similarity, not too close and not totally irrelevant
    scored_candidates = sorted(
        scored_candidates,
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


def generate_hints(article, question, vectorizer, top_k=3):
    sentences = split_sentences(article)

    if len(sentences) == 0:
        return [
            "Read the passage carefully.",
            "Look for information related to the question.",
            "The answer is directly or indirectly present in the passage."
        ]

    question_vec = vectorizer.transform([clean_text(question)])
    sentence_vecs = vectorizer.transform([clean_text(s) for s in sentences])

    similarities = cosine_similarity(question_vec, sentence_vecs)[0]

    ranked = sorted(
        zip(sentences, similarities),
        key=lambda x: x[1]
    )

    # low to high similarity = general to specific hint
    selected = ranked[-top_k:]

    selected = sorted(selected, key=lambda x: x[1])

    hints = [sentence for sentence, score in selected]

    while len(hints) < top_k:
        hints.insert(0, "Think about the main idea of the passage.")

    return hints


def train_model_b_vectorizer():
    print("Loading training dataset...")

    train_df = pd.read_csv(RAW_DIR / "train.csv")

    # sample for speed
    train_df = train_df.sample(10000, random_state=42)

    corpus = []

    for _, row in train_df.iterrows():
        corpus.append(clean_text(row["article"]))
        corpus.append(clean_text(row["question"]))
        corpus.append(clean_text(row["A"]))
        corpus.append(clean_text(row["B"]))
        corpus.append(clean_text(row["C"]))
        corpus.append(clean_text(row["D"]))

    print("Training Model B TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    vectorizer.fit(corpus)

    joblib.dump(
        vectorizer,
        MODEL_B_DIR / "model_b_vectorizer.pkl"
    )

    print("Model B vectorizer saved successfully!")

    return vectorizer


def test_model_b(vectorizer):
    print("\nTesting Model B...")

    article = """
    The Earth revolves around the Sun once every year.
    This movement causes seasons and changes in daylight.
    The Moon revolves around the Earth.
    """

    question = "What does the Earth revolve around?"

    correct_answer = "The Sun"

    distractors = generate_distractors(
        article,
        correct_answer,
        vectorizer
    )

    hints = generate_hints(
        article,
        question,
        vectorizer
    )

    print("\nGenerated Distractors:")
    for d in distractors:
        print("-", d)

    print("\nGenerated Hints:")
    for h in hints:
        print("-", h)


def main():
    vectorizer = train_model_b_vectorizer()
    test_model_b(vectorizer)


if __name__ == "__main__":
    main()