"""
Classical template-based question generation for Model A.

This module uses only traditional NLP techniques:
- regex sentence splitting
- keyword extraction with stopword filtering
- TF-IDF and cosine similarity for sentence importance
- rule/template based Wh-question generation
- hand-built lexical ranking features

No neural networks, transformers, BERT, or T5 are used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_A_DIR = BASE_DIR / "models" / "model_a"
QG_MODEL_PATH = MODEL_A_DIR / "question_generation_ranker.joblib"

STOPWORDS = {
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
    "from", "then", "there", "here", "because", "if", "while",
    "although", "however", "also", "any", "even", "over", "under",
    "again", "once", "one", "two", "three", "four", "five", "new",
}

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"
)

PERSON_CUES = {
    "Mr", "Mrs", "Ms", "Dr", "Professor", "President", "King", "Queen",
    "Sir", "Lady", "Captain", "General", "Officer", "Teacher",
}

PLACE_PREPOSITIONS = {"in", "at", "near", "from", "inside", "outside", "around"}

COMMON_VERBS = {
    "is", "are", "was", "were", "be", "became", "become", "has", "have",
    "had", "do", "does", "did", "made", "make", "makes", "used", "uses",
    "use", "called", "calls", "named", "went", "goes", "go", "found",
    "finds", "discovered", "discovers", "created", "creates", "built",
    "builds", "wrote", "writes", "said", "says", "argued", "argues",
    "believed", "believes", "caused", "causes", "helped", "helps",
    "allowed", "allows", "started", "starts", "ended", "ends",
}

BASE_VERBS = {
    "argued": "argue",
    "allowed": "allow",
    "became": "become",
    "believed": "believe",
    "built": "build",
    "called": "call",
    "caused": "cause",
    "created": "create",
    "discovered": "discover",
    "ended": "end",
    "found": "find",
    "helped": "help",
    "made": "make",
    "said": "say",
    "started": "start",
    "used": "use",
    "went": "go",
    "wrote": "write",
}


def clean_text(text: Any) -> str:
    """Normalize whitespace and remove control characters while preserving case."""
    if text is None or pd.isna(text):
        return ""
    cleaned = re.sub(r"[\r\t]+", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_sentences(article: str) -> List[str]:
    """Split an article into readable sentences using regex rules."""
    article = clean_text(article)
    if not article:
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"']))", article)
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip(" \n")
        if len(sentence.split()) >= 5:
            sentences.append(sentence)
    return sentences


def split_into_sentences(text: str) -> List[str]:
    """Backward-compatible alias for older code."""
    return split_sentences(text)


def _tokens(text: str) -> List[str]:
    """Return lowercase alphabetic tokens."""
    return re.findall(r"[A-Za-z][A-Za-z'-]*", clean_text(text).lower())


def _content_words(text: str) -> List[str]:
    """Return non-stopword content tokens from text."""
    return [t for t in _tokens(text) if t not in STOPWORDS and len(t) > 2]


def _dedupe(items: Iterable[str]) -> List[str]:
    """Deduplicate strings while preserving order."""
    seen = set()
    result = []
    for item in items:
        text = clean_text(item).strip(" ,.;:!?")
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def extract_keywords(sentence: str, top_k: int = 8) -> List[str]:
    """Extract important keywords and short phrases from one sentence."""
    sentence = clean_text(sentence)
    if not sentence:
        return []

    candidates: List[str] = []

    # Keep named entities and dates as phrase candidates.
    candidates.extend(re.findall(r"\b(?:[A-Z][a-z]+\.?\s+){1,3}[A-Z][a-z]+\.?\b", sentence))
    candidates.extend(re.findall(r"\b(?:" + "|".join(MONTHS) + r")\s+\d{1,2},?\s+\d{4}\b", sentence))
    candidates.extend(re.findall(r"\b(?:18|19|20)\d{2}\b", sentence))
    candidates.extend(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|years?|days?|months?|miles?|kilometers?|people|students|species)?\b", sentence, flags=re.I))

    words = _content_words(sentence)
    candidates.extend(words)
    for n in (3, 2):
        for idx in range(0, max(len(words) - n + 1, 0)):
            phrase = " ".join(words[idx:idx + n])
            if len(phrase) > 5:
                candidates.append(phrase)

    counts: Dict[str, int] = {}
    for candidate in candidates:
        key = candidate.lower().strip()
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(
        _dedupe(candidates),
        key=lambda item: (counts.get(item.lower(), 0), len(item.split()), len(item)),
        reverse=True,
    )
    return ranked[:top_k]


def _sentence_importance(article: str) -> List[Tuple[str, float]]:
    """Score sentences by TF-IDF similarity to the full article and keyword density."""
    sentences = split_sentences(article)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0], 1.0)]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(sentences + [clean_text(article)])
    article_vec = matrix[-1]
    sentence_vecs = matrix[:-1]
    similarities = cosine_similarity(sentence_vecs, article_vec).flatten()

    scored = []
    for sentence, similarity in zip(sentences, similarities):
        words = sentence.split()
        keyword_density = len(extract_keywords(sentence, top_k=12)) / max(len(words), 1)
        length_score = 1.0 - min(abs(len(words) - 18) / 28, 1.0)
        score = (0.65 * float(similarity)) + (0.20 * keyword_density) + (0.15 * length_score)
        scored.append((sentence, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _capitalized_phrases(sentence: str) -> List[str]:
    """Find simple named-entity-like capitalized phrases."""
    phrases = re.findall(r"\b(?:[A-Z][a-z]+\.?[ \t]*){1,4}\b", sentence)
    return [
        p.strip()
        for p in phrases
        if p.strip().lower() not in STOPWORDS and len(p.strip()) > 2
    ]


def _first_verb_index(words: Sequence[str]) -> int | None:
    """Return the first likely verb index in a token sequence."""
    for idx, word in enumerate(words):
        cleaned = word.strip(" ,.;:!?").lower()
        if cleaned in COMMON_VERBS or cleaned.endswith("ed"):
            return idx
    return None


def _clean_answer(answer: str) -> str:
    """Clean an answer span for display."""
    answer = clean_text(answer)
    answer = re.sub(r"^(the|a|an)\s+", "", answer, flags=re.I)
    return answer.strip(" ,.;:!?")


def _base_verb(verb: str) -> str:
    """Return a small rule-based base form for auxiliary-question templates."""
    cleaned = verb.strip(" ,.;:!?").lower()
    if cleaned in BASE_VERBS:
        return BASE_VERBS[cleaned]
    if cleaned.endswith("ies") and len(cleaned) > 4:
        return cleaned[:-3] + "y"
    if cleaned.endswith("ed") and len(cleaned) > 4:
        return cleaned[:-2]
    if cleaned.endswith("s") and len(cleaned) > 3:
        return cleaned[:-1]
    return cleaned


def _make_candidate(question: str, answer: str, sentence: str, template: str) -> Dict[str, Any]:
    """Create one generated question candidate with metadata."""
    question = re.sub(r"\s+", " ", question).strip()
    question = question.rstrip(" .")
    if not question.endswith("?"):
        question += "?"
    return {
        "question": question[0].upper() + question[1:] if question else question,
        "answer": _clean_answer(answer),
        "source_sentence": clean_text(sentence),
        "template": template,
        "keywords": extract_keywords(sentence),
    }


def _definition_candidates(sentence: str) -> List[Dict[str, Any]]:
    """Generate What is/was questions from definition-like sentences."""
    pattern = re.compile(
        r"^(?P<subject>.+?)\s+(?P<verb>is|are|was|were|became|becomes)\s+(?P<predicate>.+)$",
        flags=re.I,
    )
    match = pattern.match(sentence.rstrip(".!?"))
    if not match:
        return []
    subject = _clean_answer(match.group("subject"))
    predicate = _clean_answer(match.group("predicate"))
    verb = match.group("verb").lower()
    if (
        not subject
        or not predicate
        or len(subject.split()) > 8
        or re.search(r"\b(because|since|due to)\b", predicate, flags=re.I)
    ):
        return []
    aux = "are" if verb in {"are", "were"} else "is" if verb in {"is", "becomes"} else "was"
    return [_make_candidate(f"What {aux} {subject}", predicate, sentence, "definition")]


def _when_candidates(sentence: str) -> List[Dict[str, Any]]:
    """Generate When questions from date expressions."""
    date_pattern = r"\b(?:(?:" + "|".join(MONTHS) + r")\s+\d{1,2},?\s+\d{4}|(?:18|19|20)\d{2})\b"
    candidates = []
    for date in re.findall(date_pattern, sentence):
        base = re.sub(r"\b(?:in|on|during|around)\s+" + re.escape(date) + r"\b", "", sentence, count=1, flags=re.I)
        if base == sentence:
            base = sentence.replace(date, "", 1)
        base = base.rstrip(".!? ")
        base = re.sub(r"\s+", " ", base)
        words = base.split()
        if len(words) < 4:
            continue
        verb_idx = _first_verb_index(words)
        if verb_idx and verb_idx > 0:
            subject = " ".join(words[:verb_idx])
            verb = _base_verb(words[verb_idx])
            predicate = " ".join(words[verb_idx + 1:])
            candidates.append(_make_candidate(f"When did {subject} {verb} {predicate}", date, sentence, "when"))
        else:
            candidates.append(_make_candidate(f"When did this happen: {base}", date, sentence, "when"))
    return candidates


def _where_candidates(sentence: str) -> List[Dict[str, Any]]:
    """Generate Where questions from simple prepositional place phrases."""
    candidates = []
    pattern = re.compile(r"\b(?P<prep>in|at|near|from|inside|outside|around)\s+(?P<place>(?:the\s+)?(?:[A-Z][A-Za-z'-]*[ \t]*){1,4})")
    for match in pattern.finditer(sentence):
        prep = match.group("prep")
        place = _clean_answer(match.group("place"))
        if not place or place.split()[0] in PERSON_CUES:
            continue
        base = sentence[:match.start()].rstrip(" ,.;:!?")
        words = base.split()
        verb_idx = _first_verb_index(words)
        if len(words) < 3 or verb_idx is None or verb_idx <= 0:
            continue
        subject = " ".join(words[:verb_idx])
        verb = _base_verb(words[verb_idx])
        predicate = " ".join(words[verb_idx + 1:])
        candidates.append(_make_candidate(f"Where did {subject} {verb} {predicate}", f"{prep} {place}", sentence, "where"))
    return candidates


def _who_candidates(sentence: str) -> List[Dict[str, Any]]:
    """Generate Who questions from capitalized person-like subjects."""
    words = sentence.rstrip(".!?").split()
    verb_idx = _first_verb_index(words)
    candidates = []
    if verb_idx and verb_idx > 0:
        subject = " ".join(words[:verb_idx]).strip(" ,")
        subject_words = subject.split()
        looks_like_person = (
            subject_words[0].strip(".") in PERSON_CUES
            or (len(subject_words) >= 2 and subject in _capitalized_phrases(sentence))
        )
        if looks_like_person:
            predicate = " ".join(words[verb_idx:])
            candidates.append(_make_candidate(f"Who {predicate}", subject, sentence, "who-subject"))

    by_match = re.search(r"\bby\s+((?:[A-Z][A-Za-z'-]*\s*){1,4})", sentence)
    if by_match:
        person = _clean_answer(by_match.group(1))
        base = sentence[:by_match.start()].rstrip(" ,.;:!?")
        if len(base.split()) >= 3:
            candidates.append(_make_candidate(f"Who was {base} by", person, sentence, "who-by"))
    return candidates


def _why_candidates(sentence: str) -> List[Dict[str, Any]]:
    """Generate Why questions from because/due-to clauses."""
    match = re.search(r"\b(because|since|as|due to)\b\s+(?P<reason>.+)$", sentence.rstrip(".!?"), flags=re.I)
    if not match:
        return []
    reason = _clean_answer(match.group("reason"))
    base = sentence[:match.start()].rstrip(" ,.;:!?")
    words = base.split()
    verb_idx = _first_verb_index(words)
    if len(words) < 3 or not reason or verb_idx is None or verb_idx <= 0:
        return []
    subject = " ".join(words[:verb_idx])
    verb = _base_verb(words[verb_idx])
    predicate = " ".join(words[verb_idx + 1:])
    return [_make_candidate(f"Why did {subject} {verb} {predicate}", reason, sentence, "why")]


def _what_action_candidates(sentence: str) -> List[Dict[str, Any]]:
    """Generate general What questions from subject-verb-object sentences."""
    words = sentence.rstrip(".!?").split()
    verb_idx = _first_verb_index(words)
    if not verb_idx or verb_idx <= 0 or verb_idx >= len(words) - 1:
        return []

    subject = " ".join(words[:verb_idx]).strip(" ,")
    verb = words[verb_idx].strip(" ,.;:!?")
    obj = " ".join(words[verb_idx + 1:]).strip(" ,.;:!?")
    if not subject or not obj:
        return []

    if verb.lower() in {"is", "are", "was", "were"}:
        return [_make_candidate(f"What {verb.lower()} {subject}", obj, sentence, "what-is")]
    aux = "did" if verb.lower().endswith("ed") or verb.lower() in BASE_VERBS else "does"
    base_verb = _base_verb(verb) if aux in {"did", "does"} else verb.lower()
    return [_make_candidate(f"What {aux} {subject} {base_verb}", obj, sentence, "what-action")]


def generate_question_from_sentence(sentence: str) -> List[str]:
    """Generate one or more template-based question strings from a sentence."""
    return [candidate["question"] for candidate in generate_question_candidates_from_sentence(sentence)]


def generate_question_candidates_from_sentence(sentence: str) -> List[Dict[str, Any]]:
    """Generate question candidates with answers and source metadata."""
    sentence = clean_text(sentence)
    if not sentence:
        return []

    candidates: List[Dict[str, Any]] = []
    candidates.extend(_why_candidates(sentence))
    candidates.extend(_when_candidates(sentence))
    candidates.extend(_where_candidates(sentence))
    candidates.extend(_who_candidates(sentence))
    candidates.extend(_definition_candidates(sentence))
    candidates.extend(_what_action_candidates(sentence))

    # Keyword fallback keeps the generator useful when no structural rule fires.
    if not candidates:
        keywords = extract_keywords(sentence, top_k=3)
        if keywords:
            answer = keywords[0]
            topic = keywords[1] if len(keywords) > 1 else "this passage"
            candidates.append(
                _make_candidate(
                    f"What is stated about {topic} in the passage",
                    answer,
                    sentence,
                    "keyword-fallback",
                )
            )

    unique = []
    seen = set()
    for candidate in candidates:
        key = candidate["question"].lower()
        if candidate["answer"] and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _question_score(article: str, item: str | Dict[str, Any], tfidf_scores: Dict[str, float]) -> float:
    """Compute a classical ranking score for one generated question."""
    if isinstance(item, dict):
        question = item.get("question", "")
        sentence = item.get("source_sentence", "")
        answer = item.get("answer", "")
        sentence_score = float(item.get("sentence_score", 0.0))
    else:
        question = item
        sentence = ""
        answer = ""
        sentence_score = 0.0

    q_tokens = _content_words(question)
    article_tokens = set(_content_words(article))
    sentence_tokens = set(_content_words(sentence))
    wh_bonus = 1.0 if re.match(r"^(who|what|where|when|why|how)\b", question.lower()) else 0.0
    q_len = len(question.split())
    length_score = 1.0 - min(abs(q_len - 9) / 12, 1.0)
    article_overlap = len(set(q_tokens) & article_tokens) / max(len(set(q_tokens)), 1)
    keyword_overlap = len(set(q_tokens) & sentence_tokens) / max(len(set(q_tokens)), 1)
    tfidf_score = np.mean([tfidf_scores.get(token, 0.0) for token in q_tokens]) if q_tokens else 0.0
    answer_penalty = 0.20 if answer and answer.lower() in question.lower() else 0.0
    bad_length_penalty = 0.25 if q_len < 4 or q_len > 22 else 0.0

    return (
        (0.28 * sentence_score)
        + (0.20 * length_score)
        + (0.18 * keyword_overlap)
        + (0.16 * article_overlap)
        + (0.12 * tfidf_score)
        + (0.06 * wh_bonus)
        - answer_penalty
        - bad_length_penalty
    )


def _article_tfidf_scores(article: str) -> Dict[str, float]:
    """Build per-token TF-IDF weights from article sentences."""
    sentences = split_sentences(article)
    if not sentences:
        return {}
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), sublinear_tf=True)
    matrix = vectorizer.fit_transform(sentences)
    means = np.asarray(matrix.mean(axis=0)).ravel()
    return dict(zip(vectorizer.get_feature_names_out(), means))


def rank_generated_questions(
    article: str,
    generated_questions: Sequence[str | Dict[str, Any]],
) -> List[str | Dict[str, Any]]:
    """Rank generated questions using sentence importance, length, overlap, and TF-IDF."""
    tfidf_scores = _article_tfidf_scores(article)
    ranked = sorted(
        generated_questions,
        key=lambda item: _question_score(article, item, tfidf_scores),
        reverse=True,
    )
    return list(ranked)


def generate_questions(
    article: str,
    top_k: int = 5,
    return_metadata: bool = False,
) -> List[str] | List[Dict[str, Any]]:
    """Generate multiple ranked questions from an article."""
    scored_sentences = _sentence_importance(article)
    candidates: List[Dict[str, Any]] = []

    for sentence, sentence_score in scored_sentences[: max(top_k * 2, 4)]:
        for candidate in generate_question_candidates_from_sentence(sentence):
            candidate["sentence_score"] = sentence_score
            candidates.append(candidate)

    ranked = rank_generated_questions(article, candidates)
    selected = ranked[:top_k]
    if return_metadata:
        return selected  # type: ignore[return-value]
    return [item["question"] if isinstance(item, dict) else item for item in selected]


def extract_candidate_sentence(passage: str, correct_answer: str = "") -> Tuple[str, float]:
    """Find the sentence most relevant to an answer, or most important overall."""
    sentences = split_sentences(passage)
    if not sentences:
        return clean_text(passage), 0.0
    if not correct_answer:
        scored = _sentence_importance(passage)
        return scored[0] if scored else (sentences[0], 0.0)

    corpus = sentences + [clean_text(correct_answer)]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_idx = int(np.argmax(similarities))
    return sentences[best_idx], float(similarities[best_idx])


def apply_question_templates(sentence: str, correct_answer: str = "") -> List[str]:
    """Backward-compatible template helper for older answer-conditioned code."""
    if not correct_answer:
        return generate_question_from_sentence(sentence)

    blanked = re.sub(re.escape(correct_answer), "____", sentence, count=1, flags=re.I).rstrip(".!?")
    candidates = [
        f"What does the passage say about {extract_keywords(sentence, top_k=1)[0] if extract_keywords(sentence, top_k=1) else 'this topic'}?",
        f"What is the correct completion: {blanked}?",
    ]
    candidates.extend(generate_question_from_sentence(sentence))
    return _dedupe(candidates)


def build_question_feature_matrix(
    questions: List[str],
    correct_answer: str,
    passage: str,
) -> pd.DataFrame:
    """Build lexical feature rows for legacy ML question ranker support."""
    rows = []
    answer_tokens = set(_content_words(correct_answer))
    passage_tokens = set(_content_words(passage))
    for question in questions:
        q_tokens = _content_words(question)
        rows.append(
            {
                "q_length": float(len(question.split())),
                "starts_with_wh": float(bool(re.match(r"^(who|what|where|when|why|how)\b", question.lower()))),
                "has_question_mark": float(question.strip().endswith("?")),
                "answer_overlap": len(set(q_tokens) & answer_tokens) / max(len(answer_tokens), 1),
                "passage_overlap": len(set(q_tokens) & passage_tokens) / max(len(set(q_tokens)), 1),
                "unique_tokens": float(len(set(q_tokens))),
                "answer_absent": float(clean_text(correct_answer).lower() not in question.lower()),
            }
        )
    return pd.DataFrame(rows)


def train_question_ranker(df: pd.DataFrame, vectorizer: TfidfVectorizer | None = None):
    """Train a classical RandomForest ranker from labelled question rows."""
    required = {"question_text", "correct_answer", "passage_text", "is_good_question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    feature_rows = [
        build_question_feature_matrix(
            [row["question_text"]],
            row["correct_answer"],
            row["passage_text"],
        ).iloc[0].to_dict()
        for _, row in df.iterrows()
    ]
    x = pd.DataFrame(feature_rows)
    y = df["is_good_question"].astype(int)
    feature_cols = list(x.columns)

    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    if y_val.nunique() > 0:
        y_pred = model.predict(x_val)
        print(classification_report(y_val, y_pred, zero_division=0))
    return model, feature_cols


def rank_questions(
    questions: List[str],
    correct_answer: str,
    passage: str,
    model: RandomForestClassifier,
    feature_cols: List[str],
    top_k: int = 1,
) -> List[Tuple[str, float]]:
    """Rank legacy answer-conditioned candidate questions with a trained model."""
    features = build_question_feature_matrix(questions, correct_answer, passage)[feature_cols]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, -1]
    else:
        scores = model.predict(features)
    return sorted(zip(questions, scores.tolist()), key=lambda item: item[1], reverse=True)[:top_k]


def save_qg_model(model: Any, feature_cols: List[str]) -> None:
    """Save a classical QG ranker artifact."""
    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, feature_cols), QG_MODEL_PATH)


def load_qg_model():
    """Load a saved classical QG ranker artifact."""
    return joblib.load(QG_MODEL_PATH)


def generate_best_questions(
    passage: str,
    correct_answer: str,
    model: RandomForestClassifier | None = None,
    feature_cols: List[str] | None = None,
    num_questions: int = 3,
) -> List[str]:
    """Generate top questions for older callers; model ranking is optional."""
    sentence, _ = extract_candidate_sentence(passage, correct_answer)
    candidates = apply_question_templates(sentence, correct_answer)
    if model is not None and feature_cols is not None:
        return [q for q, _ in rank_questions(candidates, correct_answer, passage, model, feature_cols, num_questions)]
    ranked = rank_generated_questions(passage, candidates)
    return [str(q) for q in ranked[:num_questions]]


def _demo_article() -> str:
    """Return a short article for the command-line demo."""
    return (
        "Marie Curie discovered radium in 1898 while working in Paris. "
        "Radium is a radioactive element that glows faintly in the dark. "
        "Curie's research became important because it helped doctors develop new cancer treatments. "
        "The discovery changed science and medicine around the world."
    )


if __name__ == "__main__":
    demo_article = _demo_article()
    print("Generated questions")
    print("-" * 60)
    for idx, item in enumerate(generate_questions(demo_article, top_k=5, return_metadata=True), start=1):
        print(f"{idx}. {item['question']}")
        print(f"   Answer: {item['answer']}")
        print(f"   Template: {item['template']}")
