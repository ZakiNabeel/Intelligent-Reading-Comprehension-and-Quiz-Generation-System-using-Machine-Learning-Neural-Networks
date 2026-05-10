import pandas as pd
from pathlib import Path

from inference import predict_best_answer
from model_b_inference import generate_distractors, generate_hints
from model_a_qg import extract_keywords, generate_questions


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


def safe_text(value, fallback=""):
    """Convert missing pandas values to a display-safe string."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def load_race_sample(split="dev", sample_index=0):
    """
    Loads one sample question from the RACE dataset.
    """

    file_path = RAW_DIR / f"{split}.csv"

    df = pd.read_csv(file_path)

    row = df.iloc[sample_index]

    article = safe_text(row["article"])
    question = safe_text(row["question"])

    options = {
        "A": safe_text(row["A"], "Option A unavailable"),
        "B": safe_text(row["B"], "Option B unavailable"),
        "C": safe_text(row["C"], "Option C unavailable"),
        "D": safe_text(row["D"], "Option D unavailable")
    }

    correct_label = safe_text(row["answer"], "A")
    if correct_label not in options:
        correct_label = "A"
    correct_answer = options[correct_label]

    return article, question, options, correct_label, correct_answer


def build_quiz_item(article, question, options, correct_label):
    """
    Combines Model A and Model B into one quiz item.
    """

    article = safe_text(article)
    question = safe_text(question)
    options = {
        str(label): safe_text(option, f"Option {label} unavailable")
        for label, option in options.items()
    }
    correct_label = safe_text(correct_label, "A")
    if correct_label not in options:
        correct_label = next(iter(options), "A")
    correct_answer = options[correct_label]

    predicted_label, confidence, model_a_scores = predict_best_answer(
        article,
        question,
        options
    )

    distractors = generate_distractors(
        article,
        correct_answer,
        top_k=12
    )

    hints = generate_hints(
        article,
        question
    )

    quiz_item = {
        "article": article,
        "question": question,
        "original_options": options,
        "correct_label": correct_label,
        "correct_answer": correct_answer,
        "model_a_prediction": predicted_label,
        "model_a_confidence": confidence,
        "model_a_scores": model_a_scores,
        "generated_distractors": distractors,
        "generated_hints": hints
    }

    return quiz_item


def _normalise_options(correct_answer, distractors, fallback_keywords=None):
    """
    Build one correct answer plus three clean distractors for generated questions.
    """

    fallback_keywords = fallback_keywords or []
    correct_clean = str(correct_answer).strip().lower()
    choices = [str(correct_answer).strip()]
    seen = {correct_clean}

    for candidate in list(distractors) + list(fallback_keywords):
        text = str(candidate).strip()
        key = text.lower()
        if (
            not text
            or key in seen
            or "no suitable" in key
            or "unavailable" in key
            or key in correct_clean
            or correct_clean in key
        ):
            continue
        choices.append(text)
        seen.add(key)
        if len(choices) == 4:
            break

    while len(choices) < 4:
        choices.append(f"Alternative {len(choices)}")

    return {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}


def build_generated_quiz_items(article, top_k=5):
    """
    Generate quiz items directly from an article using template-based QG.

    The original RACE workflow remains unchanged. This function is an optional
    generated-question mode that still runs Model A answer verification and
    Model B distractor/hint generation through build_quiz_item().
    """

    generated = generate_questions(article, top_k=top_k, return_metadata=True)
    quiz_items = []

    for item in generated:
        question = item["question"]
        correct_answer = item["answer"]
        if not question or not correct_answer:
            continue

        distractors = generate_distractors(article, correct_answer, top_k=8)
        fallback_keywords = [
            kw for kw in extract_keywords(item.get("source_sentence", article), top_k=10)
            if kw.lower() != correct_answer.lower()
        ]
        options = _normalise_options(correct_answer, distractors, fallback_keywords)
        quiz_item = build_quiz_item(article, question, options, correct_label="A")
        quiz_item.update(
            {
                "question_source": "generated",
                "source_sentence": item.get("source_sentence", ""),
                "question_template": item.get("template", ""),
                "question_keywords": item.get("keywords", []),
            }
        )
        quiz_items.append(quiz_item)

    return quiz_items


def print_quiz_item(quiz_item):
    print("\n" + "=" * 70)
    print("QUIZ ITEM")
    print("=" * 70)

    print("\nArticle:")
    print(quiz_item["article"][:800] + "...")

    print("\nQuestion:")
    print(quiz_item["question"])

    print("\nOriginal Options:")
    for label, option in quiz_item["original_options"].items():
        print(f"{label}. {option}")

    print("\nCorrect Answer:")
    print(f"{quiz_item['correct_label']}. {quiz_item['correct_answer']}")

    print("\nModel A Prediction:")
    print(
        f"{quiz_item['model_a_prediction']} "
        f"(confidence: {quiz_item['model_a_confidence']:.4f})"
    )

    print("\nModel A Scores:")
    for label, score in quiz_item["model_a_scores"].items():
        print(f"{label}: {score:.4f}")

    print("\nModel B Generated Distractors:")
    for distractor in quiz_item["generated_distractors"]:
        print("-", distractor)

    print("\nModel B Generated Hints:")
    for i, hint in enumerate(quiz_item["generated_hints"], start=1):
        print(f"Hint {i}: {hint}")


def main():
    article, question, options, correct_label, correct_answer = load_race_sample(
        split="dev",
        sample_index=0
    )

    quiz_item = build_quiz_item(
        article,
        question,
        options,
        correct_label
    )

    print_quiz_item(quiz_item)


if __name__ == "__main__":
    main()
