import pandas as pd
from pathlib import Path

from inference import predict_best_answer
from model_b_inference import generate_distractors, generate_hints


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


def load_race_sample(split="dev", sample_index=0):
    """
    Loads one sample question from the RACE dataset.
    """

    file_path = RAW_DIR / f"{split}.csv"

    df = pd.read_csv(file_path)

    row = df.iloc[sample_index]

    article = row["article"]
    question = row["question"]

    options = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "D": row["D"]
    }

    correct_label = row["answer"]
    correct_answer = options[correct_label]

    return article, question, options, correct_label, correct_answer


def build_quiz_item(article, question, options, correct_label):
    """
    Combines Model A and Model B into one quiz item.
    """

    correct_answer = options[correct_label]

    predicted_label, confidence, model_a_scores = predict_best_answer(
        article,
        question,
        options
    )

    distractors = generate_distractors(
        article,
        correct_answer
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