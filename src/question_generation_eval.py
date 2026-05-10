"""
Optional evaluation for classical template-based question generation.

The script compares generated questions against RACE questions and saves a CSV:
models/model_a/question_generation_eval.csv

BLEU, ROUGE, and METEOR are computed when their optional dependencies are
installed. Missing dependencies are handled gracefully.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from evaluate import compute_bleu, compute_meteor, compute_rouge_l
from model_a_qg import generate_questions


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_PATH = BASE_DIR / "models" / "model_a" / "question_generation_eval.csv"


def evaluate_question_generation(split: str = "dev", sample_size: int = 100, top_k: int = 3) -> pd.DataFrame:
    """Load RACE samples, generate questions, compare with original questions, and save CSV."""
    file_path = RAW_DIR / f"{split}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"RACE file not found: {file_path}")

    df = pd.read_csv(file_path)
    if sample_size:
        df = df.head(sample_size)

    rows: List[dict] = []

    for sample_index, row in df.iterrows():
        article = row["article"]
        reference_question = row["question"]
        generated = generate_questions(article, top_k=top_k)

        for rank, generated_question in enumerate(generated, start=1):
            result = {
                "sample_index": sample_index,
                "rank": rank,
                "reference_question": reference_question,
                "generated_question": generated_question,
                "bleu": compute_bleu(reference_question, generated_question),
                "rouge_l": compute_rouge_l(reference_question, generated_question),
                "meteor": compute_meteor(reference_question, generated_question),
            }
            rows.append(result)

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved question generation evaluation to: {OUTPUT_PATH}")

    if not results.empty:
        metric_cols = [col for col in ["bleu", "rouge_l", "meteor"] if col in results.columns]
        if metric_cols:
            print(results[metric_cols].mean(numeric_only=True))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate template-based question generation.")
    parser.add_argument("--split", default="dev", help="RACE split CSV name without extension.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of rows to evaluate.")
    parser.add_argument("--top-k", type=int, default=3, help="Generated questions per article.")
    args = parser.parse_args()

    evaluate_question_generation(args.split, args.sample_size, args.top_k)


if __name__ == "__main__":
    main()
