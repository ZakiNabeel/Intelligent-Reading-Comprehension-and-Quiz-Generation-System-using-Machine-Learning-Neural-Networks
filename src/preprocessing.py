import pandas as pd
import string
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """Basic text cleaning."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def load_dataset():
    """Load RACE dataset files."""
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    dev_df = pd.read_csv(RAW_DIR / "dev.csv")
    test_df = pd.read_csv(RAW_DIR / "test.csv")

    return train_df, dev_df, test_df


def create_answer_verification_data(df):
    """
    Converts each MCQ row into 4 binary classification rows.

    Input:
        article, question, A, B, C, D, answer

    Output:
        text, label
    """

    rows = []

    for _, row in df.iterrows():
        article = clean_text(row["article"])
        question = clean_text(row["question"])
        correct_answer = str(row["answer"]).strip()

        for option_label in ["A", "B", "C", "D"]:
            option_text = clean_text(row[option_label])

            combined_text = f"{article} {question} {option_text}"

            label = 1 if option_label == correct_answer else 0

            rows.append({
                "article": article,
                "question": question,
                "option_label": option_label,
                "option_text": option_text,
                "combined_text": combined_text,
                "label": label
            })

    return pd.DataFrame(rows)


def main():
    train_df, dev_df, test_df = load_dataset()

    print("Original Train:", train_df.shape)
    print("Original Dev:", dev_df.shape)
    print("Original Test:", test_df.shape)

    # Use smaller samples first for development
    train_df = train_df.sample(5000, random_state=42)
    dev_df = dev_df.sample(1000, random_state=42)
    test_df = test_df.sample(1000, random_state=42)

    train_processed = create_answer_verification_data(train_df)
    dev_processed = create_answer_verification_data(dev_df)
    test_processed = create_answer_verification_data(test_df)

    train_processed.to_csv(PROCESSED_DIR / "train_model_a.csv", index=False)
    dev_processed.to_csv(PROCESSED_DIR / "dev_model_a.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test_model_a.csv", index=False)

    print("Processed Train:", train_processed.shape)
    print("Processed Dev:", dev_processed.shape)
    print("Processed Test:", test_processed.shape)

    print("\nSample processed data:")
    print(train_processed.head())


if __name__ == "__main__":
    main()