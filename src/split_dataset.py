import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

REQUIRED_COLS = {"article", "question", "A", "B", "C", "D", "answer"}


def split_dataset(source_csv=None, random_state=42):
    """
    Split a single Kaggle RACE CSV into train/dev/test using an 80/10/10 ratio.

    If source_csv is None, the script auto-detects any CSV in data/raw/ that is
    not already named train.csv, dev.csv, or test.csv.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if source_csv:
        source_path = Path(source_csv)
        if not source_path.is_absolute():
            source_path = BASE_DIR / source_path
    else:
        candidates = [
            p for p in RAW_DIR.glob("*.csv")
            if p.name not in ("train.csv", "dev.csv", "test.csv")
        ]
        if not candidates:
            raise FileNotFoundError(
                "No source CSV found in data/raw/. "
                "Download the RACE dataset from Kaggle and place it there "
                "(e.g. data/raw/race.csv), then re-run this script."
            )
        source_path = candidates[0]

    print(f"Source file : {source_path}")
    df = pd.read_csv(source_path)
    print(f"Total rows  : {len(df)}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Source CSV is missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=random_state, shuffle=True
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, shuffle=True
    )

    train_df.to_csv(RAW_DIR / "train.csv", index=False)
    dev_df.to_csv(RAW_DIR / "dev.csv", index=False)
    test_df.to_csv(RAW_DIR / "test.csv", index=False)

    total = len(df)
    print(f"\nSplit complete (random_state={random_state}):")
    print(f"  data/raw/train.csv : {len(train_df):>6} rows  ({len(train_df)/total*100:.0f}%)")
    print(f"  data/raw/dev.csv   : {len(dev_df):>6} rows  ({len(dev_df)/total*100:.0f}%)")
    print(f"  data/raw/test.csv  : {len(test_df):>6} rows  ({len(test_df)/total*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a single RACE CSV into train/dev/test (80/10/10)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Path to the source CSV (default: auto-detect any .csv in data/raw/ "
            "that is not already train/dev/test.csv)"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    split_dataset(source_csv=args.source, random_state=args.random_state)
