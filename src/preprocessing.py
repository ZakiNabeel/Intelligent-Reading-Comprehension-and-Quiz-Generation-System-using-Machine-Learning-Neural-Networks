import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
dev_df = pd.read_csv(RAW_DATA_DIR / "dev.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("Dev Shape:", dev_df.shape)

print(train_df.head())
