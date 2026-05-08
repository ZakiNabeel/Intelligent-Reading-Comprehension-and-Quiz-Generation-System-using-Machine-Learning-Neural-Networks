"""
EDA_builder.py
==============
Generates notebooks/EDA.ipynb programmatically via nbformat.
Run once: python EDA_builder.py  → produces the notebook file.
"""

import nbformat as nbf
import os

os.makedirs("notebooks", exist_ok=True)
nb = nbf.v4.new_notebook()

cells = []

# ── Cell 0: Title markdown ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# Exploratory Data Analysis — RACE Reading Comprehension Dataset
**Course Lab Project · Model A & B Pipeline**

This notebook loads `data/raw/train.csv`, computes summary statistics, and
produces three required visualisations:
1. Distribution of **passage lengths** (token count)
2. Distribution of **question types** (Wh- vs. fill-in-the-blank)
3. Distribution of **answer labels** (A / B / C / D balance)

> *No deep learning or transformers are used anywhere in this notebook.*
"""))

# ── Cell 1: Imports ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Plot style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

print("Libraries loaded ✓")
"""))

# ── Cell 2: Load data ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1 · Load Data"))
cells.append(nbf.v4.new_code_cell("""\
DATA_PATH = "data/raw/train.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
df.head(3)
"""))

# ── Cell 3: Schema check ───────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ── Validate required columns exist ─────────────────────────────────────
REQUIRED_COLS = {"passage", "question", "answer"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Check your CSV schema and adjust column names below."
    )

# ── Normalise column names (edit these if your CSV differs) ─────────────
PASSAGE_COL = "passage"
QUESTION_COL = "question"
ANSWER_COL   = "answer"        # Expected values: A / B / C / D

# Drop rows where core fields are null
df = df.dropna(subset=[PASSAGE_COL, QUESTION_COL, ANSWER_COL]).copy()
print(f"After dropping nulls: {len(df):,} rows")
"""))

# ── Cell 4: Feature Engineering for EDA ───────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2 · Feature Engineering for EDA"))
cells.append(nbf.v4.new_code_cell("""\
# ── 2a. Passage length (word count) ─────────────────────────────────────
def count_words(text: str) -> int:
    \"\"\"Count whitespace-delimited tokens.\"\"\"
    return len(str(text).split())

df["passage_length"] = df[PASSAGE_COL].apply(count_words)

# ── 2b. Question type classification ────────────────────────────────────
WH_PATTERN = re.compile(
    r"^\\s*(what|who|where|when|why|how|which|whose|whom)\\b",
    re.IGNORECASE
)
BLANK_PATTERN = re.compile(r"___+|_{2,}|\\.\\.\\.", re.IGNORECASE)

def classify_question(q: str) -> str:
    \"\"\"
    Classify a question into one of three types:
      - 'Wh- Question'      : starts with a Wh-word
      - 'Fill-in-the-Blank' : contains ___ or ... placeholder
      - 'Other'             : declarative / unclear form
    \"\"\"
    q = str(q).strip()
    if WH_PATTERN.match(q):
        return "Wh- Question"
    if BLANK_PATTERN.search(q):
        return "Fill-in-the-Blank"
    return "Other"

df["question_type"] = df[QUESTION_COL].apply(classify_question)

# ── 2c. Normalise answer label ───────────────────────────────────────────
df["answer_label"] = df[ANSWER_COL].astype(str).str.strip().str.upper()

print("Feature columns added: passage_length, question_type, answer_label")
df[["passage_length", "question_type", "answer_label"]].head(5)
"""))

# ── Cell 5: Summary statistics ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3 · Summary Statistics"))
cells.append(nbf.v4.new_code_cell("""\
# ── Passage length stats ────────────────────────────────────────────────
passage_stats = df["passage_length"].describe().rename("Passage Length (words)")

# ── Question type counts ─────────────────────────────────────────────────
q_type_counts = (
    df["question_type"]
    .value_counts()
    .rename_axis("Question Type")
    .reset_index(name="Count")
)
q_type_counts["Proportion (%)"] = (
    100 * q_type_counts["Count"] / q_type_counts["Count"].sum()
).round(1)

# ── Answer balance ───────────────────────────────────────────────────────
answer_counts = (
    df["answer_label"]
    .value_counts()
    .sort_index()
    .rename_axis("Answer Label")
    .reset_index(name="Count")
)
answer_counts["Proportion (%)"] = (
    100 * answer_counts["Count"] / answer_counts["Count"].sum()
).round(1)

# ── Display ──────────────────────────────────────────────────────────────
print("=" * 50)
print("  PASSAGE LENGTH SUMMARY")
print("=" * 50)
display(passage_stats.to_frame())

print("\\n" + "=" * 50)
print("  QUESTION TYPE DISTRIBUTION")
print("=" * 50)
display(q_type_counts)

print("\\n" + "=" * 50)
print("  ANSWER LABEL BALANCE (A / B / C / D)")
print("=" * 50)
display(answer_counts)
"""))

# ── Cell 6: Plot 1 — Passage Length Distribution ──────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4 · Visualisations\n### 4.1 — Passage Length Distribution"))
cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
fig.suptitle("Passage Length Distribution (Word Count)", fontsize=14, fontweight="bold", y=1.01)

# ── Left: Histogram + KDE ───────────────────────────────────────────────
ax = axes[0]
sns.histplot(
    df["passage_length"],
    bins=40,
    kde=True,
    color="#4878CF",
    ax=ax,
    line_kws={"linewidth": 2},
)
ax.axvline(
    df["passage_length"].median(),
    color="#d62728", linestyle="--", linewidth=1.8, label=f"Median = {df['passage_length'].median():.0f}"
)
ax.axvline(
    df["passage_length"].mean(),
    color="#ff7f0e", linestyle=":",  linewidth=1.8, label=f"Mean   = {df['passage_length'].mean():.0f}"
)
ax.set_xlabel("Passage Length (words)")
ax.set_ylabel("Count")
ax.set_title("Histogram + KDE")
ax.legend(frameon=False)

# ── Right: Box plot ──────────────────────────────────────────────────────
ax2 = axes[1]
sns.boxplot(
    y=df["passage_length"],
    color="#4878CF",
    width=0.4,
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    ax=ax2,
)
ax2.set_ylabel("Passage Length (words)")
ax2.set_title("Box Plot (outlier detection)")

# Annotate IQR
q1, q3 = df["passage_length"].quantile([0.25, 0.75])
ax2.annotate(
    f"IQR = {q3-q1:.0f} words",
    xy=(0.55, (q1 + q3) / 2),
    fontsize=9,
    color="grey",
)

plt.tight_layout()
plt.savefig("notebooks/fig1_passage_length.png", bbox_inches="tight")
plt.show()
print(f"Skewness: {df['passage_length'].skew():.3f}  |  "
      f"Kurtosis: {df['passage_length'].kurt():.3f}")
"""))

# ── Cell 7: Plot 2 — Question Type Distribution ────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 4.2 — Question Type Distribution"))
cells.append(nbf.v4.new_code_cell("""\
q_counts = df["question_type"].value_counts()
labels   = q_counts.index.tolist()
counts   = q_counts.values
colors   = ["#4878CF", "#6ACC65", "#D65F5F"][:len(labels)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Question Type Distribution", fontsize=14, fontweight="bold", y=1.01)

# ── Left: Horizontal bar chart ───────────────────────────────────────────
ax = axes[0]
bars = ax.barh(labels, counts, color=colors, edgecolor="white", height=0.5)
ax.bar_label(bars, labels=[f"{c:,}  ({100*c/counts.sum():.1f}%)" for c in counts],
             padding=4, fontsize=10)
ax.set_xlabel("Count")
ax.set_title("Absolute Counts")
ax.invert_yaxis()

# ── Right: Pie chart ─────────────────────────────────────────────────────
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    counts,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(10)
ax2.set_title("Proportional Split")

plt.tight_layout()
plt.savefig("notebooks/fig2_question_types.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 8: Plot 3 — Answer Label Balance ─────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 4.3 — Answer Label Balance (A / B / C / D)"))
cells.append(nbf.v4.new_code_cell("""\
ans_counts = (
    df["answer_label"]
    .value_counts()
    .reindex(["A", "B", "C", "D"], fill_value=0)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Answer Label Distribution (A / B / C / D)",
             fontsize=14, fontweight="bold", y=1.01)

palette = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

# ── Left: Count bar chart ────────────────────────────────────────────────
ax = axes[0]
sns.barplot(
    x=ans_counts.index,
    y=ans_counts.values,
    palette=palette,
    edgecolor="white",
    ax=ax,
)
ax.set_xlabel("Answer Label")
ax.set_ylabel("Count")
ax.set_title("Absolute Counts")

# Annotate bars
for bar, count in zip(ax.patches, ans_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{count:,}",
        ha="center", va="bottom", fontsize=10,
    )

# ── Right: Deviation from uniform baseline ──────────────────────────────
ax2 = axes[1]
expected_uniform = len(df) / 4
deviations = (ans_counts.values - expected_uniform) / expected_uniform * 100

bar_colors = ["#d62728" if d < 0 else "#2ca02c" for d in deviations]
bars2 = ax2.bar(ans_counts.index, deviations, color=bar_colors, edgecolor="white")
ax2.axhline(0, color="black", linewidth=1, linestyle="--")
ax2.set_xlabel("Answer Label")
ax2.set_ylabel("Deviation from Uniform (%)")
ax2.set_title("Balance Check\\n(Green = over-represented, Red = under-represented)")
ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("notebooks/fig3_answer_balance.png", bbox_inches="tight")
plt.show()

# Chi-squared test for label balance
from scipy.stats import chisquare
chi2, p_val = chisquare(ans_counts.values)
print(f"\\nChi-squared test for uniform distribution:")
print(f"  χ² = {chi2:.3f},  p = {p_val:.4f}")
if p_val < 0.05:
    print("  → Labels are NOT uniformly distributed (p < 0.05).")
else:
    print("  → Labels are approximately balanced (p ≥ 0.05).")
"""))

# ── Cell 9: Passage length by question type ────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 4.4 — Passage Length by Question Type (Bonus)"))
cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(9, 4))
sns.violinplot(
    data=df,
    x="question_type",
    y="passage_length",
    palette="muted",
    inner="box",
    ax=ax,
)
ax.set_title("Passage Length Distribution by Question Type", fontweight="bold")
ax.set_xlabel("Question Type")
ax.set_ylabel("Passage Length (words)")
plt.tight_layout()
plt.savefig("notebooks/fig4_length_by_qtype.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 10: Full summary table ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5 · Final Summary Statistics Table"))
cells.append(nbf.v4.new_code_cell("""\
summary_rows = []

# Passage length stats
for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    val = df["passage_length"].describe()[stat_name]
    summary_rows.append({
        "Category": "Passage Length (words)",
        "Metric":   stat_name.capitalize(),
        "Value":    f"{val:.1f}" if "%" in stat_name or stat_name in ["mean","std"] else f"{int(val):,}",
    })

# Question type distribution
for qt, cnt in df["question_type"].value_counts().items():
    summary_rows.append({
        "Category": "Question Type",
        "Metric":   qt,
        "Value":    f"{cnt:,}  ({100*cnt/len(df):.1f}%)",
    })

# Answer label distribution
for lbl, cnt in df["answer_label"].value_counts().sort_index().items():
    summary_rows.append({
        "Category": "Answer Label",
        "Metric":   f"Option {lbl}",
        "Value":    f"{cnt:,}  ({100*cnt/len(df):.1f}%)",
    })

summary_df = pd.DataFrame(summary_rows)

# Style the table
styled = (
    summary_df.style
    .set_table_styles([
        {"selector": "th",
         "props": [("background-color", "#4878CF"),
                   ("color", "white"),
                   ("font-weight", "bold"),
                   ("text-align", "center")]},
        {"selector": "tr:nth-child(even)",
         "props": [("background-color", "#f0f4ff")]},
        {"selector": "td",
         "props": [("text-align", "left"), ("padding", "6px 12px")]},
    ])
    .set_caption("RACE Training Set — EDA Summary Statistics")
    .hide(axis="index")
)
display(styled)
"""))

# ── Cell 11: Save summary CSV ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import os
os.makedirs("data/processed", exist_ok=True)
summary_df.to_csv("data/processed/eda_summary.csv", index=False)
print("Summary table saved → data/processed/eda_summary.csv")
print("All visualisation PNGs saved → notebooks/fig*.png")
print("\\nEDA complete ✓")
"""))

nb.cells = cells

output_path = "notebooks/EDA.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written → {output_path}")
