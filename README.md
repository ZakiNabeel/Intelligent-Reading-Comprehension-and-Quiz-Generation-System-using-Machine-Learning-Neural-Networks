# Intelligent Reading Comprehension and Quiz Generation System

Classical ML-based reading comprehension system using the RACE dataset.
Implements Model A (answer verification + question generation) and Model B (distractor + hint generation) with a Streamlit UI.

---

## Project Structure

```
AI Project/
├── data/
│   ├── raw/          ← RACE CSV files go here (train.csv, dev.csv, test.csv)
│   └── processed/    ← auto-generated after preprocessing
├── models/
│   ├── model_a/traditional/   ← saved .pkl files for Model A
│   └── model_b/traditional/   ← saved .pkl files for Model B
├── src/
│   ├── split_dataset.py       ← Step 0: split Kaggle CSV into train/dev/test (80/10/10)
│   ├── preprocessing.py       ← Step 1: preprocess RACE data
│   ├── model_a_train.py       ← Step 2: train Model A (LR, SVM, KMeans, LP, GMM)
│   ├── model_b_train.py       ← Step 3: train Model B TF-IDF vectorizer
│   ├── model_a_qg.py          ← Template-based question generation
│   ├── model_b_hints.py       ← Rule-based hint extraction
│   ├── inference.py           ← Model A inference (ensemble LR+SVM)
│   ├── model_b_inference.py   ← Model B inference (distractors + hints)
│   ├── quiz_pipeline.py       ← Combines Model A + B into one quiz item
│   └── evaluate.py            ← All evaluation metrics
├── ui/
│   └── app.py                 ← Streamlit application (run this)
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
├── tests/
│   └── test_inference.py
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the RACE dataset

Download the single CSV from Kaggle: https://www.kaggle.com/datasets/ankitdhiman7/race-dataset

Place the downloaded file inside `data/raw/` (e.g. `data/raw/race.csv`).

### Step 0 — Split the dataset (80/10/10)

Run the split script to produce the three split files required by the rest of the pipeline:

```bash
cd src
python split_dataset.py
```

This auto-detects any `.csv` in `data/raw/` that is not already named `train.csv`,
`dev.csv`, or `test.csv`, and writes:

```
data/raw/train.csv   ← 80 %
data/raw/dev.csv     ← 10 %
data/raw/test.csv    ← 10 %
```

You can also point it at a specific file:
```bash
python split_dataset.py --source data/raw/race.csv
```

> `train.csv`, `dev.csv`, and `test.csv` are **always** produced from this 80/10/10
> split of the single Kaggle file — they are never separate downloads.

---

## Training Pipeline (run in order)

### Step 1 — Preprocess
```bash
cd src
python preprocessing.py
```
Generates `data/processed/train_model_a.csv`, `dev_model_a.csv`, `test_model_a.csv`.

### Step 2 — Train Model A
```bash
python model_a_train.py
```
Trains Logistic Regression, SVM, KMeans, Label Propagation, and GMM.
Saves all models to `models/model_a/traditional/`.
Prints comparison table at the end.

### Step 3 — Train Model B
```bash
python model_b_train.py
```
Trains the TF-IDF vectorizer used for distractor and hint generation.
Saves to `models/model_b/traditional/model_b_vectorizer.pkl`.

### Step 4 — Run Evaluation
```bash
python evaluate.py
```
Prints Accuracy, Macro F1, Precision, Recall, Exact Match, Confusion Matrix for Model A,
and Precision, Recall, F1, Accuracy for Model B distractors.

---

## Running the UI

```bash
cd ui
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**UI Tabs:**
1. **Article Input** — Load a RACE sample or paste a custom article
2. **Quiz** — Answer MCQs with Model B distractors
3. **Hints** — Reveal progressive hints from Model B
4. **Developer Dashboard** — Model A & B metrics, inference latency, session history CSV export

---

## Using Google Colab (for training on the full dataset)

Training on the full RACE dataset (~88k rows) is slow on a laptop. Use Colab for that.

### Setup on Colab

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone or upload your project to Drive, then:
import os
os.chdir('/content/drive/MyDrive/AI Project')

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Upload the dataset to data/raw/ via Colab file upload or Drive
```

### Run training on Colab

```python
# Run each training step as a cell:
!cd src && python split_dataset.py
!cd src && python preprocessing.py
!cd src && python model_a_train.py
!cd src && python model_b_train.py
```

### Download trained models

After training, download the `models/` folder from Drive to your local machine,
then run the Streamlit UI locally.

> Colab free tier tip: Label Propagation and GMM use a sample of 3000 rows by default,
> so they finish quickly. The main training bottleneck is Model A TF-IDF + GridSearchCV —
> expect ~10–20 minutes on the full dataset.

---

## Notes

- The RACE dataset is loaded from `data/raw/dev.csv` by the UI at runtime.
  Make sure this file exists before running `streamlit run app.py`.
- All trained models must be present in `models/` before running the UI.
  Run Steps 1–3 first, or copy pre-trained `.pkl` files from a teammate.
