import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_A_DIR = BASE_DIR / "models" / "model_a" / "traditional"
MODEL_B_DIR = BASE_DIR / "models" / "model_b" / "traditional"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# ── Pre-flight: check that all required .pkl files exist before importing ──────
_REQUIRED_MODELS = [
    MODEL_A_DIR / "logistic_regression.pkl",
    MODEL_A_DIR / "linear_svm.pkl",
    MODEL_A_DIR / "tfidf_vectorizer.pkl",
    MODEL_B_DIR / "model_b_vectorizer.pkl",
    MODEL_B_DIR / "model_b_rf_ranker.pkl",
    MODEL_B_DIR / "model_b_hint_lr.pkl",
]

_missing_models = [str(p) for p in _REQUIRED_MODELS if not p.exists()]


def _import_quiz_pipeline():
    """Import quiz_pipeline only after confirming model files exist."""
    try:
        from quiz_pipeline import build_generated_quiz_items, build_quiz_item
        return build_quiz_item, build_generated_quiz_items
    except Exception as exc:
        return exc


st.set_page_config(
    page_title="AI Reading Comprehension Quiz System",
    page_icon="\U0001f4da",
    layout="wide",
)


# ── Dataset loader — cached so the CSV is only read once per session ──────────
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(split: str = "dev") -> pd.DataFrame:
    file_path = RAW_DIR / f"{split}.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n\n"
            "Run the data download step first (see README — Step 0b)."
        )
    return pd.read_csv(file_path)


def get_random_sample(df: pd.DataFrame):
    valid_df = get_valid_samples(df)
    if valid_df.empty:
        raise ValueError("No valid RACE samples found; article, question, options, and answer are required.")
    return valid_df.iloc[random.randint(0, len(valid_df) - 1)]


def get_sample_by_index(df: pd.DataFrame, index: int):
    return df.iloc[index]


def get_valid_samples(df: pd.DataFrame) -> pd.DataFrame:
    required = ["article", "question", "A", "B", "C", "D", "answer"]
    valid = df.dropna(subset=required)
    valid = valid[valid["answer"].isin(["A", "B", "C", "D"])]
    return valid


def build_options(row) -> dict:
    return {
        "A": "" if pd.isna(row["A"]) else str(row["A"]),
        "B": "" if pd.isna(row["B"]) else str(row["B"]),
        "C": "" if pd.isna(row["C"]) else str(row["C"]),
        "D": "" if pd.isna(row["D"]) else str(row["D"]),
    }


# ── Session state initialisation ───────────────────────────────────────────────
def initialise_state():
    defaults = {
        "quiz_item": None,
        "quiz_items": [],
        "quiz_set_index": 0,
        "selected_answers": {},
        "checked_variants": {},
        "hint_count": 0,
        "history": [],
        "quiz_id": 0,
        "logged_attempt_keys": [],
        "use_generated_distractors": True,
        "mcq_count": 1,
        "inference_latencies": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz_state():
    st.session_state.selected_answers = {}
    st.session_state.checked_variants = {}
    st.session_state.hint_count = 0
    st.session_state.logged_attempt_keys = []


# ── Distractor normalisation ───────────────────────────────────────────────────
def normalise_distractors(
    distractors: list, correct_answer: str, fallback_options: dict, needed: int = 3
) -> list:
    """Return exactly `needed` distractor strings, falling back to original options."""
    correct_clean = str(correct_answer).strip().lower()
    fallback_values = [
        str(v).strip()
        for v in fallback_options.values()
        if str(v).strip().lower() != correct_clean
    ]

    cleaned, seen = [], {correct_clean}
    for item in distractors:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen or "no suitable" in key or "unavailable" in key:
            continue
        cleaned.append(text)
        seen.add(key)
        if len(cleaned) == needed:
            break

    for item in fallback_values:
        key = item.lower()
        if key not in seen:
            cleaned.append(item)
            seen.add(key)
        if len(cleaned) == needed:
            break

    while len(cleaned) < needed:
        cleaned.append("(distractor unavailable)")

    return cleaned[:needed]


# ── Option set builders ────────────────────────────────────────────────────────
def build_display_options(quiz_item: dict, use_generated_distractors: bool = True):
    correct_answer = str(quiz_item["correct_answer"]).strip()
    if use_generated_distractors:
        distractors = normalise_distractors(
            quiz_item.get("generated_distractors", []),
            correct_answer,
            quiz_item["original_options"],
        )
        option_texts = distractors + [correct_answer]
        source = "Model B generated distractors"
    else:
        option_texts = list(quiz_item["original_options"].values())
        source = "Original RACE options"

    random.shuffle(option_texts)
    labels = ["A", "B", "C", "D"]
    display_options = dict(zip(labels, option_texts))
    correct_label = next(
        (lbl for lbl, txt in display_options.items()
         if str(txt).strip().lower() == correct_answer.lower()),
        labels[0],
    )
    return display_options, correct_label, source


def build_mcq_variants(
    quiz_item: dict, use_generated_distractors: bool = True, variant_count: int = 1
) -> list:
    """Build exactly one answer-option set for one question."""
    correct_answer = str(quiz_item["correct_answer"]).strip()
    labels = ["A", "B", "C", "D"]
    variants = []
    variant_count = 1

    if use_generated_distractors:
        pool = normalise_distractors(
            quiz_item.get("generated_distractors", []),
            correct_answer,
            quiz_item["original_options"],
            needed=3,
        )
        option_source = "Model B generated distractors"
    else:
        pool = [
            str(v).strip()
            for v in quiz_item["original_options"].values()
            if str(v).strip().lower() != correct_answer.lower()
        ]
        option_source = "Original RACE options"

    for idx in range(variant_count):
        if use_generated_distractors:
            start = idx * 3
            distractors = pool[start : start + 3]
            if len(distractors) < 3:
                distractors = (distractors + pool)[:3]
        else:
            distractors = pool[:3]

        option_texts = distractors + [correct_answer]
        random.shuffle(option_texts)
        display_options = dict(zip(labels, option_texts))
        correct_label = next(
            (lbl for lbl, txt in display_options.items()
             if str(txt).strip().lower() == correct_answer.lower()),
            labels[0],
        )
        variants.append(
            {
                "variant_index": idx,
                "display_options": display_options,
                "display_correct_label": correct_label,
                "option_source": option_source,
            }
        )

    return variants


# ── Quiz item construction ─────────────────────────────────────────────────────
def prepare_quiz_item(
    article: str, question: str, options: dict, correct_label: str
) -> dict:
    t0 = time.time()
    quiz_item = _build_quiz_item(article, question, options, correct_label)
    latency = time.time() - t0
    st.session_state.inference_latencies.append(latency)
    return enrich_quiz_item(quiz_item)


def enrich_quiz_item(quiz_item: dict) -> dict:
    """Attach UI display options to one quiz item without creating duplicates."""
    quiz_item["mcq_variants"] = build_mcq_variants(
        quiz_item,
        st.session_state.use_generated_distractors,
        1,
    )
    first = quiz_item["mcq_variants"][0]
    quiz_item["display_options"] = first["display_options"]
    quiz_item["display_correct_label"] = first["display_correct_label"]
    quiz_item["option_source"] = first["option_source"]
    return quiz_item


def set_quiz_item(quiz_item: dict):
    st.session_state.quiz_items = [quiz_item]
    st.session_state.quiz_set_index = 0
    st.session_state.quiz_item = quiz_item
    st.session_state.quiz_id += 1
    reset_quiz_state()


def set_quiz_items(quiz_items: list):
    st.session_state.quiz_items = quiz_items
    st.session_state.quiz_set_index = 0
    st.session_state.quiz_item = quiz_items[0] if quiz_items else None
    st.session_state.quiz_id += 1
    reset_quiz_state()


def select_quiz_item(index: int):
    """Switch between distinct generated questions from the same article."""
    quiz_items = st.session_state.quiz_items
    if not quiz_items:
        return
    st.session_state.quiz_set_index = max(0, min(index, len(quiz_items) - 1))
    st.session_state.quiz_item = quiz_items[st.session_state.quiz_set_index]
    st.session_state.quiz_id += 1
    reset_quiz_state()


def create_quiz_from_row(row):
    quiz_item = prepare_quiz_item(
        row["article"], row["question"], build_options(row), row["answer"]
    )
    set_quiz_item(quiz_item)


def create_custom_quiz(
    article: str, question: str, options: dict, correct_label: str
):
    quiz_item = prepare_quiz_item(article, question, options, correct_label)
    set_quiz_item(quiz_item)


def create_generated_quizzes(article: str, top_k: int = 5):
    t0 = time.time()
    quiz_items = _build_generated_quiz_items(article, top_k=top_k)
    st.session_state.inference_latencies.append(time.time() - t0)
    quiz_items = [enrich_quiz_item(item) for item in quiz_items]
    set_quiz_items(quiz_items)


# ── History helper ─────────────────────────────────────────────────────────────
def append_attempt_once(variant: dict, selected_label: str, is_correct: bool):
    attempt_key = f"{st.session_state.quiz_id}:{variant['variant_index']}"
    if attempt_key in st.session_state.logged_attempt_keys:
        return
    qi = st.session_state.quiz_item
    st.session_state.history.append(
        {
            "question": qi["question"],
            "mcq_number": variant["variant_index"] + 1,
            "option_source": variant["option_source"],
            "selected_label": selected_label,
            "selected_answer": variant["display_options"][selected_label],
            "correct_label": variant["display_correct_label"],
            "correct_answer": qi["correct_answer"],
            "is_correct": is_correct,
            "model_a_prediction": qi["model_a_prediction"],
            "model_a_confidence": qi["model_a_confidence"],
        }
    )
    st.session_state.logged_attempt_keys.append(attempt_key)


# ── Shared widget ──────────────────────────────────────────────────────────────
def render_model_a_scores(quiz_item: dict):
    score_df = pd.DataFrame(
        {
            "Option": list(quiz_item["model_a_scores"].keys()),
            "Original option text": [
                quiz_item["original_options"].get(lbl, "")
                for lbl in quiz_item["model_a_scores"].keys()
            ],
            "Score": list(quiz_item["model_a_scores"].values()),
        }
    )
    st.dataframe(score_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

initialise_state()

st.title("Intelligent Reading Comprehension and Quiz Generation System")
st.markdown(
    "Classical ML-based reading comprehension system using TF-IDF, "
    "Logistic Regression, SVM, cosine similarity, distractors, and hints."
)

# ── Guard: missing model files ─────────────────────────────────────────────────
if _missing_models:
    st.error(
        "**Required model files are missing.** "
        "Train the models first by running the pipeline in order:\n\n"
        "```\n"
        "cd src\n"
        "python preprocessing.py\n"
        "python model_a_train.py\n"
        "python model_b_train.py\n"
        "```\n\n"
        "**Missing files:**\n"
        + "\n".join(f"- `{p}`" for p in _missing_models)
    )
    st.stop()

# ── Import quiz pipeline after model check ─────────────────────────────────────
_pipeline_import = _import_quiz_pipeline()
if isinstance(_pipeline_import, Exception):
    st.error(
        f"**Failed to load quiz pipeline:** {_pipeline_import}\n\n"
        "Make sure all model files are present and dependencies are installed."
    )
    st.stop()

_build_quiz_item, _build_generated_quiz_items = _pipeline_import

# ── Load dataset with a clear error if missing ────────────────────────────────
try:
    df = load_dataset("dev")
except FileNotFoundError as exc:
    st.error(
        f"**Dataset not found.**\n\n{exc}\n\n"
        "Download the RACE dataset and place `dev.csv` in `data/raw/`."
    )
    st.stop()
except Exception as exc:
    st.error(f"**Unexpected error loading dataset:** {exc}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Article Input", "2. Quiz", "3. Hints", "4. Developer Dashboard"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARTICLE INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Article Input")
    st.markdown(
        "Choose how to load an article, then head to the **Quiz** tab to answer."
    )

    st.session_state.use_generated_distractors = st.checkbox(
        "Use Model B generated distractors",
        value=st.session_state.use_generated_distractors,
        help=(
            "When enabled, wrong answer options are generated by Model B from the article text. "
            "When disabled, the original RACE answer options are used."
        ),
    )

    st.divider()

    input_mode = st.radio(
        "Input mode:",
        ["Load RACE Sample", "Paste Custom Article"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Load RACE Sample":
        sample_index = st.number_input(
            f"Sample index (0 – {len(df) - 1})",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
        )
        btn_col1, btn_col2 = st.columns(2)
        load_selected_btn = btn_col1.button(
            "Load Selected Sample", key="load_selected", use_container_width=True
        )
        random_btn = btn_col2.button(
            "Load Random Sample", key="load_random", use_container_width=True
        )

        if load_selected_btn:
            with st.spinner("Running Model A + B inference…"):
                try:
                    create_quiz_from_row(get_sample_by_index(df, sample_index))
                    st.success(
                        f"Sample {sample_index} loaded. "
                        "Head to the **Quiz** tab to answer."
                    )
                except Exception as exc:
                    st.error(f"Could not generate quiz: {exc}")
                    st.exception(exc)

        if random_btn:
            with st.spinner("Running Model A + B inference…"):
                try:
                    create_quiz_from_row(get_random_sample(df))
                    st.success("Random sample loaded. Head to the **Quiz** tab to answer.")
                except Exception as exc:
                    st.error(f"Could not generate quiz: {exc}")
                    st.exception(exc)

    else:
        st.info(
            "Paste or upload any article and provide a question with four answer options. "
            "Model A will predict the correct answer and Model B will generate hints."
        )
        uploaded_file = st.file_uploader(
            "Upload article (.txt file):",
            type=["txt"],
            help="Upload a plain-text reading passage. Its content will be placed in the text area below.",
            key="article_uploader",
        )
        default_article = ""
        if uploaded_file is not None:
            try:
                default_article = uploaded_file.read().decode("utf-8")
                st.success(f"Loaded '{uploaded_file.name}' ({len(default_article.split())} words).")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
        article = st.text_area(
            "Article:",
            value=default_article,
            height=250,
            placeholder="Paste your reading passage here…",
        )
        generated_count = st.number_input(
            "Generated questions from this article",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Creates distinct template-based questions from the article instead of shuffled variants.",
        )
        if st.button("Generate Article MCQs", key="generate_custom_questions", type="primary"):
            if not str(article).strip():
                st.error("Please paste an article first.")
            else:
                with st.spinner("Generating template-based MCQs"):
                    try:
                        create_generated_quizzes(article, top_k=int(generated_count))
                        if st.session_state.quiz_item is None:
                            st.warning("No generated questions were produced for this article.")
                        else:
                            st.success(
                                f"Generated {len(st.session_state.quiz_items)} distinct MCQ(s). "
                                "Head to the **Quiz** tab to answer."
                            )
                    except Exception as exc:
                        st.error(f"Could not generate MCQs: {exc}")
                        st.exception(exc)

        st.divider()
        st.caption("Or provide one manual MCQ for this article.")
        question = st.text_input("Question:", placeholder="Enter the comprehension question")

        col1, col2 = st.columns(2)
        with col1:
            option_a = st.text_input("Option A", key="opt_a")
            option_b = st.text_input("Option B", key="opt_b")
        with col2:
            option_c = st.text_input("Option C", key="opt_c")
            option_d = st.text_input("Option D", key="opt_d")

        correct_label = st.selectbox(
            "Correct answer label",
            ["A", "B", "C", "D"],
            help="Which option above is the ground-truth answer?",
        )

        if st.button("Create Custom Quiz", key="create_custom", type="primary"):
            required_values = [article, question, option_a, option_b, option_c, option_d]
            if not all(str(v).strip() for v in required_values):
                st.error("Please fill in all fields (article, question, and all four options).")
            else:
                options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}
                with st.spinner("Running Model A + B inference…"):
                    try:
                        create_custom_quiz(article, question, options, correct_label)
                        st.success("Custom quiz created. Head to the **Quiz** tab to answer.")
                    except Exception as exc:
                        st.error(f"Could not generate custom quiz: {exc}")
                        st.exception(exc)

    if st.session_state.quiz_item is not None:
        qi = st.session_state.quiz_item
        with st.expander("Loaded quiz preview", expanded=False):
            st.markdown(f"**Question:** {qi['question']}")
            article_preview = qi["article"][:400]
            st.caption(article_preview + ("…" if len(qi["article"]) > 400 else ""))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Question and Answer Quiz")

    _total = len(st.session_state.history)
    _correct = sum(1 for h in st.session_state.history if h["is_correct"])
    _acc = f"{_correct / _total * 100:.0f}%" if _total else "—"
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Questions Answered", _total)
    _c2.metric("Correct", _correct)
    _c3.metric("Session Accuracy", _acc)

    quiz_item = st.session_state.quiz_item

    if quiz_item is None:
        st.info("Load or create a quiz from the **Article Input** tab first.")
    else:
        q_num = st.session_state.quiz_id
        variants = quiz_item.get("mcq_variants", [])
        quiz_items = st.session_state.quiz_items
        set_index = st.session_state.quiz_set_index
        if len(quiz_items) > 1:
            st.caption(f"Generated question {set_index + 1} of {len(quiz_items)}")
            nav_prev, nav_next = st.columns(2)
            if nav_prev.button(
                "Previous Question",
                key=f"prev_generated_{st.session_state.quiz_id}",
                disabled=set_index <= 0,
                use_container_width=True,
            ):
                select_quiz_item(set_index - 1)
                st.rerun()
            if nav_next.button(
                "Next Question",
                key=f"next_generated_{st.session_state.quiz_id}",
                disabled=set_index >= len(quiz_items) - 1,
                use_container_width=True,
            ):
                select_quiz_item(set_index + 1)
                st.rerun()
        else:
            st.caption(f"Question #{q_num}")

        word_count = len(quiz_item["article"].split())
        with st.expander(f"Show Article ({word_count} words)", expanded=False):
            st.write(quiz_item["article"])

        st.markdown(
            "<div style='background:#1e2a3a;border-left:4px solid #4a9eff;"
            "padding:12px 16px;border-radius:4px;margin:12px 0'>"
            f"<strong>{quiz_item['question']}</strong></div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Choose your answer:**")
        for variant in variants:
            variant_index = variant["variant_index"]
            selected_key = str(variant_index)
            display_options = variant["display_options"]
            labels = list(display_options.keys())
            selected_answer = st.session_state.selected_answers.get(selected_key)
            default_index = (
                labels.index(selected_answer) if selected_answer in labels else None
            )

            with st.container(border=True):
                st.markdown(
                    f"**Answer Options**  "
                    f"<span style='color:#888;font-size:0.85em'>"
                    f"({variant['option_source']})</span>",
                    unsafe_allow_html=True,
                )
                selected = st.radio(
                    "Options",
                    labels,
                    index=default_index,
                    format_func=lambda lbl: f"{lbl}. {display_options[lbl]}",
                    key=f"answer_radio_{st.session_state.quiz_id}_{variant_index}",
                    label_visibility="collapsed",
                )
                st.session_state.selected_answers[selected_key] = selected

                already_checked = st.session_state.checked_variants.get(selected_key, False)
                if st.button(
                    f"Check Answer",
                    key=f"check_answer_{st.session_state.quiz_id}_{variant_index}",
                    disabled=already_checked or selected is None,
                    type="primary",
                ):
                    is_correct = selected == variant["display_correct_label"]
                    st.session_state.checked_variants[selected_key] = True
                    append_attempt_once(variant, selected, is_correct)
                    st.rerun()

                if st.session_state.checked_variants.get(selected_key, False):
                    sel = st.session_state.selected_answers.get(selected_key)
                    is_correct = sel == variant["display_correct_label"]
                    if is_correct:
                        st.success("Correct!")
                    else:
                        st.error(
                            f"Incorrect. The correct answer is "
                            f"**{variant['display_correct_label']}**: "
                            f"{quiz_item['correct_answer']}"
                        )

        if any(st.session_state.checked_variants.values()):
            first_variant = variants[0] if variants else {}
            predicted_orig = quiz_item["model_a_prediction"]
            gold_orig = quiz_item["correct_label"]
            display_correct = first_variant.get("display_correct_label", gold_orig)

            # Find which shuffled display label holds Model A's predicted option text
            predicted_text = quiz_item["original_options"].get(predicted_orig, "")
            predicted_display = next(
                (lbl for lbl, txt in first_variant.get("display_options", {}).items()
                 if str(txt).strip().lower() == str(predicted_text).strip().lower()),
                None,
            )

            model_agrees = (predicted_orig == gold_orig)
            agree_icon = "✓" if model_agrees else "✗"

            if predicted_display:
                pred_note = f"**{predicted_orig}** (shown as **{predicted_display}** in the shuffled quiz)"
            else:
                pred_note = f"**{predicted_orig}** (replaced by a generated distractor in this quiz)"

            st.info(
                f"**Model A** scored the original RACE options and predicted "
                f"{pred_note} "
                f"(confidence {quiz_item['model_a_confidence']:.3f}).  "
                f"{agree_icon} {'Matches' if model_agrees else 'Does not match'} "
                f"the dataset gold label (RACE **{gold_orig}**, "
                f"displayed as **{display_correct}** in this quiz)."
            )
            with st.expander("Model A Scores on Original RACE Options"):
                render_model_a_scores(quiz_item)

            st.divider()
            if st.button("Next Random Question →", key="next_question_btn", type="primary"):
                with st.spinner("Running Model A + B inference…"):
                    try:
                        create_quiz_from_row(get_random_sample(df))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate quiz: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HINTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Hint Panel")
    st.markdown(
        "Reveal hints one at a time to guide you to the answer. "
        "Hints are ranked from most general to most specific."
    )

    quiz_item = st.session_state.quiz_item

    if quiz_item is None:
        st.info("Load or create a quiz from the **Article Input** tab first.")
    else:
        st.markdown(f"**Question:** {quiz_item['question']}")
        hints = quiz_item.get("generated_hints", [])

        if not hints:
            st.warning("No hints were generated for this question.")
        else:
            revealed = st.session_state.hint_count
            total_hints = len(hints)

            # Graduated hint level metadata
            _hint_levels = [
                ("General",       "#2d4a2d", "🟢"),
                ("Specific",      "#4a3a1a", "🟡"),
                ("Near-explicit", "#4a1a1a", "🔴"),
            ]

            hint_col, prog_col = st.columns([2, 1])
            if hint_col.button(
                "Show Next Hint",
                key=f"show_hint_{st.session_state.quiz_id}",
                disabled=revealed >= total_hints,
                type="secondary",
            ):
                st.session_state.hint_count += 1
                revealed = st.session_state.hint_count

            prog_col.markdown(
                f"<div style='padding-top:8px;color:#aaa'>"
                f"{revealed} / {total_hints} hints revealed</div>",
                unsafe_allow_html=True,
            )

            st.divider()
            for i in range(st.session_state.hint_count):
                level_label, bg_color, icon = _hint_levels[i] if i < len(_hint_levels) else ("Hint", "#2d2d4a", "💡")
                hint_text = hints[i]
                # Strip backend prefix if present (e.g. "Hint 1 (General): ...")
                for prefix in [f"Hint {i+1} (General): ", f"Hint {i+1} (Specific): ",
                                f"Hint {i+1} (Near-explicit): ", f"Hint {i+1}: "]:
                    if hint_text.startswith(prefix):
                        hint_text = hint_text[len(prefix):]
                        break
                st.markdown(
                    f"<div style='background:{bg_color};border-radius:6px;"
                    f"padding:12px 16px;margin:6px 0'>"
                    f"{icon} <strong>Hint {i+1} — {level_label}</strong><br/>"
                    f"<span style='font-size:0.97em'>{hint_text}</span></div>",
                    unsafe_allow_html=True,
                )

            if revealed < total_hints:
                st.caption(f"Click **Show Next Hint** to reveal hint {revealed + 1} of {total_hints}.")

            # Reveal Answer only appears after ALL hints have been viewed
            if st.session_state.hint_count >= total_hints:
                st.divider()
                st.caption("✅ All hints revealed. You may now reveal the answer.")
                if st.button(
                    "Reveal Answer",
                    key=f"reveal_answer_{st.session_state.quiz_id}",
                    type="primary",
                ):
                    st.success(
                        f"**Answer: {quiz_item['display_correct_label']}.**  "
                        f"{quiz_item['correct_answer']}"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEVELOPER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Developer / Analytics Dashboard")

    quiz_item = st.session_state.quiz_item

    # Top-level session metrics
    total_attempts = len(st.session_state.history)
    correct_attempts = sum(1 for h in st.session_state.history if h["is_correct"])
    user_acc = correct_attempts / total_attempts if total_attempts else 0
    latencies = st.session_state.inference_latencies
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Attempts", total_attempts)
    c2.metric("User Accuracy", f"{user_acc * 100:.1f}%")
    c3.metric("Dataset Samples", len(df))
    c4.metric("Avg Inference (s)", f"{avg_latency:.2f}s")

    if latencies:
        with st.expander("Inference Latency per Request", expanded=False):
            lat_df = pd.DataFrame(
                {"Request #": list(range(1, len(latencies) + 1)), "Latency (s)": latencies}
            )
            st.line_chart(lat_df.set_index("Request #"))
            st.caption(
                f"Min: {min(latencies):.2f}s  |  Max: {max(latencies):.2f}s  |  Avg: {avg_latency:.2f}s"
            )

    # Model A — current inference
    if quiz_item is not None:
        st.subheader("Model A — Current Inference")
        ma_c1, ma_c2, ma_c3 = st.columns(3)
        ma_c1.metric("Predicted Option", quiz_item["model_a_prediction"])
        ma_c2.metric("Confidence", f"{quiz_item['model_a_confidence']:.4f}")
        ma_c3.metric(
            "Correct?",
            "Yes" if quiz_item["model_a_prediction"] == quiz_item["correct_label"] else "No",
        )
        render_model_a_scores(quiz_item)
        score_chart_df = pd.DataFrame(
            {
                "Option": list(quiz_item["model_a_scores"].keys()),
                "Score": list(quiz_item["model_a_scores"].values()),
            }
        )
        st.bar_chart(score_chart_df.set_index("Option"))

    # Model A — session accuracy
    if total_attempts:
        st.subheader("Model A — Session Accuracy")
        model_a_correct = sum(
            1 for h in st.session_state.history
            if h.get("model_a_prediction") == h.get("correct_label")
        )
        model_a_acc = model_a_correct / total_attempts
        ma_s1, ma_s2, ma_s3 = st.columns(3)
        ma_s1.metric("Model A Accuracy", f"{model_a_acc * 100:.1f}%")
        ma_s2.metric("Model A Correct", model_a_correct)
        ma_s3.metric("Total Evaluated", total_attempts)

    # Model B — distractor & hint quality
    if quiz_item is not None:
        st.subheader("Model B — Distractor & Hint Quality")
        distractors = quiz_item.get("generated_distractors", [])
        hints = quiz_item.get("generated_hints", [])
        correct_answer = str(quiz_item.get("correct_answer", "")).strip().lower()

        valid_dist = sum(
            1 for d in distractors
            if str(d).strip().lower() != correct_answer
            and "no suitable" not in str(d).lower()
            and "unavailable" not in str(d).lower()
        )
        total_dist = max(len(distractors), 1)
        dist_acc = valid_dist / total_dist
        dist_recall = min(valid_dist / 3, 1.0)

        mb_m1, mb_m2, mb_m3, mb_m4 = st.columns(4)
        mb_m1.metric("Distractor Accuracy", f"{dist_acc * 100:.1f}%")
        mb_m2.metric("Distractor Precision", f"{dist_acc * 100:.1f}%")
        mb_m3.metric("Distractor Recall", f"{dist_recall * 100:.1f}%")
        mb_m4.metric("Hints Generated", len(hints))

        mb_c1, mb_c2 = st.columns(2)
        with mb_c1:
            st.markdown("**Generated Distractors**")
            for d in distractors:
                is_valid = (
                    str(d).strip().lower() != correct_answer
                    and "no suitable" not in str(d).lower()
                )
                icon = "✅" if is_valid else "⚠️"
                st.write(f"{icon} {d}")
        with mb_c2:
            st.markdown("**Generated Hints**")
            for i, h in enumerate(hints, 1):
                st.write(f"**{i}.** {h}")

    # Session history & CSV export
    if st.session_state.history:
        st.subheader("Session History")
        history_df = pd.DataFrame(st.session_state.history).rename(
            columns={
                "question": "Question",
                "mcq_number": "MCQ #",
                "option_source": "Option Source",
                "selected_label": "Your Label",
                "selected_answer": "Your Answer",
                "correct_label": "Correct Label",
                "correct_answer": "Correct Answer",
                "is_correct": "Correct?",
                "model_a_prediction": "Model A Prediction",
                "model_a_confidence": "Model A Confidence",
            }
        )
        st.dataframe(history_df, use_container_width=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Session History CSV",
            csv,
            f"quiz_history_{timestamp}.csv",
            "text/csv",
        )
