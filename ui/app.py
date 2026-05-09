import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DIR = BASE_DIR / "data" / "raw"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from quiz_pipeline import build_quiz_item


st.set_page_config(
    page_title="AI Reading Comprehension Quiz System",
    page_icon="📚",
    layout="wide",
)


@st.cache_data
def load_dataset(split="dev"):
    file_path = RAW_DIR / f"{split}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_csv(file_path)


def get_random_sample(df):
    return df.iloc[random.randint(0, len(df) - 1)]


def get_sample_by_index(df, index):
    return df.iloc[index]


def build_options(row):
    return {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "D": row["D"],
    }


def initialise_state():
    defaults = {
        "quiz_item": None,
        "selected_answer": None,
        "checked": False,
        "hint_count": 0,
        "history": [],
        "quiz_id": 0,
        "logged_quiz_id": None,
        "use_generated_distractors": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz_state():
    st.session_state.selected_answer = None
    st.session_state.checked = False
    st.session_state.hint_count = 0
    st.session_state.logged_quiz_id = None


def normalise_distractors(distractors, correct_answer, fallback_options):
    correct_clean = str(correct_answer).strip().lower()
    fallback_values = [
        str(option).strip()
        for option in fallback_options.values()
        if str(option).strip().lower() != correct_clean
    ]

    cleaned = []
    seen = {correct_clean}

    for item in distractors:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen or "no suitable" in key:
            continue
        cleaned.append(text)
        seen.add(key)
        if len(cleaned) == 3:
            break

    for item in fallback_values:
        key = item.lower()
        if key not in seen:
            cleaned.append(item)
            seen.add(key)
        if len(cleaned) == 3:
            break

    while len(cleaned) < 3:
        cleaned.append("Distractor unavailable")

    return cleaned[:3]


def build_display_options(quiz_item, use_generated_distractors=True):
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
        (label for label, text in display_options.items()
         if str(text).strip().lower() == correct_answer.lower()),
        labels[0],
    )

    return display_options, correct_label, source


def prepare_quiz_item(article, question, options, correct_label):
    quiz_item = build_quiz_item(article, question, options, correct_label)
    display_options, display_correct_label, option_source = build_display_options(
        quiz_item,
        st.session_state.use_generated_distractors,
    )

    quiz_item["display_options"] = display_options
    quiz_item["display_correct_label"] = display_correct_label
    quiz_item["option_source"] = option_source
    return quiz_item


def set_quiz_item(quiz_item):
    st.session_state.quiz_item = quiz_item
    st.session_state.quiz_id += 1
    reset_quiz_state()


def create_quiz_from_row(row):
    article = row["article"]
    question = row["question"]
    options = build_options(row)
    correct_label = row["answer"]
    quiz_item = prepare_quiz_item(article, question, options, correct_label)
    set_quiz_item(quiz_item)


def create_custom_quiz(article, question, options, correct_label):
    quiz_item = prepare_quiz_item(article, question, options, correct_label)
    set_quiz_item(quiz_item)


def append_attempt_once(selected_label, is_correct):
    if st.session_state.logged_quiz_id == st.session_state.quiz_id:
        return

    quiz_item = st.session_state.quiz_item
    selected_text = quiz_item["display_options"][selected_label]

    st.session_state.history.append({
        "question": quiz_item["question"],
        "option_source": quiz_item["option_source"],
        "selected_label": selected_label,
        "selected_answer": selected_text,
        "correct_label": quiz_item["display_correct_label"],
        "correct_answer": quiz_item["correct_answer"],
        "is_correct": is_correct,
        "model_a_prediction": quiz_item["model_a_prediction"],
        "model_a_confidence": quiz_item["model_a_confidence"],
    })
    st.session_state.logged_quiz_id = st.session_state.quiz_id


def render_model_a_scores(quiz_item):
    score_df = pd.DataFrame({
        "Option": list(quiz_item["model_a_scores"].keys()),
        "Original option text": [
            quiz_item["original_options"].get(label, "")
            for label in quiz_item["model_a_scores"].keys()
        ],
        "Score": list(quiz_item["model_a_scores"].values()),
    })
    st.dataframe(score_df, use_container_width=True)


initialise_state()

st.title("Intelligent Reading Comprehension and Quiz Generation System")
st.markdown(
    "Classical ML-based reading comprehension system using TF-IDF, "
    "Logistic Regression, SVM, cosine similarity, distractors, and hints."
)

try:
    df = load_dataset("dev")
except Exception as exc:
    st.error(f"Could not load the RACE dev dataset: {exc}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Article Input",
    "2. Quiz",
    "3. Hints",
    "4. Developer Dashboard",
])


with tab1:
    st.header("Article Input")

    st.session_state.use_generated_distractors = st.checkbox(
        "Use Model B generated distractors in the quiz",
        value=st.session_state.use_generated_distractors,
        help="When disabled, the quiz uses the original RACE answer options.",
    )

    input_mode = st.radio(
        "Choose input mode:",
        ["Load RACE Sample", "Paste Custom Article"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Load RACE Sample":
        sample_index = st.number_input(
            "Sample index",
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
            with st.spinner("Generating quiz from selected sample..."):
                try:
                    create_quiz_from_row(get_sample_by_index(df, sample_index))
                    st.success(f"Sample {sample_index} loaded. Head to the **Quiz** tab to answer.")
                except Exception as exc:
                    st.error(f"Could not generate quiz: {exc}")

        if random_btn:
            with st.spinner("Generating quiz from random sample..."):
                try:
                    create_quiz_from_row(get_random_sample(df))
                    st.success("Random sample loaded. Head to the **Quiz** tab to answer.")
                except Exception as exc:
                    st.error(f"Could not generate quiz: {exc}")

    else:
        st.warning(
            "Custom article mode requires a question and four answer options. "
            "Automatic question generation can be connected later."
        )

        article = st.text_area("Paste your article here:", height=250)
        question = st.text_input("Enter question:")

        col1, col2 = st.columns(2)
        with col1:
            option_a = st.text_input("Option A")
            option_b = st.text_input("Option B")
        with col2:
            option_c = st.text_input("Option C")
            option_d = st.text_input("Option D")

        correct_label = st.selectbox("Correct answer label", ["A", "B", "C", "D"])

        if st.button("Create Custom Quiz", key="create_custom"):
            required_values = [article, question, option_a, option_b, option_c, option_d]
            if not all(str(value).strip() for value in required_values):
                st.error("Please fill all fields before creating the quiz.")
            else:
                options = {
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": option_d,
                }
                with st.spinner("Generating custom quiz..."):
                    try:
                        create_custom_quiz(article, question, options, correct_label)
                        st.success("Custom quiz created. Head to the **Quiz** tab to answer.")
                    except Exception as exc:
                        st.error(f"Could not generate custom quiz: {exc}")

    if st.session_state.quiz_item is not None:
        qi = st.session_state.quiz_item
        with st.expander("Loaded quiz preview", expanded=False):
            st.markdown(f"**Question:** {qi['question']}")
            article_preview = qi["article"][:400]
            st.caption(article_preview + ("…" if len(qi["article"]) > 400 else ""))


with tab2:
    st.header("Question and Answer Quiz")

    _total = len(st.session_state.history)
    _correct = sum(1 for h in st.session_state.history if h["is_correct"])
    _acc = f"{_correct / _total * 100:.0f}%" if _total else "—"
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Questions Answered", _total)
    _c2.metric("Correct", _correct)
    _c3.metric("Accuracy", _acc)

    quiz_item = st.session_state.quiz_item

    if quiz_item is None:
        st.info("Load or create a quiz from the **Article Input** tab first.")
    else:
        q_num = st.session_state.quiz_id
        st.caption(
            f"Question #{q_num} · Options source: {quiz_item['option_source']}"
        )

        word_count = len(quiz_item["article"].split())
        with st.expander(f"Show Article ({word_count} words)", expanded=False):
            st.write(quiz_item["article"])

        st.markdown(
            f"<div style='background:#1e2a3a;border-left:4px solid #4a9eff;"
            f"padding:12px 16px;border-radius:4px;margin:12px 0'>"
            f"<strong>{quiz_item['question']}</strong></div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Choose your answer:**")
        display_options = quiz_item["display_options"]
        labels = list(display_options.keys())
        default_index = (
            labels.index(st.session_state.selected_answer)
            if st.session_state.selected_answer in labels
            else None
        )

        selected = st.radio(
            "Options",
            labels,
            index=default_index,
            format_func=lambda label: f"{label}. {display_options[label]}",
            key=f"answer_radio_{st.session_state.quiz_id}",
            label_visibility="collapsed",
        )
        st.session_state.selected_answer = selected

        check_disabled = st.session_state.checked or selected is None
        if st.button("Check Answer", key="check_answer", disabled=check_disabled, type="primary"):
            st.session_state.checked = True
            is_correct = selected == quiz_item["display_correct_label"]
            append_attempt_once(selected, is_correct)

        if st.session_state.checked:
            selected = st.session_state.selected_answer
            is_correct = selected == quiz_item["display_correct_label"]

            if is_correct:
                st.success("Correct answer!")
            else:
                st.error(
                    f"Incorrect. Correct answer is "
                    f"{quiz_item['display_correct_label']}. {quiz_item['correct_answer']}"
                )

            model_agrees = quiz_item["model_a_prediction"] == quiz_item["correct_label"]
            st.info(
                f"Model A predicted original RACE option **{quiz_item['model_a_prediction']}** "
                f"(confidence {quiz_item['model_a_confidence']:.4f}). "
                f"{'Matches' if model_agrees else 'Does not match'} "
                f"the dataset gold label ({quiz_item['correct_label']})."
            )

            with st.expander("Model A Scores on Original RACE Options"):
                render_model_a_scores(quiz_item)

            st.divider()
            if st.button("Next Random Question →", key="next_question_btn", type="primary"):
                with st.spinner("Generating next quiz..."):
                    try:
                        create_quiz_from_row(get_random_sample(df))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate quiz: {exc}")


with tab3:
    st.header("Hint Panel")

    quiz_item = st.session_state.quiz_item

    if quiz_item is None:
        st.info("Load or create a quiz first.")
    else:
        hints = quiz_item.get("generated_hints", [])
        st.write("Reveal hints gradually before revealing the answer.")

        if not hints:
            st.warning("No hints were generated for this quiz.")
        else:
            revealed = st.session_state.hint_count
            total_hints = len(hints)
            next_hint_disabled = revealed >= total_hints

            hint_col, prog_col = st.columns([2, 1])
            if hint_col.button(
                "Show Next Hint",
                key=f"show_hint_{st.session_state.quiz_id}",
                disabled=next_hint_disabled,
            ):
                st.session_state.hint_count += 1
                revealed = st.session_state.hint_count

            prog_col.markdown(
                f"<div style='padding-top:8px;color:#aaa'>"
                f"{revealed} / {total_hints} hints revealed</div>",
                unsafe_allow_html=True,
            )

            for i in range(st.session_state.hint_count):
                st.info(f"Hint {i + 1}: {hints[i]}")

            if st.session_state.hint_count >= len(hints):
                st.caption("All hints revealed.")
                if st.button(
                    "Reveal Answer",
                    key=f"reveal_answer_{st.session_state.quiz_id}",
                ):
                    st.success(
                        f"Answer: {quiz_item['display_correct_label']}. "
                        f"{quiz_item['correct_answer']}"
                    )


with tab4:
    st.header("Developer / Analytics Dashboard")

    quiz_item = st.session_state.quiz_item

    col1, col2, col3 = st.columns(3)

    total_attempts = len(st.session_state.history)
    correct_attempts = sum(1 for item in st.session_state.history if item["is_correct"])
    accuracy = correct_attempts / total_attempts if total_attempts else 0

    col1.metric("Total Attempts", total_attempts)
    col2.metric("User Accuracy", f"{accuracy * 100:.1f}%")
    col3.metric("Dataset Samples", len(df))

    if quiz_item is not None:
        st.subheader("Current Model A Output")
        ma_c1, ma_c2 = st.columns(2)
        ma_c1.metric("Predicted Option", quiz_item["model_a_prediction"])
        ma_c2.metric("Confidence", f"{quiz_item['model_a_confidence']:.4f}")

        render_model_a_scores(quiz_item)

        score_chart_df = pd.DataFrame({
            "Option": list(quiz_item["model_a_scores"].keys()),
            "Score": list(quiz_item["model_a_scores"].values()),
        })
        st.bar_chart(score_chart_df.set_index("Option"))

        st.subheader("Model B Output")
        distractors = quiz_item.get("generated_distractors", [])
        hints = quiz_item.get("generated_hints", [])

        mb_c1, mb_c2 = st.columns(2)
        with mb_c1:
            st.markdown("**Generated Distractors**")
            for d in distractors:
                st.write(f"- {d}")
        with mb_c2:
            st.markdown("**Generated Hints**")
            for h in hints:
                st.write(f"- {h}")

    if st.session_state.history:
        st.subheader("Session History")
        history_df = pd.DataFrame(st.session_state.history).rename(columns={
            "question": "Question",
            "option_source": "Option Source",
            "selected_label": "Your Label",
            "selected_answer": "Your Answer",
            "correct_label": "Correct Label",
            "correct_answer": "Correct Answer",
            "is_correct": "Correct?",
            "model_a_prediction": "Model A Prediction",
            "model_a_confidence": "Model A Confidence",
        })
        st.dataframe(history_df, use_container_width=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Session History CSV",
            csv,
            f"quiz_history_{timestamp}.csv",
            "text/csv",
        )
