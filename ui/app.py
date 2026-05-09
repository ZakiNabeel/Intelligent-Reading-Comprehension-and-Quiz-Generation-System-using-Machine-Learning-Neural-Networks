import random
import sys
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
    page_icon="book",
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
        label
        for label, text in display_options.items()
        if str(text).strip().lower() == correct_answer.lower()
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
        col1, col2 = st.columns([1, 1])

        with col1:
            sample_index = st.number_input(
                "Sample index",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1,
            )

        with col2:
            random_btn = st.button("Load Random Sample", key="load_random")

        if random_btn:
            with st.spinner("Generating quiz from random sample..."):
                try:
                    create_quiz_from_row(get_random_sample(df))
                    st.success("Random sample loaded successfully.")
                except Exception as exc:
                    st.error(f"Could not generate quiz: {exc}")

        if st.button("Load Selected Sample", key="load_selected"):
            with st.spinner("Generating quiz from selected sample..."):
                try:
                    create_quiz_from_row(get_sample_by_index(df, sample_index))
                    st.success(f"Sample {sample_index} loaded successfully.")
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
                        st.success("Custom quiz created successfully.")
                    except Exception as exc:
                        st.error(f"Could not generate custom quiz: {exc}")


with tab2:
    st.header("Question and Answer Quiz")

    quiz_item = st.session_state.quiz_item

    if quiz_item is None:
        st.info("Load or create a quiz from the Article Input tab first.")
    else:
        st.caption(f"Quiz options source: {quiz_item['option_source']}")

        st.subheader("Article")
        with st.expander("Show Article", expanded=True):
            st.write(quiz_item["article"])

        st.subheader("Question")
        st.write(quiz_item["question"])

        st.subheader("Options")
        display_options = quiz_item["display_options"]
        labels = list(display_options.keys())
        default_index = (
            labels.index(st.session_state.selected_answer)
            if st.session_state.selected_answer in labels
            else None
        )

        selected = st.radio(
            "Choose your answer:",
            labels,
            index=default_index,
            format_func=lambda label: f"{label}. {display_options[label]}",
            key=f"answer_radio_{st.session_state.quiz_id}",
        )
        st.session_state.selected_answer = selected

        check_disabled = st.session_state.checked or selected is None
        if st.button("Check Answer", key="check_answer", disabled=check_disabled):
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
                f"Model A predicted original RACE option {quiz_item['model_a_prediction']} "
                f"with confidence {quiz_item['model_a_confidence']:.4f}. "
                f"Model A {'matched' if model_agrees else 'did not match'} "
                f"the dataset gold label ({quiz_item['correct_label']})."
            )

            st.subheader("Model A Scores on Original RACE Options")
            render_model_a_scores(quiz_item)


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
            next_hint_disabled = st.session_state.hint_count >= len(hints)
            if st.button(
                "Show Next Hint",
                key=f"show_hint_{st.session_state.quiz_id}",
                disabled=next_hint_disabled,
            ):
                st.session_state.hint_count += 1

            for i in range(st.session_state.hint_count):
                st.info(f"Hint {i + 1}: {hints[i]}")

            if st.session_state.hint_count >= len(hints):
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
        st.write("Prediction:", quiz_item["model_a_prediction"])
        st.write("Confidence:", round(quiz_item["model_a_confidence"], 4))
        render_model_a_scores(quiz_item)

        score_df = pd.DataFrame({
            "Option": list(quiz_item["model_a_scores"].keys()),
            "Score": list(quiz_item["model_a_scores"].values()),
        })
        st.bar_chart(score_df.set_index("Option"))

        st.subheader("Model B Output")

        st.write("Generated Distractors:")
        for distractor in quiz_item.get("generated_distractors", []):
            st.write("-", distractor)

        st.write("Generated Hints:")
        for hint in quiz_item.get("generated_hints", []):
            st.write("-", hint)

    if st.session_state.history:
        st.subheader("Session History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Session History CSV",
            csv,
            "session_history.csv",
            "text/csv",
        )
