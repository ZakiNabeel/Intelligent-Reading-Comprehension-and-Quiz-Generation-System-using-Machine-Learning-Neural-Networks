"""
ui/app.py
=========
Reading Comprehension AI — Streamlit Front-End
================================================
A four-screen interactive quiz application that connects to the ML
back-end pipelines in src/.

Screens (sidebar navigation)
-----------------------------
  📄  Article Input       — paste a passage, load a sample, run inference
  🧠  Quiz                — question + MCQ options + answer verification
  💡  Hints               — 3 graduated hints from Model B
  📊  Developer Dashboard — model metrics and inference diagnostics

Run:
  streamlit run ui/app.py

Requirements:
  streamlit, pandas, matplotlib, seaborn, joblib, numpy
  src/ directory with model_a_train.py, model_a_qg.py,
       model_b_hints.py, model_b_train.py, inference.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import random
import logging
import traceback
import textwrap
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — add project root and src/ to sys.path
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReadMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — editorial dark theme (navy × amber × slate)
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject a cohesive dark-editorial theme via st.markdown."""
    st.markdown("""
    <style>
    /* ── Google Font import ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Root palette ── */
    :root {
        --bg-base:      #0B0F19;
        --bg-card:      #131929;
        --bg-elevated:  #1A2236;
        --border:       #2A3450;
        --amber:        #F5A623;
        --amber-dim:    #C4821A;
        --amber-glow:   rgba(245,166,35,0.12);
        --green:        #3DD68C;
        --green-dim:    rgba(61,214,140,0.12);
        --red:          #F05D5E;
        --red-dim:      rgba(240,93,94,0.12);
        --blue:         #4E9FE5;
        --blue-dim:     rgba(78,159,229,0.12);
        --text-primary: #E8EDF5;
        --text-muted:   #8A94A6;
        --text-dim:     #4A5568;
        --radius:       10px;
        --radius-lg:    16px;
    }

    /* ── Global reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: var(--text-primary) !important;
    }
    .brand-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: var(--amber);
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }
    .brand-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* ── Screen title ── */
    .screen-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 20px 0 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }
    .screen-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.55rem;
        color: var(--text-primary);
        margin: 0;
    }
    .screen-pill {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        background: var(--amber-glow);
        color: var(--amber);
        border: 1px solid var(--amber-dim);
        border-radius: 20px;
        padding: 2px 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* ── Cards ── */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 22px 26px;
        margin-bottom: 18px;
    }
    .card-accent {
        border-left: 3px solid var(--amber);
    }

    /* ── Passage display ── */
    .passage-box {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-left: 4px solid var(--amber);
        border-radius: var(--radius);
        padding: 20px 24px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.92rem;
        line-height: 1.75;
        color: var(--text-primary);
        max-height: 200px;
        overflow-y: auto;
        margin-bottom: 20px;
    }

    /* ── Question display ── */
    .question-box {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 24px 28px;
        margin-bottom: 22px;
    }
    .question-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: var(--amber);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .question-text {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: var(--text-primary);
        line-height: 1.4;
    }

    /* ── MCQ options ── */
    .option-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    div[data-testid="stRadio"] > label {
        display: none;
    }
    div[data-testid="stRadio"] > div {
        gap: 10px !important;
        flex-direction: column !important;
    }
    div[data-testid="stRadio"] > div > label {
        display: flex !important;
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 14px 18px !important;
        cursor: pointer !important;
        transition: border-color 0.15s, background 0.15s !important;
        font-size: 0.9rem !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stRadio"] > div > label:hover {
        border-color: var(--amber) !important;
        background: var(--amber-glow) !important;
    }

    /* ── Result banners ── */
    .result-correct {
        background: var(--green-dim);
        border: 1px solid var(--green);
        border-radius: var(--radius);
        padding: 14px 20px;
        color: var(--green);
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 14px;
    }
    .result-wrong {
        background: var(--red-dim);
        border: 1px solid var(--red);
        border-radius: var(--radius);
        padding: 14px 20px;
        color: var(--red);
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 14px;
    }

    /* ── Hint cards ── */
    .hint-card {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .hint-card:hover { border-color: var(--amber-dim); }
    .hint-number {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: var(--amber);
        line-height: 1;
        min-width: 28px;
    }
    .hint-text {
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.6;
        padding-top: 3px;
    }
    .hint-score {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-top: 5px;
    }

    /* ── Metric tiles ── */
    .metric-tile {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 20px 22px;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: var(--amber);
        line-height: 1.1;
    }
    .metric-delta {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--green);
        margin-top: 2px;
    }
    .metric-name {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Status badge ── */
    .badge {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.08em;
    }
    .badge-ok  { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }
    .badge-err { background: var(--red-dim);   color: var(--red);   border: 1px solid var(--red); }
    .badge-warn{ background: var(--amber-glow);color: var(--amber); border: 1px solid var(--amber-dim); }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        border-color: var(--amber) !important;
        color: var(--amber) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--amber) !important;
        color: #0B0F19 !important;
        border-color: var(--amber) !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--amber-dim) !important;
        color: #0B0F19 !important;
    }

    /* ── Text inputs ── */
    .stTextArea > div > textarea {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTextArea > div > textarea:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 3px var(--amber-glow) !important;
    }
    .stTextInput > div > input {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
    }

    /* ── Expander ── */
    details {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    summary {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
    }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--amber) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sample passages (hardcoded for demo / "Load Sample" button)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PASSAGES = [
    {
        "title": "The Amazon Rainforest",
        "passage": (
            "The Amazon rainforest is the world's largest tropical rainforest, "
            "covering over five million square kilometres across nine countries in "
            "South America. It is home to an estimated three million species of "
            "plants and animals, including jaguars, anacondas, and poison dart frogs. "
            "The forest plays a critical role in regulating the global climate by "
            "absorbing vast amounts of carbon dioxide and releasing oxygen through "
            "photosynthesis. Despite its importance, deforestation driven by agriculture, "
            "logging, and mining threatens the entire ecosystem. Scientists argue that "
            "protecting the Amazon is essential for maintaining global biodiversity "
            "and slowing the acceleration of climate change."
        ),
        "answer": "absorbing carbon dioxide and releasing oxygen",
        "question": "What role does the Amazon rainforest play in regulating the global climate?",
    },
    {
        "title": "The Industrial Revolution",
        "passage": (
            "The Industrial Revolution began in Britain during the late eighteenth century "
            "and rapidly spread across Europe and North America. Steam engines powered "
            "factories and transformed the manufacturing process, replacing hand tools "
            "with machinery. Cotton production rose dramatically, and workers migrated "
            "from rural farms to crowded urban centres. Railways expanded at an "
            "unprecedented pace, connecting distant regions and accelerating trade. "
            "Coal mining became the essential fuel source for the new industrial economy. "
            "While industrial growth created unprecedented wealth, it also widened social "
            "inequalities and increased urban poverty among the working class."
        ),
        "answer": "steam engines",
        "question": "What powered factories during the Industrial Revolution?",
    },
    {
        "title": "Deep Sea Exploration",
        "passage": (
            "The deep ocean remains one of the least explored environments on Earth. "
            "Below 200 metres, sunlight cannot penetrate the water, creating a world "
            "of perpetual darkness known as the midnight zone. Despite the crushing "
            "pressure and freezing temperatures, unique organisms such as bioluminescent "
            "fish, giant squid, and deep-sea anglerfish thrive in these conditions. "
            "Scientists use remotely operated vehicles, known as ROVs, to study these "
            "ecosystems without risking human life. Recent expeditions to the Mariana "
            "Trench, the deepest point on Earth at nearly 11 kilometres, have discovered "
            "entirely new species previously unknown to science."
        ),
        "answer": "remotely operated vehicles (ROVs)",
        "question": "What technology do scientists use to study deep-sea ecosystems?",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ML Model loading — graceful fallback to demo mode
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    """
    Attempt to load all trained joblib artifacts from the models/ directory.
    Returns a dict with loaded objects and a status report per model.
    Falls back to demo mode if any artifact is missing.

    Returns
    -------
    dict with keys:
        model_a_verifier, model_a_verifier_tfidf,
        model_a_qg, model_a_qg_feature_cols,
        model_b_ranker, model_b_tfidf,
        status: dict[str -> "ok" | "missing" | "error"]
    """
    import joblib

    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    artifacts = {
        "model_a_verifier":       os.path.join(MODELS_DIR, "model_a_verifier.joblib"),
        "model_a_verifier_tfidf": os.path.join(MODELS_DIR, "model_a_tfidf.joblib"),
        "model_a_qg":             os.path.join(MODELS_DIR, "model_a_qg_ranker.joblib"),
        "model_b_ranker":         os.path.join(MODELS_DIR, "model_b_ranker.joblib"),
        "model_b_tfidf":          os.path.join(MODELS_DIR, "model_b_tfidf.joblib"),
    }

    result = {}
    status = {}

    for key, path in artifacts.items():
        if not os.path.exists(path):
            status[key] = "missing"
            result[key] = None
        else:
            try:
                result[key] = joblib.load(path)
                status[key] = "ok"
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                status[key] = "error"
                result[key] = None

    # Unpack QG tuple (model, feature_cols) if present
    if isinstance(result.get("model_a_qg"), tuple):
        result["model_a_qg"], result["model_a_qg_feature_cols"] = result["model_a_qg"]
    else:
        result["model_a_qg_feature_cols"] = None

    result["status"] = status
    result["demo_mode"] = any(v != "ok" for v in status.values())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Inference layer — wraps src/ functions with fallback to demo data
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    passage: str,
    correct_answer: str,
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the full pipeline for a given passage and correct answer.

    Returns
    -------
    dict with keys:
        question       : str
        options        : List[str]   — shuffled A/B/C/D options
        correct_index  : int         — index of the correct answer in options
        hints          : List[Tuple[str, float]]
        inference_ms   : float
        source         : "live" | "demo"
    """
    t0 = time.perf_counter()

    if models["demo_mode"]:
        result = _demo_inference(passage, correct_answer)
        result["source"] = "demo"
        result["inference_ms"] = (time.perf_counter() - t0) * 1000
        return result

    # ── Live inference ────────────────────────────────────────────────────
    try:
        from model_a_qg     import generate_best_question
        from model_b_train  import generate_distractors
        from model_b_hints  import extract_hints

        # 1. Generate 3 questions
        from model_a_qg import generate_best_questions 
        
        questions = generate_best_questions(
            passage, correct_answer,
            models["model_a_qg"],
            models["model_a_qg_feature_cols"],
            num_questions=3
        )

        # 2. Generate 3 distractors
        distractors = generate_distractors(
            passage, correct_answer,
            models["model_b_ranker"],
            models["model_b_tfidf"],
        )
        while len(distractors) < 3:
            distractors.append(f"[distractor unavailable {len(distractors)+1}]")
        distractors = distractors[:3]

        # 3. Build shuffled option list
        options = distractors + [correct_answer]
        random.shuffle(options)
        correct_index = options.index(correct_answer)

        # 4. Extract hints
        hints = extract_hints(passage, questions[0], n_hints=3)

        return {
            "questions":     questions,  # <--- Changed to a list of questions
            "options":       options,
            "correct_index": correct_index,
            "hints":         hints,
            "inference_ms":  (time.perf_counter() - t0) * 1000,
            "source":        "live",
        }

    except Exception as exc:
        logger.error(f"Live inference failed: {exc}\n{traceback.format_exc()}")
        result = _demo_inference(passage, correct_answer)
        result["source"] = "demo (fallback)"
        result["inference_ms"] = (time.perf_counter() - t0) * 1000
        return result


def _demo_inference(passage: str, correct_answer: str) -> Dict[str, Any]:
    """
    Return deterministic demo inference output using the hardcoded samples.
    Activated when models are not loaded or live inference fails.
    """
    # Look for a matching sample, else use the first
    sample = next(
        (s for s in SAMPLE_PASSAGES if s["answer"] in passage or s["passage"][:40] in passage),
        SAMPLE_PASSAGES[0],
    )

    demo_distractors = {
        "absorbing carbon dioxide and releasing oxygen": [
            "reflecting sunlight back into space",
            "producing rainfall through evaporation only",
            "storing minerals in root systems",
        ],
        "steam engines": [
            "coal mining equipment",
            "cotton looms",
            "railway carriages",
        ],
        "remotely operated vehicles (ROVs)": [
            "pressurised diving suits",
            "sonar mapping buoys",
            "deep-sea submarines with crews",
        ],
    }.get(correct_answer, [
        "Plausible distractor A",
        "Plausible distractor B",
        "Plausible distractor C",
    ])

    options = demo_distractors[:3] + [correct_answer]
    random.shuffle(options)
    correct_index = options.index(correct_answer)

    hints_text = [s.strip() for s in passage.replace("!", ".").replace("?", ".").split(".") if len(s.strip()) > 30]
    hints = [(h, round(0.9 - i * 0.2, 2)) for i, h in enumerate(hints_text[:3])]

    return {
        "questions":     [sample["question"], f"What is the significance of {correct_answer}?", f"Where is {correct_answer} mentioned in the passage?"],
        "options":       options,
        "correct_index": correct_index,
        "hints":         hints,
    }


def verify_answer(
    selected_option: str,
    correct_answer: str,
    passage: str,
    models: Dict[str, Any],
) -> Tuple[bool, float, str]:
    """
    Use Model A's verifier to check whether the selected option is correct.

    Returns
    -------
    Tuple[bool, float, str] — (is_correct, confidence_score, explanation)
    """
    # Always compare the selected text against the stored correct answer
    is_correct = selected_option.strip().lower() == correct_answer.strip().lower()

    if models["demo_mode"] or models.get("model_a_verifier") is None:
        confidence = 0.97 if is_correct else 0.11
        explanation = (
            "Answer verified against the passage (demo mode — verifier model not loaded)."
        )
        return is_correct, confidence, explanation

    try:
        from model_a_train import build_feature_matrix as build_verify_features
        feat = build_verify_features(
            [selected_option], passage, models["model_a_verifier_tfidf"]
        )
        proba = models["model_a_verifier"].predict_proba(feat)[0]
        confidence = float(proba[1])
        is_correct  = confidence > 0.5
        explanation = f"Model A confidence: {confidence:.1%}"
        return is_correct, confidence, explanation
    except Exception as e:
        logger.warning(f"Verifier failed: {e}")
        confidence = 0.92 if is_correct else 0.08
        return is_correct, confidence, "Verification via string match (model fallback)."


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_state() -> None:
    """Initialise all required session_state keys once."""
    defaults = {
        # Navigation
        "current_screen": "📄 Article Input",
        # Input
        "passage_text":   "",
        "correct_answer": "",
        # Inference output
        "inference_result": None,
        "inference_done":   False,
        # Quiz interaction
        "selected_option":  None,
        "answer_checked":   False,
        "verify_result":    None,   # (is_correct, confidence, explanation)
        # Sample tracking
        "loaded_sample_idx": -1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation + model status
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(models: Dict[str, Any]) -> str:
    """
    Render the sidebar with branding, navigation, and model status indicators.

    Returns
    -------
    str — The currently selected screen name.
    """
    with st.sidebar:
        # Brand
        st.markdown(
            '<p class="brand-title">ReadMind</p>'
            '<p class="brand-sub">Reading Comprehension AI · RACE Dataset</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Navigation
        screens = [
            "📄 Article Input",
            "🧠 Quiz",
            "💡 Hints",
            "📊 Developer Dashboard",
        ]
        selected = st.radio(
            "Navigate",
            screens,
            index=screens.index(st.session_state.current_screen),
            label_visibility="collapsed",
        )
        st.session_state.current_screen = selected
        st.markdown("---")

        # Model status
        st.markdown(
            '<span style="font-family:DM Mono,monospace;font-size:0.65rem;'
            'color:#8A94A6;letter-spacing:0.1em;text-transform:uppercase;">'
            'Model Status</span>',
            unsafe_allow_html=True,
        )

        status_icons = {"ok": "🟢", "missing": "🟡", "error": "🔴"}
        status_labels = {
            "model_a_verifier":       "A  · Answer Verifier",
            "model_a_verifier_tfidf": "A  · Verifier TF-IDF",
            "model_a_qg":             "A  · Question Gen",
            "model_b_ranker":         "B  · Distractor Ranker",
            "model_b_tfidf":          "B  · Distractor TF-IDF",
        }
        for key, label in status_labels.items():
            s = models["status"].get(key, "missing")
            icon = status_icons.get(s, "🟡")
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
                f'color:#8A94A6;padding:2px 0;">{icon} {label}</div>',
                unsafe_allow_html=True,
            )

        # Demo mode banner
        if models["demo_mode"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div style="background:rgba(245,166,35,0.08);border:1px solid #C4821A;'
                'border-radius:8px;padding:10px 12px;font-size:0.75rem;color:#F5A623;'
                'font-family:DM Sans,sans-serif;">'
                '⚠️ <strong>Demo Mode</strong><br>'
                '<span style="color:#8A94A6;">Train and save models to<br>'
                '<code style="color:#F5A623;">models/</code> for live inference.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Passage status indicator at bottom
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.inference_done:
            st.markdown(
                '<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
                'color:#3DD68C;">✓ Passage processed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
                'color:#8A94A6;">○ Awaiting passage input</div>',
                unsafe_allow_html=True,
            )

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Screen 1 — Article Input
# ─────────────────────────────────────────────────────────────────────────────
def screen_article_input(models: Dict[str, Any]) -> None:
    """Render the Article Input screen."""
    st.markdown(
        '<div class="screen-header">'
        '<p class="screen-title">Article Input</p>'
        '<span class="screen-pill">Step 1</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sample loader ──────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.7rem;'
        'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
        'margin-bottom:10px;">Quick Start — Load a Sample Passage</p>',
        unsafe_allow_html=True,
    )

    sample_cols = st.columns(len(SAMPLE_PASSAGES))
    for i, (col, sample) in enumerate(zip(sample_cols, SAMPLE_PASSAGES)):
        with col:
            if st.button(f"📖 {sample['title']}", use_container_width=True, key=f"sample_{i}"):
                st.session_state.passage_text    = sample["passage"]
                st.session_state.correct_answer  = sample["answer"]
                st.session_state.loaded_sample_idx = i
                st.session_state.inference_done  = False
                st.session_state.inference_result = None
                st.session_state.answer_checked   = False
                st.session_state.selected_option  = None
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input columns ──────────────────────────────────────────────────────
    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.markdown(
            '<p style="font-family:DM Mono,monospace;font-size:0.68rem;'
            'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
            'margin-bottom:6px;">Reading Passage</p>',
            unsafe_allow_html=True,
        )
        passage_input = st.text_area(
            label="passage",
            label_visibility="collapsed",
            value=st.session_state.passage_text,
            height=260,
            placeholder=(
                "Paste your reading passage here…\n\n"
                "The passage should be 3–10 sentences long for best results. "
                "Try loading one of the sample passages above to see the system in action."
            ),
            key="passage_input_widget",
        )
        st.session_state.passage_text = passage_input

    with right_col:
        st.markdown(
            '<p style="font-family:DM Mono,monospace;font-size:0.68rem;'
            'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
            'margin-bottom:6px;">Correct Answer</p>',
            unsafe_allow_html=True,
        )
        answer_input = st.text_input(
            label="answer",
            label_visibility="collapsed",
            value=st.session_state.correct_answer,
            placeholder="e.g. absorbing carbon dioxide",
            key="answer_input_widget",
        )
        st.session_state.correct_answer = answer_input

        # Live word count
        word_count = len(passage_input.split()) if passage_input.strip() else 0
        sent_count = len(
            [s for s in passage_input.replace("!",".").replace("?",".").split(".")
             if len(s.strip()) > 10]
        ) if passage_input.strip() else 0
        st.markdown(
            f'<div class="card" style="margin-top:16px;">'
            f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
            f'color:#8A94A6;margin-bottom:10px;text-transform:uppercase;'
            f'letter-spacing:0.08em;">Passage Stats</div>'
            f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
            f'<div><div style="font-family:DM Serif Display,serif;font-size:1.5rem;'
            f'color:#F5A623;">{word_count}</div>'
            f'<div style="font-size:0.72rem;color:#8A94A6;">words</div></div>'
            f'<div><div style="font-family:DM Serif Display,serif;font-size:1.5rem;'
            f'color:#F5A623;">{sent_count}</div>'
            f'<div style="font-size:0.72rem;color:#8A94A6;">sentences</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Tips box
        st.markdown(
            '<div class="card" style="margin-top:12px;">'
            '<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
            'color:#8A94A6;text-transform:uppercase;letter-spacing:0.08em;'
            'margin-bottom:8px;">Tips</div>'
            '<ul style="font-size:0.8rem;color:#8A94A6;padding-left:16px;margin:0;">'
            '<li style="margin-bottom:5px;">Aim for 5–8 sentences for best distractor quality.</li>'
            '<li style="margin-bottom:5px;">The correct answer should appear verbatim in the passage.</li>'
            '<li>Avoid single-word answers for richer questions.</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Submit button ──────────────────────────────────────────────────────
    submit_col, _, nav_col = st.columns([2, 3, 2])

    with submit_col:
        submit_clicked = st.button(
            "⚡ Run Inference Pipeline",
            type="primary",
            use_container_width=True,
        )

    with nav_col:
        if st.session_state.inference_done:
            if st.button("Go to Quiz →", use_container_width=True):
                st.session_state.current_screen = "🧠 Quiz"
                st.rerun()

    # ── Validation + inference ─────────────────────────────────────────────
    if submit_clicked:
        passage = st.session_state.passage_text.strip()
        answer  = st.session_state.correct_answer.strip()

        # Input validation
        errors = []
        if not passage:
            errors.append("Passage cannot be empty.")
        elif len(passage.split()) < 15:
            errors.append("Passage is too short (minimum ~15 words).")
        if not answer:
            errors.append("Correct answer cannot be empty.")
        elif answer.lower() not in passage.lower():
            errors.append(
                "⚠️ The correct answer was not found verbatim in the passage. "
                "Results may be degraded."
            )

        hard_errors = [e for e in errors if not e.startswith("⚠️")]
        soft_errors = [e for e in errors if e.startswith("⚠️")]

        if hard_errors:
            for e in hard_errors:
                st.markdown(
                    f'<div class="result-wrong">✗ {e}</div>',
                    unsafe_allow_html=True,
                )
        else:
            for e in soft_errors:
                st.warning(e)

            with st.spinner("Running ML pipelines…"):
                result = run_inference(passage, answer, models)

            st.session_state.inference_result = result
            st.session_state.inference_done   = True
            st.session_state.answer_checked   = False
            st.session_state.selected_option  = None
            st.session_state.verify_result    = None

            src = result.get("source", "demo")
            src_color = "#3DD68C" if src == "live" else "#F5A623"
            st.markdown(
                f'<div style="background:rgba(61,214,140,0.08);border:1px solid #3DD68C;'
                f'border-radius:10px;padding:14px 18px;margin-top:14px;">'
                f'<span style="color:#3DD68C;font-weight:600;">✓ Pipeline complete</span> '
                f'<span style="color:#8A94A6;font-size:0.85rem;"> — '
                f'{result["inference_ms"]:.1f} ms · </span>'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;'
                f'color:{src_color};">{src}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.info("✓ Navigate to **🧠 Quiz** in the sidebar to start the quiz.")


# ─────────────────────────────────────────────────────────────────────────────
# Screen 2 — Quiz
# ─────────────────────────────────────────────────────────────────────────────
def screen_quiz(models: Dict[str, Any]) -> None:
    """Render the interactive Quiz screen."""
    st.markdown(
        '<div class="screen-header">'
        '<p class="screen-title">Reading Comprehension Quiz</p>'
        '<span class="screen-pill">Step 2</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Guard: ensure inference has been run
    if not st.session_state.inference_done or st.session_state.inference_result is None:
        st.markdown(
            '<div class="card card-accent" style="text-align:center;padding:40px;">'
            '<div style="font-size:2.5rem;margin-bottom:12px;">📄</div>'
            '<div style="font-family:DM Serif Display,serif;font-size:1.2rem;'
            'color:#E8EDF5;margin-bottom:8px;">No passage loaded yet</div>'
            '<div style="font-size:0.88rem;color:#8A94A6;">'
            'Go to <strong>Article Input</strong> and submit a passage first.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("← Go to Article Input"):
            st.session_state.current_screen = "📄 Article Input"
            st.rerun()
        return

    result        = st.session_state.inference_result
    questions     = result.get("questions", ["Error: No questions generated"]) # <--- Pull the list
    options       = result["options"]
    correct_index = result["correct_index"]
    correct_text  = options[correct_index]

    # ── Layout: passage on right, quiz on left ─────────────────────────────
    quiz_col, passage_col = st.columns([3, 2], gap="large")

    with passage_col:
        st.markdown(
            '<p style="font-family:DM Mono,monospace;font-size:0.65rem;'
            'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
            'margin-bottom:8px;">Source Passage</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="passage-box">{st.session_state.passage_text}</div>',
            unsafe_allow_html=True,
        )

        # Hint expander
        hints = result.get("hints", [])
        if hints:
            with st.expander("💡 Show Hints", expanded=False):
                for i, (hint_text, score) in enumerate(hints, 1):
                    revealed = (
                        st.session_state.answer_checked or
                        st.session_state.get(f"hint_revealed_{i}", False)
                    )
                    if not revealed:
                        st.markdown(
                            f'<div class="hint-card">'
                            f'<div class="hint-number">{i}</div>'
                            f'<div>'
                            f'<div style="font-size:0.8rem;color:#8A94A6;font-style:italic;">'
                            f'Hint {i} — hidden until revealed</div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Reveal Hint {i}", key=f"reveal_hint_{i}"):
                            st.session_state[f"hint_revealed_{i}"] = True
                            st.rerun()
                    else:
                        st.markdown(
                            f'<div class="hint-card">'
                            f'<div class="hint-number">{i}</div>'
                            f'<div>'
                            f'<div class="hint-text">{hint_text}</div>'
                            f'<div class="hint-score">relevance {score:.2f}</div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

    with quiz_col:
        # Question selection dropdown
        st.markdown(
            '<p class="option-label">Select a generated question:</p>',
            unsafe_allow_html=True,
        )
        selected_q = st.selectbox(
            "Select Question", 
            options=questions, 
            label_visibility="collapsed"
        )

        # Question box
        st.markdown(
            f'<div class="question-box">'
            f'<div class="question-label">Generated Question</div>'
            f'<div class="question-text">{selected_q}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # MCQ options
        option_labels = ["A", "B", "C", "D"]
        formatted_options = [
            f"{option_labels[i]}.  {opt}" for i, opt in enumerate(options)
        ]

        st.markdown(
            '<p class="option-label">Choose the correct answer:</p>',
            unsafe_allow_html=True,
        )

        selected_fmt = st.radio(
            label="options",
            options=formatted_options,
            index=(
                formatted_options.index(
                    next((f for f in formatted_options
                          if st.session_state.correct_answer in f or
                             (st.session_state.selected_option and
                              st.session_state.selected_option in f)), formatted_options[0])
                )
                if st.session_state.answer_checked else 0
            ),
            label_visibility="collapsed",
            key="quiz_radio",
            disabled=st.session_state.answer_checked,
        )

        # Parse selected option text (strip the "A.  " prefix)
        selected_raw = selected_fmt.split(".", 1)[-1].strip() if selected_fmt else ""
        st.session_state.selected_option = selected_raw

        st.markdown("<br>", unsafe_allow_html=True)

        # Check / Reset buttons
        btn_l, btn_r = st.columns(2)

        with btn_l:
            check_clicked = st.button(
                "✓ Check Answer",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.answer_checked,
            )

        with btn_r:
            if st.button("↺ New Attempt", use_container_width=True):
                st.session_state.answer_checked = False
                st.session_state.verify_result  = None
                for k in ["hint_revealed_1", "hint_revealed_2", "hint_revealed_3"]:
                    st.session_state[k] = False
                st.rerun()

        # ── Answer checking ────────────────────────────────────────────────
        if check_clicked and not st.session_state.answer_checked:
            if not selected_raw:
                st.warning("Please select an option before checking.")
            else:
                is_correct, confidence, explanation = verify_answer(
                    selected_raw, correct_text, st.session_state.passage_text, models
                )
                st.session_state.verify_result  = (is_correct, confidence, explanation)
                st.session_state.answer_checked = True
                st.rerun()

        # ── Result display ─────────────────────────────────────────────────
        if st.session_state.answer_checked and st.session_state.verify_result:
            is_correct, confidence, explanation = st.session_state.verify_result

            if is_correct:
                st.markdown(
                    f'<div class="result-correct">'
                    f'✓ Correct! &nbsp; '
                    f'<span style="font-weight:400;font-size:0.85rem;">'
                    f'Confidence: {confidence:.1%} · {explanation}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-wrong">'
                    f'✗ Incorrect. &nbsp; '
                    f'<span style="font-weight:400;font-size:0.85rem;">'
                    f'The correct answer was: <strong>{correct_text}</strong>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
                # Reveal correct option highlight
                correct_label = option_labels[correct_index]
                st.markdown(
                    f'<div style="margin-top:10px;background:rgba(61,214,140,0.08);'
                    f'border:1px solid #3DD68C;border-radius:8px;padding:12px 16px;'
                    f'font-size:0.88rem;color:#3DD68C;">'
                    f'Correct answer: <strong>{correct_label}. {correct_text}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Regenerate question ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳ Regenerate Question", use_container_width=False):
            with st.spinner("Re-running pipeline…"):
                new_result = run_inference(
                    st.session_state.passage_text,
                    st.session_state.correct_answer,
                    models,
                )
            st.session_state.inference_result = new_result
            st.session_state.answer_checked   = False
            st.session_state.verify_result    = None
            for k in ["hint_revealed_1", "hint_revealed_2", "hint_revealed_3"]:
                st.session_state[k] = False
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Screen 3 — Hints
# ─────────────────────────────────────────────────────────────────────────────
def screen_hints() -> None:
    """Render the dedicated Hints panel screen."""
    st.markdown(
        '<div class="screen-header">'
        '<p class="screen-title">Graduated Hints</p>'
        '<span class="screen-pill">Model B</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.inference_done or st.session_state.inference_result is None:
        st.markdown(
            '<div class="card card-accent" style="text-align:center;padding:40px;">'
            '<div style="font-size:2.5rem;margin-bottom:12px;">💡</div>'
            '<div style="font-family:DM Serif Display,serif;font-size:1.2rem;'
            'color:#E8EDF5;margin-bottom:8px;">No hints available yet</div>'
            '<div style="font-size:0.88rem;color:#8A94A6;">'
            'Submit a passage first to generate hints.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    result   = st.session_state.inference_result
    hints    = result.get("hints", [])
    question = result.get("question", "")

    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.65rem;'
        'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
        'margin-bottom:12px;">Question</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="question-box" style="margin-bottom:28px;">'
        f'<div class="question-text">{question}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not hints:
        st.info("No hints were extracted for this passage.")
        return

    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.65rem;'
        'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
        'margin-bottom:16px;">Hints — most relevant first</p>',
        unsafe_allow_html=True,
    )

    # Explainer
    st.markdown(
        '<div style="background:rgba(245,166,35,0.06);border:1px solid #2A3450;'
        'border-radius:10px;padding:14px 18px;margin-bottom:22px;font-size:0.82rem;'
        'color:#8A94A6;line-height:1.6;">'
        '<strong style="color:#F5A623;">How hints work:</strong> '
        'Model B extracts the three most relevant sentences from the passage using '
        'TF-IDF cosine similarity and Jaccard word overlap. Hints are ordered from '
        'most to least directly related to the question. Use them sparingly!'
        '</div>',
        unsafe_allow_html=True,
    )

    # Hint cards
    col_l, col_r = st.columns([2, 1], gap="large")

    with col_l:
        for i, (hint_text, score) in enumerate(hints, 1):
            bar_w = max(5, int(score * 100))
            st.markdown(
                f'<div class="hint-card">'
                f'<div class="hint-number">{i}</div>'
                f'<div style="flex:1;">'
                f'<div class="hint-text">{hint_text}</div>'
                f'<div style="margin-top:8px;">'
                f'<div style="height:3px;border-radius:2px;background:#2A3450;'
                f'overflow:hidden;">'
                f'<div style="width:{bar_w}%;height:100%;background:#F5A623;'
                f'border-radius:2px;"></div></div>'
                f'<div class="hint-score">Relevance score: {score:.3f}</div>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

    with col_r:
        # Relevance bar chart
        labels = [f"Hint {i}" for i in range(1, len(hints) + 1)]
        scores = [s for _, s in hints]

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        bars = ax.barh(labels[::-1], scores[::-1], color="#F5A623", alpha=0.85, height=0.5)
        ax.set_xlim(0, max(scores) * 1.3 if scores else 1)
        ax.set_xlabel("Relevance Score", color="#8A94A6", fontsize=8)
        ax.tick_params(colors="#8A94A6", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2A3450")

        for bar, v in zip(bars, scores[::-1]):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", color="#8A94A6", fontsize=7)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Screen 4 — Developer Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def screen_dashboard(models: Dict[str, Any]) -> None:
    """Render the Developer Dashboard with dummy + live metrics."""
    st.markdown(
        '<div class="screen-header">'
        '<p class="screen-title">Developer Dashboard</p>'
        '<span class="screen-pill">Diagnostics</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Dummy evaluation metrics (replace with eval script output) ─────────
    DUMMY_METRICS = {
        "Model A — Answer Verifier": {
            "Accuracy":  ("87.3%", "+2.1% vs baseline"),
            "F1 Score":  ("0.861", "+0.034"),
            "Precision": ("0.889", "—"),
            "Recall":    ("0.835", "—"),
        },
        "Model A — Question Gen": {
            "BLEU-1":    ("0.412", "template ranker"),
            "Wh-Rate":   ("78.5%", "Wh-word questions"),
            "Avg Length":("11.2 tok", "per question"),
            "Coverage":  ("94.1%", "answer in sentence"),
        },
        "Model B — Distractor Gen": {
            "Accuracy":  ("79.6%", "+5.3% vs baseline"),
            "F1 Score":  ("0.783", "+0.041"),
            "Precision": ("0.801", "—"),
            "Recall":    ("0.766", "—"),
        },
        "Model B — Hint Extraction": {
            "Coverage":  ("96.7%", "answer in hints"),
            "Avg Score": ("0.341", "TF-IDF similarity"),
            "Top-1 Hit": ("71.2%", "answer in hint 1"),
            "Top-3 Hit": ("96.7%", "answer in hints 1-3"),
        },
    }

    # ── Live inference timing ──────────────────────────────────────────────
    inf_time = None
    if st.session_state.inference_result:
        inf_time = st.session_state.inference_result.get("inference_ms")

    # ── Model status summary ───────────────────────────────────────────────
    status_map = models["status"]
    n_ok      = sum(1 for v in status_map.values() if v == "ok")
    n_total   = len(status_map)

    # Top KPI row
    kpi_cols = st.columns(4)
    kpis = [
        (f"{n_ok}/{n_total}", "Models Loaded", "#F5A623" if n_ok < n_total else "#3DD68C"),
        (f"{inf_time:.1f} ms" if inf_time else "—", "Last Inference", "#4E9FE5"),
        ("87.3%", "Model A Accuracy", "#3DD68C"),
        ("79.6%", "Model B Accuracy", "#3DD68C"),
    ]
    for col, (val, label, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-value" style="color:{color};">{val}</div>'
                f'<div class="metric-name">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-model metric tables ────────────────────────────────────────────
    left_col, right_col = st.columns(2, gap="large")
    model_names = list(DUMMY_METRICS.keys())

    for idx, (col, name) in enumerate(zip([left_col, right_col, left_col, right_col], model_names)):
        with col:
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
                f'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
                f'margin:18px 0 8px;">{name}</div>',
                unsafe_allow_html=True,
            )
            metrics = DUMMY_METRICS[name]
            rows_html = ""
            for metric, (value, delta) in metrics.items():
                rows_html += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:9px 0;border-bottom:1px solid #1A2236;">'
                    f'<span style="font-size:0.84rem;color:#8A94A6;">{metric}</span>'
                    f'<span style="display:flex;flex-direction:column;align-items:flex-end;">'
                    f'<span style="font-family:DM Mono,monospace;font-size:0.88rem;'
                    f'color:#E8EDF5;font-weight:600;">{value}</span>'
                    f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;'
                    f'color:#3DD68C;">{delta}</span>'
                    f'</span></div>'
                )
            st.markdown(
                f'<div class="card">{rows_html}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training curves (dummy — replace with real eval history) ──────────
    st.markdown(
        '<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
        'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
        'margin-bottom:16px;">Training Curves (Dummy — replace with eval history)</div>',
        unsafe_allow_html=True,
    )

    chart_l, chart_r = st.columns(2, gap="large")

    with chart_l:
        _render_training_curve(
            title="Model A — Answer Verifier",
            train_scores=[0.61, 0.69, 0.74, 0.79, 0.82, 0.85, 0.872],
            val_scores  =[0.58, 0.65, 0.71, 0.76, 0.80, 0.84, 0.861],
        )

    with chart_r:
        _render_training_curve(
            title="Model B — Distractor Ranker",
            train_scores=[0.55, 0.63, 0.68, 0.72, 0.76, 0.78, 0.796],
            val_scores  =[0.52, 0.60, 0.65, 0.69, 0.73, 0.76, 0.783],
        )

    # ── Feature importance ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
        'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
        'margin-bottom:16px;">Model B — Feature Importances</div>',
        unsafe_allow_html=True,
    )

    fi_col, _ = st.columns([2, 1])
    with fi_col:
        features    = ["TF-IDF Similarity", "Passage Frequency", "Char Overlap"]
        importances = [0.412, 0.356, 0.232]

        fig, ax = plt.subplots(figsize=(6, 2.4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        colors = ["#F5A623", "#4E9FE5", "#6ACC65"]
        bars = ax.barh(features[::-1], importances[::-1], color=colors[::-1], height=0.45)
        ax.set_xlabel("Importance", color="#8A94A6", fontsize=9)
        ax.tick_params(colors="#8A94A6", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#2A3450")
        for bar, v in zip(bars, importances[::-1]):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", color="#8A94A6", fontsize=8)
        ax.set_xlim(0, 0.55)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Model status table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
        'color:#8A94A6;letter-spacing:0.08em;text-transform:uppercase;'
        'margin-bottom:12px;">Loaded Artifact Status</div>',
        unsafe_allow_html=True,
    )

    status_rows = ""
    status_badge = {"ok": "badge-ok", "missing": "badge-warn", "error": "badge-err"}
    status_text  = {"ok": "LOADED", "missing": "MISSING", "error": "ERROR"}
    for key, s in status_map.items():
        badge_cls = status_badge.get(s, "badge-warn")
        badge_lbl = status_text.get(s, s.upper())
        status_rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:9px 0;border-bottom:1px solid #1A2236;">'
            f'<span style="font-family:DM Mono,monospace;font-size:0.78rem;'
            f'color:#8A94A6;">{key}</span>'
            f'<span class="badge {badge_cls}">{badge_lbl}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="card">{status_rows}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.75rem;color:#4A5568;margin-top:8px;font-style:italic;">'
        '* Dummy metrics shown. Replace DUMMY_METRICS dict with output from your '
        'evaluation script once models are trained.</div>',
        unsafe_allow_html=True,
    )


def _render_training_curve(
    title: str,
    train_scores: List[float],
    val_scores: List[float],
) -> None:
    """Helper: render a small training curve chart."""
    epochs = list(range(1, len(train_scores) + 1))
    fig, ax = plt.subplots(figsize=(5.5, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(epochs, train_scores, color="#F5A623", linewidth=2,
            marker="o", markersize=4, label="Train")
    ax.plot(epochs, val_scores,   color="#4E9FE5", linewidth=2,
            marker="s", markersize=4, linestyle="--", label="Validation")

    ax.set_title(title, color="#E8EDF5", fontsize=9, pad=8)
    ax.set_xlabel("Estimators (×25)", color="#8A94A6", fontsize=8)
    ax.set_ylabel("F1 Score",         color="#8A94A6", fontsize=8)
    ax.tick_params(colors="#8A94A6", labelsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor="#8A94A6")
    ax.set_ylim(0.45, 0.95)

    for spine in ax.spines.values():
        spine.set_color("#2A3450")
    ax.grid(True, color="#1A2236", linewidth=0.8, linestyle=":")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    init_state()

    # Load models (cached — only runs once per session)
    with st.spinner("Loading ML models…"):
        models = load_models()

    # Render sidebar navigation
    current_screen = render_sidebar(models)

    # Route to the correct screen
    if current_screen == "📄 Article Input":
        screen_article_input(models)
    elif current_screen == "🧠 Quiz":
        screen_quiz(models)
    elif current_screen == "💡 Hints":
        screen_hints()
    elif current_screen == "📊 Developer Dashboard":
        screen_dashboard(models)


if __name__ == "__main__":
    main()