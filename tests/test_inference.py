"""
test_inference.py — Unit tests for Model A and Model B inference modules.

Run with:
    cd tests
    python -m pytest test_inference.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Helpers ────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLE = (
    "The Earth revolves around the Sun once every year. "
    "This orbital movement causes the seasons and changes in daylight hours. "
    "The Moon orbits the Earth and controls ocean tides."
)

SAMPLE_QUESTION = "What does the Earth revolve around?"

SAMPLE_OPTIONS = {
    "A": "The Moon",
    "B": "Mars",
    "C": "The Sun",
    "D": "Jupiter",
}

CORRECT_LABEL = "C"
CORRECT_ANSWER = "The Sun"


# ══════════════════════════════════════════════════════════════════════════════
# Model B — no model files needed, pure Python logic
# ══════════════════════════════════════════════════════════════════════════════

class TestModelBInferenceMocked(unittest.TestCase):
    """Tests for Model B using a mocked vectorizer — no .pkl files required."""

    def setUp(self):
        import numpy as np
        from unittest.mock import MagicMock

        # Build a minimal mock vectorizer that returns zero vectors
        mock_vec = MagicMock()
        mock_vec.transform.side_effect = lambda texts: MagicMock(shape=(len(texts), 10))

        # Patch the module-level vectorizer before importing
        self.vec_patcher = patch("model_b_inference.vectorizer", mock_vec)
        self.vec_patcher.start()

        import model_b_inference as mb
        self.mb = mb

    def tearDown(self):
        self.vec_patcher.stop()

    def test_clean_text_lowercase(self):
        result = self.mb.clean_text("Hello World!")
        self.assertEqual(result, "hello world")

    def test_clean_text_punctuation_removed(self):
        result = self.mb.clean_text("it's great, really.")
        self.assertNotIn(".", result)
        self.assertNotIn(",", result)

    def test_clean_text_nan(self):
        import math
        result = self.mb.clean_text(math.nan)
        self.assertEqual(result, "")

    def test_clean_text_whitespace_normalised(self):
        result = self.mb.clean_text("  hello   world  ")
        self.assertEqual(result, "hello world")

    def test_split_sentences_basic(self):
        sents = self.mb.split_sentences(SAMPLE_ARTICLE)
        self.assertIsInstance(sents, list)
        self.assertGreater(len(sents), 0)
        for s in sents:
            self.assertGreater(len(s), 20)

    def test_split_sentences_empty(self):
        sents = self.mb.split_sentences("")
        self.assertIsInstance(sents, list)

    def test_extract_candidate_phrases_no_stopwords(self):
        candidates = self.mb.extract_candidate_phrases("the quick brown fox")
        stopwords = {"the", "quick"}
        for c in candidates:
            self.assertNotIn(c, stopwords)

    def test_extract_candidate_phrases_uniqueness(self):
        cands = self.mb.extract_candidate_phrases("word word word other")
        self.assertEqual(len(cands), len(set(cands)), "Candidates should be unique")

    def test_extract_candidate_phrases_min_length(self):
        cands = self.mb.extract_candidate_phrases("a ab abc abcd abcde")
        for c in cands:
            self.assertGreater(len(c), 3)

    def test_generate_hints_fallback_on_empty_article(self):
        hints = self.mb.generate_hints("", SAMPLE_QUESTION)
        self.assertEqual(len(hints), 3)
        self.assertIsInstance(hints[0], str)

    def test_generate_hints_returns_list(self):
        hints = self.mb.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertIsInstance(hints, list)
        self.assertEqual(len(hints), 3)

    def test_generate_distractors_fallback_returns_three(self):
        # With a mocked vectorizer that gives zero vectors, the similarity scoring
        # path runs; we just verify the return contract.
        dists = self.mb.generate_distractors(SAMPLE_ARTICLE, CORRECT_ANSWER)
        self.assertEqual(len(dists), 3)

    def test_generate_distractors_excludes_correct_answer(self):
        import numpy as np
        # Make cosine_similarity return distinguishable values
        with patch("model_b_inference.cosine_similarity") as mock_cos:
            mock_cos.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            dists = self.mb.generate_distractors(SAMPLE_ARTICLE, CORRECT_ANSWER)
        clean_correct = self.mb.clean_text(CORRECT_ANSWER)
        for d in dists:
            self.assertNotEqual(d.lower(), clean_correct)


# ── Tests that do require model files (skipped if missing) ────────────────────

def _models_present() -> bool:
    base = Path(__file__).resolve().parent.parent
    required = [
        base / "models" / "model_a" / "traditional" / "logistic_regression.pkl",
        base / "models" / "model_a" / "traditional" / "linear_svm.pkl",
        base / "models" / "model_a" / "traditional" / "tfidf_vectorizer.pkl",
        base / "models" / "model_b" / "traditional" / "model_b_vectorizer.pkl",
    ]
    return all(p.exists() for p in required)


SKIP_IF_NO_MODELS = unittest.skipUnless(
    _models_present(),
    "Trained model .pkl files not found — run the training pipeline first."
)


@SKIP_IF_NO_MODELS
class TestModelAInference(unittest.TestCase):
    """Integration tests for Model A inference using real trained models."""

    @classmethod
    def setUpClass(cls):
        import inference as inf
        cls.inf = inf

    # Return type contracts
    def test_predict_returns_tuple_of_three(self):
        result = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_best_answer_is_valid_label(self):
        best, _, _ = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        self.assertIn(best, SAMPLE_OPTIONS.keys())

    def test_confidence_in_zero_one(self):
        _, conf, _ = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_scores_dict_has_all_labels(self):
        _, _, scores = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        self.assertEqual(set(scores.keys()), set(SAMPLE_OPTIONS.keys()))

    def test_best_answer_matches_argmax(self):
        best, _, scores = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        expected = max(scores, key=scores.get)
        self.assertEqual(best, expected)

    def test_confidence_equals_best_score(self):
        best, conf, scores = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        self.assertAlmostEqual(conf, scores[best], places=6)

    def test_feature_matrix_shape(self):
        feats = self.inf.prepare_features(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, CORRECT_ANSWER
        )
        self.assertEqual(feats.shape[0], 1)
        self.assertGreater(feats.shape[1], 3)  # at least TF-IDF + 3 cosine

    def test_cosine_features_shape(self):
        import numpy as np
        cosine = self.inf.compute_cosine_features(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, CORRECT_ANSWER
        )
        self.assertEqual(cosine.shape, (1, 3))

    def test_scores_all_positive(self):
        _, _, scores = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        for lbl, score in scores.items():
            self.assertGreaterEqual(score, 0.0, f"Score for {lbl} is negative")

    # Edge case: single option
    def test_single_option_still_returns(self):
        best, conf, scores = self.inf.predict_best_answer(
            SAMPLE_ARTICLE, SAMPLE_QUESTION, {"A": CORRECT_ANSWER}
        )
        self.assertEqual(best, "A")
        self.assertGreater(conf, 0.0)

    # Robustness: short article
    def test_very_short_article(self):
        best, conf, scores = self.inf.predict_best_answer(
            "Short text.", SAMPLE_QUESTION, SAMPLE_OPTIONS
        )
        self.assertIn(best, SAMPLE_OPTIONS.keys())
        self.assertIsInstance(conf, float)

    def test_nan_text_inputs_do_not_crash(self):
        import math
        best, conf, scores = self.inf.predict_best_answer(
            SAMPLE_ARTICLE,
            math.nan,
            {"A": "The Moon", "B": math.nan, "C": CORRECT_ANSWER, "D": "Mars"},
        )
        self.assertIn(best, {"A", "B", "C", "D"})
        self.assertIsInstance(conf, float)
        self.assertEqual(set(scores.keys()), {"A", "B", "C", "D"})


@SKIP_IF_NO_MODELS
class TestModelBInferenceReal(unittest.TestCase):
    """Integration tests for Model B using the real trained vectorizer."""

    @classmethod
    def setUpClass(cls):
        import model_b_inference as mb
        cls.mb = mb

    def test_generate_distractors_count(self):
        dists = self.mb.generate_distractors(SAMPLE_ARTICLE, CORRECT_ANSWER)
        self.assertEqual(len(dists), 3)

    def test_generate_distractors_custom_top_k(self):
        dists = self.mb.generate_distractors(SAMPLE_ARTICLE, CORRECT_ANSWER, top_k=5)
        self.assertEqual(len(dists), 5)

    def test_distractors_differ_from_correct(self):
        clean = self.mb.clean_text(CORRECT_ANSWER)
        dists = self.mb.generate_distractors(SAMPLE_ARTICLE, CORRECT_ANSWER)
        for d in dists:
            self.assertNotEqual(self.mb.clean_text(d), clean)

    def test_generate_hints_count(self):
        hints = self.mb.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertEqual(len(hints), 3)

    def test_hints_are_non_empty_strings(self):
        hints = self.mb.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        for h in hints:
            self.assertIsInstance(h, str)
            self.assertGreater(len(h.strip()), 0)

    def test_distractors_are_strings(self):
        dists = self.mb.generate_distractors(SAMPLE_ARTICLE, CORRECT_ANSWER)
        for d in dists:
            self.assertIsInstance(d, str)


@SKIP_IF_NO_MODELS
class TestEvaluationUtils(unittest.TestCase):
    """Unit tests for evaluate.py utility functions."""

    @classmethod
    def setUpClass(cls):
        import evaluate as ev
        cls.ev = ev

    def test_exact_match_perfect(self):
        score = self.ev.exact_match(["a", "b", "c"], ["a", "b", "c"])
        self.assertAlmostEqual(score, 1.0)

    def test_exact_match_zero(self):
        score = self.ev.exact_match(["a", "b"], ["x", "y"])
        self.assertAlmostEqual(score, 0.0)

    def test_exact_match_case_insensitive(self):
        score = self.ev.exact_match(["Apple"], ["apple"])
        self.assertAlmostEqual(score, 1.0)

    def test_exact_match_empty(self):
        score = self.ev.exact_match([], [])
        self.assertAlmostEqual(score, 0.0)

    def test_evaluate_model_a_returns_dict(self):
        import numpy as np
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 1, 0, 1]
        result = self.ev.evaluate_model_a(y_true, y_pred, "TestModel")
        for key in ["accuracy", "macro_f1", "precision", "recall", "exact_match"]:
            self.assertIn(key, result)

    def test_compare_model_a_results_returns_dataframe(self):
        import pandas as pd
        dummy = [
            {"model": "A", "accuracy": 0.9, "macro_f1": 0.88,
             "precision": 0.87, "recall": 0.89, "exact_match": 0.9},
            {"model": "B", "accuracy": 0.8, "macro_f1": 0.78,
             "precision": 0.77, "recall": 0.79, "exact_match": 0.8},
        ]
        df = self.ev.compare_model_a_results(dummy)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_evaluate_model_b_distractors_all_valid(self):
        distractors = [["moon", "mars", "jupiter"]]
        correct = ["The Sun"]
        result = self.ev.evaluate_model_b_distractors(distractors, correct)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertEqual(result["valid_distractors"], 3)

    def test_evaluate_model_b_distractors_invalid_fallback(self):
        distractors = [["No suitable distractor", "No suitable distractor"]]
        correct = ["some answer"]
        result = self.ev.evaluate_model_b_distractors(distractors, correct)
        self.assertAlmostEqual(result["accuracy"], 0.0)

    def test_compute_session_metrics_none_on_empty(self):
        result = self.ev.compute_session_metrics([])
        self.assertIsNone(result)

    def test_compute_session_metrics_accuracy(self):
        history = [
            {"is_correct": True,  "model_a_prediction": "A", "correct_label": "A"},
            {"is_correct": False, "model_a_prediction": "B", "correct_label": "A"},
            {"is_correct": True,  "model_a_prediction": "A", "correct_label": "A"},
        ]
        result = self.ev.compute_session_metrics(history)
        self.assertAlmostEqual(result["user_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["model_a_accuracy"], 2 / 3)
        self.assertEqual(result["total_attempts"], 3)

    def test_question_text_metrics_include_meteor(self):
        result = self.ev.evaluate_question_text_metrics(
            ["What does Earth revolve around?"],
            ["What does the Earth revolve around?"],
        )
        for key in ["bleu", "rouge_l", "meteor"]:
            self.assertIn(key, result.columns)
            self.assertGreaterEqual(float(result.iloc[0][key]), 0.0)
            self.assertLessEqual(float(result.iloc[0][key]), 1.0)

    def test_meteor_identical_question_is_high(self):
        score = self.ev.compute_meteor(
            "What does Earth revolve around?",
            "What does Earth revolve around?",
        )
        self.assertGreater(score, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
