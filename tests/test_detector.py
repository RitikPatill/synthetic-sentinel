"""Integration tests for sentinel/detector.py — Detector."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentinel.detector import Detector
from sentinel.ml import MLDetector


@pytest.fixture(autouse=True)
def reset_ml_pipeline():
    """Reset MLDetector class-level cache between tests."""
    MLDetector._pipeline = None
    yield
    MLDetector._pipeline = None


class TestDetectorSchema:
    """Detector returns the correct schema (ML unavailable / fallback path)."""

    def test_returns_expected_keys(self, real_jpeg):
        with patch.object(MLDetector, "detect", return_value=None):
            result = Detector().detect(real_jpeg)

        assert set(result.keys()) == {"score", "label", "signals", "reasoning"}

    def test_score_in_range(self, real_jpeg):
        with patch.object(MLDetector, "detect", return_value=None):
            result = Detector().detect(real_jpeg)

        assert 0.0 <= result["score"] <= 1.0

    def test_label_valid(self, real_jpeg):
        with patch.object(MLDetector, "detect", return_value=None):
            result = Detector().detect(real_jpeg)

        assert result["label"] in ("AI", "REAL", "UNCERTAIN")


class TestDetectorWithML:
    """Detector correctly fuses heuristic + ML scores when ML is available."""

    def test_weighted_fusion(self, real_jpeg):
        # First get the heuristic score on its own
        from sentinel.heuristics import HeuristicDetector

        h_result = HeuristicDetector().detect(real_jpeg)
        h_score = h_result["score"]

        ml_score = 0.8
        synthetic_ml = {
            "score": ml_score,
            "label": "AI",
            "signals": {"ml_classifier": ml_score},
            "reasoning": "ML says AI.",
        }

        with patch.object(MLDetector, "detect", return_value=synthetic_ml):
            result = Detector().detect(real_jpeg)

        expected = round(0.4 * h_score + 0.6 * ml_score, 4)
        assert result["score"] == pytest.approx(expected, abs=1e-3)

    def test_ml_available_flag_true(self, real_jpeg):
        synthetic_ml = {
            "score": 0.7,
            "label": "AI",
            "signals": {"ml_classifier": 0.7},
            "reasoning": "ML says AI.",
        }
        with patch.object(MLDetector, "detect", return_value=synthetic_ml):
            result = Detector().detect(real_jpeg)

        assert result["signals"]["ml_available"] is True

    def test_ml_classifier_signal_in_signals(self, real_jpeg):
        synthetic_ml = {
            "score": 0.7,
            "label": "AI",
            "signals": {"ml_classifier": 0.7},
            "reasoning": "ML says AI.",
        }
        with patch.object(MLDetector, "detect", return_value=synthetic_ml):
            result = Detector().detect(real_jpeg)

        assert "ml_classifier" in result["signals"]


class TestDetectorFallback:
    """Detector falls back to heuristics-only when ML is unavailable."""

    def test_ml_available_flag_false(self, real_jpeg):
        with patch.object(MLDetector, "detect", return_value=None):
            result = Detector().detect(real_jpeg)

        assert result["signals"]["ml_available"] is False

    def test_score_equals_heuristic_score(self, real_jpeg):
        from sentinel.heuristics import HeuristicDetector

        h_result = HeuristicDetector().detect(real_jpeg)

        with patch.object(MLDetector, "detect", return_value=None):
            result = Detector().detect(real_jpeg)

        assert result["score"] == pytest.approx(h_result["score"], abs=1e-4)

    def test_reasoning_mentions_fallback(self, real_jpeg):
        with patch.object(MLDetector, "detect", return_value=None):
            result = Detector().detect(real_jpeg)

        assert "heuristics" in result["reasoning"].lower()
