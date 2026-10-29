"""Tests for sentinel/ml.py — MLDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sentinel.ml import MLDetector


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Reset class-level pipeline cache between tests to avoid cross-test pollution."""
    MLDetector._pipeline = None
    yield
    MLDetector._pipeline = None


class TestMLDetectorSchema:
    """MLDetector returns the correct schema when the pipeline is available."""

    def test_returns_expected_keys(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "artificial", "score": 0.82}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert result is not None
        assert set(result.keys()) == {"score", "label", "signals", "reasoning"}

    def test_score_in_range(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "artificial", "score": 0.82}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert 0.0 <= result["score"] <= 1.0

    def test_label_valid(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "artificial", "score": 0.82}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert result["label"] in ("AI", "REAL", "UNCERTAIN")

    def test_high_score_gives_ai_label(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "artificial", "score": 0.82}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert result["label"] == "AI"
        assert result["score"] == pytest.approx(0.82, abs=1e-4)

    def test_signals_contains_ml_classifier(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "artificial", "score": 0.82}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert "ml_classifier" in result["signals"]


class TestMLDetectorFallback:
    """MLDetector returns None when the pipeline is unavailable."""

    def test_returns_none_when_pipeline_unavailable(self, real_jpeg):
        with patch.object(MLDetector, "_get_pipeline", return_value=None):
            result = MLDetector().detect(real_jpeg)

        assert result is None


class TestMLDetectorLabelMapping:
    """MLDetector correctly normalises unfamiliar label names."""

    def test_human_label_inverted(self, real_jpeg):
        # Model returns "human" with score 0.9 → AI confidence ≈ 0.1
        mock_pipe = MagicMock(return_value=[{"label": "human", "score": 0.9}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert result is not None
        assert result["score"] == pytest.approx(0.1, abs=1e-4)
        assert result["label"] == "REAL"

    def test_artificial_label_used_directly(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "artificial", "score": 0.45}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert result["score"] == pytest.approx(0.45, abs=1e-4)
        assert result["label"] == "UNCERTAIN"

    def test_fake_label_matched(self, real_jpeg):
        mock_pipe = MagicMock(return_value=[{"label": "FAKE", "score": 0.75}])
        with patch.object(MLDetector, "_get_pipeline", return_value=mock_pipe):
            result = MLDetector().detect(real_jpeg)

        assert result["score"] == pytest.approx(0.75, abs=1e-4)
        assert result["label"] == "AI"
