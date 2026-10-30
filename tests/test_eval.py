"""Tests for sentinel.eval.evaluate()."""

from __future__ import annotations

from unittest.mock import patch

from sentinel.detector import Detector
from sentinel.eval import evaluate

_AI_RESULT = {
    "score": 0.9,
    "label": "AI",
    "signals": {},
    "reasoning": "AI.",
}
_REAL_RESULT = {
    "score": 0.1,
    "label": "REAL",
    "signals": {},
    "reasoning": "REAL.",
}
_AI_AS_REAL = {
    "score": 0.1,
    "label": "REAL",
    "signals": {},
    "reasoning": "REAL.",
}
_REAL_AS_AI = {
    "score": 0.9,
    "label": "AI",
    "signals": {},
    "reasoning": "AI.",
}


def test_perfect_classifier(real_jpeg, synthetic_jpeg):
    side_effects = [_REAL_RESULT, _AI_RESULT]
    with patch.object(Detector, "detect", side_effect=side_effects):
        metrics = evaluate([real_jpeg], [synthetic_jpeg])
    assert metrics["accuracy"] == 1.0


def test_rows_have_expected_keys(real_jpeg, synthetic_jpeg):
    side_effects = [_REAL_RESULT, _AI_RESULT]
    with patch.object(Detector, "detect", side_effect=side_effects):
        metrics = evaluate([real_jpeg], [synthetic_jpeg])
    for row in metrics["rows"]:
        assert "target" in row
        assert "true_label" in row
        assert "pred_label" in row
        assert "score" in row


def test_empty_inputs_returns_zero_accuracy():
    metrics = evaluate([], [])
    assert metrics["accuracy"] == 0.0
    assert metrics["rows"] == []


def test_all_wrong_gives_zero_accuracy(real_jpeg, synthetic_jpeg):
    # AI image predicted REAL, real image predicted AI
    side_effects = [_REAL_AS_AI, _AI_AS_REAL]
    with patch.object(Detector, "detect", side_effect=side_effects):
        metrics = evaluate([real_jpeg], [synthetic_jpeg])
    assert metrics["accuracy"] == 0.0
