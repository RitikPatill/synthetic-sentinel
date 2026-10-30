"""Tests for sentinel.cli — exit-code contract and table output."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from sentinel.cli import main
from sentinel.detector import Detector

REAL_RESULT = {
    "score": 0.2,
    "label": "REAL",
    "signals": {"exif_absence": 0.1, "dct_artifacts": 0.2, "noise_floor": 0.2, "ml_available": False},
    "reasoning": "REAL.",
}
AI_RESULT = {
    "score": 0.8,
    "label": "AI",
    "signals": {"exif_absence": 0.8, "dct_artifacts": 0.8, "noise_floor": 0.8, "ml_available": False},
    "reasoning": "AI.",
}


def test_exit_0_for_real(real_jpeg, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sentinel", str(real_jpeg)])
    with patch.object(Detector, "detect", return_value=REAL_RESULT):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_exit_1_for_ai(real_jpeg, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sentinel", str(real_jpeg)])
    with patch.object(Detector, "detect", return_value=AI_RESULT):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_table_output_contains_label(real_jpeg, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sentinel", str(real_jpeg)])
    with patch.object(Detector, "detect", return_value=REAL_RESULT):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "REAL" in out


def test_multiple_targets_any_ai_exits_1(real_jpeg, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sentinel", str(real_jpeg), str(real_jpeg)])
    side_effects = [REAL_RESULT, AI_RESULT]
    with patch.object(Detector, "detect", side_effect=side_effects):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_multiple_targets_all_real_exits_0(real_jpeg, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sentinel", str(real_jpeg), str(real_jpeg)])
    with patch.object(Detector, "detect", return_value=REAL_RESULT):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_table_output_contains_score(real_jpeg, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sentinel", str(real_jpeg)])
    with patch.object(Detector, "detect", return_value=REAL_RESULT):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "0.2" in out
