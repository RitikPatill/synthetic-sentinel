"""Tests for sentinel.heuristics — M2."""

import numpy as np
import pytest
from PIL import Image

from sentinel.heuristics import (
    HeuristicDetector,
    score_exif_absence,
    score_noise_floor,
)


class TestDetectSchema:
    def test_returns_required_keys(self, real_jpeg):
        result = HeuristicDetector().detect(real_jpeg)
        assert set(result.keys()) == {"score", "label", "signals", "reasoning"}

    def test_signals_has_three_sub_keys(self, real_jpeg):
        signals = HeuristicDetector().detect(real_jpeg)["signals"]
        assert set(signals.keys()) == {"exif_absence", "dct_artifacts", "noise_floor"}

    def test_score_in_unit_interval(self, real_jpeg):
        score = HeuristicDetector().detect(real_jpeg)["score"]
        assert 0.0 <= score <= 1.0

    def test_label_is_valid(self, real_jpeg):
        label = HeuristicDetector().detect(real_jpeg)["label"]
        assert label in {"AI", "REAL", "UNCERTAIN"}

    def test_reasoning_is_nonempty_string(self, real_jpeg):
        reasoning = HeuristicDetector().detect(real_jpeg)["reasoning"]
        assert isinstance(reasoning, str) and len(reasoning) > 0


class TestOrdering:
    def test_real_scores_lower_than_synthetic(self, real_jpeg, synthetic_jpeg):
        detector = HeuristicDetector()
        real_score = detector.detect(real_jpeg)["score"]
        synth_score = detector.detect(synthetic_jpeg)["score"]
        assert real_score < synth_score, (
            f"Expected real ({real_score:.4f}) < synthetic ({synth_score:.4f})"
        )


class TestScorerEdgeCases:
    def test_score_exif_absence_no_exif(self):
        # A freshly created PIL image has no EXIF → should score >= 0.5
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        score = score_exif_absence(img)
        assert score >= 0.5

    def test_score_noise_floor_smooth_image(self):
        # Solid-colour image → zero noise → score should be close to 0.95
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        score = score_noise_floor(img)
        assert score > 0.7

    def test_score_noise_floor_noisy_image(self):
        # Heavy random noise → high std → score should be low
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        score = score_noise_floor(img)
        assert score < 0.5

    def test_score_noise_floor_clamped(self):
        # Regardless of input, result must be within [0.05, 0.95]
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        score = score_noise_floor(img)
        assert 0.05 <= score <= 0.95
