"""
Unified AI image detector.

Combines the heuristics layer and ML layer into a single weighted verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sentinel.heuristics import HeuristicDetector
from sentinel.ml import MLDetector


class Detector:
    HEURISTIC_WEIGHT = 0.4
    ML_WEIGHT = 0.6

    def __init__(self):
        self._heuristic = HeuristicDetector()
        self._ml = MLDetector()

    def detect(self, path: str | Path) -> dict:
        """Analyse an image and return a unified detection result.

        Parameters
        ----------
        path:
            File-system path to a JPEG, PNG, or WebP image.

        Returns
        -------
        dict with keys:
            score     – weighted confidence in [0, 1] (1 = AI)
            label     – "AI" | "REAL" | "UNCERTAIN"
            signals   – merged per-signal scores + ml_available bool
            reasoning – concatenated reasoning from both layers
        """
        h_result = self._heuristic.detect(path)
        ml_result = self._ml.detect(path)

        ml_available = ml_result is not None

        if ml_available:
            combined_score = float(
                np.clip(
                    self.HEURISTIC_WEIGHT * h_result["score"]
                    + self.ML_WEIGHT * ml_result["score"],
                    0.0,
                    1.0,
                )
            )
            combined_score = round(combined_score, 4)
            reasoning = h_result["reasoning"] + " | " + ml_result["reasoning"]
            signals = {**h_result["signals"], **ml_result["signals"], "ml_available": True}
        else:
            combined_score = h_result["score"]
            reasoning = (
                h_result["reasoning"] + " | ML classifier unavailable, using heuristics only."
            )
            signals = {**h_result["signals"], "ml_available": False}

        if combined_score >= 0.6:
            label = "AI"
        elif combined_score <= 0.4:
            label = "REAL"
        else:
            label = "UNCERTAIN"

        return {
            "score": combined_score,
            "label": label,
            "signals": signals,
            "reasoning": reasoning,
        }
