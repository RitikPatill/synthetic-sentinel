"""
ML-based AI image detector.

Wraps the `umm-maybe/AI-image-detector` HuggingFace checkpoint via a
lazy-loaded transformers pipeline. Falls back gracefully (returns None)
when the model or its dependencies are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class MLDetector:
    MODEL_ID = "umm-maybe/AI-image-detector"

    # None = not yet attempted; False = load failed (permanent)
    _pipeline = None

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline

                cls._pipeline = hf_pipeline(
                    "image-classification", model=cls.MODEL_ID, device=-1
                )
            except Exception:
                cls._pipeline = False
        return cls._pipeline if cls._pipeline is not False else None

    def detect(self, path: str | Path) -> Optional[dict]:
        """Analyse an image with the ML classifier.

        Returns None when the pipeline is unavailable. Otherwise returns:
            score     – confidence image is AI [0, 1]
            label     – "AI" | "REAL" | "UNCERTAIN"
            signals   – {"ml_classifier": float}
            reasoning – human-readable explanation
        """
        pipe = self._get_pipeline()
        if pipe is None:
            return None

        from PIL import Image

        img = Image.open(Path(path)).convert("RGB")
        results = pipe(img)

        # Find the AI-confidence score by label name
        ai_score: float = 0.5
        for item in results:
            label_lower = item["label"].lower()
            if any(k in label_lower for k in ("artif", "fake", "ai")):
                ai_score = float(item["score"])
                break
        else:
            # No AI label found — invert the human score if present
            for item in results:
                if "human" in item["label"].lower():
                    ai_score = round(1.0 - float(item["score"]), 4)
                    break

        ai_score = max(0.0, min(1.0, ai_score))

        if ai_score >= 0.6:
            label = "AI"
        elif ai_score <= 0.4:
            label = "REAL"
        else:
            label = "UNCERTAIN"

        reasoning = f"ML classifier scored {ai_score:.2f} ({label})."

        return {
            "score": round(ai_score, 4),
            "label": label,
            "signals": {"ml_classifier": round(ai_score, 4)},
            "reasoning": reasoning,
        }
