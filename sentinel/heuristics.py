"""
Heuristics-based AI image detector.

Three independent signals, each returning a float in [0, 1] where
1.0 = confident the image is AI-generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Empirical boundary: noise std ≈ 8 separates real photos from smooth AI outputs.
NOISE_THRESHOLD = 8.0


def score_exif_absence(img: Image.Image) -> float:
    """Return how likely the image is AI-generated based on missing/sparse EXIF.

    Scoring rules:
    - No EXIF at all           → 0.8  (strong AI signal)
    - EXIF with camera tags    → 0.1  (strong REAL signal)
    - EXIF but no camera tags  → 0.4  (weak signal either way)
    - Non-JPEG / unsupported   → 0.5  (neutral)
    """
    # Tags: DateTimeOriginal=0x9003, Make=0x010F, Model=0x0110
    CAMERA_TAGS = {0x9003, 0x010F, 0x0110}

    try:
        exif = dict(img.getexif())
    except Exception:
        return 0.5  # can't read → neutral

    if not exif:
        return 0.8

    if CAMERA_TAGS & exif.keys():
        return 0.1

    return 0.4


def score_dct_artifacts(path: str | Path) -> float:
    """Return how likely the image is AI-generated from JPEG quantisation tables.

    A real camera JPEG attenuates high frequencies heavily → large quant values
    and a high max/min ratio in the luminance table.
    A diffusion output saved as JPEG tends to have a flatter (more uniform) table.

    Maps max/min ratio to [0, 1]:
    - ratio == 1   (perfectly flat) → 1.0  (AI)
    - ratio >= 16  (steep roll-off) → 0.0  (REAL)
    - Linearly interpolated between.

    Non-JPEG formats → 0.5 (neutral).
    """
    try:
        img = Image.open(Path(path))
        img.load()
    except Exception:
        return 0.5

    if not hasattr(img, "quantization") or not img.quantization:
        return 0.5

    luma_table = img.quantization.get(0)
    if luma_table is None:
        return 0.5

    values = list(luma_table) if not isinstance(luma_table, list) else luma_table
    min_val = min(values)
    max_val = max(values)

    if min_val <= 0:
        return 0.5  # degenerate table → neutral

    ratio = max_val / min_val
    # ratio=1 → score=1.0 (AI); ratio≥16 → score=0.0 (REAL)
    score = 1.0 - min((ratio - 1.0) / 15.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def score_noise_floor(img: Image.Image) -> float:
    """Return how likely the image is AI-generated based on noise floor.

    Real photos carry sensor noise; diffusion outputs are unnaturally smooth.
    Uses a manual 2-D Laplacian (no scipy dependency) to measure noise std.

    Score = 1 / (1 + std / NOISE_THRESHOLD)  — high std → low score (REAL).
    Clamped to [0.05, 0.95].
    """
    gray = np.array(img.convert("L"), dtype=float)

    # 2-D Laplacian via array slicing (equivalent to kernel [[0,1,0],[1,-4,1],[0,1,0]])
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    std = float(np.std(laplacian))
    score = 1.0 / (1.0 + std / NOISE_THRESHOLD)
    return float(np.clip(score, 0.05, 0.95))


class HeuristicDetector:
    """Ensemble of heuristic scorers for AI-generated image detection."""

    # Signal weights (must sum to 1.0)
    _WEIGHTS = {"exif_absence": 0.3, "dct_artifacts": 0.3, "noise_floor": 0.4}

    def detect(self, path: str | Path) -> dict:
        """Analyse an image and return a structured detection result.

        Parameters
        ----------
        path:
            File-system path to a JPEG, PNG, or WebP image.

        Returns
        -------
        dict with keys:
            score     – weighted average confidence in [0, 1] (1 = AI)
            label     – "AI" | "REAL" | "UNCERTAIN"
            signals   – per-scorer scores
            reasoning – human-readable explanation
        """
        path = Path(path)

        with Image.open(path) as img:
            img.load()  # ensure data is read before file closes
            exif_score = score_exif_absence(img)
            noise_score = score_noise_floor(img)

        dct_score = score_dct_artifacts(path)

        signals = {
            "exif_absence": round(float(np.clip(exif_score, 0.0, 1.0)), 4),
            "dct_artifacts": round(float(np.clip(dct_score, 0.0, 1.0)), 4),
            "noise_floor": round(float(np.clip(noise_score, 0.0, 1.0)), 4),
        }

        w = self._WEIGHTS
        score = (
            w["exif_absence"] * signals["exif_absence"]
            + w["dct_artifacts"] * signals["dct_artifacts"]
            + w["noise_floor"] * signals["noise_floor"]
        )
        score = round(float(np.clip(score, 0.0, 1.0)), 4)

        if score >= 0.6:
            label = "AI"
        elif score <= 0.4:
            label = "REAL"
        else:
            label = "UNCERTAIN"

        parts = []
        if signals["exif_absence"] >= 0.7:
            parts.append("no camera EXIF metadata")
        elif signals["exif_absence"] <= 0.2:
            parts.append("camera EXIF metadata present")

        if signals["dct_artifacts"] >= 0.7:
            parts.append("flat JPEG quantisation table")
        elif signals["dct_artifacts"] <= 0.2:
            parts.append("natural JPEG quantisation roll-off")

        if signals["noise_floor"] >= 0.7:
            parts.append("unnaturally smooth noise floor")
        elif signals["noise_floor"] <= 0.3:
            parts.append("natural sensor noise detected")

        if parts:
            reasoning = f"Scored {score:.2f} ({label}): {', '.join(parts)}."
        else:
            reasoning = f"Scored {score:.2f} ({label}): mixed signals, low confidence."

        return {
            "score": score,
            "label": label,
            "signals": signals,
            "reasoning": reasoning,
        }
