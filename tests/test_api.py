"""Smoke tests for the FastAPI /detect endpoint."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from sentinel.api import app

client = TestClient(app)

FIXTURES = Path(__file__).parent / "fixtures"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def test_detect_base64_returns_schema():
    """POST with base64 image returns 200 and correct schema."""
    b64 = _b64(FIXTURES / "synthetic.jpg")
    response = client.post("/detect", json={"base64_image": b64})
    assert response.status_code == 200, response.text

    data = response.json()
    assert "score" in data
    assert "label" in data
    assert "signals" in data
    assert "reasoning" in data

    assert data["label"] in {"AI", "REAL", "UNCERTAIN"}
    assert 0.0 <= data["score"] <= 1.0


def test_detect_missing_fields_returns_422():
    """POST with no fields returns 422."""
    response = client.post("/detect", json={})
    assert response.status_code == 422


def test_detect_both_fields_returns_422():
    """POST with both url and base64_image returns 422."""
    b64 = _b64(FIXTURES / "synthetic.jpg")
    response = client.post(
        "/detect",
        json={"url": "http://example.com/image.jpg", "base64_image": b64},
    )
    assert response.status_code == 422
