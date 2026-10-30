"""FastAPI application for AI image detection.

Run with:
    python -m sentinel.api
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.detector import Detector

app = FastAPI(title="Synthetic Sentinel", version="0.1.0")
_detector = Detector()


class DetectRequest(BaseModel):
    url: Optional[str] = None
    base64_image: Optional[str] = None


def _to_serializable(obj):
    """Recursively cast numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    # numpy floats / ints
    try:
        import numpy as np
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return obj


@app.post("/detect")
def detect(request: DetectRequest) -> dict:
    """Detect whether an image is AI-generated or real.

    Accepts exactly one of `url` or `base64_image`.
    """
    has_url = request.url is not None
    has_b64 = request.base64_image is not None

    if has_url == has_b64:  # both set or neither set
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'url' or 'base64_image'.",
        )

    tmp_path = None
    try:
        if has_url:
            with httpx.Client(follow_redirects=True, timeout=30) as client:
                response = client.get(request.url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to fetch URL: HTTP {response.status_code}",
                )
            image_bytes = response.content
            content_type = response.headers.get("content-type", "")
            suffix = ".png" if "png" in content_type else ".jpg"
        else:
            try:
                image_bytes = base64.b64decode(request.base64_image)
            except Exception:
                raise HTTPException(status_code=422, detail="Invalid base64 data.")
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = _detector.detect(tmp_path)
        return _to_serializable(result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sentinel.api:app", host="0.0.0.0", port=8000, reload=False)
