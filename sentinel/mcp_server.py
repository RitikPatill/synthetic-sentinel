"""MCP server for AI image detection.

Registers a `detect_image(image_url)` tool callable by any MCP-compatible host.

Run with:
    python -m sentinel.mcp_server
"""

from __future__ import annotations

import os
import tempfile

import httpx
from mcp.server.fastmcp import FastMCP

from sentinel.detector import Detector

mcp = FastMCP("synthetic-sentinel")
_detector = Detector()


def _to_serializable(obj):
    """Recursively cast numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return obj


@mcp.tool()
def detect_image(image_url: str) -> dict:
    """Return AI-vs-real verdict for the image at image_url.

    Parameters
    ----------
    image_url:
        HTTP/HTTPS URL pointing to a JPEG, PNG, or WebP image.

    Returns
    -------
    dict with keys: score, label, signals, reasoning
    """
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        response = client.get(image_url)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    suffix = ".png" if "png" in content_type else ".jpg"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        result = _detector.detect(tmp_path)
        return _to_serializable(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    mcp.run()
