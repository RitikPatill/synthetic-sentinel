# Synthetic Sentinel

> **Is this image real or AI-generated?**

A self-contained, privacy-preserving detector for AI-generated images. Runs fully offline on CPU. Exposes results via a FastAPI REST endpoint and an MCP tool so both humans and LLM assistants can query it.

## Problem

AI-generated images are ubiquitous in 2026 and detection tooling is fragmented: most detectors are paywalled SaaS or research demos that won't run offline. Journalists, developers, and AI agents need a locally runnable detector they can pipe images through without sending data to a third party.

## Architecture

Detection runs in two sequential layers:

1. **Heuristics layer** — EXIF metadata checks, JPEG DCT artifact scoring, noise-pattern analysis (Pillow + NumPy). Fast, zero model download.
2. **ML layer** — a small CPU-only HuggingFace classifier (≤500 MB) fine-tuned on real vs. diffusion images. Falls back gracefully if the model isn't cached.

The REST API (`sentinel.api`) accepts `POST /detect` with a URL or base64 image and returns the detection JSON. The MCP server (`sentinel.mcp_server`) registers a `detect_image(image_url)` tool callable by Claude Desktop or any MCP-compatible host.

## What works (M4)

### Unified detector (`sentinel/detector.py`)

`Detector.detect(path)` is the single entry point for callers. It runs both layers and returns a weighted fusion:

| Weight | Layer |
|--------|-------|
| 0.4 | Heuristics (`HeuristicDetector`) |
| 0.6 | ML classifier (`MLDetector`) |

When the ML model is unavailable the detector falls back to heuristics-only and sets `signals.ml_available = false`.

```json
{
  "score": 0.76,
  "label": "AI",
  "signals": {
    "exif_absence": 0.8,
    "dct_artifacts": 0.65,
    "noise_floor": 0.72,
    "ml_classifier": 0.81,
    "ml_available": true
  },
  "reasoning": "Scored 0.73 (AI): no camera EXIF metadata, unnaturally smooth noise floor. | ML classifier scored 0.81 (AI)."
}
```

### ML layer (`sentinel/ml.py`)

`MLDetector` wraps the [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) HuggingFace checkpoint via a `transformers` image-classification pipeline. The pipeline is lazy-loaded on first call (CPU, `device=-1`) and cached at the class level. If `transformers` is not installed or the model download fails, `MLDetector.detect()` returns `None` and the unified detector falls back to heuristics.

Label normalisation handles model-specific label names: scores under keys containing `artif`, `fake`, or `ai` are used directly; a `human` label is inverted (`1 − score`).

### Heuristics layer (`sentinel/heuristics.py`)

Three independent scorers, each returning a `float` in `[0, 1]` (1 = confident AI):

| Signal | Function | Idea |
|--------|----------|------|
| EXIF absence | `score_exif_absence` | AI images almost never carry camera EXIF tags |
| JPEG DCT artifacts | `score_dct_artifacts` | Diffusion outputs saved as JPEG show flatter quant tables |
| Noise floor | `score_noise_floor` | Real photos carry sensor noise; AI outputs are unnaturally smooth |

`HeuristicDetector.detect(path)` combines all three with weights `exif=0.3, dct=0.3, noise=0.4`.

Labels: `score >= 0.6` → **AI**, `score <= 0.4` → **REAL**, otherwise **UNCERTAIN**.

### REST API (`sentinel/api.py`)

`POST /detect` accepts exactly one of `url` or `base64_image` and returns the detection JSON:

```bash
# Start the server
python -m sentinel.api          # listens on http://localhost:8000

# Detect via URL
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.jpg"}'

# Detect via base64
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"base64_image": "<base64-encoded-bytes>"}'
```

Response schema:

```json
{
  "score": 0.76,
  "label": "AI",
  "signals": { "exif_absence": 0.8, "dct_artifacts": 0.65, "noise_floor": 0.72, "ml_available": true },
  "reasoning": "..."
}
```

### MCP Server (`sentinel/mcp_server.py`)

Registers a `detect_image(image_url)` tool over stdio for MCP-compatible hosts (Claude Desktop, Cursor, etc.):

```bash
python -m sentinel.mcp_server
```

Add to Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "synthetic-sentinel": {
      "command": "python",
      "args": ["-m", "sentinel.mcp_server"]
    }
  }
}
```

### Tests

```bash
pytest tests/
```

Source files added in M4:
- `sentinel/api.py` — FastAPI application; `POST /detect` handler with URL fetch and base64 decode paths
- `sentinel/mcp_server.py` — MCP server built on `FastMCP`; exposes the `detect_image` tool over stdio
- `sentinel/__main__.py` — package-level entrypoint (`python -m sentinel`); prints available sub-commands and exits

Test coverage through M4:
- `tests/test_api.py` — smoke tests for `POST /detect`: schema validation, base64 input, 422 on missing/both fields
- `tests/test_detector.py` — integration tests for `Detector`: schema validation, weighted fusion, ML-available flag, heuristics-only fallback
- `tests/test_ml.py` — unit tests for `MLDetector`: schema, score range, label mapping, None-return when pipeline unavailable
- `tests/test_heuristics.py` — one test per scorer plus an end-to-end `HeuristicDetector.detect()` test
- `tests/conftest.py` — shared fixtures exposing `real_jpeg` and `synthetic_jpeg` paths
- `tests/fixtures/real.jpg` — a real camera JPEG (carries EXIF, sensor noise)
- `tests/fixtures/synthetic.jpg` — a Stable Diffusion output (no EXIF, smooth noise floor)
- `tests/create_fixtures.py` — script used to generate the fixture pair; not part of the test run

### M1 foundation

- `sentinel/` package skeleton with `sentinel/__init__.py` and a stub `sentinel/cli.py`
- `pyproject.toml` — hatchling build backend, `pip install -e ".[dev]"` installs ruff and pytest
- `requirements.txt` — runtime deps; M4 expanded it to include fastapi, uvicorn, mcp, httpx, transformers, and torch (CPU); streamlit remains commented for M5
- `tests/test_scaffold.py` — import smoke test confirming the package is importable
- MIT license

## Install

```bash
git clone https://github.com/you/synthetic-sentinel
cd synthetic-sentinel
pip install -e ".[dev]"
```

Or install runtime deps only:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```bash
# REST API
python -m sentinel.api           # http://localhost:8000
# curl -X POST http://localhost:8000/detect -H "Content-Type: application/json" \
#   -d '{"url": "https://example.com/photo.jpg"}'

# MCP server (stdio transport for Claude Desktop / Cursor)
python -m sentinel.mcp_server

# CLI (available after M5)
python -m sentinel.cli --help
```

## Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Scaffold + README | ✅ done |
| M2 | Heuristics layer | ✅ done |
| M3 | ML layer | ✅ done |
| M4 | FastAPI + MCP server | ✅ done |
| M5 | Streamlit UI + CLI | planned |
| M6 | Eval script + README GIF | planned |

## License

MIT © 2026 Ritik
