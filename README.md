# Synthetic Sentinel

> **Is this image real or AI-generated?**

A self-contained, privacy-preserving detector for AI-generated images. Runs fully offline on CPU. Exposes results via a FastAPI REST endpoint and an MCP tool so both humans and LLM assistants can query it.

## Problem

AI-generated images are ubiquitous in 2026 and detection tooling is fragmented: most detectors are paywalled SaaS or research demos that won't run offline. Journalists, developers, and AI agents need a locally runnable detector they can pipe images through without sending data to a third party.

## Architecture

Detection runs in two sequential layers:

1. **Heuristics layer** — EXIF metadata checks, JPEG DCT artifact scoring, noise-pattern analysis (Pillow + NumPy). Fast, zero model download.
2. **ML layer** — a small CPU-only HuggingFace classifier (≤500 MB) fine-tuned on real vs. diffusion images. Falls back gracefully if the model isn't cached.

The REST API (`sentinel.api`) and MCP server (`sentinel.mcp_server`) are thin wrappers over these two layers. Both are planned for M4.

<!-- TODO: add component diagram once API shape is finalised in M4 -->

## What works (M2)

### Heuristics layer (`sentinel/heuristics.py`)

Three independent scorers, each returning a `float` in `[0, 1]` (1 = confident AI):

| Signal | Function | Idea |
|--------|----------|------|
| EXIF absence | `score_exif_absence` | AI images almost never carry camera EXIF tags |
| JPEG DCT artifacts | `score_dct_artifacts` | Diffusion outputs saved as JPEG show flatter quant tables |
| Noise floor | `score_noise_floor` | Real photos carry sensor noise; AI outputs are unnaturally smooth |

`HeuristicDetector.detect(path)` combines all three with weights `exif=0.3, dct=0.3, noise=0.4` and returns:

```json
{
  "score": 0.73,
  "label": "AI",
  "signals": {
    "exif_absence": 0.8,
    "dct_artifacts": 0.65,
    "noise_floor": 0.72
  },
  "reasoning": "Scored 0.73 (AI): no camera EXIF metadata, unnaturally smooth noise floor."
}
```

Labels: `score >= 0.6` → **AI**, `score <= 0.4` → **REAL**, otherwise **UNCERTAIN**.

### Tests

```bash
pytest tests/
```

Coverage targets for M2:
- `tests/test_heuristics.py` — one test per scorer plus an end-to-end `detect()` test
- `tests/conftest.py` — shared fixtures exposing `real_path` and `synthetic_path`
- `tests/fixtures/real.jpg` — a real camera JPEG (carries EXIF, sensor noise)
- `tests/fixtures/synthetic.jpg` — a Stable Diffusion output (no EXIF, smooth noise floor)
- `tests/create_fixtures.py` — script used to generate the fixture pair; not part of the test run

### M1 foundation

- `sentinel/` package skeleton with `sentinel/__init__.py` and a stub `sentinel/cli.py`
- `pyproject.toml` — hatchling build backend, `pip install -e ".[dev]"` installs ruff and pytest
- `requirements.txt` — pillow + numpy as runtime deps; future deps commented with their target milestone
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
# CLI (available after M5)
python -m sentinel.cli --help

# REST API (available after M4)
uvicorn sentinel.api:app --reload
# POST /detect  { "url": "https://..." }

# MCP server (available after M4)
python -m sentinel.mcp_server
```

## Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Scaffold + README | ✅ done |
| M2 | Heuristics layer | ✅ done |
| M3 | ML layer | planned |
| M4 | FastAPI + MCP server | planned |
| M5 | Streamlit UI + CLI | planned |
| M6 | Eval script + README GIF | planned |

## License

MIT © 2026 Ritik
