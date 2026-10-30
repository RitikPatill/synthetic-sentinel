# Synthetic Sentinel

> Local REST + MCP server that classifies images as AI-generated or real using heuristics and a CPU-only ML model.

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

Synthetic Sentinel answers one question: **is this image real or AI-generated?** It runs entirely offline on CPU and returns a scored verdict—`AI`, `REAL`, or `UNCERTAIN`—along with per-signal confidence values and a plain-English reasoning string.

Detection runs in two sequential layers. The heuristics layer (EXIF metadata checks, JPEG DCT artifact scoring, noise-floor analysis) fires instantly without any model download. The ML layer wraps the [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) HuggingFace checkpoint and activates when the model is cached locally; if it is not, the detector falls back to heuristics only. Both layers feed a weighted-fusion `Detector` class that four entry points share: a CLI, a Streamlit UI, a FastAPI REST endpoint, and an MCP stdio server.

## Quickstart

```bash
git clone https://github.com/RitikPatill/synthetic-sentinel.git
cd synthetic-sentinel

# Install PyTorch CPU wheel first to avoid pulling in the large CUDA binaries
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install the package and dev dependencies
pip install -e ".[dev]"

# Verify everything is wired up
pytest tests/

# Run the REST API
python -m sentinel.api          # listens on http://localhost:8000

# Or launch the interactive UI
streamlit run sentinel/app.py
```

## Usage

**CLI** — scan one or more local files or URLs; exits with code `1` if any image is labelled AI:

```bash
sentinel photo.jpg ai_art.png https://example.com/image.jpg
sentinel --threshold 0.7 photo.jpg
```

**REST API** — POST a URL or a base64-encoded image:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.jpg"}'
```

Response:

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

**MCP server** — add to Claude Desktop `claude_desktop_config.json` to expose a `detect_image` tool to any MCP-compatible host:

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

**Streamlit UI** — drag-drop a JPEG, PNG, or WebP file to see a colour-coded verdict badge, a per-signal bar chart, and the reasoning string.

## Architecture

All four entry points call `Detector` directly; none route through each other.

```
┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌────────────┐
│   CLI    │  │ Streamlit │  │  FastAPI     │  │ MCP server │
│ cli.py   │  │  app.py   │  │  /detect     │  │  stdio     │
└────┬─────┘  └─────┬─────┘  └──────┬───────┘  └─────┬──────┘
     └──────────────┴───────────────┴─────────────────┘
                              │
                       ┌──────▼──────┐
                       │  Detector   │
                       │  (weighted  │
                       │   fusion)   │
                       └──────┬──────┘
                  ┌───────────┴───────────┐
           ┌──────▼──────┐         ┌──────▼──────┐
           │ Heuristics  │         │     ML      │
           │ EXIF / DCT  │         │  classifier │
           │ noise floor │         │ (HuggingFace│
           └─────────────┘         └─────────────┘
```

Fusion weights: heuristics 0.4, ML classifier 0.6. Thresholds: `>= 0.6` → AI, `<= 0.4` → REAL, otherwise UNCERTAIN.

## Project structure

```
synthetic-sentinel/
├── sentinel/
│   ├── detector.py      # weighted fusion of both layers; primary public API
│   ├── heuristics.py    # EXIF absence, DCT artifact, noise-floor scorers
│   ├── ml.py            # HuggingFace pipeline wrapper with graceful fallback
│   ├── api.py           # FastAPI app; POST /detect
│   ├── mcp_server.py    # MCP stdio server; exposes detect_image tool
│   ├── cli.py           # argparse CLI; multi-target, exit-code contract
│   ├── app.py           # Streamlit demo UI
│   └── eval.py          # batch evaluation script; outputs accuracy/F1
├── tests/
│   ├── fixtures/        # real.jpg (camera EXIF present) + synthetic.jpg (none)
│   └── test_*.py        # unit + integration tests for every layer
├── docs/
│   └── demo.gif
├── pyproject.toml       # hatchling build; `sentinel` CLI entry point
├── requirements.txt     # pinned runtime deps; see torch note above
└── LICENSE
```

## Roadmap

- [ ] Expand eval dataset to 50 real + 50 AI images and publish benchmark numbers
- [ ] Add ONNX export of the ML classifier for faster CPU inference without PyTorch
- [ ] Support batch `/detect` endpoint accepting a list of URLs in one request
- [ ] Add C2PA / Content Credentials watermark check as a fourth heuristic signal
- [ ] Package as a single Docker image so the full stack runs with one command

## License

MIT — see LICENSE.

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
