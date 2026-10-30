"""Streamlit demo UI for Synthetic Sentinel.

Run with:
    streamlit run sentinel/app.py
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from sentinel.detector import Detector


def _render_badge(label: str, score: float) -> None:
    if label == "AI":
        st.error(f"AI-GENERATED — confidence {score:.0%}")
    elif label == "REAL":
        st.success(f"REAL PHOTO — confidence {1 - score:.0%}")
    else:
        st.warning(f"UNCERTAIN — score {score:.2f}")


def main() -> None:
    st.set_page_config(page_title="Synthetic Sentinel", layout="centered")
    st.title("Synthetic Sentinel")
    st.caption("Detect whether an image is real or AI-generated — fully offline.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        return

    tmp_path = None
    try:
        suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.image(tmp_path, use_container_width=True)

        with st.spinner("Analysing…"):
            result = Detector().detect(tmp_path)

        _render_badge(result["label"], result["score"])

        numeric_signals = {k: v for k, v in result["signals"].items() if isinstance(v, float)}
        if numeric_signals:
            st.subheader("Signal breakdown")
            st.bar_chart(numeric_signals)

        st.caption(result["reasoning"])

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
