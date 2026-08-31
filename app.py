"""Streamlit viewer for the Track 5 ``output.json`` submission contract.

Inference stays in ``aigc-predict`` so the JSON artifact and dashboard can be
checked independently. Validation and path resolution live in the package so
they remain unit-testable without importing Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from aigc_detector.dashboard import load_results, resolve_image


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
OUTPUT_FILE = ROOT / "output.json"

st.set_page_config(page_title="AIGC Detector", page_icon="🔍", layout="wide")
st.markdown(
    """
    <style>
    .probability-spectrum { margin: .35rem 0 1rem; }
    .spectrum-track {
        position: relative; height: 14px; border-radius: 999px;
        background: linear-gradient(90deg, #2e9d59 0%, #f0c84b 50%, #d64b4b 100%);
    }
    .spectrum-marker {
        position: absolute; top: -4px; width: 20px; height: 20px;
        border: 3px solid white; border-radius: 50%;
        background: #222; box-shadow: 0 1px 5px #555;
        transform: translateX(-50%);
    }
    .spectrum-labels {
        display: flex; justify-content: space-between;
        font-size: .75rem; color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_results(output_file: str, modified_time_ns: int) -> list[dict[str, str | float]]:
    """Invalidate cached JSON whenever inference rewrites the file."""
    del modified_time_ns
    return load_results(output_file)


def render_record(record: dict[str, str | float]) -> None:
    """Render one image with its calibrated AIGC probability."""
    image_path, probability = str(record["image_path"]), float(record["pred"])
    image_file = resolve_image(image_path, ROOT, IMAGES_DIR)
    if image_file is None:
        st.warning(f"Image not found: `{image_path}`")
        return
    try:
        with Image.open(image_file) as source:
            st.image(source.convert("RGB"), use_container_width=True)
    except (OSError, ValueError) as error:
        st.error(f"Could not open `{image_path}`: {error}")
        return

    label = "AIGC likely" if probability >= 0.5 else "Likely real"
    st.metric(label, f"{probability:.1%}")
    st.markdown(
        f"""
        <div class="probability-spectrum" aria-label="AIGC probability spectrum">
          <div class="spectrum-track">
            <div class="spectrum-marker" style="left: {probability * 100:.2f}%"></div>
          </div>
          <div class="spectrum-labels"><span>Human / real</span><span>AIGC / AI</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"`{image_path}`")


st.title("🔍 Robust AIGC Detector")
st.caption("Submission dashboard — calibrated predictions loaded from output.json")
if st.button("Refresh results"):
    st.cache_data.clear()
    st.rerun()
if not OUTPUT_FILE.is_file():
    st.info("No output.json found. Put images in images/ and run ./infer.sh first.")
    st.stop()
try:
    results = cached_results(str(OUTPUT_FILE), OUTPUT_FILE.stat().st_mtime_ns)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Invalid {OUTPUT_FILE.name}: {error}")
    st.stop()

st.write(f"Showing {len(results)} prediction(s) from `{OUTPUT_FILE.name}`")
if not results:
    st.info("The result file is empty.")
    st.stop()
columns = st.columns(3)
for index, record in enumerate(results):
    with columns[index % len(columns)]:
        render_record(record)
