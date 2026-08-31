"""Streamlit dashboard for viewing directory inference results.

Inference is intentionally kept outside the dashboard. Run ``infer.sh`` to
create ``output.json``; this app then reads that file and displays its images.
"""

import json
from pathlib import Path

import streamlit as st
from PIL import Image


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
    .spectrum-labels { display: flex; justify-content: space-between; font-size: .75rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _image_candidates(image_path: str) -> list[Path]:
    """Return useful locations for paths produced by the inference script."""
    raw = Path(image_path)
    candidates = [raw if raw.is_absolute() else ROOT / raw]
    if not raw.is_absolute():
        candidates.extend((IMAGES_DIR / raw, IMAGES_DIR / raw.name))
    return list(dict.fromkeys(path.resolve() for path in candidates))


@st.cache_data
def load_results(output_file: str, modified_time_ns: int) -> list[dict]:
    """Load and validate the required ``image_path``/``pred`` JSON records."""
    del modified_time_ns
    with Path(output_file).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("output.json must contain a JSON array")

    records = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or "image_path" not in row or "pred" not in row:
            raise ValueError(f"Record {index} must contain image_path and pred")
        probability = float(row["pred"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Record {index} has pred outside [0, 1]")
        records.append({"image_path": str(row["image_path"]), "pred": probability})
    return records


def resolve_image(image_path: str) -> Path | None:
    for candidate in _image_candidates(image_path):
        if candidate.is_file():
            return candidate
    return None


def render_record(record: dict) -> None:
    image_path = record["image_path"]
    probability = record["pred"]
    image_file = resolve_image(image_path)

    if image_file is None:
        st.warning(f"Image not found: `{image_path}`")
        return
    try:
        with Image.open(image_file) as source:
            image = source.convert("RGB")
        st.image(image, use_container_width=True)
    except (OSError, ValueError) as error:
        st.error(f"Could not open `{image_path}`: {error}")
        return

    label = "AIGC likely" if probability >= 0.5 else "Likely real"
    st.metric(label, f"{probability:.1%}")
    marker_position = probability * 100
    st.markdown(
        f"""
        <div class="probability-spectrum" aria-label="AIGC probability spectrum">
          <div class="spectrum-track">
            <div class="spectrum-marker" style="left: {marker_position:.2f}%"></div>
          </div>
          <div class="spectrum-labels"><span>Human / real</span><span>AIGC / AI</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"`{image_path}`")


st.title("🔍 AIGC Detector")
st.caption("Directory inference dashboard — predictions come from output.json")

if st.button("Refresh results"):
    st.cache_data.clear()
    st.rerun()

if not OUTPUT_FILE.is_file():
    st.info("No output.json found yet. Add images to images/ and run ./infer.sh.")
    st.stop()

try:
    results = load_results(str(OUTPUT_FILE), OUTPUT_FILE.stat().st_mtime_ns)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Could not load {OUTPUT_FILE.name}: {error}")
    st.stop()

st.write(f"Showing {len(results)} prediction(s) from `{OUTPUT_FILE.name}`")
if not results:
    st.info("output.json is empty. Add images to images/ and run ./infer.sh.")
    st.stop()

columns = st.columns(3)
for index, record in enumerate(results):
    with columns[index % len(columns)]:
        render_record(record)
