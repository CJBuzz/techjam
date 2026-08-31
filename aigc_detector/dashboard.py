"""Validation and path resolution for the submission results dashboard.

The challenge contract deliberately separates inference from presentation:
``aigc-predict`` writes JSON and Streamlit only renders those records. Keeping
these helpers free of Streamlit makes the contract easy to test.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_results(output_file: str | Path) -> list[dict[str, str | float]]:
    """Read and validate exact ``image_path``/``pred`` result records."""
    with Path(output_file).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    # A top-level array preserves one stable record per input image.
    if not isinstance(payload, list):
        raise ValueError("output JSON must contain an array")

    records: list[dict[str, str | float]] = []
    for index, row in enumerate(payload):
        # Reject extra keys so the demo exercises the exact submission schema.
        if not isinstance(row, dict) or set(row) != {"image_path", "pred"}:
            raise ValueError(f"Record {index} must contain exactly image_path and pred")
        probability = float(row["pred"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Record {index} has pred outside [0, 1]")
        records.append({"image_path": str(row["image_path"]), "pred": probability})
    return records


def resolve_image(image_path: str, root: Path, images_dir: Path) -> Path | None:
    """Resolve predictor paths without allowing the dashboard to guess content."""
    raw = Path(image_path)
    candidates = [raw if raw.is_absolute() else root / raw]
    if not raw.is_absolute():
        # Accept predictor-relative paths and bare filenames copied to images/.
        candidates.extend((images_dir / raw, images_dir / raw.name))
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_file():
            return candidate
    return None
