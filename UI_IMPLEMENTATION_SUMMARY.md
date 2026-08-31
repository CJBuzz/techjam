# UI implementation summary

## Scope

The UI is a results dashboard for Track 5. Inference and visualization are separate processes:

1. Users place images in `images/`.
2. `infer.sh` invokes `scripts/predict_directory.py`.
3. The predictor loads one checkpoint and writes `output.json`.
4. `app.py` reads the JSON and displays the referenced images.

## Implemented files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit results viewer |
| `infer.sh` | Unix/macOS inference launcher |
| `scripts/predict_directory.py` | Simple entry point for directory inference |
| `aigc_detector/predict.py` | Batched calibrated prediction and JSON writing |
| `images/.gitkeep` | Placeholder for user-provided images |
| `run_ui.sh`, `run_ui.bat` | Dashboard launchers |

## Dashboard behavior

- Reads `output.json` instead of running inference.
- Validates that every record has `image_path` and `pred`.
- Rejects probabilities outside the inclusive range 0–1.
- Resolves paths relative to the repository or `images/`.
- Displays a three-column image grid.
- Shows the numeric probability and a green-to-red spectrum:
  - green = likely human/real;
  - yellow = uncertain;
  - red = likely AIGC/AI.
- Shows a warning for missing images and an error for invalid image files.
- Supports manual refresh and automatically invalidates cached results when `output.json` changes.

## Inference output

The required output shape is:

```json
[
  {"image_path": "images/example.jpg", "pred": 0.82}
]
```

The value is a calibrated AIGC likelihood, not a guaranteed fact or provenance assertion.

## Run commands

```bash
uv sync
./infer.sh
uv run streamlit run app.py
```

Override defaults with:

```bash
./infer.sh images artifacts/model.pt output.json
```

## Verification

The implementation was checked with:

- Python compilation for `app.py` and prediction scripts;
- `bash -n infer.sh`;
- Core unit tests;
- Git whitespace validation.
