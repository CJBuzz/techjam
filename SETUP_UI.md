# Dashboard setup

This dashboard is a local Streamlit viewer. Model inference runs separately and writes results to `output.json`.

## Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)
- A trained checkpoint, for example `artifacts/robust_laplacian_fft.pt`

## Install

From the repository root:

```bash
uv sync
```

If no checkpoint exists, train the smoke-test model:

```bash
uv run python scripts/download_cifake_smoke.py --per-class 50
uv run aigc-train \
  --data-dir data/cifake_smoke \
  --output artifacts/hybrid_detector.pt \
  --cache artifacts/cifake_smoke_features.pt \
  --augmentation-repeats 2 \
  --epochs 30
```

## Run the workflow

1. Put supported images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, or `.tif`) in `images/`.
2. Generate predictions:

   ```bash
   ./infer.sh
   ```

   On Windows, use Git Bash or run the equivalent:

   ```bash
   uv run python scripts/predict_directory.py images \
     --checkpoint artifacts/robust_laplacian_fft.pt \
     --output output.json
   ```

3. Start the dashboard:

   ```bash
   uv run streamlit run app.py
   ```

4. Open `http://localhost:8501`.
5. Click **Refresh results** after regenerating `output.json`.

The script accepts overrides:

```bash
./infer.sh [image_dir] [checkpoint] [output_json]
```

For example:

```bash
./infer.sh images artifacts/hybrid_detector.pt output.json
```

## Expected files

```text
images/
  image_001.jpg
  image_002.png
artifacts/
  robust_laplacian_fft.pt
output.json
app.py
infer.sh
```

`output.json` is generated locally and ignored by Git.
