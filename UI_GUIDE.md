# UI Quick Start Guide

## Overview

The AIGC Detector now includes a modern web-based UI built with Streamlit. This allows you to easily test your trained models on new images without using command-line tools.

## Prerequisites

1. Python 3.10+ (same as the main project)
2. A trained model checkpoint (`.pt` file) in the `artifacts/` directory
3. Dependencies installed via `uv sync`

## Getting Started

### Step 1: Train a Model

First, you need to train a model if you haven't already. Here's the quickest way:

```bash
# Download smoke test data (50 real + 50 AI images)
uv run python scripts/download_cifake_smoke.py --per-class 50

# Train the hybrid detector
uv run aigc-train \
  --data-dir data/cifake_smoke \
  --output artifacts/hybrid_detector.pt \
  --cache artifacts/cifake_smoke_features.pt \
  --augmentation-repeats 2 \
  --epochs 30
```

This takes about 5-10 minutes on a modern CPU.

### Step 2: Launch the UI

```bash
uv run streamlit run app.py
```

Streamlit will start a local web server and automatically open your browser to `http://localhost:8501`.

### Step 3: Test Images

1. Click the **"Upload an Image"** box to select a JPG, PNG, WebP, or other image file
2. The app will automatically run inference and display:
   - The uploaded image
   - AI probability score (0 = real, 1 = AI-generated)
   - Confidence level
   - Results from each loaded model

## Features

### Single Model Usage
If you only have one trained model, the UI will display its predictions prominently with:
- Probability score
- Confidence metric
- Visual progress bar

### Ensemble Mode
If you have trained multiple models (in the `artifacts/` directory), you can:
- **Enable Ensemble Predictions**: The UI will average predictions from all models
- **View Individual Results**: Expand sections to see each model's prediction
- **Compare Models**: See how different architectures perform on the same image

### Supported Transformations
The model is robust to real-world image transformations:
- JPEG compression
- Blur and resizing
- Color shifts and jitter
- Cropping
- Gaussian noise
- Combinations of the above

## Understanding Results

### Probability Score
- **0.0 - 0.4**: Likely real image
- **0.4 - 0.6**: Uncertain, could be either
- **0.6 - 1.0**: Likely AI-generated

### Confidence
- **>80%**: High confidence prediction
- **50-80%**: Moderate confidence
- **<50%**: Low confidence, prediction is uncertain

## Training Multiple Models

For better ensemble predictions, try training multiple models with different configurations:

```bash
# Model 1: Laplacian-only features
uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/model_laplacian.pt \
  --cache artifacts/cache_laplacian.pt \
  --forensic-mode laplacian \
  --epochs 40

# Model 2: Laplacian + FFT features
uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/model_laplacian_fft.pt \
  --cache artifacts/cache_laplacian_fft.pt \
  --forensic-mode laplacian_fft \
  --epochs 40
```

Then launch the UI - it will automatically load both models and offer ensemble predictions!

## Tips & Tricks

### Performance
- The first inference run loads model weights (slightly slower)
- Subsequent inferences are faster due to Streamlit caching
- For batch prediction, use the command-line `aigc-predict` tool instead

### Memory Usage
- The UI loads all available models into memory
- If you have limited RAM, keep only the best performing model in `artifacts/`
- Consider using a GPU if available for faster inference

### Troubleshooting

**"No trained models found"**
- Make sure you have a `.pt` file in the `artifacts/` directory
- Check that model training completed successfully
- Verify the file path: should be `artifacts/model_name.pt`

**Slow inference**
- Try using `--device cuda` when training if you have a GPU
- The UI will use the same device as the model was trained on
- First run loads weights; subsequent runs are cached

**"Failed to load model"**
- Check the error message for details
- Ensure the checkpoint was saved correctly
- Try retraining the model

## Advanced Usage

### Custom Model Path
To use models from a different directory, edit the model loading code in `app.py`:
```python
artifacts_dir = Path("path/to/your/models")
```

### Batch Processing
For processing many images at once:
```bash
uv run aigc-predict path/to/images/ \
  --checkpoint artifacts/hybrid_detector.pt \
  --output results.json
```

### Integration with Other Tools
The `app.py` file can be modified to:
- Save predictions to a database
- Send alerts for AI-generated images
- Integrate with existing ML pipelines
- Add custom preprocessing

## Next Steps

1. **Explore the competition** - See [Track 5 Details](https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc)
2. **Improve your model** - Train on larger, more diverse datasets
3. **Ensemble strategies** - Experiment with different model combinations
4. **Real-world testing** - Test on transformed images to validate robustness

## Questions?

- Check the main `README.md` for model training details
- See `aigc_detector/predict.py` for the inference pipeline
- Review model config options in `aigc_detector/model.py`
