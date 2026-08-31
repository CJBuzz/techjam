# AIGC Detector UI - Setup & Usage Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- UV package manager (install from https://github.com/astral-sh/uv)
- A trained model checkpoint

### 1. Install Dependencies

```bash
cd path/to/techjam
uv sync
```

This installs all required packages including the newly added `streamlit`.

### 2. Train a Model (if you haven't already)

Quick 5-minute smoke test:
```bash
uv run python scripts/download_cifake_smoke.py --per-class 50
uv run aigc-train \
  --data-dir data/cifake_smoke \
  --output artifacts/hybrid_detector.pt \
  --cache artifacts/cifake_smoke_features.pt \
  --augmentation-repeats 2 \
  --epochs 30
```

### 3. Launch the UI

**Windows:**
```bash
# Option A: Use the batch script
run_ui.bat

# Option B: Manual
uv run streamlit run app.py
```

**macOS/Linux:**
```bash
# Option A: Use the shell script
chmod +x run_ui.sh
./run_ui.sh

# Option B: Manual
uv run streamlit run app.py
```

The UI will open automatically at `http://localhost:8501`

## 📋 What's Included

### Main Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application |
| `UI_GUIDE.md` | Detailed user guide for the web interface |
| `scripts/batch_inference.py` | Batch processing script for multiple images |
| `run_ui.bat` | Windows launcher script |
| `run_ui.sh` | Unix/macOS launcher script |
| `.streamlit/config.toml` | Streamlit configuration |

### Updated Files

| File | Changes |
|------|---------|
| `pyproject.toml` | Added streamlit>=1.28 dependency |
| `README.md` | Added UI section with quick start instructions |

## 🎯 Features

### Web UI (`app.py`)
- 📤 Drag-and-drop image upload
- 🤖 Real-time AI vs Real detection
- 📊 Confidence scores and visualizations
- 🎯 Multi-model ensemble support
- 💾 Automatic model discovery and loading
- 🎨 Modern, responsive design

### Batch Processing (`scripts/batch_inference.py`)
- Process multiple images at once
- JSON output for integration with other tools
- Optional ensemble predictions
- Summary statistics and filtering

### Key Improvements Over Command-Line

| Feature | CLI (`aigc-predict`) | Web UI | Batch Script |
|---------|---------------------|--------|--------------|
| Single image upload | ✗ | ✓ | ✗ |
| Visual feedback | ✗ | ✓ | Partial |
| Real-time results | ✗ | ✓ | ✗ |
| Batch processing | ✓ | ✗ | ✓ |
| Model ensemble | ✗ | ✓ | ✓ |
| JSON output | ✓ | ✗ | ✓ |

## 💡 Usage Tips

### Using Multiple Models

Train multiple models with different configurations:
```bash
# Model 1
uv run aigc-train --data-dir data/mixed_5k --output artifacts/model_v1.pt ...

# Model 2  
uv run aigc-train --data-dir data/mixed_5k --output artifacts/model_v2.pt ...

# Model 3
uv run aigc-train --data-dir data/mixed_5k --output artifacts/model_v3.pt ...
```

The UI will automatically detect all `.pt` files and allow ensemble predictions!

### Batch Processing Examples

```bash
# Process directory with all default settings
uv run python scripts/batch_inference.py path/to/images

# With ensemble predictions
uv run python scripts/batch_inference.py path/to/images --ensemble

# With multiple checkpoints
uv run python scripts/batch_inference.py path/to/images \
  --checkpoint artifacts/model_v1.pt artifacts/model_v2.pt \
  --ensemble \
  --output results.json

# Custom threshold
uv run python scripts/batch_inference.py path/to/images \
  --threshold 0.6 \
  --output results_threshold60.json
```

## 🔧 Configuration

### Streamlit Settings (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#667eea"        # Purple accent
backgroundColor = "#ffffff"     # White background
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200            # Max upload size in MB
enableXsrfProtection = true
```

Modify these values to customize the UI appearance.

### Model Selection in `app.py`

Default behavior: loads all `.pt` files from `artifacts/` directory

To use a custom directory:
```python
# In app.py, modify:
artifacts_dir = Path("your/custom/path")
```

## 🐛 Troubleshooting

### Issue: "No models found"
**Solution:** Ensure a trained model exists at `artifacts/your_model.pt`

### Issue: Slow first inference
**Solution:** This is normal - the first run loads model weights. Subsequent runs are cached.

### Issue: "ModuleNotFoundError"
**Solution:** Run `uv sync` to install all dependencies

### Issue: Port 8501 already in use
**Solution:** Run on a different port: `uv run streamlit run app.py --server.port 8502`

### Issue: Memory errors
**Solution:** 
- Reduce batch size in `app.py`
- Remove unused model checkpoints from `artifacts/`
- Use a machine with more RAM or a GPU

## 📈 Performance Tips

1. **First Run**: 30-60 seconds (model loading)
2. **Cached Runs**: 2-5 seconds per image
3. **With GPU**: 1-2 seconds per image
4. **Batch Processing**: 50-100 images per minute (CPU)

## 🔗 Integration Examples

### Save to Database
```python
# In app.py, after getting predictions:
import sqlite3
conn = sqlite3.connect("predictions.db")
conn.execute("INSERT INTO predictions VALUES (?, ?, ?)", 
             (image_path, probability, timestamp))
conn.commit()
```

### Send Alerts
```python
# For high-confidence AI predictions:
if prediction["is_ai"] and prediction["confidence"] > 0.9:
    send_alert(f"AI image detected: {image_path}")
```

### API Integration
```bash
# Use batch_inference.py to generate JSON
uv run python scripts/batch_inference.py images/ --output api_payload.json

# Send to external API
curl -X POST https://api.example.com/detect -d @api_payload.json
```

## 📚 Additional Resources

- **Main README**: [README.md](README.md) - Complete technical details
- **UI Guide**: [UI_GUIDE.md](UI_GUIDE.md) - Detailed user guide
- **Competition**: [Track 5 Details](https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc)
- **Streamlit Docs**: https://docs.streamlit.io/

## ✨ What's Next?

1. **Train better models** with larger, more diverse datasets
2. **Experiment with ensemble strategies** (weighted averaging, voting)
3. **Test robustness** on transformed images
4. **Deploy to production** using Streamlit Cloud or Docker
5. **Contribute back** to the TechJam competition!

## 🆘 Need Help?

- Check `UI_GUIDE.md` for user documentation
- Review `README.md` for model training details
- See `scripts/batch_inference.py` for integration patterns
- Streamlit errors often have solutions in their [docs](https://docs.streamlit.io/)
