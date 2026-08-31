"""
Streamlit UI for Robust AIGC Detector
Allows users to upload images and detect AI-generated content
"""

import streamlit as st
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
from typing import Optional

from aigc_detector.model import FrozenEncoders, load_checkpoint, ModelConfig
from aigc_detector.train import choose_device

# Configure Streamlit page
st.set_page_config(
    page_title="AIGC Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-meter {
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all available models from artifacts directory"""
    device = choose_device("auto")
    models = {}
    
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        st.warning("⚠️ No artifacts directory found. Please train a model first.")
        return models, device
    
    # Look for checkpoint files
    checkpoint_files = list(artifacts_dir.glob("*.pt"))
    
    for checkpoint_path in checkpoint_files:
        try:
            model_name = checkpoint_path.stem
            head, config, temperature, metadata = load_checkpoint(checkpoint_path, device)
            encoders = FrozenEncoders(config, device)
            models[model_name] = {
                "head": head,
                "encoders": encoders,
                "config": config,
                "temperature": temperature,
                "metadata": metadata,
                "path": checkpoint_path
            }
            st.sidebar.success(f"✅ Loaded model: {model_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {checkpoint_path.name}: {str(e)}")
    
    return models, device

def predict_single_image(image: Image.Image, models: dict, device: torch.device, use_ensemble: bool = False) -> dict:
    """
    Predict whether an image is AI-generated or real
    
    Args:
        image: PIL Image to predict
        models: Dictionary of loaded models
        device: Torch device
        use_ensemble: Whether to average predictions from all models
    
    Returns:
        Dictionary with prediction results
    """
    if not models:
        return {"error": "No models loaded"}
    
    # Convert image to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    predictions = {}
    
    for model_name, model_dict in models.items():
        try:
            with torch.inference_mode():
                # Extract features
                features = model_dict["encoders"]([image]).to(device)
                
                # Get head prediction
                logits = model_dict["head"](features)
                
                # Apply temperature calibration and sigmoid
                temperature = model_dict["temperature"]
                probability = torch.sigmoid(logits / temperature).cpu().numpy()[0]
                
                predictions[model_name] = {
                    "probability": float(probability),
                    "is_ai": probability > 0.5,
                    "confidence": abs(probability - 0.5) * 2  # Confidence between 0 and 1
                }
        except Exception as e:
            predictions[model_name] = {"error": str(e)}
    
    # Ensemble prediction if multiple models and requested
    if use_ensemble and len(predictions) > 1:
        valid_probs = [p["probability"] for p in predictions.values() if "probability" in p]
        if valid_probs:
            ensemble_prob = np.mean(valid_probs)
            predictions["ensemble"] = {
                "probability": float(ensemble_prob),
                "is_ai": ensemble_prob > 0.5,
                "confidence": abs(ensemble_prob - 0.5) * 2,
                "num_models": len(valid_probs)
            }
    
    return predictions

def display_result(predictions: dict, image: Image.Image):
    """Display prediction results in a formatted way"""
    if "error" in predictions:
        st.error(f"Error during prediction: {predictions['error']}")
        return
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
    
    with col2:
        # Display results for each model
        if "ensemble" in predictions:
            result = predictions["ensemble"]
            st.markdown(f"### 🎯 Ensemble Prediction ({result.get('num_models', 1)} models)")
            
            col_prob, col_conf = st.columns(2)
            with col_prob:
                label = "🤖 AI-Generated" if result["is_ai"] else "👤 Real Image"
                st.metric(label, f"{result['probability']:.1%}")
            with col_conf:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            st.divider()
        
        # Display individual model results
        for model_name, result in predictions.items():
            if model_name == "ensemble":
                continue
            
            if "error" in result:
                st.warning(f"**{model_name}**: {result['error']}")
                continue
            
            with st.expander(f"📊 {model_name}", expanded=len(predictions) == 2):
                col_prob, col_conf = st.columns(2)
                
                with col_prob:
                    label = "🤖 AI-Generated" if result["is_ai"] else "👤 Real Image"
                    st.metric(label, f"{result['probability']:.1%}")
                
                with col_conf:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                # Progress bar for visualization
                prob = result["probability"]
                st.progress(prob, text=f"{prob:.1%} toward AI-Generated")

# Main UI
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔍 AIGC Detector</div>', unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right; margin-top: 1rem;'>Track 5 - TechJam</div>", unsafe_allow_html=True)

st.markdown('<div class="subtitle">Detect AI-Generated Images Under Real-World Transformations</div>', unsafe_allow_html=True)

# Sidebar for model selection and info
st.sidebar.markdown("## ⚙️ Settings")

# Load models
models, device = load_models()

if not models:
    st.error("❌ No trained models found in artifacts/ directory. Please train a model first using the README instructions.")
    st.info("To get started, run: `uv run python scripts/download_cifake_smoke.py --per-class 50` and then `uv run aigc-train ...`")
    st.stop()

st.sidebar.markdown("### Loaded Models")
for model_name in models.keys():
    st.sidebar.text(f"• {model_name}")

# Model selection
use_ensemble = len(models) > 1
if use_ensemble:
    st.sidebar.markdown("---")
    use_ensemble = st.sidebar.checkbox("Use Ensemble (average all models)", value=True)

# Main content area
st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, PNG, etc.)",
    type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
    help="Images will be processed to detect whether they are AI-generated or real"
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    
    # Run prediction
    with st.spinner("🔄 Analyzing image..."):
        predictions = predict_single_image(image, models, device, use_ensemble=use_ensemble)
    
    # Display results
    display_result(predictions, image)
    
    # Additional info
    st.divider()
    st.markdown("### ℹ️ About This Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Model Architecture:**
        - Frozen CLIP ViT-B/32 for semantic features
        - Frozen EfficientNet-B0 with forensic views (Laplacian, FFT)
        - Small MLP head for fusion
        
        **Features:**
        - Robust to real-world transformations
        - Temperature-calibrated probabilities
        - Lightweight and fast inference
        """)
    
    with col2:
        st.markdown("""
        **Confidence Interpretation:**
        - **High confidence (>80%)**: Strong prediction
        - **Medium confidence (50-80%)**: Moderate prediction
        - **Low confidence (<50%)**: Uncertain, borderline case
        
        **Data Sources:**
        - CIFAKE, SID_Set, COCO+DALL-E
        - Diverse AI generation methods
        """)

else:
    # Show welcome message
    st.markdown("""
    ### 👋 Welcome!
    
    This detector uses advanced deep learning to identify AI-generated images even after real-world transformations.
    
    **Supported transformations:**
    - ✓ JPEG compression
    - ✓ Blur and resizing
    - ✓ Color shifts
    - ✓ Cropping
    - ✓ Gaussian noise
    
    **To get started:**
    1. Upload an image using the file uploader above
    2. The model will analyze the image
    3. You'll receive a probability score indicating if the image is AI-generated
    4. Higher scores mean more likely to be AI-generated
    
    **Questions?**
    - See the README.md for model details
    - Check the sidebar for loaded models and settings
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About Track 5")
st.sidebar.markdown("""
**Robust Detection of AI‑Generated Images Under Real‑World Transformations**

This competition focuses on detecting AI-generated images that have been subject to real-world transformations, 
making the task more challenging than detecting pristine AI generations.

[View Competition Details](https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem;'>
Built with ❤️ for TikTok TechJam 2024
</div>
""", unsafe_allow_html=True)
