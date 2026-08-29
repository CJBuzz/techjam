# Robust AIGC Detector

A lightweight two-stream detector for TikTok TechJam Track 5. It fuses a frozen CLIP ViT-B/32 semantic embedding with frozen EfficientNet-B0 features computed from a reproducible Laplacian high-pass image. Only a small MLP is trained. A held-out split is used for early stopping and post-hoc temperature calibration.

The model stays far below the 2-billion-parameter limit and feature caching makes the initial frozen-encoder stage practical on modest hardware.

## Setup

```bash
uv sync
```

Training data must use this layout (folder aliases `fake`, `aigc`, and `authentic` are also accepted):

```text
data/my_dataset/
├── ai/
└── real/
```

Do not use the challenge's COCO val2017 / DALL-E Advanced demonstration set for training.

## 100-image smoke test

Download 50 real and 50 AI-generated CIFAKE images, then train the frozen hybrid model:

```bash
uv run python scripts/download_cifake_smoke.py --per-class 50
uv run aigc-train \
  --data-dir data/cifake_smoke \
  --output artifacts/hybrid_detector.pt \
  --cache artifacts/cifake_smoke_features.pt \
  --augmentation-repeats 2 \
  --epochs 30
```

The cache stores one clean and one randomly transformed feature row per training image. On Kaggle, increase `--augmentation-repeats` to 4-8 and use a larger, source-diverse dataset. Delete or change the cache path whenever the source images, split seed, encoder, or augmentation configuration changes.

## Reproduced mixed 5K CPU run

The verified local run uses 1,250 real and 1,250 synthetic images from each of
CIFAKE and SID_Set. SID_Set's tampered class is excluded because this prototype
targets fully generated images. Images are split independently within every
class/source group: 3,496 train (70%), 752 validation (15%), and 752 test (15%).

```bash
uv run python scripts/download_mixed_5k.py --per-class-source 1250

uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/mixed_5k_detector.pt \
  --cache artifacts/mixed_5k_features.pt \
  --validation-fraction 0.15 --test-fraction 0.15 --seed 42 \
  --augmentation-repeats 2 --feature-batch-size 16 \
  --head-batch-size 64 --epochs 40 --patience 7 --device cpu

uv run aigc-evaluate \
  --data-dir data/mixed_5k \
  --checkpoint artifacts/mixed_5k_detector.pt \
  --output artifacts/mixed_5k_test_robustness.json \
  --validation-fraction 0.15 --test-fraction 0.15 --seed 42 \
  --batch-size 16 --device cpu
```

On the tested CPU, training plus feature extraction took 12m36s and deterministic
test robustness evaluation took 7m47s. Test accuracy was 95.7% clean, 94.4% JPEG,
89.6% blur, 89.5% resize, 92.0% noise, 94.8% color, and 91.9% crop. These are
same-source held-out results, not proof of unseen-generator or cross-dataset
generalization.

## Required JSON inference

```bash
uv run aigc-predict path/to/images \
  --checkpoint artifacts/hybrid_detector.pt \
  --output predictions.json
```

The output is a JSON array of records with exactly `image_path` and calibrated AIGC probability `pred` fields.

## Quick classification demo

Use the cached untouched test features to show several correct examples from each
class and report precision, recall, F1, accuracy, specificity, ROC-AUC, average
precision, the confusion matrix, and the full per-class report:

```bash
uv run python scripts/demo_classification.py --examples-per-class 3
```

The examples are illustrative; every reported metric is calculated over all 752
clean test images, not only the displayed correct predictions.

## Robustness evaluation

```bash
uv run aigc-evaluate \
  --data-dir path/to/held_out_data \
  --checkpoint artifacts/hybrid_detector.pt \
  --output artifacts/robustness.json
```

The evaluator recreates the source-stratified 70/15/15 split and evaluates only
the untouched test originals. Its JSON contains aggregate and per-source metrics
for every transformation. Keep the split fractions and seed identical to training.
The model automatically discovers weights in this workspace's `.hf-cache` and
`.torch-cache`, so `HF_HUB_OFFLINE=1` works after the initial download. On a new
machine, omit the offline flag for the first run so the pretrained weights can be
downloaded.

For a final report, use a held-out dataset/generator and report clean plus individual JPEG, blur, resizing, noise, color, and crop settings. The smoke subset only proves that the software path works; its metrics are not scientifically meaningful.

## Kaggle recommendations

- Enable a GPU. The encoders automatically use CUDA; only the fusion head is optimized.
- Use generator/source-separated training and validation data. Never split transformed copies of one original across splits.
- Prefer SID_Set or a mixture of properly licensed sources at realistic resolution. CIFAKE is only 32×32 and has severe source shortcuts, so it is unsuitable as the sole final dataset.
- Cache 4-8 feature variants per training image using `--augmentation-repeats`; the first is clean and the rest receive a challenge-style random transform.
- Compare CLIP-only, forensic-only, and fused features before claiming a hybrid gain.

## Limitations and future work

Frozen ImageNet EfficientNet was not pretrained specifically on Laplacian inputs, and high-frequency evidence is fragile under blur and compression. CLIP can learn content or dataset bias. Temperature calibration cannot repair domain shift. A stronger version should fine-tune the last EfficientNet block on a diverse high-resolution corpus, measure per-generator generalization, tune thresholds for moderation costs, and include representative false-positive/false-negative analysis.
