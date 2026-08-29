# Robust AIGC Detector

A lightweight multi-view detector for TikTok TechJam Track 5. The strongest version fuses a frozen CLIP ViT-B/32 semantic embedding with frozen EfficientNet-B0 features from both a reproducible Laplacian high-pass image and a centered log-magnitude FFT. Only a small MLP is trained. A held-out validation split is used for early stopping, candidate selection, and post-hoc temperature calibration; the test split is reserved for one final evaluation.

The model stays far below the 2-billion-parameter limit and feature caching makes the initial frozen-encoder stage practical on modest hardware.

## Robustness-first training

The current training path can cover every challenge severity deterministically, keep clean/transformed pairs together, penalize prediction drift, and optimize the worst transformation group. Checkpoint selection uses clean loss plus the mean and worst losses on the hardest validation severities. The recommended starting point is:

```bash
uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/robust_laplacian.pt \
  --cache artifacts/robust_laplacian_features.pt \
  --forensic-mode laplacian \
  --augmentation-policy balanced --augmentation-repeats 6 \
  --consistency-weight 0.05 --worst-group-weight 0.50 \
  --robust-validation-weight 0.70 \
  --feature-batch-size 16 --head-batch-size 64 \
  --epochs 50 --patience 8 --device auto

uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/robust_laplacian_fft.pt \
  --cache artifacts/robust_laplacian_fft_features.pt \
  --forensic-mode laplacian_fft \
  --initialize-from-laplacian artifacts/robust_laplacian.pt \
  --augmentation-policy balanced --augmentation-repeats 6 \
  --consistency-weight 0.05 --worst-group-weight 0.50 \
  --robust-validation-weight 0.70 --learning-rate 1e-4 \
  --feature-batch-size 16 --head-batch-size 64 \
  --epochs 50 --patience 8 --device auto
```

The balanced policy includes every official JPEG, blur, resize, noise, color, and crop severity, plus a small set of composed redistribution transforms. Feature caches include an experiment manifest and are rejected when the dataset, split, encoder, or augmentation configuration changes.

### Adaptive three-expert ensemble

The experimental three-expert head combines complementary evidence:

- a CLIP-only semantic expert intended to remain useful when pixel artifacts are damaged;
- a CLIP + Laplacian expert for local edge and residual evidence;
- a CLIP + FFT expert for frequency evidence;
- a learned softmax gate, optionally conditioned on inexpensive image-quality statistics.

Initialize it from the two robust checkpoints and train it against the same robustness-aware validation objective:

```bash
uv run aigc-train-mixture \
  --data-dir data/mixed_5k \
  --cache artifacts/robust_laplacian_fft_features.pt \
  --laplacian-checkpoint artifacts/robust_laplacian.pt \
  --fused-checkpoint artifacts/robust_laplacian_fft.pt \
  --output artifacts/robust_three_expert.pt \
  --experts three --gate-mode quality \
  --gate-prior-weight 0.01 --robust-validation-weight 0.70 \
  --learning-rate 5e-5 --epochs 40 --patience 8 --device auto
```

Treat this ensemble as a candidate until it beats the fused model on the untouched validation robustness matrix. Model selection must not use the test split.

### Exact robustness and error analysis

```bash
uv run aigc-evaluate \
  --data-dir data/mixed_5k \
  --checkpoint artifacts/robust_laplacian_fft.pt \
  --checkpoint artifacts/robust_three_expert.pt \
  --split validation --profile full \
  --output artifacts/validation_robustness.json \
  --error-analysis-output artifacts/validation_errors.json \
  --batch-size 16 --device auto
```

`--profile full` scores all 15 official non-clean severities independently. The report includes precision, recall, F1, specificity, false-positive/false-negative rates, confusion matrices, mean transformed accuracy, worst transformed accuracy, and the drop from clean accuracy. After selecting one final checkpoint, run the same command once with `--split test`.

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

The cache stores one clean and one randomly transformed feature row per training image. On Kaggle, prefer the balanced robustness-first configuration above and use a larger, source-diverse dataset. Caches are automatically checked against their experiment manifest.

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

## Improved Laplacian + FFT model

The selected improvement uses three frozen views: CLIP, Laplacian, and FFT. The FFT view is the centered log-magnitude spectrum after mean removal and a Hann window, robustly scaled by its per-image 1st and 99th percentiles. Each augmented training copy composes between one and two distinct challenge transformations. The combined head is initialized from the strong Laplacian head: CLIP and Laplacian weights are copied, new FFT input weights start at zero, and the head is fine-tuned at `1e-4`. This initialization was important; training the three-view head from scratch was worse.

Reproduce the selected model without scoring the test set during candidate training:

```bash
uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/ablation_laplacian_composed.pt \
  --cache artifacts/ablation_laplacian_composed_features.pt \
  --forensic-mode laplacian \
  --augmentation-repeats 3 --augmentation-depth 2 \
  --modality-dropout 0.1 \
  --validation-fraction 0.15 --test-fraction 0.15 --seed 42 \
  --feature-batch-size 16 --head-batch-size 64 \
  --epochs 40 --patience 7 --device cpu

uv run aigc-train \
  --data-dir data/mixed_5k \
  --output artifacts/ablation_laplacian_fft_init_lr1e4.pt \
  --cache artifacts/ablation_laplacian_fft_composed_features.pt \
  --forensic-mode laplacian_fft \
  --initialize-from-laplacian artifacts/ablation_laplacian_composed.pt \
  --augmentation-repeats 3 --augmentation-depth 2 \
  --learning-rate 1e-4 \
  --validation-fraction 0.15 --test-fraction 0.15 --seed 42 \
  --feature-batch-size 16 --head-batch-size 64 \
  --epochs 40 --patience 7 --device cpu

uv run aigc-evaluate \
  --data-dir data/mixed_5k \
  --checkpoint artifacts/ablation_laplacian_fft_init_lr1e4.pt \
  --output artifacts/improved_laplacian_fft_test_robustness.json \
  --split test --validation-fraction 0.15 --test-fraction 0.15 \
  --seed 42 --batch-size 16 --device cpu
```

The combined feature extraction and head training took 20m57s on the tested CPU; the final seven-condition test evaluation took 8m47s. Compared with the original checkpoint, mean accuracy across clean/JPEG/blur/resize/noise/color/crop rose from 92.57% to 93.62%, and mean ROC-AUC from 97.84% to 98.33%.

| Test condition | Original accuracy | Improved accuracy | Improved ROC-AUC |
|---|---:|---:|---:|
| Clean | 95.74% | 95.88% | 99.20% |
| JPEG | 94.41% | 94.28% | 98.80% |
| Blur | 89.63% | 93.09% | 97.96% |
| Resize/upscale | 89.49% | 92.29% | 97.34% |
| Noise | 92.02% | 92.42% | 97.45% |
| Color jitter | 94.81% | 94.95% | 98.86% |
| Crop | 91.89% | 92.42% | 98.69% |

These remain same-source held-out results. The next scientifically important test is a generator- or source-held-out dataset.

## Required JSON inference

```bash
uv run aigc-predict path/to/images \
  --checkpoint artifacts/ablation_laplacian_fft_init_lr1e4.pt \
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

The evaluator recreates the source-stratified 70/15/15 split and defaults to only
the untouched test originals. Use `--split validation` while comparing candidates,
then `--split test` once for the selected model. Repeat `--checkpoint` to compare
multiple checkpoints with the same encoder configuration while extracting each
transformed feature only once. Its JSON contains aggregate and per-source metrics
for every exact severity, plus a robustness summary. Keep the split fractions and
seed identical to training. Earlier seven-condition results in this README used one
sampled severity per transformation family and should not be compared directly with
the new full-severity report.
The model automatically discovers weights in this workspace's `.hf-cache` and
`.torch-cache`, so `HF_HUB_OFFLINE=1` works after the initial download. On a new
machine, omit the offline flag for the first run so the pretrained weights can be
downloaded.

For a final report, use a held-out dataset/generator and report clean plus individual JPEG, blur, resizing, noise, color, and crop settings. The smoke subset only proves that the software path works; its metrics are not scientifically meaningful.

For a completely separate source or generator dataset, pass `--split all`; this
evaluates every labeled image without recreating an internal train/validation/test
split.

## Kaggle recommendations

- Enable a GPU. The encoders automatically use CUDA; only the fusion head is optimized.
- Use generator/source-separated training and validation data. Never split transformed copies of one original across splits.
- Prefer SID_Set or a mixture of properly licensed sources at realistic resolution. CIFAKE is only 32×32 and has severe source shortcuts, so it is unsuitable as the sole final dataset.
- Cache 4-8 feature variants per training image using `--augmentation-repeats`; the first is clean and the rest receive one or more challenge-style transforms. Start with `--augmentation-depth 2` rather than stacking every transform at once.
- Compare CLIP-only, forensic-only, and fused features before claiming a hybrid gain.

## Limitations and future work

Frozen ImageNet EfficientNet was not pretrained specifically on Laplacian or FFT inputs, and high-frequency evidence remains vulnerable to redistribution. CLIP can learn content or dataset bias. Temperature calibration cannot repair domain shift. A stronger version should fine-tune the last EfficientNet block on a diverse high-resolution corpus, measure per-generator generalization, tune thresholds for moderation costs, and include representative false-positive/false-negative analysis.
