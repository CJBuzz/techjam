# Robust AIGC Detector

A lightweight multi-view detector for TikTok TechJam Track 5. The strongest version fuses a frozen CLIP ViT-B/32 semantic embedding with frozen EfficientNet-B0 features from both a reproducible Laplacian high-pass image and a centered log-magnitude FFT. Only a small MLP is trained. Disjoint model-selection and calibration splits are used for early stopping and post-hoc temperature calibration; the test split is reserved for one final evaluation.

The model stays far below the 2-billion-parameter limit and feature caching makes the initial frozen-encoder stage practical on modest hardware.

## Cross-generator evaluation

The current mixed-5K results measure same-source held-out performance. They do not establish generalization to unseen image generators. The next experiment freezes the selected checkpoint and evaluates the untouched local test split followed by the B-Free FLUX/Stable Diffusion 3.5 external benchmark, without retraining or external threshold tuning.

See [the cross-generator generalization protocol](docs/GENERALIZATION_PROTOCOL.md) for the verified research rationale, license/download notes, safe dataset preparation, paired-generator metrics, and the one-command Stage 1–2 evaluation workflow.

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

A compatible two-checkpoint ensemble can be selected on named external
development datasets with `aigc-select-ensemble` and served with
`aigc-predict-ensemble`. See `docs/ENSEMBLE_SELECTION.md`. Development datasets
used to choose its weights or threshold must not subsequently be reported as
untouched final tests.

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

### Local RTX 4090 workflow (Windows)

The local project lock deliberately selects CPU-only PyTorch for portability. Do not run
`uv sync` for a CUDA environment. On a Windows machine with an RTX 4090, install `uv`, then use
the dedicated bootstrap script to create an isolated CUDA environment:

```powershell
winget install --id=astral-sh.uv -e
PowerShell -ExecutionPolicy Bypass -File scripts\setup_4090.ps1
PowerShell -ExecutionPolicy Bypass -File scripts\run_scale_4090.ps1 -Stage Preflight
```

The staged runner defaults to the recommended 100K corpus. Preparation downloads 25K images for
each SID/CIFAKE class-source cell, excludes SID label 2, deduplicates before assigning duplicate-
atomic splits, stores lossless PNGs to avoid adding a common JPEG signature, and writes a
reproducible manifest. Its bounded streaming shuffle prevents high-resolution SID rows from
exhausting RAM, while per-source preparation journals allow an interrupted download to resume.
Run each long stage separately so failures are
easy to resume and inspect:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts\run_scale_4090.ps1 -Scale 100k -Stage Prepare
PowerShell -ExecutionPolicy Bypass -File scripts\run_scale_4090.ps1 -Scale 100k -Stage Extract
PowerShell -ExecutionPolicy Bypass -File scripts\run_scale_4090.ps1 -Scale 100k -Stage Train
PowerShell -ExecutionPolicy Bypass -File scripts\run_scale_4090.ps1 -Scale 100k -Stage Analyze
```

Use `-FeatureBatchSize 16` if another application occupies substantial VRAM. A clean 24 GiB 4090
should normally start at 32. The head is small and normally supports `-HeadBatchSize 256`.
For a deadline-oriented run, `-Scale 40k` uses 10K images per class-source cell. After the 100K
run completes comfortably, repeat with `-Scale 200k`; it uses 50K images per class-source cell.
`-Stage All` is available, but separate stages are recommended for the first run.

Extraction includes train, model-selection, calibration, and robust model-selection features but
deliberately excludes reserved-test features. Training and analysis therefore cannot accidentally
inspect the reserved test set. Do not run the final test or B-Free evaluation until the checkpoint,
calibration choice, threshold, and augmentation policy are locked.

### Audited 100K GPU handoff

The reproducible route is to upload the already prepared `data/mixed_100k` directory once as a
private Kaggle Dataset. This preserves the exact duplicate groups and the train/model-selection/
calibration/reserved-test assignments instead of generating a different random sample in the
notebook.

On this machine, authenticate with a token from Kaggle's API settings, validate the corpus, and
upload it (replace the handle with your Kaggle username and a new dataset slug):

```bash
uv run --with kagglehub python scripts/kaggle_dataset.py upload \
  --handle YOUR_USERNAME/aigc-mixed-100k \
  --data-dir data/mixed_100k \
  --dry-run
uv run --with kagglehub python scripts/kaggle_dataset.py upload \
  --handle YOUR_USERNAME/aigc-mixed-100k \
  --data-dir data/mixed_100k \
  --login
```

`--with kagglehub` installs KaggleHub only for that `uv run`; it does not add it to this project's
dependencies. `--login` is intentionally part of the upload command: KaggleHub's interactive login
is process-local, so logging in with one `uv run` and uploading with another loses the credential.
The uploader calls `whoami()` and verifies that `YOUR_USERNAME` owns the requested handle before it
starts transferring files. As a persistent alternative, place a generated API token in
`~/.kaggle/access_token` or set `KAGGLE_API_TOKEN`, then omit `--login`.

Upload the repository itself as a second, small private Kaggle Dataset. The helper excludes data,
artifacts, model caches, `.venv`, Git history, logs, and common credential files:

```bash
uv run --with kagglehub python scripts/kaggle_repo.py \
  --handle YOUR_USERNAME/techjam-source \
  --dry-run
uv run --with kagglehub python scripts/kaggle_repo.py \
  --handle YOUR_USERNAME/techjam-source \
  --login
```

Verify in Kaggle that **both** Datasets are Private. In a new Kaggle notebook, select a GPU, turn
Internet on for the initial model download, and add `techjam-source` and `aigc-mixed-100k` through
the notebook's **Add Input** panel. Kaggle inputs are read-only, so copy only the small source
Dataset to the writable working directory:

```bash
find /kaggle/input -maxdepth 4 -type f \
  \( -name pyproject.toml -o -name split_manifest.csv \) -print

SOURCE_FILE=$(find /kaggle/input -maxdepth 4 -type f -name pyproject.toml -print -quit)
SOURCE_ROOT=$(dirname "$SOURCE_FILE")
mkdir -p /kaggle/working/techjam
cp -a "$SOURCE_ROOT"/. /kaggle/working/techjam/
cd /kaggle/working/techjam
python -m pip install -q -e . --no-deps

DATA_FILE=$(find /kaggle/input -maxdepth 4 -type f -name split_manifest.csv -print -quit)
DATA_ROOT=$(dirname "$DATA_FILE")
python scripts/kaggle_dataset.py validate --data-dir "$DATA_ROOT"
```

The Kaggle image already includes CUDA-enabled PyTorch and the common scientific dependencies.
Using `pip --no-deps` here is intentional: this repository's local `uv` configuration pins the CPU
PyTorch index, so running `uv sync` in Kaggle could replace its CUDA build. If an import is missing,
install only that package rather than reinstalling `torch` or `torchvision`.

Kaggle P100 sessions may currently start with a CUDA 12.8 PyTorch wheel that omits Pascal `sm_60`
kernels. If `torch.cuda.get_arch_list()` does not include `sm_60`, install the official CUDA 12.6
build before importing the detector, then verify a real CUDA operation:

```bash
python -m pip install -q --no-cache-dir --force-reinstall \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -c 'import torch; x=torch.ones(1, device="cuda"); print(torch.__version__, torch.cuda.get_arch_list(), x)'
```

The last line printed is `DATA_ROOT=...`. Use that exact path without copying the images into
`/kaggle/working`, for example:

```bash
bash scripts/kaggle_run_100k.sh \
  "$DATA_ROOT" \
  /kaggle/working/aigc_100k
```

If KaggleHub returns a versioned cache path instead of `/kaggle/input/aigc-mixed-100k`, pass the
printed `DATA_ROOT` value. `FEATURE_BATCH_SIZE=16` can be used if 32 exhausts GPU memory. The runner
extracts three balanced training views plus clean model-selection and calibration features, trains
the Laplacian initializer, then trains the paired-consistency Laplacian+FFT head. It deliberately
does not extract or score the reserved test split. Save a notebook version when finished so the
feature caches and checkpoints under `/kaggle/working/aigc_100k` become reusable notebook output.

After the baseline runner completes, run the prepared analyses sequentially on the same GPU:

```bash
bash scripts/kaggle_analyze_100k.sh \
  "$DATA_ROOT" \
  /kaggle/working/aigc_100k
```

This performs three model-selection-safe tasks without reading reserved test images:

- fits both clean-only and deterministic mixed-condition temperatures on the separate 4,987-image
  calibration split, preselects mixed calibration, and records high-recall/high-precision
  thresholds;
- verifies exact hashes and near-duplicate groups remain split-atomic, trains linear source probes
  for CLIP/Laplacian/FFT/fused features, reports detector metrics by source and native resolution,
  and records representative false positives/negatives with nearest-training-image dHash distance;
- evaluates all 16 explicit challenge cells: clean, four JPEG qualities, three blur sigmas, two
  resize scales, three noise sigmas, two color magnitudes, and the 80% center crop.

The severity output is saved after every cell and `--resume` skips completed cells, so a Kaggle
session can safely be continued. The full severity matrix processes 159,600 model-selection views;
the mixed calibration pass adds 4,987 transformed views. This is inference-only but is still a
substantial GPU job. The shortcut audit reuses the baseline feature cache and is comparatively
cheap. Preserve these additional output artifacts:

```text
mixed_100k_balanced_consistency_w01_mixed_calibrated.pt
mixed_100k_calibration.json
mixed_100k_shortcut_audit.json
mixed_100k_model_selection_severity.json
```

Do not invoke `aigc-severity --split test --allow-test` until preprocessing, calibration, thresholds,
and every candidate comparison have been locked.

WildFake is not substituted automatically. Its official release is on ModelScope, not as a
verified first-party Kaggle Dataset, and it is distributed in multi-gigabyte generator archives.
The smallest useful multi-generator experiment also needs a new generator-held-out manifest;
mixing an unofficial mirror into this fixed split would weaken both licensing and leakage checks.
Use the audited CIFAKE+SID 100K run first. Treat an official WildFake import as a separate
experiment, preserve its official hierarchy CSVs, and never use the challenge's demonstration-only
COCO/DALL-E subset for training.

- Enable a GPU. The encoders automatically use CUDA; only the fusion head is optimized.
- Use generator/source-separated training and validation data. Never split transformed copies of one original across splits.
- Prefer SID_Set or a mixture of properly licensed sources at realistic resolution. CIFAKE is only 32×32 and has severe source shortcuts, so it is unsuitable as the sole final dataset.
- Cache 4-8 feature variants per training image using `--augmentation-repeats`; the first is clean and the rest receive one or more challenge-style transforms. Start with `--augmentation-depth 2` rather than stacking every transform at once.
- Compare CLIP-only, forensic-only, and fused features before claiming a hybrid gain.

## Limitations and future work

Frozen ImageNet EfficientNet was not pretrained specifically on Laplacian or FFT inputs, and high-frequency evidence remains vulnerable to redistribution. CLIP can learn content or dataset bias. Temperature calibration cannot repair domain shift. A stronger version should fine-tune the last EfficientNet block on a diverse high-resolution corpus, measure per-generator generalization, tune thresholds for moderation costs, and include representative false-positive/false-negative analysis.
