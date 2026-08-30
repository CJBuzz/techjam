# Integrated changes from `main` and Track 5 experiments

> This repository does not have a `master` branch. This report began as a comparison with the earlier `main` baseline at `f043567`; the current integration also includes the 100K/Kaggle scaling work from `main` at `bbf3667` and the later `track5-experiments` work.

## Summary

The detector was changed from a basic frozen two-stream classifier into a robustness-first, multi-view AIGC detector tailored to TechJam Problem 5. The implementation now reproduces every official corruption severity, trains against balanced clean/transformed views, selects checkpoints using robust validation, supports an additional FFT forensic stream and an experimental three-expert ensemble, produces richer robustness/error reports, and runs natively on Apple Silicon through MPS.

The result combines the robustness-first detector, controlled Track 5 experiments, cross-generator evaluation, and the duplicate-aware large-scale data pipeline in one tested branch.

## Latest branch integration

- Kept the Track 5 exact corruption names, balanced augmentation, robust checkpoint selection, TTA, streamed diverse-data caches, and controlled phase-two experiments.
- Added `main`'s persisted duplicate-aware train/model-selection/calibration/test manifests, 100K preparation and Kaggle handoff tools, exact-severity reports, calibration reports, ECE, operational thresholds, and shortcut/leakage audit.
- Unified both severity implementations behind one canonical 16-condition matrix so training and analysis cannot apply subtly different color or corruption definitions.
- Kept calibration independent from model-selection data when a four-way split manifest is supplied, while retaining the existing robust-validation calibration fallback for legacy three-way runs.
- Extended the new analysis commands to Apple MPS and made severity/audit decisions use the checkpoint's calibrated threshold.
- Verified the merged result with all 75 unit tests, Python compilation, shell-script syntax checks, and Git whitespace checks.

## Robustness improvements

- Added exact deterministic conditions for all official transformations:
  - JPEG quality 90, 70, 50, and 30;
  - Gaussian blur sigma 0.5, 1.0, and 2.0;
  - resize to 0.5 and 0.25 scale followed by upscaling;
  - Gaussian noise sigma 0.02, 0.05, and 0.10;
  - color factors 0.8 and 1.2;
  - centered 80% crop followed by resizing.
- Added balanced augmentation scheduling so hard severities are not missed through random sampling.
- Added deterministic composed transforms such as resize plus JPEG and noise plus JPEG.
- Kept clean and transformed copies grouped during training.
- Added prediction-consistency loss between views of the same original image.
- Added a worst-transformation-group loss to prevent strong easy-condition performance from hiding a weak corruption family.
- Added robust checkpoint selection using clean validation loss together with mean and worst hard-condition validation loss.
- Added calibration across clean and robust validation views.
- Added configurable threshold selection for balanced accuracy or F1 instead of assuming a fixed threshold of 0.5.

## Model and creativity improvements

- Added a centered log-magnitude FFT forensic representation with mean removal, a Hann window, and robust percentile scaling.
- Added `laplacian_fft` mode, combining:
  - frozen CLIP ViT-B/32 semantic features;
  - frozen EfficientNet-B0 Laplacian features;
  - frozen EfficientNet-B0 FFT features.
- Added initialization of the fused Laplacian+FFT head from a trained Laplacian checkpoint. Existing semantic/Laplacian weights are copied and new FFT input weights begin at zero.
- Added modality and FFT dropout controls to reduce dependence on a single evidence stream.
- Added an experimental adaptive three-expert head containing:
  - a CLIP-only semantic expert;
  - a CLIP plus Laplacian expert;
  - a CLIP plus FFT expert;
  - a learned softmax gate optionally conditioned on image-quality statistics.
- Added gate priors and robust-validation-aware training for the expert mixture.

## Evaluation and reporting improvements

- Added `full` and `worst` robustness profiles.
- Full evaluation now measures every official severity independently.
- Added per-condition and per-source metrics, including:
  - accuracy, precision, recall, F1, and specificity;
  - false-positive and false-negative rates;
  - ROC-AUC, average precision, and Brier score;
  - confusion matrices.
- Added summary fields for clean accuracy, mean transformed accuracy, worst transformed accuracy, worst condition, and mean drop from clean.
- Added ranked error analysis for high-confidence false positives and false negatives.
- Added support for evaluating multiple compatible checkpoints from one shared feature extraction pass.
- Preserved the required prediction JSON format with `image_path` and calibrated `pred` fields.

## Reproducibility and safety improvements

- Added feature-cache manifests containing dataset fingerprints, split settings, encoder configuration, augmentation settings, and seed.
- Stale or incompatible caches are now rejected instead of silently reused.
- Mixture training validates that its cache and initialization checkpoints have compatible configurations.
- Candidate selection defaults to validation data; test scoring remains an explicit final step.
- Expanded unit tests to cover every exact transformation and the three-expert gate.

## Apple Silicon support

- Added `mps` as an explicit device choice for training, mixture training, evaluation, and prediction.
- Automatic device selection now prefers CUDA, then Apple MPS, then CPU.
- Added clear errors when CUDA or MPS is explicitly requested but unavailable.

## M4 Pro training performed

The following local setup and validation work was completed:

- installed the locked Python environment with `uv`;
- downloaded pretrained CLIP ViT-B/32 and EfficientNet-B0 weights;
- downloaded a balanced 5,000-image dataset with 1,250 examples for each class/source combination from CIFAKE and SID_Set;
- verified that all 5,000 images decode successfully;
- ran a 32-image end-to-end MPS smoke training;
- trained the Laplacian baseline, Laplacian+FFT model, and adaptive three-expert model;
- evaluated the two compatible final candidates over every official severity;
- ran all 10 unit tests successfully.

Large datasets, feature caches, and model checkpoints are intentionally ignored by Git and remain local under `data/` and `artifacts/`.

## Validation results

Validation used 752 held-out images from the balanced mixed dataset. These are same-source held-out results and should not be interpreted as proof of unseen-generator generalization.

| Metric | Laplacian + FFT | Three-expert ensemble |
|---|---:|---:|
| Clean accuracy | 97.34% | 97.34% |
| Mean transformed accuracy | **94.80%** | 94.56% |
| Worst transformed accuracy | **90.82%** | 90.43% |
| Worst condition | Noise sigma 0.10 | Noise sigma 0.10 |

The Laplacian+FFT checkpoint is the recommended candidate because it has the stronger mean and worst-case robustness. The three-expert model improved JPEG quality 90 by 0.40 percentage points and color factor 1.2 by 0.13 points, but slightly reduced performance on most other conditions.

### Recommended model by official condition

| Condition | Laplacian + FFT accuracy |
|---|---:|
| Clean | 97.34% |
| JPEG Q90 | 97.07% |
| JPEG Q70 | 96.41% |
| JPEG Q50 | 96.94% |
| JPEG Q30 | 94.55% |
| Blur sigma 0.5 | 96.81% |
| Blur sigma 1.0 | 94.55% |
| Blur sigma 2.0 | 91.89% |
| Resize 0.5 | 94.55% |
| Resize 0.25 | 91.62% |
| Noise sigma 0.02 | 95.74% |
| Noise sigma 0.05 | 94.02% |
| Noise sigma 0.10 | 90.82% |
| Color 0.8 | 96.68% |
| Color 1.2 | 95.61% |
| Center crop 0.8 | 94.68% |

## Files changed relative to `main`

- `README.md`: robustness-first commands, ensemble workflow, evaluation instructions, results, and limitations.
- `aigc_detector/data.py`: exact challenge transformations and balanced deterministic schedules.
- `aigc_detector/features.py`: condition-specific feature and quality-statistic extraction.
- `aigc_detector/model.py`: FFT representation and adaptive three-expert architecture.
- `aigc_detector/train.py`: robust losses, robust validation, calibration, thresholds, manifests, and MPS support.
- `aigc_detector/train_mixture.py`: three-expert training, robust selection, cache validation, and MPS support.
- `aigc_detector/evaluate.py`: complete severity matrix, summaries, multi-checkpoint evaluation, error analysis, and MPS support.
- `aigc_detector/metrics.py`: expanded classification metrics and threshold selection.
- `aigc_detector/predict.py`: MPS device option.
- `tests/test_core.py`: exact-transform and three-expert tests.

## Commits included in the comparison

- `1140fb0` — robustness, FFT, ensemble, evaluation, cache, metric, documentation, and test improvements.
- `3b305ec` — Apple MPS support for training, evaluation, mixture training, and prediction.

## Subsequent cross-generator evaluation support

After external review identified unseen-generator generalization as the largest remaining risk, the evaluation workflow was extended without changing or retraining the architecture:

- added safe preparation of the B-Free RAISE/FLUX/SD3.5 archives with optional checksum verification and no image re-encoding;
- added a paired-generator protocol that evaluates each fake generator against the shared real set;
- added macro and worst-generator balanced accuracy so B-Free's 1:2 real/fake ratio does not inflate the headline score;
- kept checkpoint calibration and thresholds frozen during external evaluation;
- added a runner for the untouched local test followed by B-Free external evaluation;
- documented the scientific protocol, data-license restriction, and Stage 3 decision rule;
- expanded the test suite from 10 to 14 tests.
