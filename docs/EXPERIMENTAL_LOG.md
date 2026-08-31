# Experimental log and decision record

This is the human-readable summary of experiments that shaped the submitted
model. `AGENTS.md` remains the append-only audit record with exact commands,
runtimes, artifacts, failures, and historical metrics.

## Selected submission model

`artifacts/diverse_initialized_40k_calibrated.pt` is the only submission
default. It uses frozen CLIP ViT-B/32 semantics plus separate EfficientNet-B0
Laplacian and FFT forensic views, followed by an 820,225-parameter fusion MLP.

The head was initialized from the calibrated mixed-40K model and trained on
three balanced views per original with paired-logit consistency `0.1`, modality
dropout `0.1`, and FFT-block dropout `0.15`. It was fitted on 47,994 originals,
selected on 9,988 originals, and temperature-calibrated on a separate 2,000
original split. The reserved 6,000-image test split was not extracted during
development. Mixed-condition calibration selected temperature `0.6637356` and
balanced threshold `0.7572185`.

The locked checkpoint's external demonstration-only WildFake check used 500
COCO-val2017 real images and 500 DALL·E 3 Advanced synthetic images. At the
neutral 0.5 probability threshold it achieved 86.00% accuracy, 93.59% ROC-AUC,
94.73% average precision, and 86.19% F1. The data was not used for training,
selection, calibration, or tuning by the evaluation command. Because this is
only one real source and one generator, it is a stress check rather than broad
cross-generator evidence. The adapted checkpoint references a missing
`mixed_wildfake_66k` training manifest, so exact training provenance and
non-overlap with the supplied demonstration corpus remain unverified locally.

## Experiments that changed the design

### Frozen CLIP + Laplacian baseline

The first 5K mixed-source model established CPU feasibility and strong clean
performance, but blur and resize were the main weaknesses. This motivated a
frequency view and transformation-aware training while retaining frozen
backbones and reusable feature caches.

### FFT fusion and residual initialization

Adding FFT features from scratch improved blur but overfit noise. FFT-block
dropout alone did not fix this. Copying compatible CLIP+Laplacian head weights,
zero-initializing only the new FFT columns, and fine-tuning at `1e-4` improved
the seven-condition mean test accuracy from 92.57% to 93.62%, with the largest
gains on blur and resize. Residual initialization became the standard method.

### Transform-balanced paired consistency

Random robust augmentation was replaced by deterministic balanced assignment
over the official transformations and selected two-step compositions. Keeping
three views of each original in one minibatch enabled a logit-consistency loss.
Weight `0.1` improved mean validation accuracy and calibration; adding a
max-group loss did not help and was rejected.

### Quality-aware and three-expert mixtures

Fixed and learned mixtures of Laplacian and FFT experts did not beat the fused
head. Quality statistics moved in plausible directions for blur/resize, but the
gate encoded CIFAKE-versus-SID source identity more strongly than corruption.
The later semantic/Laplacian/FFT three-expert model won a few isolated severity
cells but lost on the overall robustness objective. Neither is in the submitted
inference path; their code is under `aigc_detector/experiments/`.

### 40K scaling

The duplicate-atomic mixed-40K run used 28K train, 4K model-selection, 2K
calibration, and 6K reserved-test originals. On its model-selection matrix it
reached 98.33% clean accuracy, 95.99% mean transformed accuracy, and 91.13%
worst-condition accuracy. CIFAKE and SID were both present in every split, so
these remain shared-source results. The source audit could identify CIFAKE
versus SID almost perfectly, largely because CIFAKE is 32×32 while SID images
are high resolution.

After the checkpoint and threshold were frozen, its 6,000-image reserved test
was opened once. It reached 98.62% clean accuracy, 96.13% mean accuracy across
the clean plus 15 severity cells, and 90.85% worst-condition accuracy at resize
0.25. This is held-out same-source evidence for the 40K checkpoint, not evidence
of generalization to unseen generators.

### Provisional 100K comparison

Restored 100K weights scored higher than the 40K model on the mixed-40K
model-selection matrix. The subsequently recovered manifest confirmed exact
training contamination: 2,838 of the 40K model-selection images and 4,190 of
its reserved-test images occur in the 100K training split. Those results prove
checkpoint compatibility, not held-out improvement, and the 100K checkpoint
was not made the submission default.

### Diverse adaptation and calibration

The selected model initialized from the locked 40K fusion head and adapted on
the larger diverse manifest. Mixed clean/transformed calibration was chosen
before external demonstration evaluation. The external COCO/DALL·E check showed
materially better balance than the shared-source 40K and 100K checkpoints,
supporting the choice for the submission demo while leaving the limitations
explicit.

## Supporting experiments and utilities

- `aigc_detector/experiments/e1b.py` — initialization/transfer studies.
- `e1c.py` and `e2b.py` — test-time and transformation diagnostics.
- `e4a.py`, `e4b.py`, and `e4c.py` — adaptive fusion and intervention studies.
- `e5.py` — quality-conditioned calibration experiments.
- `e6.py` — multi-scale consistency research.
- `e7.py` — radial FFT descriptor experiments.
- `aigc_detector/analysis/` — shortcut audits plus external and response-based
  evaluation tools that are not needed for ordinary prediction.
- `aigc_detector/tooling/` — streamed feature-cache support used by large-data
  preparation workflows.
- `scripts/kaggle/` — Kaggle dataset/repository upload, feature extraction,
  training, and analysis launchers.
- remaining `scripts/` — local data preparation, 4090 handoff, external
  evaluation, and historical compatibility launchers.

## Data and evaluation rules

- Exact decoded-pixel duplicates and near-duplicate groups stay within one
  persisted split.
- Augmentation occurs only after the original-level split.
- Model-selection labels may select checkpoints; calibration labels only fit
  temperature and operational thresholds; reserved-test labels are never used
  for either purpose.
- The supplied COCO/DALL·E WildFake demonstration set is external evaluation
  only and must never enter training, model-selection, or calibration.
- Shared-source results are never presented as unseen-generator evidence.

## Remaining limitations

- Frozen ImageNet EfficientNet was not pretrained on Laplacian or FFT images.
- Dataset source, resolution, and generator identity can be confounded.
- Calibration does not repair domain shift.
- A stronger final study needs properly licensed, generator-disjoint training,
  validation, and test sets, plus nested selection for any learned gate.
- False positives and false negatives should be reviewed by source, resolution,
  and transformation severity before any moderation deployment.

## Consolidated historical documents

This file supersedes the former top-level `EVALUATION_REPORT.md`,
`CHANGES_FROM_MAIN.md`, `docs/MIXED_40K_BENCHMARK.md`, and
`docs/GENERALIZATION_PROTOCOL.md`. Their durable commands and detailed results
remain recorded in `AGENTS.md`; the README carries only the selected submission
story and commands.
