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

## Consolidated later investigations

### GenImage GLIDE and ensemble selection

GLIDE was used as external development data, so it is not an untouched final
test. The locked 40K and 100K checkpoints reached clean ROC-AUC 68.77% and
69.96%; fake recall was only 14.15% and 8.83%. A compatible 15%/85% 40K/100K
probability blend selected on GLIDE increased ROC-AUC to 70.97% under a 5% real
false-positive ceiling, but fake recall remained 19.93%. The ensemble was
therefore provisional and was not selected for submission.

### Generator-diverse WildFake adaptation

The diverse experiment contained 65,982 usable originals: 47,994 train, 9,988
model selection, 2,000 calibration, and 6,000 reserved test. It retained legacy
40K assignments, added ImageNet reals plus DDIM, DDPM, BigGAN, and StyleGAN
fakes to training, and held ADM out for model selection. DALL-E Advanced and
COCO val2017 demonstration images were excluded; decoded-pixel and dHash audits
rejected cross-split conflicts.

The 40K-initialized candidate beat the 100K-initialized candidate on model
selection (ROC-AUC 0.8480 versus 0.8407) and became the selected checkpoint. On
external GLIDE it reached clean ROC-AUC 0.9078 and mean severe ROC-AUC 0.8252.
Mixed-condition calibration used only the separate 2,000-image split and chose
temperature 0.6637356. The exact training manifest is absent from this checkout,
so its reported hash and demonstration-set exclusion still require provenance
recovery before submission.

### SD1.5 VAE residual FFT study

The independent `research/sd15_vae_fft/` study compared raw-image spectra with
FFT energy in Stable Diffusion 1.5 VAE reconstruction residuals on 11,841 COCO
real/DALL-E 3 fake examples. VAE residual AUROC was 0.9994 clean, 1.0000 at JPEG
Q30, 0.9311 under blur 2, and 0.9951 after 0.25× resizing. Raw FFT fell to
0.4347 and 0.4913 for blur and resize. Fixed-clean-threshold accuracy still
collapsed after smoothing because scores shifted downward; this is a
calibration warning rather than a ranking failure. This training-free study is
not part of the submitted detector; its protocol and artifact index remain in
`research/sd15_vae_fft/README.md`.

## Deferred work

- TextureCrop remains a validation-only preprocessing comparison for native
  images at least 224×224, with global resize retained for smaller inputs.
- A generator/source-disjoint official WildFake study remains the most useful
  next training experiment. Preserve official hierarchy and licensing, keep
  duplicate groups atomic, and separate selection, calibration, and test.
- Selective final-block EfficientNet fine-tuning should follow only if a frozen
  generator-held-out baseline justifies its cost; initially keep CLIP frozen.

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

This file supersedes the former status/UI guides, experiment roadmap, one-off
evaluation reports, branch note, and individual ensemble, GLIDE, WildFake, and
mixed-40K documents. Durable commands and detailed results remain in
`AGENTS.md` and machine-readable artifacts; the README carries only the
selected submission story and commands.
