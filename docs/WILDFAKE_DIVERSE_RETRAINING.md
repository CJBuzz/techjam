# WildFake-diverse retraining result

## Outcome

The 40K-initialized, generator-diverse candidate is the strongest cross-generator model tested in this repository so far. It should be treated as the primary candidate for another untouched validation run, while the locked 40K and 100K checkpoints remain preserved as fallbacks.

Candidate checkpoint:

`artifacts/mixed_wildfake_66k/diverse_initialized_40k_calibrated.pt`

- SHA-256: `f219384c54ed2f57fdb9cdc3cdd92dbb6c7038a14c6d4ac467865cf08587d02f`
- Size: 3,296,775 bytes
- Inference parameter count: 92,676,989 total (87,849,216 CLIP vision, 4,007,548 EfficientNet, 820,225 fusion head)
- The total is about 0.093B parameters, comfortably below the competition's 2B-parameter limit.

## Dataset and leakage controls

The combined corpus has 65,982 usable images. The persisted split counts are:

| Split | Real | Synthetic | Total |
|---|---:|---:|---:|
| Train | 23,997 | 23,997 | 47,994 |
| Model selection | 4,994 | 4,994 | 9,988 |
| Calibration | 1,000 | 1,000 | 2,000 |
| Reserved test | 3,000 | 3,000 | 6,000 |

Training contains the original mixed-40K CIFAKE/SID training rows plus:

- 9,997 ImageNet real images;
- 2,999 DDIM fake images;
- 2,998 DDPM fake images;
- 2,000 BigGAN fake images;
- 2,000 StyleGAN fake images.

The new generator-held-out model-selection pair contains 2,994 ADM fake images and 2,994 ImageNet real images. ADM was not used for training. All new rows came from the official WildFake training metadata; WildFake official test rows were excluded.

The TechJam demonstration categories DALL-E Advanced and COCO val2017 were explicitly excluded from both training and model selection. They remain eligible for an untouched final validation and must not be trained on.

Exact and perceptual duplicate auditing rejected 14 new cross-split conflicts at dHash radius 4. Four additional rows were removed to preserve exact class balance. The locked mixed-40K split assignments and duplicate groups were retained. The reserved 6,000-image test split was not feature-extracted or scored during this experiment.

Audit artifacts:

- `data/mixed_wildfake_66k/audit.json`
- `data/mixed_wildfake_66k/split_manifest.csv`
- split-manifest SHA-256: `6214c16fc37b7652d2b8a5a059506b7257b3bb6eaae005bbe649c3c40c6bf028`

## Training

The frozen encoder is CLIP ViT-B/32 plus EfficientNet-B0 Laplacian and FFT views. Only the 820K-parameter fusion head was optimized. Training used three balanced paired views, consistency weight 0.1, modality dropout 0.1, FFT dropout 0.15, learning rate `1e-4`, batch size 256, seed 42, and robust-validation checkpoint selection. Both 40K and 100K initializations early-stopped after epoch 10.

The 40K initialization won model selection:

| Initialization | Overall AUC | Overall AP | ADM vs ImageNet AUC | ADM vs ImageNet AP | Legacy AUC |
|---|---:|---:|---:|---:|---:|
| Locked 40K (no diverse retraining) | 0.7247 | 0.7923 | 0.5921 | 0.5876 | 0.9989 |
| Locked 100K (no diverse retraining) | 0.6932 | 0.7759 | 0.5248 | 0.5518 | 0.9997 |
| Diverse, initialized from 40K | **0.8480** | **0.8791** | **0.7402** | **0.7588** | 0.9978 |
| Diverse, initialized from 100K | 0.8407 | 0.8754 | 0.7316 | 0.7443 | 0.9988 |

The tiny familiar-source AUC reduction is outweighed by the much larger held-out-generator improvement. Full source-level results are in `artifacts/mixed_wildfake_66k/model_selection_comparison.json` (SHA-256 `451f1682c5b93bc5fe55e6114fb7aaa7f4100e0a44351761fc106ab62e963aca`).

## Calibration

Mixed-condition temperature calibration used only the separate 2,000-image calibration split. The selected temperature is 0.6637356. On the 4,000 clean-plus-transformed calibration rows, ROC-AUC is 0.9935 and AP is 0.9940.

The checkpoint also contains a threshold for operational classification reports. The competition submission uses probabilities, so model selection should emphasize ranking metrics such as ROC-AUC/AP; the threshold does not change their ordering. Temperature calibration still matters because it changes the quality of submitted confidence values.

## External GLIDE comparison

GLIDE contains 6,000 real and 6,000 synthetic images and was not used for training or calibration.

| Model | Clean AUC | Clean AP | Clean BAcc | Mean severe AUC | Worst severe AUC | Mean severe BAcc | Worst severe BAcc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Locked 40K | 0.6877 | 0.6846 | 0.5537 | 0.7117 | 0.6122 | 0.5346 | 0.4941 |
| Locked 100K | 0.6996 | 0.6954 | 0.5352 | 0.7310 | 0.6062 | 0.5330 | 0.4969 |
| WildFake-diverse candidate | **0.9078** | **0.8978** | **0.7458** | **0.8252** | **0.7074** | **0.6759** | **0.5763** |

For the new model, the worst balanced-accuracy condition is JPEG quality 30. At the frozen calibrated threshold, clean GLIDE fake recall is 0.5332, precision is 0.9278, and real false-positive rate is 0.0415.

Detailed new-model results are in `artifacts/mixed_wildfake_66k/glide_worst/detailed_metrics.json` (SHA-256 `ce8a1a38d0fdf79c7570774a1ed89c5d7af9c0db16d9cac62b87eecd036346ad`).

## Recommendation and remaining check

Use `diverse_initialized_40k_calibrated.pt` as the primary candidate, but do not delete or overwrite the locked 40K/100K weights. Before final submission, evaluate this candidate once on the exact untouched TechJam DALL-E Advanced plus COCO val2017 validation set that was used for the earlier 40K/100K report. Those images are not currently present in this workspace, so that exact paired comparison could not be run here.

Do not tune, calibrate, or retrain after inspecting that final validation if it is intended to remain the final unbiased benchmark. Generate the required per-image probabilities from the selected checkpoint and preserve the checkpoint hash with the submission.
