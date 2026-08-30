# Mixed 40K RTX 4090 benchmark

## Outcome

The mixed-40K Laplacian + FFT run is the strongest recorded same-source
robustness result in this repository. Compared with the previous mixed-5K
Laplacian + FFT benchmark, it improves clean accuracy, mean transformed
accuracy, and worst-condition accuracy.

This is not a strict apples-to-apples scaling result. The earlier benchmark
used 752 model-selection images from the mixed-5K experiment, while this run
uses a separately prepared, duplicate-atomic 4,000-image model-selection split
from the mixed-40K corpus. Both contain CIFAKE and SID_Set, so the comparison
measures progress on the same source families rather than unseen-generator
generalization.

| Summary metric | Previous mixed 5K | Mixed 40K | Change |
|---|---:|---:|---:|
| Clean accuracy | 97.34% | **98.33%** | **+0.99 pp** |
| Mean transformed accuracy (15 conditions) | 94.80% | **95.99%** | **+1.19 pp** |
| Worst transformed accuracy | 90.82% | **91.13%** | **+0.31 pp** |
| Worst condition | Noise sigma 0.10 | Resize 0.25 | -- |

The mixed-40K model improves 14 of the 15 transformed conditions. The one
regression is resize 0.25, which falls from 91.62% to 91.13% (-0.50 percentage
points).

## Exact condition comparison

| Condition | Previous mixed 5K | Mixed 40K | Change |
|---|---:|---:|---:|
| Clean | 97.34% | **98.33%** | +0.99 pp |
| JPEG Q90 | 97.07% | **98.23%** | +1.16 pp |
| JPEG Q70 | 96.41% | **98.05%** | +1.64 pp |
| JPEG Q50 | 96.94% | **97.25%** | +0.31 pp |
| JPEG Q30 | 94.55% | **96.00%** | +1.45 pp |
| Blur sigma 0.5 | 96.81% | **97.08%** | +0.27 pp |
| Blur sigma 1.0 | 94.55% | **96.65%** | +2.10 pp |
| Blur sigma 2.0 | 91.89% | **92.30%** | +0.41 pp |
| Resize 0.5 | 94.55% | **94.88%** | +0.33 pp |
| Resize 0.25 | **91.62%** | 91.13% | -0.50 pp |
| Noise sigma 0.02 | 95.74% | **97.63%** | +1.89 pp |
| Noise sigma 0.05 | 94.02% | **96.98%** | +2.96 pp |
| Noise sigma 0.10 | 90.82% | **92.88%** | +2.06 pp |
| Color 0.8 | 96.68% | **97.58%** | +0.90 pp |
| Color 1.2 | 95.61% | **97.38%** | +1.77 pp |
| Center crop 0.8 | 94.68% | **95.83%** | +1.15 pp |

## Run configuration

- Hardware: NVIDIA GeForce RTX 4090 with 24 GiB VRAM.
- Corpus: 40,000 originals, balanced across CIFAKE real, CIFAKE synthetic,
  SID_Set real, and SID_Set fully synthetic.
- SID_Set label 2 (locally tampered) was excluded because this training target
  is fully generated images versus authentic images.
- Split sizes: 28,000 train, 4,000 model selection, 2,000 calibration, and
  6,000 reserved test originals.
- Duplicate handling: no cross-split near-duplicate groups, exact hashes, or
  repeated manifest paths were found by the shortcut audit.
- Encoders: frozen CLIP ViT-B/32 and frozen EfficientNet-B0 forensic streams.
- Learned component: a small Laplacian + FFT fusion head initialized from the
  Laplacian checkpoint.
- Training: three balanced cached training views per original, robust checkpoint
  selection, and early stopping. The Laplacian initializer stopped after 25
  epochs and the fused head after 28 epochs.
- Calibration: mixed clean/transformed calibration policy, temperature
  0.70715034, and selected threshold 0.36389595.
- Selected checkpoint:
  `artifacts/mixed_40k/balanced_consistency_w01_calibrated.pt`.

The full reports are stored locally in:

- `artifacts/mixed_40k/calibration.json`
- `artifacts/mixed_40k/shortcut_audit.json`
- `artifacts/mixed_40k/model_selection_severity.json`

Artifacts and datasets remain ignored by Git; this document records the
reproducible headline results.

## Important limitation

The audit can identify SID versus CIFAKE almost perfectly. CIFAKE is 32 x 32,
whereas the selected SID images are high resolution, so dataset source and
native resolution are strongly confounded. Both sources are internally balanced
between authentic and synthetic classes, which prevents source alone from being
the class label, but the high same-source scores still do not establish
generalization to unseen generators or realistic competition image
distributions.

The calibrated checkpoint, threshold, and preprocessing should now be frozen.
The next scientifically useful measurement is the untouched reserved test,
followed by the B-Free FLUX and Stable Diffusion 3.5 benchmark under the protocol
in `docs/GENERALIZATION_PROTOCOL.md`. Results from those sets must not be used to
retune this checkpoint if they are to remain final-test evidence.

## Provisional 100K checkpoint compatibility comparison

The unpacked Kaggle checkpoint
`artifacts/mixed_100k_balanced_consistency_w01_mixed_calibrated` was restored to
the loadable file
`artifacts/mixed_100k_balanced_consistency_w01_mixed_calibrated.pt`. It contains
the trained Laplacian + FFT fusion-head weights, temperature 0.77410257, and
embedded validation metadata. Because it does not store a selected balanced
threshold, evaluation preserves its default threshold of 0.5.

Both the 40K and 100K weights were run over the same 4,000-image mixed-40K
model-selection set and the same deterministic 16-condition matrix:

| Metric | 40K weights | 100K weights | Change |
|---|---:|---:|---:|
| Clean accuracy | 98.33% | **99.33%** | +1.00 pp |
| Mean transformed accuracy | 95.99% | **96.94%** | +0.95 pp |
| Worst transformed accuracy | 91.13% | **93.38%** | +2.25 pp |
| Worst condition | Resize 0.25 | Resize 0.25 | -- |

The 100K weights improve 14 of 15 transformed conditions. The small regressions
are color 0.8 (-0.08 percentage points) and color 1.2 (-0.23 percentage points).
The largest gains are resize 0.25 (+2.25 points), noise sigma 0.10 (+1.83
points), blur sigma 2.0 (+1.58 points), and resize 0.5 (+1.48 points). The full
diagnostic output is stored in
`artifacts/mixed_100k_on_mixed_40k_model_selection_severity.json`.

This comparison is **not an uncontaminated held-out benchmark**. The local copy
does not include the mixed-100K split manifest, so exact hashes and duplicate
groups cannot be checked against the 40K evaluation rows. Both corpora were
sampled from CIFAKE and SID_Set, and some or all of the 40K model-selection
images may have appeared in the 100K training data. The numbers establish that
the restored weights load correctly and are operationally stronger on this
shared-source diagnostic set; they must not be used as evidence of
cross-generator generalization. A fair model-selection comparison requires the
original mixed-100K manifest, and final generalization evidence still requires
the untouched B-Free evaluation.
