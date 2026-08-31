# GenImage GLIDE external evaluation

## Decision summary

The mixed-40K and mixed-100K checkpoints were evaluated on the official
GenImage GLIDE validation split: 6,000 GLIDE-generated images and 6,000 real
ImageNet images. Neither GLIDE nor this validation set was used to train either
checkpoint.

The 100K checkpoint ranks GLIDE images slightly better, but both checkpoints
are overly conservative at their frozen decision thresholds and miss most
GLIDE fakes. This is evidence of a substantial unseen-generator gap, not a
successful deployment result.

| Metric | 40K checkpoint | 100K checkpoint |
|---|---:|---:|
| Clean ROC-AUC | 68.77% | **69.96%** |
| Clean balanced accuracy | **55.37%** | 53.52% |
| GLIDE fake recall | **14.15%** | 8.83% |
| Real false-positive rate | 3.42% | **1.80%** |
| Mean transformed ROC-AUC | 67.90% | **69.30%** |
| Worst transformed ROC-AUC | 54.76% | **55.14%** |
| Mean transformed balanced accuracy | **52.87%** | 52.39% |
| Worst transformed balanced accuracy | 49.00% | **49.47%** |

The worst balanced-accuracy condition for both models is Gaussian noise with
sigma 0.05. The top-ranked errors are overwhelmingly false negatives: GLIDE
images assigned very low fake probabilities. This agrees with the low fake
recall and shows that changing only the threshold could trade additional fake
recall for more false positives. No threshold was retuned on GLIDE for this
report.

## Interpretation

ROC-AUC is the fairest ranking comparison here because the checkpoints preserve
different frozen threshold policies. The 40K checkpoint uses its calibrated
threshold of 0.36389595; the downloaded 100K checkpoint does not contain a
selected threshold and therefore uses 0.5. On clean GLIDE, the 100K model gains
1.19 percentage points of AUC, but its stricter effective decision policy gives
5.32 points less fake recall. Neither result is strong enough to claim robust
cross-generator detection.

Because both checkpoints have now been compared on GLIDE, this set is a
model-selection benchmark for this project. It must not subsequently be
described as an untouched final test. A separate unseen-generator set is still
needed for final evidence after choosing and freezing a checkpoint and decision
policy.

## Protocol and provenance

- Dataset: official GenImage GLIDE validation split.
- Images: 12,000 total, balanced by class.
- Fake generator: GLIDE.
- Real images: the paired ImageNet validation images distributed with GLIDE.
- Conditions: clean plus JPEG, blur, resizing, Gaussian noise, color, and crop
  transformations from the repository's full evaluation profile.
- Evaluation seed: 42.
- Batch size: 32 on an NVIDIA GeForce RTX 4090.
- Checkpoints:
  - `artifacts/mixed_40k/balanced_consistency_w01_calibrated.pt`
  - `artifacts/mixed_100k_balanced_consistency_w01_mixed_calibrated/mixed_100k_balanced_consistency_w01_mixed_calibrated.pt`
- Full metrics: `artifacts/genimage_glide_40k_100k_full.json`.
- Top-error audit: `artifacts/genimage_glide_40k_100k_errors.json`.

The source archives, extracted images, checkpoints, and result JSON files are
local ignored artifacts. The extraction utility verifies ZIP CRC and decoded
all 12,000 extracted images successfully.

## Internal-overlap audit

The mixed-100K manifest confirms that all 40,000 mixed-40K originals also occur
in the 100K corpus by exact content SHA-256. More importantly, 2,838 images from
the 40K model-selection split and 4,190 images from the 40K reserved-test split
were assigned to training in the 100K experiment. Therefore the earlier
100K-on-40K diagnostic is training-contaminated and cannot compare the two
models fairly. The GLIDE result above is the first common external comparison.
