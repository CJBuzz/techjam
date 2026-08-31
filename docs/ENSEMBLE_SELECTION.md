# Cross-generator ensemble selection

## Current status

The 40K and 100K fusion heads use compatible frozen encoders, so they can be
scored from one CLIP + EfficientNet feature pass and blended without duplicating
the expensive encoder computation. The development selector searches weights
from 0% to 100% 40K in five-point increments and selects one global threshold.
Every development dataset contributes equally, irrespective of its image count,
and the requested real-image false-positive ceiling must hold separately on
every dataset.

The first run used the official GenImage GLIDE validation set as development
data and imposed a 5% real false-positive ceiling:

| Metric | Selected GLIDE-only ensemble |
|---|---:|
| 40K weight | 15% |
| 100K weight | 85% |
| Threshold | 0.08917413 |
| ROC-AUC | 70.97% |
| Balanced accuracy | 57.47% |
| GLIDE fake recall | 19.93% |
| Real specificity | 95.00% |
| Confusion matrix | TN 5,700; FP 300; FN 4,804; TP 1,196 |

This is a modest improvement over either frozen checkpoint on GLIDE, but fake
recall remains poor. The policy is provisional and GLIDE is model-selection
data. It is not evidence from an untouched final test.

The separately reported 1,000-image COCO-real/DALL-E-3-fake WildFake evaluation
cannot yet be included in the ensemble calculation because neither those images
nor their per-image 40K/100K scores are present in this workspace. Aggregate
accuracy, AUC, and confusion matrices are not sufficient to reconstruct a
probability ensemble.

## Add the WildFake development set

Place the exact audited images in this layout:

```text
data/wildfake_dalle3_external/
  real/coco/       # 500 images
  ai/dalle3/       # 500 images
```

Then run the selector below. It reuses the saved GLIDE per-image predictions,
extracts WildFake features once, and selects a joint policy without rescoring
GLIDE:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m aigc_detector.ensemble `
  --predictions-input artifacts\ensemble_glide_predictions.json `
  --dataset WildFake-DALLE3=data\wildfake_dalle3_external `
  --checkpoint-40k artifacts\mixed_40k\balanced_consistency_w01_calibrated.pt `
  --checkpoint-100k artifacts\mixed_100k_balanced_consistency_w01_mixed_calibrated\mixed_100k_balanced_consistency_w01_mixed_calibrated.pt `
  --max-real-fpr 0.05 `
  --batch-size 32 `
  --device cuda `
  --seed 42 `
  --output artifacts\ensemble_glide_wildfake_development.json `
  --predictions-output artifacts\ensemble_glide_wildfake_predictions.json
```

Do not evaluate multiple weights on the final untouched generator. After the
joint policy is selected, freeze its two weights and threshold, then evaluate
that single policy once on a different generator family.

## Local candidate bundle

`artifacts/submission_candidates` contains:

- the original 40K checkpoint as a fallback;
- both checkpoints and the provisional ensemble policy;
- the required detector source;
- a SHA-256 manifest; and
- local inference instructions.

The packaged ensemble was smoke-tested from inside the bundle. This is not yet
a competition-format archive because the official submission file layout has
not been supplied, and the pretrained encoder files are external dependencies.
