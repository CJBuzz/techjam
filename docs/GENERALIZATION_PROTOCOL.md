# Cross-generator generalization protocol

## Why this is the next experiment

The current 97.34% clean accuracy and 94.80% mean transformed accuracy were measured on 752 validation images drawn from the same CIFAKE and SID_Set sources as training. They demonstrate strong same-source corruption robustness, but they do not demonstrate generalization to generators absent from training.

This distinction is supported by recent benchmarks:

- [Community Forensics (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html) contains 2.7 million images from 4,803 generator models and reports that increasing generator count and diversity improves generalization.
- [NTIRE 2026 Robust AI-Generated Image Detection](https://openaccess.thecvf.com/CVPR2026_workshops/NTIRE) evaluates 42 generators together with 36 transformations, treating generator diversity and post-processing robustness as distinct dimensions.
- [B-Free (CVPR 2025)](https://github.com/grip-unina/B-Free) releases a new-generators evaluation set with 1,000 RAISE real images, 1,000 FLUX images, and 1,000 Stable Diffusion 3.5 images. Its fake-image prompts are derived from the real images to reduce semantic-content shortcuts.

## Important evaluation rules

1. Freeze the selected checkpoint, temperature, and decision threshold before evaluating either stage.
2. Do not recalibrate or tune the threshold on the local test set or B-Free labels.
3. Stage 1 is an untouched **in-domain test**, not an unseen-generator test.
4. Stage 2 is the first unseen-generator test for this project.
5. Because B-Free contains one real set and two fake generators, do not use raw overall accuracy as the headline result. Pair each fake generator with the shared real set, calculate balanced accuracy independently, then macro-average the two generators.
6. Once B-Free results influence training or model selection, B-Free is no longer an untouched external test. Reserve a different generator benchmark for the next final evaluation.

## Prepare B-Free without changing forensic traces

Review the B-Free license before downloading. The official repository states that the material is limited to informational and nonprofit use. Dataset files should not be committed or redistributed with this repository.

Open the [official new-generators directory](https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/) and download:

- `real_RAISE_1k.zip` (approximately 1.5 GB)
- `sd3_flux.zip` (approximately 3.2 GB)
- `checksum.txt`

The separate `latent-diffusion.zip` file is not required for this experiment.

After downloading, run:

```bash
uv run python scripts/prepare_bfree_new_generators.py \
  --real-archive ~/Downloads/real_RAISE_1k.zip \
  --fake-archive ~/Downloads/sd3_flux.zip \
  --checksum ~/Downloads/checksum.txt \
  --output data/bfree_new_generators
```

The preparer:

- verifies the official checksums when supplied;
- rejects unsafe ZIP paths;
- verifies that every file is a readable image;
- requires exactly 1,000 RAISE, 1,000 FLUX, and 1,000 SD3.5 images;
- writes `real/raise`, `ai/flux`, and `ai/sd35` source folders;
- copies the original compressed bytes without re-encoding images.

The last point matters because re-saving images could create or remove forensic evidence.

## Run Stages 1 and 2 without training

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/evaluate_generalization.py \
  --checkpoint artifacts/robust_laplacian_fft.pt \
  --local-data data/mixed_5k \
  --external-data data/bfree_new_generators \
  --output-dir artifacts/generalization \
  --profile full \
  --batch-size 16 \
  --device mps
```

This performs inference only:

- Stage 1 evaluates the untouched 752-image local test split.
- Stage 2 evaluates all B-Free images with the `paired-generators` protocol.
- Both stages evaluate every official TechJam transformation severity when `--profile full` is used.
- A combined summary is written to `artifacts/generalization/summary.json`.

For a quicker preliminary check, use `--profile worst`. Use the full profile for the final report.

## Stage 2 metrics to report

For clean and transformed images, report:

- RAISE false-positive rate;
- FLUX recall, false-negative rate, balanced accuracy, and ROC-AUC when paired with RAISE;
- SD3.5 recall, false-negative rate, balanced accuracy, and ROC-AUC when paired with RAISE;
- macro generator balanced accuracy;
- worst-generator balanced accuracy;
- mean and worst transformed macro generator balanced accuracy.

The evaluator also retains the class-imbalanced overall metrics for transparency, but they should not be the headline external-generalization score.

## Stage 3 decision

Do not change the architecture before reviewing Stages 1 and 2. If external performance is weak:

1. add generator-diverse training data;
2. keep CLIP and EfficientNet frozen;
3. retrain only the small fusion head using the existing robustness losses;
4. reserve a generator that was not used for training, calibration, threshold selection, or architecture decisions;
5. report whether cross-generator performance improves without materially reducing the existing transformation robustness.

The current training code already freezes both encoders and trains only the head, so no architectural change is required for this stage. A new dataset split and a genuinely untouched generator benchmark are required.

