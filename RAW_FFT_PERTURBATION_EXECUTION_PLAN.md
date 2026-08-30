# Raw-Image FFT Robustness Experiment: Autonomous Execution Runbook

## 1. Objective

Run a raw-image FFT baseline on the same images and, whenever recoverable, the exact same perturbed and canonicalized pixels used by the completed Stable Diffusion 1.5 VAE residual-FFT experiment. Compare the two methods on the same rows under:

1. `clean`
2. `jpeg_q30`
3. `blur_2`
4. `resize_0.25x`

The central question is whether VAE reconstruction residuals preserve real/fake frequency separation under corruption better than the raw image spectrum itself.

This document is intended to be executable by an autonomous agent while the user is unavailable. Continue through recoverable failures. Use the fallback ladder in Section 12 rather than abandoning the experiment. Never silently substitute data, transforms, metrics, or labels.

The agent should make safe, local, reversible implementation and diagnostic decisions without waiting for routine clarification. It must not delete existing artifacts, overwrite user work, rewrite shared reference code, download a replacement dataset without authorization, or conceal a protocol deviation. If an action requires unavailable permission, continue with the best local fallback and record the limitation.

## 2. Expected scientific output

The main table must be:

| Condition | Raw FFT AUROC | VAE residual FFT AUROC | VAE minus Raw AUROC | Raw change from clean | VAE change from clean | Difference in degradation |
|---|---:|---:|---:|---:|---:|---:|
| clean | recompute | 0.9994416117 | compute | 0 | 0 | 0 |
| JPEG Q30 | compute | 0.9999971298 | compute | compute | +0.0005555181 | compute |
| Blur radius 2 | compute | 0.9311046094 | compute | compute | -0.0683370022 | compute |
| Resize 0.25x | compute | 0.9951183357 | compute | compute | -0.0043232760 | compute |

Define the difference in degradation for condition `c` as:

```text
(VAE_AUROC[c] - VAE_AUROC[clean])
    - (RAW_AUROC[c] - RAW_AUROC[clean])
```

A positive value means the VAE method retained more of its clean AUROC than raw FFT. Also report average precision and fixed-clean-threshold balanced accuracy, but do not replace AUROC as the primary endpoint after seeing results.

## 3. Important correction to the historical clean baseline

`sd15_fft.ipynb` reports raw FFT AUROC `0.989585`, but this is not directly eligible for the paired table. The notebook used:

- 13,841 images: 4,998 real and 8,843 fake;
- direct resize to 1024 by 1024;
- its own image loading and RGBA handling;
- filename-derived labels;
- a different population from the frozen VAE production subset.

The VAE comparison uses:

- exactly 11,841 images: 3,998 real and 7,843 fake;
- the frozen extraction-manifest order;
- perturbation at source resolution followed by aspect-preserving resize and center crop to 512 by 512;
- labels joined by `image_id`.

Therefore, recompute raw FFT for `clean` using the VAE input pipeline. Present `0.989585` only as a historical notebook result. Do not place it in the exact paired comparison table.

The notebook itself is currently untracked user work. Preserve it and do not rewrite, clear, reformat, or add outputs to it.

## 4. Repository layout and immutable reference material

Assume the repository layout below, resolving all paths from the location of the scripts rather than from the shell's current directory:

```text
/home/xinnan/codejam/
    data/
    sb15_fft/
    sd15_vae_fft/
```

Primary project:

```text
/home/xinnan/codejam/sd15_vae_fft
```

Shared, pinned implementation and environment:

```text
/home/xinnan/codejam/sb15_fft
```

Treat the following files as read-only scientific references. Import them; do not edit them for this experiment:

```text
sb15_fft/sb15_fft/perturbations.py
sb15_fft/sb15_fft/preprocessing.py
sb15_fft/sb15_fft/spectral_metrics.py
sb15_fft/results/manifests/frozen_extraction.parquet
sb15_fft/uv.lock
sd15_vae_fft/scripts/run_severity_matrix.py
sd15_vae_fft/results/selected_evaluation_registry.json
sd15_vae_fft/results/scores/production/.../*.parquet
data/manifests/wildfake_test_labels.parquet
```

Known reference hashes at planning time:

| Item | SHA-256 |
|---|---|
| Frozen extraction manifest | `097bb22382f950afdb314d894c8ddd0e69e22b6f73bcf3f80006485db58b16e4` |
| Label manifest | `0e0ec3902ce3f07e08b3651c8972af71fae07447f9049e62c7c7176e9d525c7f` |
| Selected VAE registry | `35264847b76ae536ae808e62ac7119a1f5200e1cb76e402db9146cfbf33991d0` |
| Shared perturbations code | `785b9ac44318bc9931fa4e73f93ad940505b356e3a9f7c2f760bbf28c0321ca6` |
| Shared preprocessing code | `df5769325b99dd0d75f4c9389056ae842211276f1b992ceeae29c5669f52b468` |
| Shared spectral code | `3884a5748aaa2bbb07a786ea205444d69bdcc920ec11087e8816bdd550c8c9ae` |
| Shared environment lock | `0c70d2821298593327072d96c1e13b2e4082a46e0a3bef4fe8cd2605c922e7fd` |
| Original VAE runner | `0e08a3ebe1c04da01ce009aff73c0b16b99fff7cfde89b18025b35d1c689a3bc` |
| Historical FFT notebook | `17d2bbb8aa1292af4b1ec53fdd05f22745d346c873ed097f94ffcd34eb603713` |

Reference VAE score files:

| Condition | Relative path from `/home/xinnan/codejam` | SHA-256 |
|---|---|---|
| clean | `sd15_vae_fft/results/scores/production/9199b8664536bcec9c0d68aa28a98911486cc282f1fdeadbaa639dcd9ac36db4/clean.parquet` | `3ed422cf054f76406b0485aa45104a401f8fe669ee42e98f5e65b61bd184c5f6` |
| jpeg_q30 | `sd15_vae_fft/results/scores/production/94bed5836585cf21612a250296de72016ee4db90b4920f26e85c1da9dff2ffad/jpeg_q30.parquet` | `4380d440a5d12aaf0ab560abb240c0bff76d72e576d73e75a6edfb87259eb9d7` |
| blur_2 | `sd15_vae_fft/results/scores/production/94bed5836585cf21612a250296de72016ee4db90b4920f26e85c1da9dff2ffad/blur_2.parquet` | `d024a968123646f92b52359134a774fa8fd30f69277b611c87f6f8474d98a5d3` |
| resize_0.25x | `sd15_vae_fft/results/scores/production/94bed5836585cf21612a250296de72016ee4db90b4920f26e85c1da9dff2ffad/resize_0.25x.parquet` | `f463ef79b47c078e60a04913f3e7ce38c7f8e7844d562b577410ba68b0017f5e` |

Hashes are validation targets, not instructions to revert user changes. At startup, run `git status --short`. Preserve all pre-existing changes and untracked files. If reference code has changed, do not overwrite it; follow Section 12.2.

## 5. Locked experiment specification

### 5.1 Dataset and row identity

Use `sb15_fft/results/manifests/frozen_extraction.parquet` as the sole extraction inventory. Production must contain all 11,841 rows in `ordinal` order `0..11840`.

Required manifest fields:

```text
ordinal
image_id
relative_path
source_path
severity_seed_key
source_bytes
source_sha256
```

Do not discover images with `rglob`, basename matching, directory sorting, or filename prefixes. DALLE3 basenames are not unique, so basename joins are invalid.

### 5.2 Conditions

Primary production conditions are locked to:

```python
(
    ("clean", 0.0),
    ("jpeg", 30.0),
    ("blur", 2.0),
    ("resize", 0.25),
)
```

Use global perturbation seed `42` and each row's `severity_seed_key` exactly as the VAE runner did.

### 5.3 Exact image loading

Match the VAE runner:

```python
with Image.open(path) as source:
    image = ImageOps.exif_transpose(source).convert("RGB")
```

Do not use the notebook's `torchvision.io.read_image` path or its white alpha compositing in the primary experiment.

### 5.4 Exact perturbations

Import and call:

```python
ExactSeverityTransform(operation, value, 42, row.severity_seed_key)(image)
```

The locked behavior is:

- `clean`: RGB image unchanged.
- `jpeg_q30`: in-memory Pillow JPEG encode at `quality=30`, decode, convert to RGB.
- `blur_2`: Pillow `ImageFilter.GaussianBlur(2.0)`.
- `resize_0.25x`: resize original dimensions to `round(width*0.25), round(height*0.25)` with Pillow bilinear, then resize back to original dimensions with Pillow bilinear.

Do not describe the exact blur as Gaussian sigma 2 unless supported by Pillow's own definition. In reports, prefer `Pillow GaussianBlur radius 2`, matching the implementation.

### 5.5 Exact canonicalization

After perturbation:

1. Convert to RGB.
2. Set `scale = 512 / min(width, height)`.
3. Resize to `(round(width*scale), round(height*scale))` using Pillow bicubic.
4. Center crop 512 by 512 using integer floor offsets.
5. Convert the resulting uint8 RGB array to CHW float32 in `[0,1]` by division by 255.

Import `canonical` from `sb15_fft.preprocessing` instead of reimplementing this path whenever possible.

### 5.6 Raw FFT score

Let `x` be the canonical float32 RGB tensor with shape `N x 3 x 512 x 512`. Compute:

```python
gray = x.mean(1)
fft = torch.fft.rfft2(gray, norm="ortho").abs().square()
fy = abs(torch.fft.fftfreq(512))[:, None]
fx = abs(torch.fft.rfftfreq(512))[None, :]
radius = sqrt(fx**2 + fy**2) / sqrt(0.5**2 + 0.5**2)
low  = radius < 0.25
mid  = (radius >= 0.25) & (radius < 0.5)
high = radius >= 0.5
band_means = [fft[:, mask].mean(1) for mask in (low, mid, high)]
raw_fft_high_bandmean_ratio = band_means[2] / clamp_min(sum(band_means), 1e-12)
```

This is a ratio of the three band means, not a ratio of coefficient sums. Also calculate the separately named coefficient-sum ratio, but never conflate the two.

The preferred reuse path is:

```python
metrics = image_metrics(x, torch.zeros_like(x))
```

Because `image_metrics` defines its signal as `x-y`, setting `y=0` makes the signal exactly the raw canonical image while reusing the previously tested spectral implementation. Rename output fields with a `raw_` prefix before persistence.

Do not add mean subtraction, windowing, logarithms, spectrum shifting, per-image normalization, or channel-wise FFTs.

### 5.7 Labels and score direction

Extraction must not read labels. Evaluation may read `data/manifests/wildfake_test_labels.parquet` only after score files and an evaluation registry are frozen.

Label convention:

```text
0 = real COCO
1 = fake advanced DALLE3
```

Prespecify `raw_fft_high_bandmean_ratio` direction as positive: higher means more likely fake. This is equivalent to the notebook's use of the negated score for a real-positive AUROC. Never choose score direction separately for each perturbation.

Primary metric: AUROC.

Secondary metrics:

- average precision;
- accuracy;
- balanced accuracy;
- precision;
- recall/TPR;
- specificity;
- FPR;
- F1;
- confusion-matrix counts.

Choose one post-hoc raw clean-oracle threshold maximizing balanced accuracy using the same deterministic tie-break rule as `scripts/evaluate_selected.py`. Apply that threshold unchanged to all raw perturbations. Retain the existing VAE clean threshold for VAE fixed-threshold metrics.

## 6. Files to implement

Create new files without altering the historical notebook or existing VAE result files:

```text
configs/raw_fft_experiment.yaml
scripts/run_raw_fft_severity.py
scripts/evaluate_raw_vs_vae.py
tests/test_raw_fft_experiment.py
```

Expected generated outputs:

```text
results/raw_fft/scores/smoke/<configuration_hash>/<condition>.parquet
results/raw_fft/scores/production/<configuration_hash>/<condition>.parquet
results/raw_fft/logs/smoke_report.json
results/raw_fft/logs/production_report.json
results/raw_fft/selected_evaluation_registry.json
results/metrics/raw_fft_vs_vae_selected_production.json
results/metrics/raw_fft_vs_vae_selected_production.csv
RAW_FFT_COMPARISON.md
```

Do not write raw images into `artifacts/`. Existing `artifacts/` PNGs are VAE reconstructions, not perturbed inputs, and must not be used as raw-FFT inputs.

## 7. Runner design

### 7.1 CLI

Support at least:

```text
--mode smoke|production
--only CONDITION           repeatable
--resume
--batch-size INTEGER
--reference-registry PATH
--allow-approximate-inputs
```

Approximate mode must never activate implicitly. The autonomous agent may explicitly activate it only after the exact recovery attempts in Section 12 have failed and the divergence has been logged.

Suggested defaults:

```text
mode: required
batch_size: 16
reference_registry: results/selected_evaluation_registry.json
checkpoint interval: 100 rows
device: cpu
```

CPU is preferred because the shared spectral function returns NumPy arrays directly and the workload does not need the VAE or CUDA. CUDA is optional only if a tested wrapper produces equivalent scores and offers a material speedup.

### 7.2 Configuration identity

Compute a configuration hash over a canonical JSON object containing:

- parsed `configs/raw_fft_experiment.yaml`;
- batch size and compute backend;
- runner SHA-256;
- shared perturbation, preprocessing, and spectral-code SHA-256 hashes;
- frozen-manifest SHA-256;
- `uv.lock` SHA-256;
- selected VAE registry SHA-256;
- every referenced VAE Parquet SHA-256;
- fallback tier and every approximation flag;
- relevant package versions at runtime.

Exact and approximate runs must always receive different configuration hashes and output directories.

### 7.3 Reference loading

Resolve condition-specific VAE Parquet paths from `results/selected_evaluation_registry.json`. Do not hard-code the two VAE configuration hashes in operational code.

For each reference frame, require:

- 11,841 rows for production;
- unique `image_id`;
- ordinals exactly `0..11840`;
- condition name matches;
- finite VAE primary scores;
- reference file hash matches the registry.

For smoke mode, select the same ordinals as the VAE runner:

```python
np.linspace(0, len(manifest) - 1, 16, dtype=int)
```

Use those ordinals to subset the complete production reference frames. This permits smoke input-hash validation without relying on separate smoke artifacts.

### 7.4 Per-row extraction algorithm

For each condition and row:

1. Confirm manifest `ordinal` and `image_id` match the corresponding VAE row.
2. Confirm source file exists and byte length matches `source_bytes`.
3. Compute the source file SHA-256 and compare it with `source_sha256`.
4. Load using EXIF transpose and RGB conversion.
5. Apply the exact deterministic perturbation.
6. Convert perturbed image to a uint8 RGB array and calculate `transformed_sha256` with `pixel_hash`.
7. Canonicalize to 512 by 512.
8. Calculate `canonical_sha256` with `pixel_hash`.
9. Compare derived seed, transformed hash, and canonical hash with the VAE reference row.
10. If exact mode and any comparison fails, stop the condition before committing a mixed exact/approximate score file.
11. Compute raw FFT metrics.
12. Persist row metadata and scores.

Required output columns:

```text
ordinal
image_id
condition
operation
value
global_seed
derived_seed
severity_key
source_sha256
transformed_sha256
canonical_sha256
reference_transformed_sha256
reference_canonical_sha256
input_identity_status
fallback_tier
runtime_seconds
configuration_hash
method
raw_fft_low_bandmean
raw_fft_mid_bandmean
raw_fft_high_bandmean
raw_fft_high_bandmean_ratio
raw_fft_high_coeffsum_ratio
```

Use method name `raw_image_fft_high_bandmean_ratio`.

### 7.5 Atomic persistence and resume

- Write checkpoints every 100 completed rows and at condition completion.
- Write to `filename.tmp`, validate the temporary file, then use `os.replace`.
- Store one JSON completion summary next to each completed Parquet file.
- On `--resume`, validate configuration hash, row prefix, IDs, ordinals, finite scores, and hash status before continuing.
- Never append to or reuse output created under a different configuration hash.
- If a checkpoint is corrupt, move it to a timestamped `quarantine/` directory and restart that condition. Do not delete it.
- A completed condition rerun with `--resume` should validate and exit without recomputation.

## 8. Tests and validation gates

### Gate A: static integrity

Before implementation or execution:

1. Record `git status --short` in the execution log.
2. Compute current hashes of all references in Section 4.
3. Verify the frozen manifest has 11,841 unique `image_id` values and exact ordinals.
4. Verify the labels contain exactly 3,998 real and 7,843 fake rows with unique IDs.
5. Verify each of the four VAE reference Parquets is complete and matches its registry hash.

If any check fails, consult Section 12 before proceeding.

### Gate B: FFT parity unit tests

On deterministic synthetic float32 tensors, assert:

1. The notebook `hf_ratio` formula equals `image_metrics(x, zeros_like(x))["fft_high_bandmean_ratio"]` within strict numerical tolerance.
2. Batch and one-at-a-time scoring agree.
3. Constant black, constant white, impulses, gradients, and seeded random tensors produce finite scores.
4. The primary band-mean ratio is demonstrably different from the coefficient-sum ratio on at least one nontrivial tensor, preventing accidental field substitution.
5. Score arrays preserve input row order.

Use an absolute and relative tolerance no looser than `1e-7` for same-backend float32 parity. If a fallback backend is used, characterize its discrepancy and use the tightest justified tolerance.

### Gate C: exact smoke input identity

Run all four conditions over the 16 smoke ordinals. Require:

```text
source SHA mismatches:      0 / 64
derived-seed mismatches:    0 / 64
transformed hash mismatches:0 / 64
canonical hash mismatches:  0 / 64
non-finite raw scores:      0 / 64
```

Do not start exact production if this gate fails. Diagnose and follow the exact recovery ladder first.

### Gate D: production extraction completeness

Require per condition:

- 11,841 rows;
- 11,841 unique `image_id` values;
- exact ordinal and ID order matching the manifest and VAE reference;
- zero input hash mismatches for an exact run;
- zero NaN or infinite scores;
- a valid completion JSON with score-file hash.

Total required rows across four conditions: 47,364.

### Gate E: evaluation integrity

Before reading labels:

1. Write `results/raw_fft/selected_evaluation_registry.json` with score-file paths, file hashes, configuration hashes, row counts, input-identity status, and fallback tier.
2. Hash the registry.
3. Require identical IDs and order across all raw and VAE condition frames.
4. Verify the label-file hash.

Then join labels by `image_id` with `validate="one_to_one"`; never infer labels from filenames in the exact experiment.

## 9. Execution sequence and commands

Run from `/home/xinnan/codejam` when possible.

### 9.1 Environment check

```bash
cd /home/xinnan/codejam
sb15_fft/.venv/bin/python --version
uv run --project sb15_fft python -c "import torch, PIL, pandas, pyarrow, sklearn; print(torch.__version__, PIL.__version__)"
```

The expected project environment is Python 3.12 with dependencies pinned by `sb15_fft/uv.lock`. Do not update packages merely to make the run easier.

### 9.2 Tests

```bash
uv run --project sb15_fft pytest -q sd15_vae_fft/tests/test_raw_fft_experiment.py
```

### 9.3 Exact smoke

```bash
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py \
  --mode smoke \
  --only clean \
  --only jpeg_q30 \
  --only blur_2 \
  --only resize_0.25x
```

Inspect the smoke JSON report. Proceed only if Gate C passes or an explicit fallback tier has been selected and recorded.

### 9.4 Exact production

```bash
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py \
  --mode production \
  --only clean \
  --only jpeg_q30 \
  --only blur_2 \
  --only resize_0.25x \
  --resume
```

If a single long command is operationally fragile, run one condition at a time with repeated `--resume`. This does not change the scientific design:

```bash
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py --mode production --only clean --resume
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py --mode production --only jpeg_q30 --resume
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py --mode production --only blur_2 --resume
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py --mode production --only resize_0.25x --resume
```

### 9.5 Evaluation

```bash
uv run --project sb15_fft python sd15_vae_fft/scripts/evaluate_raw_vs_vae.py --allow-label-read
```

### 9.6 Final verification

Run tests again, hash all generated outputs, inspect all fallback and exclusion counts, and confirm the Markdown report agrees with JSON/CSV values.

## 10. Statistical analysis

### 10.1 Per-method metrics

For each method-condition pair, calculate AUROC and average precision using fake as the positive class. Calculate fixed-threshold metrics using each method's own clean-selected threshold applied unchanged to that method's perturbations.

Do not optimize a threshold separately on `jpeg_q30`, `blur_2`, or `resize_0.25x`.

### 10.2 Paired bootstrap

Use 2,000 paired, class-stratified bootstrap replicates with NumPy seed `20260830`, matching the VAE evaluation.

Create the bootstrap index arrays once, then reuse exactly the same arrays for:

- all four conditions;
- raw and VAE methods;
- AUROC, average precision, and balanced accuracy;
- all method differences and changes from clean.

Sample each class with replacement at its original class size, concatenate the class samples, and apply the same sampled indices to all aligned score arrays.

Report percentile 95% confidence intervals for:

- raw AUROC and AP;
- VAE AUROC and AP;
- `VAE - Raw` within each condition;
- each method's change from clean;
- the difference in degradation;
- fixed-clean-threshold balanced accuracy and its method difference.

### 10.3 Sanity reproduction of stored VAE values

Before comparing methods, reevaluate the stored VAE scores and require agreement with `results/metrics/selected_production_metrics.json` to at least 12 decimal places for AUROC. Expected values are listed in Section 2.

If they do not reproduce, diagnose row alignment, label orientation, score column, and label-file identity. Do not alter the expected values or score direction to force agreement.

### 10.4 Interpretation

Use paired confidence intervals rather than an arbitrary verbal cutoff.

- If raw FFT remains close to VAE residual FFT under resize and blur, conclude that much of the robustness is attributable to dataset-level raw spectral separation; do not claim a strong VAE-specific mechanism.
- If raw FFT degrades substantially more and the paired difference-in-degradation interval supports a VAE advantage, conclude that the VAE residual representation preserves separability better under these transformations.
- If both remain strong, emphasize that both raw spectral properties and the VAE may contribute.
- If results depend on approximate fallbacks, restrict conclusions to a sensitivity analysis and state that the exact paired question remains unresolved.

This is a comparison on COCO real versus advanced DALLE3 fake images. Do not generalize to all real and synthetic imagery without additional datasets.

## 11. Required reporting and provenance

`RAW_FFT_COMPARISON.md` must include:

1. Executive summary.
2. Exact scientific question and prespecified conditions.
3. Dataset counts and class balance.
4. Historical notebook result and why it is not directly comparable.
5. Raw FFT definition.
6. Exact or approximate perturbation/preprocessing definitions.
7. Input-identity audit counts per condition.
8. Main paired table and confidence intervals.
9. Fixed-threshold calibration results.
10. Interpretation relative to the two possible outcomes in Section 10.4.
11. Limitations.
12. Reproducibility commands and artifact hashes.
13. A divergence ledger described below.

Every result JSON should include:

```text
status
fallback_tier
exact_input_identity
manifest_sha256
label_sha256
reference_registry_sha256
code_sha256 values
environment/package versions
conditions
n per condition
class counts
score name and direction
threshold and threshold status
bootstrap seed and replicates
input mismatch counts
excluded-row counts and reasons
result rows
```

Maintain a divergence ledger even when empty:

| Time | Stage | Failure | Attempts | Chosen fallback | Scientific impact | Affected rows/conditions |
|---|---|---|---|---|---|---|

An exact successful run should explicitly say `No protocol divergences` rather than omit the section.

## 12. Autonomous recovery and fallback ladder

General rule: attempt the exact path first. For a recoverable failure, make up to three materially different diagnostic or repair attempts. Preserve evidence, record the failure, then use the highest-quality viable fallback. Do not repeatedly rerun the same failing command without a new hypothesis.

Fallback tiers are ordered from strongest to weakest:

| Tier | Meaning | Eligible for exact main table? |
|---|---|---|
| `E0` | All source, transformed, and canonical hashes match VAE references | Yes |
| `E1` | Exact pinned code and verified source hashes, but VAE pixel-hash references unavailable | Qualified; clearly mark identity not independently confirmed |
| `P1` | Partial E0 dataset; same matched rows used for raw and stored VAE comparison | Yes, as a partial-coverage analysis only |
| `A1` | Reimplemented documented transforms with intended parameters; pixel hashes differ or cannot be checked | No; approximate sensitivity analysis |
| `A2` | Nearest available substitute library/data semantics | No; exploratory only |

Never combine tiers in one aggregate metric. Write separate configuration directories and separate result sections.

### 12.1 Missing or moved source images

Recovery order:

1. Check the exact `source_path` from the frozen manifest.
2. Resolve `relative_path` under likely local WildFake roots within `/home/xinnan/codejam/data`.
3. Search local files by `source_sha256`, not basename. Use a cache mapping byte size to candidate files before hashing to avoid hashing an entire large tree repeatedly.
4. If another WildFake copy is found, accept a replacement only when its bytes match `source_sha256`.
5. If only some exact sources are recoverable, switch to tier `P1`: use the intersection of rows that are E0 across all four conditions and both methods. Recompute the VAE metrics on that identical subset using stored VAE scores. Report coverage overall and by class.
6. If no useful exact subset is available, search for a locally available equivalent COCO-real/DALLE3 WildFake subset. Run tier `A2` and label it as an external or approximate replication, not the original paired experiment.

Do not use VAE reconstruction PNGs as substitutes for raw inputs. They answer a different question.

### 12.2 Reference source code changed or unavailable

Recovery order:

1. Inspect Git history and the working-tree state without modifying user changes. Locate the blob matching the planned hash.
2. If the matching blob exists in Git, export it into a new read-only snapshot directory under `sd15_vae_fft/references/` without checking it out over current files.
3. If the code exists in another local worktree or cache, verify its hash and import or copy the verified snapshot.
4. If only changed code is available, compare its semantics with the code excerpts in this runbook and the original VAE runner.
5. Reimplement the locked semantics locally and designate the result tier `A1` unless pixel hashes prove E0 equivalence.

Pixel-hash agreement overrides code-location differences: a reimplementation that reproduces all recorded transformed and canonical hashes is tier E0 for input identity.

### 12.3 VAE reference Parquets or registry missing

Recovery order:

1. Locate files by their known SHA-256 hashes under local project/worktree/cache roots.
2. Reconstruct registry paths from the condition and configuration hashes in Section 4.
3. Use pilot or smoke VAE Parquets for preliminary hash validation if production references are temporarily unavailable.
4. If pinned source code, frozen manifest, source-file hashes, and environment all match but reference pixel hashes cannot be recovered, proceed at tier `E1`.
5. If stored VAE scores are unavailable, complete raw extraction and evaluation alone, but do not fabricate the comparison table. Report the raw results and state that paired VAE comparison is pending.

### 12.4 Pixel-hash mismatches

Diagnose in this order:

1. Confirm correct row, condition, seed key, operation value, and source SHA.
2. Confirm EXIF transpose occurs before RGB conversion and perturbation.
3. Confirm exact Pillow version from the locked environment.
4. Confirm perturbation happens before canonicalization.
5. Confirm Python `round` semantics, original dimensions, resampling modes, JPEG in-memory encode/decode, and uint8 conversion points.
6. Compare source, transformed, and canonical image dimensions.
7. Save a small number of mismatching diagnostic images under `results/raw_fft/diagnostics/<config>/`; never mix them with official inputs.
8. Find the earliest stage at which hashes diverge.

If mismatches are isolated to some source files, use tier `P1` on the exact intersection and separately report the mismatched rows. If all rows mismatch consistently after a known library difference, use tier `A1`, record the implementation and package versions, and keep approximate results separate.

Do not weaken or remove the hash check merely to finish an E0 run.

### 12.5 Approximate perturbation definitions

If exact Pillow behavior cannot be recovered, use these nearest alternatives in order. Always record the actual library, version, arguments, and output-hash mismatch counts.

#### JPEG Q30

1. Pillow in-memory JPEG with `quality=30` and otherwise default options.
2. OpenCV `imencode('.jpg', ..., [IMWRITE_JPEG_QUALITY, 30])`, preserving RGB/BGR conversions correctly.
3. Another deterministic JPEG codec at nominal quality 30.

Codec quality scales and chroma subsampling may differ. Label non-Pillow results approximate.

#### Blur radius 2

1. Pillow `ImageFilter.GaussianBlur(2.0)`.
2. OpenCV Gaussian blur with `sigmaX=2`, `sigmaY=2`, documenting kernel and border mode.
3. SciPy `ndimage.gaussian_filter` with sigma 2 per spatial axis and no channel blur, documenting boundary mode.

These are similar but not guaranteed pixel-equivalent.

#### Resize 0.25x

1. Pillow bilinear downsample to rounded quarter dimensions, then Pillow bilinear upsample to original dimensions.
2. Torchvision or PyTorch bilinear interpolation with explicit sizes, `align_corners=False`, and documented antialias behavior.
3. OpenCV linear interpolation with explicit rounded sizes.

Do not perform a single resize directly to 512; the intended corruption is source-resolution downscale/upscale before canonicalization.

#### Canonicalization

1. Pillow bicubic aspect-preserving resize and exact center crop.
2. Torchvision bicubic with explicit output dimensions and documented antialias setting.
3. OpenCV cubic with explicit dimensions and center crop.

Any non-Pillow canonicalization is approximate unless it reproduces the recorded canonical hashes.

### 12.6 Locked Python environment unavailable

Recovery order:

1. Use `/home/xinnan/codejam/sb15_fft/.venv/bin/python` directly.
2. Use `uv run --offline --project sb15_fft` if the environment exists but normal resolution attempts network access.
3. Locate the pinned packages in local uv caches and recreate the environment from `uv.lock` without upgrading versions.
4. Use a compatible existing Python environment and record every package version.
5. If PyTorch is unavailable, implement FFT with NumPy:

```python
np.fft.rfft2(gray, axes=(-2, -1), norm="ortho")
```

Validate NumPy scores against known synthetic-tensor outputs and, if possible, a small PyTorch run. A backend-only FFT change may remain scientifically usable, but report the observed numerical difference and assign `A1` unless equivalence is convincingly established.

### 12.7 CUDA, memory, or throughput failure

The default exact runner should work on CPU. If resource failures occur:

1. Set DataLoader workers to zero.
2. Reduce batch size: `16 -> 8 -> 4 -> 1`.
3. Process one condition per command with `--resume`.
4. Ensure tensors are discarded after each batch and avoid retaining FFT arrays.
5. If CPU FFT is prohibitively slow and CUDA works, add a CUDA scoring path, validate parity on smoke, and include backend in the configuration hash.
6. If a process is killed, validate the last atomic checkpoint and resume. Do not start over unless checkpoint validation fails.

Batch-size or device changes must use a new configuration hash. If scores are shown equivalent on the smoke set, completed configuration shards may be combined only through an explicit merge step that records the source configuration for every row. Prefer one consistent production configuration.

### 12.8 Corrupt or unreadable images

For each failure:

1. Confirm file bytes and SHA-256.
2. Retry opening once in a fresh process.
3. Try the exact pinned Pillow environment.
4. If the original VAE run had a score for the row but the source is now unreadable, search for a byte-identical source copy.
5. If still unavailable, exclude that row from all conditions and both methods and switch to tier `P1` for the common intersection.

Never replace one missing image with another image sharing the same basename.

### 12.9 Parquet or PyArrow failure

Recovery order:

1. Use the pinned `pyarrow` environment.
2. Write smaller atomic Parquet checkpoints.
3. Fall back to atomic CSV plus a schema JSON and file SHA-256.
4. Preserve float values with round-trip-safe formatting (`repr`/17 significant digits).
5. Add CSV support to evaluation and record storage fallback in the registry.

Storage-format fallback does not change the scientific tier if values and metadata round-trip exactly.

### 12.10 Missing or changed labels

Recovery order:

1. Locate the label file by its known SHA-256.
2. Search other local worktrees and manifests.
3. If scores are complete but exact labels cannot be recovered, freeze the score registry and defer official evaluation.
4. As an explicitly approximate fallback, reconstruct labels from the frozen `relative_path` categories (`Real/coco/...` versus the selected advanced DALLE3 path), validate expected counts, and label the analysis as reconstructed-label tier `A1`.

Do not infer labels from basenames or the notebook's `name.startswith("img")` rule for the exact experiment.

### 12.11 One or more conditions cannot complete

Continue all conditions that can be completed. The report must distinguish:

- complete exact conditions;
- partial exact conditions;
- approximate conditions;
- missing conditions.

Never replace a missing main condition with a different severity in the same row of the main table. A nearby severity may be run as a separate sensitivity analysis, for example JPEG Q25/Q35, blur 1.5/2.5, or resize 0.2/0.3, but it must have its own condition name and cannot be compared as though it were Q30, radius 2, or 0.25x.

### 12.12 Bootstrap or evaluation performance failure

1. Verify aligned NumPy arrays and remove DataFrame work from the inner loop.
2. Generate bootstrap indices once and reuse them.
3. Run replicates in deterministic chunks and checkpoint replicate outputs.
4. If 2,000 replicates cannot complete, report point estimates first and use the largest completed prespecified prefix of replicates, no fewer than 500, with the actual count prominently disclosed.
5. Do not change the seed after inspecting preliminary intervals.

## 13. Partial-coverage rules

If tier `P1` is necessary, define one common eligible set before reading labels:

```text
eligible = rows with exact source, transformed, and canonical identity
           in all four raw conditions
           and present in all four stored VAE score files
```

Freeze the eligible `image_id` list and its hash. Then read labels and report:

- total retained rows and percentage;
- retained real/fake counts;
- exclusions by condition and reason;
- whether exclusions differ by source category after labels are opened;
- VAE metrics recomputed on exactly the retained rows;
- raw metrics on exactly the retained rows.

Do not compare full-sample VAE AUROC with partial-sample raw AUROC.

## 14. Optional secondary phase

After the four production conditions are complete, an optional exploratory phase may run raw FFT over the 200-row pilot for all 16 original severity conditions:

```text
clean
jpeg_q90, jpeg_q70, jpeg_q50, jpeg_q30
blur_0.5, blur_1, blur_2
resize_0.5x, resize_0.25x
noise_0.02, noise_0.05, noise_0.1
color_pm10pct, color_pm20pct
crop_0.8x
```

This is secondary. Do not delay or obscure the four-condition production answer. Label pilot results exploratory and use the matching 200 VAE pilot rows and hashes.

## 15. Completion checklist

- [ ] Existing user changes recorded and preserved.
- [ ] Reference hashes checked and divergences logged.
- [ ] New config, runner, evaluator, and tests implemented.
- [ ] FFT parity tests pass.
- [ ] Exact 16-row/four-condition smoke passes, or fallback tier is documented.
- [ ] Four production conditions attempted independently.
- [ ] Exact conditions contain 11,841 rows each; otherwise common partial set frozen.
- [ ] Input identity counts are present in logs and report.
- [ ] Raw score registry frozen before label read.
- [ ] Stored VAE metrics reproduced.
- [ ] Raw clean score recomputed on VAE inputs.
- [ ] Paired bootstrap uses one shared set of 2,000 class-stratified resamples.
- [ ] Main table, method differences, and difference-in-degradation reported.
- [ ] Historical notebook result kept separate.
- [ ] Approximate results, if any, are separated and labeled.
- [ ] JSON, CSV, and Markdown values agree.
- [ ] Output hashes and reproducibility commands recorded.
- [ ] Final tests pass.

## 16. Final handoff format

At completion, give the user a concise summary containing:

1. Whether input identity was E0, E1, P1, A1, or A2 for each condition.
2. The main four-row AUROC table.
3. The paired VAE-minus-raw differences and confidence intervals.
4. The difference-in-degradation result for each perturbation.
5. Any fallback, exclusion, or protocol divergence.
6. Links to `RAW_FFT_COMPARISON.md`, the metrics JSON/CSV, the frozen registry, and the runner/evaluator.
7. A direct conclusion: raw spectrum explains most robustness, VAE residual adds robustness, evidence is mixed, or the exact comparison remains unresolved.

Do not claim success solely because scripts ran. Success requires validated data identity, complete or explicitly qualified coverage, correct paired evaluation, and transparent reporting of every fallback.
