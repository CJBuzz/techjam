# Robust AIGC Image Detector

This repository is our Track 5 submission for detecting AI-generated images
under common content transformations. The submitted model is
`artifacts/diverse_initialized_40k_calibrated.pt`: a calibrated, 40K-initialized
three-view detector that outputs an AIGC probability for every input image.

The repository includes the complete feature-extraction, training, calibration,
validation, test-inference, JSON-output, and dashboard path. The checkpoint has
820,225 trainable head parameters; its frozen CLIP and EfficientNet backbones
keep the complete system well below the challenge's 2-billion-parameter limit.

## Setup and quick start

### Requirements

- Git;
- Python 3.10–3.12;
- [`uv`](https://docs.astral.sh/uv/);
- approximately 2 GB of free disk space for the Python environment and frozen
  encoder weights;
- internet access for the first setup/inference run.

The submitted 3.2 MB detector head is tracked at
`artifacts/diverse_initialized_40k_calibrated.pt`. CLIP ViT-B/32 and
EfficientNet-B0 are public pretrained backbones and are downloaded separately
on first use.

### Fresh clone

```bash
git clone https://github.com/CJBuzz/techjam.git
cd techjam
uv sync --frozen
uv run --frozen python scripts/download_pretrained_models.py \
  --checkpoint artifacts/diverse_initialized_40k_calibrated.pt
```

The download command fetches the exact CLIP model named by the checkpoint and
the ImageNet EfficientNet-B0 weights. It stores them under the ignored,
project-local `.hf-cache/` and `.torch-cache/` directories. It is idempotent:
rerunning it validates or reuses downloaded files instead of downloading them
again.

### Run inference and the dashboard

Copy JPG, PNG, WebP, BMP, or TIFF files into the existing `images/` directory,
then run:

```bash
./infer.sh
uv run --no-sync streamlit run app.py
```

Open <http://localhost:8501>. `infer.sh` uses the submitted checkpoint by
default and writes `output.json`; the dashboard validates that file and renders
each image on a human/real-to-AIGC probability spectrum.

`infer.sh` is also safe to use immediately after cloning. When `uv` is
available it creates or synchronizes the locked environment, runs the
pretrained-model downloader, and then performs inference. The explicit setup
commands above are recommended because they surface installation or network
errors before the demo.

On Windows, run `infer.sh` from Git Bash. The equivalent PowerShell commands
are:

```powershell
uv sync --frozen
uv run --frozen python scripts/download_pretrained_models.py --checkpoint artifacts/diverse_initialized_40k_calibrated.pt
uv run --no-sync python -m aigc_detector.predict images --checkpoint artifacts/diverse_initialized_40k_calibrated.pt --output output.json
uv run --no-sync streamlit run app.py
```

A missing `output.json` means inference has not run. After adding images or
rerunning inference, use **Refresh results** in the dashboard.

To run only the required machine-readable inference:

```bash
uv run --no-sync aigc-predict images \
  --checkpoint artifacts/diverse_initialized_40k_calibrated.pt \
  --output output.json
```

The result is a JSON array with exactly the required fields:

```json
[
  {"image_path": "images/example.jpg", "pred": 0.9374}
]
```

`pred` is the temperature-calibrated probability of AIGC in `[0, 1]`. It is a
model estimate, not a provenance guarantee. After setup, later runs reuse the
project-local `.hf-cache` and `.torch-cache` without downloading weights again.

### Setup troubleshooting

- **`uv` is not found:** install it from the linked official uv documentation,
  reopen the terminal, and confirm with `uv --version`.
- **The checkpoint is missing:** confirm
  `artifacts/diverse_initialized_40k_calibrated.pt` exists after cloning. It is
  tracked directly by Git and does not require a separate model registry.
- **A pretrained download fails:** verify access to `huggingface.co` and
  `download.pytorch.org`, then rerun `scripts/download_pretrained_models.py`.
  Partial downloads are safe to resume.
- **Hugging Face warns about Windows symlinks:** inference still works. Enabling
  Windows Developer Mode saves disk space but is not required.
- **Running without internet:** complete the downloader once while online and
  keep `.hf-cache/` and `.torch-cache/` in the project. Do not delete these
  directories before the offline run.

## Technical approach

The detector combines three complementary frozen representations:

1. **Semantic view — CLIP ViT-B/32 (512 features).** Captures image-level
   concepts and composition that survive compression or resampling.
2. **Local forensic view — EfficientNet-B0 on a Laplacian image (1,280
   features).** The RGB image is resized to 224×224, converted to grayscale,
   filtered with a fixed 3×3 Laplacian, scaled by its per-image 99th percentile,
   and ImageNet-normalized.
3. **Frequency forensic view — EfficientNet-B0 on an FFT image (1,280
   features).** The grayscale mean is removed, a 2D Hann window is applied, and
   the centered `log1p(abs(FFT))` spectrum is robustly scaled between its
   per-image 1st and 99th percentiles.

The normalized 3,072-dimensional representation feeds a small MLP. Only this
fusion head is trained. Feature caching makes repeated head experiments cheap
and guarantees that preprocessing used for training is identical to inference.

### Robust training and selection

- Original images are assigned to deterministic, duplicate-group-atomic
  train/model-selection/calibration/test splits before augmentation.
- Every training original has one clean view and two deterministic balanced
  transformed views spanning JPEG, blur, resize, noise, color, crop, and useful
  two-step compositions.
- A paired-logit consistency penalty (`0.1`) discourages predictions from
  changing across views of the same image.
- The final fusion head starts from the selected 40K head. This retains its
  strong CLIP+Laplacian+FFT solution while adapting on the more diverse corpus;
  modality dropout `0.1` and FFT-block dropout `0.15` reduce dependence on a
  single stream.
- Early stopping and model comparison use only the model-selection split.
  Temperature (`0.6637356`) and the balanced operating threshold (`0.7572185`)
  are then fitted once on the separate clean/transformed calibration split.
- Reserved-test images are excluded from feature extraction during model
  development. The exact-severity test command requires an explicit
  `--allow-test` acknowledgement.

Checkpoint metadata records 47,994 training originals, 9,988 model-selection
originals, 2,000 calibration originals, 6,000 reserved-test originals, and
143,982 training feature rows. The checkpoint is 3.2 MB; pretrained frozen
backbones are downloaded separately.

The 40K initializer used balanced CIFAKE and SID_Set real/synthetic classes
(SID's tampered class was excluded). The adapted checkpoint names its source
manifest as `data/mixed_wildfake_66k/split_manifest.csv`, but that manifest is
not present in this checkout. Consequently, its exact additional dataset
composition and exclusion of the challenge's demonstration-only images cannot
yet be independently audited here; see **Pre-submission blockers** below.

## Results and honest scope

On the separate 2,000-image calibration split, the selected temperature reached
96.8% clean accuracy and 95.75% accuracy across the combined clean/transformed
calibration set at threshold 0.5 (ROC-AUC 0.9935 for the combined set). These
numbers are calibration diagnostics, not final-test evidence.

The compact exact-transformation summary below uses mixed-condition calibration
and the neutral 0.5 probability threshold. Each transformed calibration image
is assigned to one exact challenge severity; family rows macro-average those
cells.

| Calibration condition | Accuracy | ROC-AUC |
|---|---:|---:|
| Clean | 96.80% | 99.67% |
| JPEG (Q90/Q70/Q50/Q30) | 96.83% | 99.49% |
| Gaussian blur (σ 0.5/1.0/2.0) | 94.49% | 99.00% |
| Resize (0.5×/0.25× then upscale) | 90.23% | 97.20% |
| Gaussian noise (σ 0.02/0.05/0.10) | 94.74% | 99.21% |
| Color (0.8×/1.2×) | 93.98% | 98.86% |
| Center crop (80%) | 96.99% | 99.01% |
| All transformed calibration images | 94.70% | 98.96% |

As an external demonstration-only stress check, the locked checkpoint scored
1,000 supplied WildFake images (500 COCO-val2017 real and 500 DALL·E 3 Advanced
synthetic) at threshold 0.5:

| Metric | Result |
|---|---:|
| Accuracy / balanced accuracy | 86.00% |
| ROC-AUC | 93.59% |
| Average precision | 94.73% |
| Precision / recall / F1 | 85.02% / 87.40% / 86.19% |
| Confusion matrix (TN / FP / FN / TP) | 423 / 77 / 63 / 437 |

The external evaluation command does not fit weights, calibration, or a
threshold on this set. It is a useful end-to-end stress check, but it contains
only one real source and one synthetic generator and therefore does not prove
broad unseen-generator generalization. Because the selected checkpoint's
training manifest is currently missing, this checkout cannot certify zero
image overlap with its adapted training corpus. The evidence trail and rejected
approaches are in [`docs/EXPERIMENTAL_LOG.md`](docs/EXPERIMENTAL_LOG.md).

## Reproducing the submitted pipeline

Training data uses an audited CSV split manifest with `path`, `label`, and
`split` information. The image root can contain source-specific subdirectories;
the manifest is authoritative. Do not use the challenge's supplied
COCO-val2017/DALL·E Advanced demonstration corpus for training.

### 1. Extract frozen features

```bash
uv run aigc-extract \
  --data-dir data/mixed_wildfake_66k \
  --split-manifest data/mixed_wildfake_66k/split_manifest.csv \
  --combined-output artifacts/mixed_wildfake_66k/laplacian_fft_features.pt \
  --laplacian-output artifacts/mixed_wildfake_66k/laplacian_features.pt \
  --augmentation-repeats 3 --batch-size 32 --seed 42 --device auto
```

This stage writes train, model-selection, calibration, and robust
model-selection features. Its cache explicitly records
`test_features_extracted: false`.

### 2. Train from the locked 40K initializer

```bash
uv run aigc-train \
  --data-dir data/mixed_wildfake_66k \
  --split-manifest data/mixed_wildfake_66k/split_manifest.csv \
  --cache artifacts/mixed_wildfake_66k/laplacian_fft_features.pt \
  --output artifacts/mixed_wildfake_66k/diverse_initialized_40k.pt \
  --forensic-mode laplacian_fft \
  --initialize-from-checkpoint artifacts/mixed_40k_balanced_consistency_w01_calibrated.pt \
  --augmentation-policy balanced --augmentation-repeats 3 \
  --consistency-weight 0.1 --modality-dropout 0.1 --fft-dropout 0.15 \
  --robust-validation-weight 0.7 --learning-rate 1e-4 \
  --head-batch-size 256 --epochs 50 --patience 8 --seed 42 --device auto
```

### 3. Fit calibration on the separate calibration split

```bash
uv run aigc-calibrate \
  --data-dir data/mixed_wildfake_66k \
  --split-manifest data/mixed_wildfake_66k/split_manifest.csv \
  --checkpoint artifacts/mixed_wildfake_66k/diverse_initialized_40k.pt \
  --feature-cache artifacts/mixed_wildfake_66k/laplacian_fft_features.pt \
  --output-checkpoint artifacts/diverse_initialized_40k_calibrated.pt \
  --output-report artifacts/mixed_wildfake_66k/calibration.json \
  --selection mixed --seed 42 --device auto
```

### 4. Validate, then perform one locked test

```bash
# Candidate/diagnostic evaluation: safe to repeat.
uv run aigc-severity \
  --data-dir data/mixed_wildfake_66k \
  --split-manifest data/mixed_wildfake_66k/split_manifest.csv \
  --checkpoint artifacts/diverse_initialized_40k_calibrated.pt \
  --output artifacts/model_selection_severity.json \
  --split model_selection --device auto

# Run once only after the checkpoint and threshold are frozen.
uv run aigc-severity \
  --data-dir data/mixed_wildfake_66k \
  --split-manifest data/mixed_wildfake_66k/split_manifest.csv \
  --checkpoint artifacts/diverse_initialized_40k_calibrated.pt \
  --output artifacts/final_test_severity.json \
  --split test --allow-test --device auto
```

## Repository structure

```text
aigc_detector/
├── data.py              # split loading and deterministic transformations
├── model.py             # frozen encoders, forensic views, and fusion head
├── features.py          # batched feature extraction
├── extract.py           # leakage-safe cache CLI
├── train.py             # robust head training and checkpoint selection
├── calibrate.py         # separate-split temperature/threshold calibration
├── evaluate.py          # robustness and external evaluation
├── severity.py          # exact challenge condition matrix
├── predict.py           # required calibrated JSON inference
├── dashboard.py         # testable dashboard data contract
├── analysis/            # optional audits and external/response evaluation
├── tooling/             # streamed-cache support for data preparation
└── experiments/         # ablations and mixture training, not submission runtime
app.py                   # Streamlit result visualizer
infer.sh                 # selected-model inference launcher
images/.gitkeep          # preserves the empty demo input folder in Git
scripts/                 # local dataset preparation and hardware utilities
├── download_pretrained_models.py  # fetches frozen inference backbones
└── kaggle/                       # Kaggle upload, extraction, training, analysis
tests/                   # unit and contract tests
docs/EXPERIMENTAL_LOG.md # concise experiment decisions and limitations
AGENTS.md                 # detailed append-only experiment/audit handoff
```

Production code is documented at the point where non-obvious preprocessing,
leakage controls, calibration, or checkpoint compatibility is enforced.
Experimental `e*` modules are intentionally isolated from the submission path.
The `.gitkeep` file is intentionally empty: Git does not track directories, so
it ensures `images/` exists immediately after cloning.

## Error analysis and trade-offs

On the 1,000-image external demonstration check, the model produced 77 false
positives among COCO real images and 63 false negatives among DALL·E images.
The dashboard intentionally presents a continuous probability rather than a
definitive provenance label: lowering the threshold catches more AIGC but
increases false accusations; raising it protects authentic images but misses
more synthetic content. The calibrated operating threshold (`0.7572`) targets
balanced calibration performance, while the external table retains `0.5` so
probability quality is visible without hiding threshold effects.

The external evaluator records the five highest-confidence false positives and
false negatives for qualitative review without using those labels for tuning.
Extreme resize and blur remain the clearest quantitative weak points: both
suppress local/frequency evidence, while semantics alone cannot establish
provenance. Moderation use should therefore keep a human-review band around the
operating threshold and must not treat this score as proof.

## Notable explorations we did not select

- Training Laplacian+FFT fusion from scratch improved blur but became brittle to
  noise. Zero-initializing the new FFT columns from a strong Laplacian solution
  was consistently better.
- A learned Laplacian/FFT expert gate discovered image-source identity more
  strongly than corruption severity and underperformed the simple fused head.
- A three-expert semantic/Laplacian/FFT ensemble helped a few severe conditions
  but was weaker on the mean robustness objective.
- A provisional 100K checkpoint was strong on shared-source diagnostics, but
  overlap with the 40K evaluation rows could not be ruled out without its
  original manifest, so that comparison was not treated as held-out evidence.

## Verification

```bash
uv run --frozen python -m unittest tests.test_core tests.test_dashboard
uv run --frozen python -m compileall -q aigc_detector app.py scripts
bash -n infer.sh
```

These production/core and dashboard checks are platform-independent. The full
research suite can be run with `python -m unittest discover -s tests -v` on its
intended POSIX/Kaggle environment; a few Phase 2/Phase 3 tests assume those
paths and are not native-Windows checks.

## Limitations and future improvements

- Build a properly licensed generator- and real-source-disjoint WildFake split.
  This is because random same-source splits cannot establish unseen-generator
  performance.
- Compare global resize with validation-only texture-crop aggregation for
  eligible high-resolution images, retaining global resize for small images.
- Investigate fine-tuning only the final EfficientNet block after a frozen
  generator-held-out baseline justifies the added compute.
- Expand error review by generator, source, resolution, subject matter, and
  exact transformation before setting moderation policies.
- Train over a larger dataset. Current training efforts only went up to
  100k images due to limited compute and time. Training the model on larger
  volume and more diverse data may increase robustness.  

## Team contributions

| Team member | Repository contribution |
|---|---|
| See Jay | Core detector, robustness training, and external evaluation |
| Xuan Shan | Scaled 40K/100K training, GPU handoff, benchmarking, and documentation |
| Max | Phase-two/phase-three research experiments and validation tooling |
| Xinnan | FFT experiments and early detector implementation |
| Yu Bin | Dashboard/inference workflow, evaluation reporting, and demo video |

## Submission checklist

- Working inference produces the exact `image_path` / `pred` JSON contract.
- The dashboard visualizes that JSON and the corresponding images end to end.
- The selected checkpoint is visible to Git and all required code is present;
  both still need to be committed and pushed to the public repository.
- Code is separated into production and experimental paths and comments explain
  the non-obvious technical decisions.
- This README documents setup, architecture, reproduction, results, limitations,
  and repository structure.
- The Devpost submission still needs the written project description and public
  three-minute YouTube demo video required by the event-level deliverables.

### Pre-submission blockers

1. Recover the exact `mixed_wildfake_66k` split manifest and dataset provenance.
   Compare decoded-pixel hashes against all 13,841 supplied COCO-val2017/DALL·E
   Advanced images. If any demonstration image entered train, model-selection,
   or calibration, this checkpoint is not challenge-compliant and must not be
   submitted.
