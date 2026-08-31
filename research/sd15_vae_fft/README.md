# SD1.5 VAE residual FFT research

This folder preserves an independent, training-free experiment. It is not
imported by or required for the submitted `aigc_detector` pipeline.

## Question and method

The study tested whether high-frequency energy in Stable Diffusion 1.5 VAE
reconstruction residuals distinguishes COCO real images from advanced DALL-E 3
images, and whether the signal is more robust than raw-pixel FFT features. Each
image was deterministically perturbed, resized and center-cropped to 512×512,
encoded with the mode of the pinned VAE posterior, and decoded once. No UNet,
text encoder, tokenizer, or diffusion scheduler was used.

The signed reconstruction residual was averaged across channels and transformed
with an orthonormal 2D FFT. The primary score is high-band mean energy divided
by total low-, middle-, and high-band energy. A post-hoc clean threshold of
`0.011662438977509737` was applied unchanged to every condition. AUROC is the
primary endpoint because this threshold was not fitted on held-out calibration
data.

## Selected full-subset results

The frozen manifest contains 11,841 images. Production covered four
predeclared conditions; 2,000 paired, class-stratified bootstrap samples used
NumPy seed 20260830.

| Condition | VAE residual AUROC | AP | Balanced accuracy |
|---|---:|---:|---:|
| Clean | 0.999442 | 0.999785 | 0.999107 |
| JPEG Q30 | 0.999997 | 0.999999 | 0.999235 |
| Blur 2 | 0.931105 | 0.955471 | 0.500255 |
| Resize 0.25× | 0.995118 | 0.997339 | 0.502869 |

Blur changed VAE AUROC by −0.06834 (95% CI −0.07339 to −0.06312), while
resize changed it by −0.00432 (95% CI −0.00541 to −0.00336). Fixed-threshold
balanced accuracy fell to chance because smoothing shifted nearly every fake
score below the clean threshold, even where ranking remained strong.

## Raw FFT control

Exact input identity was verified for the paired raw/VAE comparison.

| Condition | Raw FFT AUROC | VAE residual AUROC | VAE minus raw |
|---|---:|---:|---:|
| Clean | 0.975712 | 0.999442 | 0.023730 |
| JPEG Q30 | 0.978688 | 0.999997 | 0.021309 |
| Blur 2 | 0.434741 | 0.931105 | 0.496364 |
| Resize 0.25× | 0.491331 | 0.995118 | 0.503787 |

The result supports a VAE-residual-specific robustness advantage for blur and
resize, but only for this COCO/DALL-E comparison. It is not universal detector
evidence, and its clean threshold is post-hoc.

## Reproduction and artifacts

```bash
uv run --project sb15_fft pytest -q sd15_vae_fft/tests/test_raw_fft_experiment.py
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py \
  --mode production --only clean --only jpeg_q30 --only blur_2 \
  --only resize_0.25x --resume
uv run --project sb15_fft python sd15_vae_fft/scripts/evaluate_raw_vs_vae.py \
  --allow-label-read
```

Machine-readable results and the frozen protocol remain under `results/`:

- `metrics/selected_production_metrics.{json,csv}` — VAE residual results;
- `metrics/raw_fft_vs_vae_selected_production.{json,csv}` — paired control;
- `selected_evaluation_registry.json` and
  `raw_fft/selected_evaluation_registry.json` — frozen execution registries;
- `logs/` and `raw_fft/logs/` — smoke and production reports.

Configurations live in `configs/`, executable logic in `scripts/`, contract
tests in `tests/`, and the exploratory notebook remains `sd15_fft.ipynb`.
Production generated 35,523 reconstructions in 9,138 seconds (2.54 hours), with
zero retries and 5.84 GB peak reserved VRAM.
