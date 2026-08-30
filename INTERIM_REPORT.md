# Stable Diffusion 1.5 VAE Residual Spectra for AI-Image Detection

## Interim robustness report

**Status:** pilot complete; full 11,841-image extraction in progress  
**Dataset scope:** COCO real images versus advanced DALL-E 3 fake images from the existing WildFake evaluation subset  
**Primary statistic:** high-frequency band-mean ratio of the SD1.5 VAE reconstruction residual

## Executive summary

This experiment tests a simple, training-free detector based on the reconstruction behavior of the Stable Diffusion 1.5 variational autoencoder (VAE). An image is encoded using the mode of the VAE posterior and decoded once. The difference between the original image and its reconstruction is transformed into the frequency domain, where the relative energy in the high-frequency residual band becomes the detection score. The Stable Diffusion UNet, text encoder, tokenizer, and diffusion schedulers are not loaded or used.

The method builds on an earlier complete evaluation in which the SD1.5 VAE high-frequency residual ratio was the strongest tested detector on the 11,841-image WildFake subset: AUROC 0.9451, average precision 0.9710, and balanced accuracy 0.8788. That study did not systematically measure robustness to post-processing. The present experiment applies an exact, deterministic matrix of JPEG compression, blur, resizing, noise, color adjustment, and cropping.

On the new 200-image pilot, the VAE-only FFT score perfectly ranked real and fake images under the clean condition and under most perturbations. Strong blur and resizing produced the important exception. Class ranking remained high—AUROC 0.9503 for blur radius 2 and 0.9938 for 0.25× resizing—but the score scale shifted below the clean-image threshold. Consequently, fixed-threshold balanced accuracy fell to 0.50 for both conditions. This is primarily a calibration failure under smoothing, with an additional ranking degradation under strong blur.

These figures are interim and exploratory. The pilot consists of the first 200 evaluation rows, and its threshold was selected and assessed on the same clean pilot. It must not be presented as the final 11,841-image result.

## 1. Motivation and prior work

Reconstruction-based AI-image detection rests on the idea that a generative model may reconstruct synthetic and natural images differently. Pixel error alone is often a weak signal: it mixes semantic reconstruction error, color shifts, edges, texture, and compression artifacts. Frequency analysis separates some of these effects and can reveal systematic differences in the fine-scale residual structure.

The preceding reconstruction-error study evaluated several training-free residual statistics and learned baselines on the same WildFake subset of 3,998 COCO real and 7,843 advanced DALL-E 3 fake images. Its strongest complete result was the deterministic SD1.5 VAE high-frequency residual ratio:

| Earlier complete method | AUROC | Average precision | Balanced accuracy |
|---|---:|---:|---:|
| SD1.5 VAE high-frequency residual ratio | 0.9451 | 0.9710 | 0.8788 |
| SID-trained two-layer MLP | 0.7514 | 0.8708 | 0.7097 |
| Joint-source two-layer MLP | 0.7382 | 0.8579 | 0.6923 |
| SD1.5 VAE gradient MSE | 0.6736 | 0.8067 | 0.6072 |
| SD1.5 VAE pixel MSE | 0.6608 | 0.7977 | 0.6048 |

For the earlier VAE FFT result, the 95% confidence interval was 0.9409–0.9490 for AUROC and 0.9683–0.9736 for average precision. Its threshold, 0.042689, was selected on a separate 2,000-image calibration split. The final confusion matrix was TN 3,558, FP 440, FN 1,038, and TP 6,805.

An earlier diffusion reconstruction arm also used 20 DDIM inversion and 20 denoising steps. It reached only partial coverage and was not eligible for final scoring, although its available calibration subset showed unusually strong frequency separation. A subsequent 200-image robustness pilot confirmed that the DDIM20 feature was sensitive to strong blur and downscaling and required about 72.8 minutes for 3,200 reconstructions. These results motivated testing whether the much simpler VAE-only reconstruction preserves the useful frequency signal at lower computational cost.

The earlier complete VAE arm used 256-pixel canonical inputs, whereas the present robustness experiment uses 512×512 inputs to match the companion DDIM20 protocol. Its historical threshold is therefore not transferred to this experiment. The present work is a robustness follow-up, not a new pristine held-out test. The WildFake labels were opened during prior work, and the pilot threshold is explicitly post hoc.

## 2. Method

### 2.1 Data and image order

The experiment uses the existing WildFake evaluation manifest in its frozen row order. It contains 11,841 unique images: 3,998 COCO real images and 7,843 advanced DALL-E 3 fake images. No SID_Set or CIFAKE images are used.

The label-blind extraction manifest records each ordinal, image ID, relative and absolute source path, deterministic perturbation seed key, source byte length, and source SHA-256. The source inventory contains 11,503 RGB and 338 RGBA files; all inputs are converted to RGB before perturbation and reconstruction.

### 2.2 Deterministic perturbation matrix

Every image is processed under the following 16 conditions in fixed order:

1. clean
2. JPEG quality 90, 70, 50, and 30
3. Gaussian blur radius 0.5, 1, and 2
4. bilinear downscale/upscale at 0.5× and 0.25×
5. Gaussian noise with sigma 0.02, 0.05, and 0.10
6. independent ±10% and ±20% brightness, contrast, and saturation changes
7. centered 0.8× crop restored with bicubic interpolation

Randomized operations use global seed 42 and a SHA-256-derived seed containing the absolute source path, operation, and severity. Noise uses a local NumPy generator and color adjustment uses a local Python generator, so transforms do not mutate global random state.

### 2.3 Canonical preprocessing

After perturbation, each RGB image is resized so that its shorter side is 512 pixels. Dimensions use Python's `round` rule, resizing uses Pillow bicubic interpolation, and a centered 512×512 crop is taken. The resulting unsigned 8-bit RGB array is converted to a CHW float32 tensor in `[0,1]`.

### 2.4 VAE reconstruction

The experiment pins `stable-diffusion-v1-5/stable-diffusion-v1-5` to revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. Only its `AutoencoderKL` component is loaded, in FP16 and evaluation mode.

For canonical input `x`, reconstruction is:

1. Map `x` from `[0,1]` to `[-1,1]`.
2. Encode it with the VAE.
3. Take `latent_dist.mode()` rather than sampling.
4. Decode that latent exactly once.
5. Convert the clamped float32 decoder output back to `[0,1]`.

The VAE scaling factor is not applied because the latent never enters the diffusion UNet. In particular, this is not diffusion inversion: there is no prompt, tokenizer, text embedding, UNet evaluation, noise schedule, or denoising loop.

### 2.5 Spectral score

Let `x'` be the VAE reconstruction and define the signed RGB residual

```text
r = x - x'
```

The residual is averaged across RGB channels, transformed with an orthonormal two-dimensional real FFT, and squared to obtain spectral energy. Radial frequency is normalized by the Nyquist-corner radius. Three fixed bands are used:

- low: radius below 0.25
- middle: radius from 0.25 to below 0.50
- high: radius at least 0.50

The primary score is

```text
mean(high-band energy)
────────────────────────────────────────────────────────
mean(low-band energy) + mean(mid-band energy) + mean(high-band energy)
```

Higher score is frozen a priori as evidence for the fake class. MSE, MAE, gradient MSE, and a coefficient-sum FFT ratio are retained only as prespecified secondary diagnostics; they do not determine the primary result.

### 2.6 Evaluation

The pilot contains the first 200 frozen-manifest rows: 68 real and 132 fake images. Its primary metrics are AUROC and average precision. For deployment-style metrics, one threshold was chosen to maximize balanced accuracy on the clean pilot and then applied unchanged to every perturbed condition. The selected threshold is `0.011914256261661649`.

Balanced accuracy is the mean of fake recall and real specificity, so each class contributes equally despite the pilot's unequal class counts. Confidence intervals and changes from clean use 2,000 paired, class-stratified bootstrap replicates with seed 20260830. The same resampled indices are reused across conditions.

Because the threshold was fitted and evaluated on the same clean pilot, all threshold-dependent figures are **post-hoc clean-oracle** results, not held-out estimates.

## 3. Engineering validation and cost

The smoke test processed 16 ordinal-spanning images under all 16 conditions, producing 256 valid reconstructions in 64.8 seconds. The pilot produced 3,200 valid reconstructions in 821.6 seconds, or 13.7 minutes. Both completed with zero retries.

All pilot score rows were checked for finite values and exact manifest order. Every pilot reconstruction was reopened and verified as an RGB 512×512 PNG with matching byte length and SHA-256.

| Quantity | Pilot result |
|---|---:|
| Batch size | 4 |
| Peak reserved VRAM | 5.84 GB |
| Reconstructions | 3,200 |
| Elapsed time | 821.6 s |
| Retries/failures | 0 |
| Projected production time | 48,643 s / 13.51 h |
| Projected production PNG storage | 73.65 GB |

The projected workspace footprint, including a 20% output margin, is approximately 166.3 GiB and passes the experiment's 180 GiB ceiling.

## 4. Pilot results

| Condition | AUROC | AP | Accuracy | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| clean | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| JPEG Q90 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| JPEG Q70 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| JPEG Q50 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| JPEG Q30 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| blur 0.5 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| blur 1 | 0.9978 | 0.9990 | 0.940 | 0.9545 | 0.9524 |
| blur 2 | 0.9503 | 0.9675 | 0.340 | 0.5000 | 0.0000 |
| resize 0.5× | 0.9990 | 0.9995 | 0.805 | 0.8523 | 0.8267 |
| resize 0.25× | 0.9938 | 0.9973 | 0.340 | 0.5000 | 0.0000 |
| noise 0.02 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| noise 0.05 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| noise 0.10 | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| color ±10% | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| color ±20% | 1.0000 | 1.0000 | 1.000 | 1.0000 | 1.0000 |
| crop 0.8× | 1.0000 | 1.0000 | 0.995 | 0.9962 | 0.9962 |

Clean pilot classification was perfect: TN 68, FP 0, FN 0, TP 132. JPEG compression at every tested quality, mild blur, Gaussian noise, and both color severities produced the same perfect confusion matrix. Cropping introduced one false negative.

Blur radius 1 retained near-perfect ranking but moved 12 fake images below the clean threshold, yielding TN 68, FP 0, FN 12, and TP 120. Resize 0.5× similarly retained AUROC 0.9990 but moved 39 fakes below threshold, yielding balanced accuracy 0.8523.

The strongest shifts occurred under blur radius 2 and resize 0.25×. In both cells, the unchanged clean threshold classified every image as real: TN 68, FP 0, FN 132, TP 0. Ordinary accuracy was therefore 0.34—the real-class prevalence—while balanced accuracy and F1 fell to 0.50 and 0.0. The ranking information was not completely lost: resize 0.25× retained AUROC 0.9938, whereas blur radius 2 fell more materially to AUROC 0.9503.

Selected 95% pilot bootstrap intervals are:

| Condition | AUROC (95% CI) | Balanced accuracy (95% CI) |
|---|---:|---:|
| clean | 1.0000 (1.0000–1.0000) | 1.0000 (1.0000–1.0000) |
| blur 1 | 0.9978 (0.9923–1.0000) | 0.9545 (0.9280–0.9773) |
| blur 2 | 0.9503 (0.9135–0.9805) | 0.5000 (0.5000–0.5000) |
| resize 0.5× | 0.9990 (0.9967–1.0000) | 0.8523 (0.8144–0.8939) |
| resize 0.25× | 0.9938 (0.9826–1.0000) | 0.5000 (0.5000–0.5000) |

## 5. Interpretation

Three findings stand out.

First, the VAE-only residual is computationally attractive. It removes 40 UNet evaluations per image relative to DDIM20 and reduced pilot runtime from 72.8 minutes to 13.7 minutes while using less peak VRAM.

Second, its pilot ranking is remarkably robust. JPEG compression down to quality 30, additive noise up to sigma 0.10, color changes, cropping, and mild blur did not reduce pilot AUROC. This result is consistent with a source-dependent residual-frequency difference that survives several common transformations.

Third, rank robustness does not imply threshold robustness. Blur and resizing remove high-frequency content and shift both classes' absolute scores downward. A threshold calibrated on clean images can therefore fail catastrophically even while real and fake images remain correctly ordered. This matters operationally: recalibrating a threshold for every unknown post-processing pipeline is generally unavailable in deployment.

Strong blur is more damaging than aggressive resizing in the pilot. Under resize 0.25×, AUROC remains 0.9938, suggesting a largely monotonic score shift. Under blur radius 2, AUROC drops to 0.9503, indicating both a scale shift and genuine deterioration in class separation.

## 6. Limitations

- The pilot covers only 200 images and uses the first 200 manifest rows rather than a newly randomized sample.
- The clean threshold is optimized on the same pilot used to report clean accuracy, making threshold-dependent metrics optimistic.
- Evaluation labels were already opened in prior work. This is a post-hoc follow-up, not a pristine held-out experiment.
- The dataset contrasts one real-image source family, COCO, with one generator family, advanced DALL-E 3. Source acquisition and processing differences may contribute to the signal.
- Perfect pilot scores should not be interpreted as universal AI-image detection. The earlier full clean evaluation, AUROC 0.9451, is a more realistic indication of expected clean performance.
- The coefficient masks in the real FFT are intentionally unweighted; the primary statistic gives equal weight to band means rather than to individual coefficients.
- Robustness to screenshots, sharpening, WebP, social-media transcoding, combined transformations, and unseen resize kernels remains untested.

## 7. Interim conclusion

The SD1.5 VAE high-frequency residual ratio remains a compelling training-free signal in this narrowly defined WildFake comparison. The pilot indicates excellent rank robustness under JPEG, noise, color manipulation, cropping, and mild blur. Its key weakness is clean-threshold instability after operations that suppress high frequencies, especially blur and downscaling.

The full production extraction is underway and will determine whether these patterns persist across all 11,841 images. Final claims should be based on full-subset AUROC, average precision, paired changes from clean, and the unchanged clean threshold, with the pilot retained only as interim evidence.

## Reproducibility artifacts

- Experiment configuration: `sd15_vae_fft/configs/experiment.yaml`
- Extraction runner: `sd15_vae_fft/scripts/run_severity_matrix.py`
- Frozen source manifest: `sb15_fft/results/manifests/frozen_extraction.parquet`
- Pilot metrics: `sd15_vae_fft/results/metrics/pilot_exploratory_metrics.json`
- Pilot registry: `sd15_vae_fft/results/pilot_evaluation_registry.json`
- Historical study: `overnight/summary.md` and `overnight/metrics.csv`
- Experiment configuration hash: `9199b8664536bcec9c0d68aa28a98911486cc282f1fdeadbaa639dcd9ac36db4`
