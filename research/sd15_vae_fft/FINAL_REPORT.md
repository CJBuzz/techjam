# Final Report: SD1.5 VAE Residual FFT Robustness

## Scope

This training-free experiment evaluated whether the high-frequency reconstruction residual of the Stable Diffusion 1.5 VAE distinguishes COCO real images from advanced DALL-E 3 fake images in the 11,841-image WildFake evaluation subset. By user request, production was limited to four complete conditions: clean, JPEG quality 30, Gaussian blur radius 2, and 0.25× downscale/upscale.

## Method

Each image was converted to RGB, deterministically perturbed, resized and center-cropped to 512×512, encoded using the mode of the pinned SD1.5 VAE posterior, and decoded once. The UNet, text encoder, tokenizer, and diffusion schedulers were not used.

The signed reconstruction residual was averaged across color channels and transformed with an orthonormal 2D FFT. The primary score was the mean energy in the high-frequency band divided by the sum of the low-, middle-, and high-band mean energies. Higher scores were interpreted as fake.

One post-hoc clean-oracle threshold, `0.011662438977509737`, was selected on all clean evaluation rows and applied unchanged to every perturbation. Confidence intervals and changes from clean used 2,000 paired, class-stratified bootstrap replicates.

## Results

| Condition | AUROC | Average precision | Accuracy | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| clean | 0.999442 | 0.999785 | 0.998818 | 0.999107 | 0.999107 |
| JPEG Q30 | 0.999997 | 0.999999 | 0.998987 | 0.999235 | 0.999234 |
| blur 2 | 0.931105 | 0.955471 | 0.337978 | 0.500255 | 0.001019 |
| resize 0.25× | 0.995118 | 0.997339 | 0.341441 | 0.502869 | 0.011410 |

JPEG Q30 preserved the signal. Its ranking was effectively perfect, and its fixed-threshold performance matched clean performance.

Blur radius 2 caused a material ranking loss: AUROC changed by −0.06834 relative to clean, with a paired 95% confidence interval of −0.07339 to −0.06312. Only 4 of 7,843 fake images remained above the clean threshold.

Resize 0.25× caused a smaller but conclusive ranking loss: AUROC changed by −0.00432, with a paired 95% confidence interval of −0.00541 to −0.00336. Only 45 fake images remained above the clean threshold.

The low threshold accuracy under blur and resizing is primarily a calibration failure. These transformations shift scores downward, causing nearly every image to be classified as real even though AUROC shows that much of the real-versus-fake ordering remains intact.

## Execution

The three requested perturbation cells produced 35,523 reconstructions in 9,138 seconds (2.54 hours) with zero retries. Peak reserved VRAM was 5.84 GB. Each cell contains all 11,841 images in frozen manifest order.

## Conclusion

For this narrowly defined COCO-real versus advanced-DALL-E-3 comparison, the SD1.5 VAE high-frequency residual is a strong training-free ranking signal. It is robust to severe JPEG compression, remains highly discriminative after aggressive resizing, and weakens more substantially after strong blur. Its main operational weakness is that smoothing invalidates a decision threshold calibrated on clean images.

These findings apply only to the four completed conditions and this two-source WildFake subset. The threshold is post-hoc rather than held out, and the results should not be presented as universal AI-image detection performance or as completion of the original 16-condition matrix.

## Artifacts

- Metrics: `sd15_vae_fft/results/metrics/selected_production_metrics.json`
- Tabular metrics: `sd15_vae_fft/results/metrics/selected_production_metrics.csv`
- Frozen registry: `sd15_vae_fft/results/selected_evaluation_registry.json`
- Detailed results: `sd15_vae_fft/RESULTS.md`
