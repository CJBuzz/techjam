# Results

Production extraction was intentionally limited by user request. Clean, JPEG Q30, blur 2, and resize 0.25× are complete for all 11,841 images; the other robustness cells are incomplete or unrun.

## Selected full-subset results

The primary score is `fft_high_bandmean_ratio`. One post-hoc clean-oracle threshold (`0.011662438977509737`) was selected on all 11,841 clean rows and applied unchanged to the three completed perturbations.

| Condition | AUROC | AP | Accuracy | Balanced accuracy | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.999442 | 0.999785 | 0.998818 | 0.999107 | 0.999107 | 3998 / 0 / 14 / 7829 |
| JPEG Q30 | 0.999997 | 0.999999 | 0.998987 | 0.999235 | 0.999234 | 3998 / 0 / 12 / 7831 |
| blur 2 | 0.931105 | 0.955471 | 0.337978 | 0.500255 | 0.001019 | 3998 / 0 / 7839 / 4 |
| resize 0.25× | 0.995118 | 0.997339 | 0.341441 | 0.502869 | 0.011410 | 3998 / 0 / 7798 / 45 |

JPEG Q30 preserves and slightly improves ranking relative to clean. Blur 2 materially degrades ranking (paired AUROC change −0.06834; 95% CI −0.07339 to −0.06312). Resize 0.25× causes a smaller but conclusive ranking loss (−0.00432; 95% CI −0.00541 to −0.00336). Both smoothing conditions shift almost every fake score below the clean threshold, so fixed-threshold balanced accuracy falls to approximately chance despite strong AUROC.

These four cells are complete, but they do not constitute the originally planned 16-condition matrix. The threshold is post-hoc full-clean oracle rather than held out.

## Exploratory 200-image pilot

The pilot contains 68 real and 132 fake images. Labels were read only after a dedicated pilot registry was frozen. The primary score is `fft_high_bandmean_ratio` with higher = fake. A post-hoc clean-oracle threshold of `0.011914256261661649` was selected on the clean pilot and applied unchanged to all perturbations.

| Condition | AUROC | AP | Accuracy | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| clean | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| jpeg_q90 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| jpeg_q70 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| jpeg_q50 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| jpeg_q30 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| blur_0.5 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| blur_1 | 0.9978 | 0.9990 | 0.940 | 0.9545 | 0.9524 |
| blur_2 | 0.9503 | 0.9675 | 0.340 | 0.5000 | 0.0000 |
| resize_0.5x | 0.9990 | 0.9995 | 0.805 | 0.8523 | 0.8267 |
| resize_0.25x | 0.9938 | 0.9973 | 0.340 | 0.5000 | 0.0000 |
| noise_0.02 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| noise_0.05 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| noise_0.1 | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| color_pm10pct | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| color_pm20pct | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 |
| crop_0.8x | 1.0000 | 1.0000 | 0.995 | 0.9962 | 0.9962 |

These are exploratory results on the first 200 of 11,841 rows, not the final evaluation. The clean threshold was optimized on the same pilot and is not held out. Strong blur and resizing shift scores below that clean threshold even where AUROC remains high.
