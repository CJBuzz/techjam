# Raw-image FFT versus VAE residual FFT

## Executive summary

Exact E0 input identity was achieved for all four conditions. The conclusion below is based on paired 11,841-image evaluation. Raw spectral separation explains some of the clean and JPEG performance, but the VAE residual adds substantial robustness under blur and resize: raw AUROC falls to 0.435 and 0.491 while VAE AUROC remains 0.931 and 0.995.

## Main paired AUROC results

| Condition | Raw FFT AUROC | VAE residual FFT AUROC | VAE minus Raw | Raw change from clean | VAE change from clean | Difference in degradation |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.9757120368 | 0.9994416117 | 0.0237295748 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| jpeg_q30 | 0.9786882476 | 0.9999971298 | 0.0213088822 | 0.0029762108 | 0.0005555181 | -0.0024206927 |
| blur_2 | 0.4347408308 | 0.9311046094 | 0.4963637786 | -0.5409712060 | -0.0683370022 | 0.4726342038 |
| resize_0.25x | 0.4913309485 | 0.9951183357 | 0.5037873871 | -0.4843810883 | -0.0043232760 | 0.4800578123 |

## Paired 95% confidence intervals

| Condition | Raw AUROC CI | VAE AUROC CI | VAE minus Raw CI | Difference-in-degradation CI |
|---|---:|---:|---:|---:|
| clean | [0.973439, 0.978038] | [0.998969, 0.999807] | [0.021464, 0.025946] | [0, 0] |
| jpeg_q30 | [0.976574, 0.980809] | [0.999994, 0.999999] | [0.019190, 0.023424] | [-0.002852, -0.001914] |
| blur_2 | [0.424223, 0.445275] | [0.926005, 0.936251] | [0.485177, 0.508247] | [0.462097, 0.483811] |
| resize_0.25x | [0.480535, 0.502307] | [0.994031, 0.996127] | [0.492547, 0.514772] | [0.469805, 0.490346] |

The 2,000 class-stratified paired resamples used NumPy seed 20260830 and the same sampled indices for every method and condition. Average precision, fixed-clean-threshold balanced accuracy, confusion counts, and their intervals are retained in the JSON and CSV outputs.

## Interpretation

JPEG Q30 slightly improves both methods; its negative difference in degradation means raw FFT improves slightly more, although VAE remains better in absolute AUROC. For Pillow GaussianBlur radius 2 and source-resolution 0.25× down/up-resize, the paired intervals strongly support a VAE-specific robustness advantage. The evidence is therefore mixed by corruption type, but decisively favors the VAE residual representation for blur and resize.

## Protocol and provenance

The frozen set contains 3,998 real COCO and 7,843 advanced DALLE3 images. Images were EXIF-transposed, converted to RGB, perturbed at source resolution, then aspect-resized and center-cropped to 512×512. Raw FFT is the high-band mean divided by the sum of low-, mid-, and high-band means. Higher scores are prespecified as fake-positive.

The historical notebook AUROC 0.989585 used 13,841 differently preprocessed images and is not included in the paired table.

Raw clean-oracle threshold: `4.5911116103525274e-05`. VAE clean-oracle threshold: `0.011662438977509737`. Both were fixed across perturbations. Confidence intervals and all secondary metrics are in the JSON artifact.

## Input-identity audit

All 47,364 source, transformed, canonical, and derived-seed comparisons matched; no rows were excluded.

## Limitations

This comparison is specific to COCO real versus advanced DALLE3 fake imagery. Thresholds are post-hoc clean oracles, while AUROC is the primary endpoint.

## Reproducibility

```bash
uv run --project sb15_fft pytest -q sd15_vae_fft/tests/test_raw_fft_experiment.py
uv run --project sb15_fft python sd15_vae_fft/scripts/run_raw_fft_severity.py --mode production --only clean --only jpeg_q30 --only blur_2 --only resize_0.25x --resume
uv run --project sb15_fft python sd15_vae_fft/scripts/evaluate_raw_vs_vae.py --allow-label-read
```

Artifact SHA-256 values:

- Frozen raw registry: `04f8f1fb750860e2ee11b93e00dbd0e917683184cff8a2ec177a817fd16d090e`
- Metrics JSON: `22e14e0a8a21620b40697deddf1128eb8977ee308b105de950ad3a212271a71d`
- Metrics CSV: `c86b81f1438179c19617ad599bd44fb7ef461e8228b321aba65061807c6cb038`
- Production report: `90d73d4c6b43390b324289acb9328d8acdaec4218673a93c54b5c91e7a88e4c1`
- Runner: `7626e87f1e75e5bb0d13354ad13c86c1aff90feb26906bc7efe5c69b60783b67`
- Evaluator: `e759f8341c6e7301ff978297ba7a60ed5947a3b2a4b919c85332e60ce06e8368`

## Divergence ledger

No protocol divergences.
