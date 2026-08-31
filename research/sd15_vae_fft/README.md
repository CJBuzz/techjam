# SD1.5 VAE FFT robustness experiment

Companion to `sb15_fft`, using only the pinned Stable Diffusion 1.5 VAE. Each image is canonically cropped to 512×512, encoded with `latent_dist.mode()`, and decoded once. The UNet, text encoder, tokenizer, and diffusion schedulers are never loaded.

The experiment intentionally imports the already-tested deterministic perturbation, preprocessing, hashing, and spectral-metric implementations from `../sb15_fft/sb15_fft/` to prevent scientific drift. It uses the locked environment at `../sb15_fft/` and the same frozen, label-blind extraction manifest.

```bash
uv run --project sb15_fft python sd15_vae_fft/scripts/run_severity_matrix.py --mode smoke
uv run --project sb15_fft python sd15_vae_fft/scripts/run_severity_matrix.py --mode pilot
uv run --project sb15_fft python sd15_vae_fft/scripts/run_severity_matrix.py --mode production --resume
```

Labels are not read by extraction. Pilot or production scoring must be a separate, explicitly acknowledged command.

