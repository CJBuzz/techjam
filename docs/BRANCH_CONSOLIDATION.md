# Branch consolidation

The project branches were consolidated into `main` on 2026-08-31.

## Integrated branches

- `testing` (`3334c79`): WildFake-diverse training preparation, external-generator evaluation, ensemble inference, submission packaging, and benchmark documentation.
- `track5-experiments` (`0ec016b`): Track 5 Phase-2/Phase-3 research modules, Kaggle job configurations, promotion logic, controllers, and tests.
- `origin/front-end` (`bc7b55a`): Streamlit interface, batch/directory inference utilities, launch scripts, and UI documentation.
- `origin/sd15_vae_fft` (`ea2c73d`): imported under `research/sd15_vae_fft/` because the branch has unrelated Git history and conflicting root-level project files. Its reports, configurations, scripts, tests, recorded metrics, and notebook are preserved without changing the production package layout.

Generated local environments (`.python-runtime/` and `.tools/`) are intentionally excluded. Datasets, feature caches, model checkpoints, and generated predictions remain ignored and are not part of source control.

## Conflict resolution

The only content conflict was `aigc_detector/predict.py`. The consolidated version retains:

- calibrated probability JSON output using `image_path` and `pred`;
- explicit checkpoint existence checks;
- positive batch-size validation;
- skipping unreadable images without misaligning paths and probabilities;
- strict path/probability pairing.
