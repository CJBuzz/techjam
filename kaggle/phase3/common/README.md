# Track-5 Phase-3 Kaggle runtime

Phase-3 kernels are private, offline GPU jobs. Attach project source, image manifests,
pretrained weights, and prior-job outputs as Kaggle inputs. Production launch uses
`torchrun --standalone --nproc_per_node=2 entrypoint.py --config config.json`; runtime
selection falls back to one CUDA GPU or CPU smoke mode.

The artifact contract and ranking policy live in `aigc_detector.phase3`. R1-R7
entrypoints stay thin and must not duplicate training implementation.
