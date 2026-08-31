# Experimental modules

These modules implement ablations and post-selection research. They are kept
inside a separate package so the submitted path remains easy to identify:
`extract` → `train` → `calibrate` → `evaluate`/`predict`.

The `e*` names match the historical experiment identifiers in
[`docs/EXPERIMENTAL_LOG.md`](../../docs/EXPERIMENTAL_LOG.md). None is required
to load or run `diverse_initialized_40k_calibrated.pt`. Mixture-head training
also lives here because it was evaluated and rejected for the submitted model.
