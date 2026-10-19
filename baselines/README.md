# Baselines

See the [evaluation guide](../EXPERIMENTS.md#baselines) for all six methods, live and
cached inference, segmentation configuration, and automatic analysis.

Shared public settings live in `config.yaml`. Method-specific parameters remain
under `methods.geometry`, `methods.POMDP`, `methods.GIRAF`, `methods.VLM-Grid`,
`methods.VLM-PIVOT`, and `methods.VLM-SoM`.

Example: `python -m baselines POMDP --batch --analyze`.
