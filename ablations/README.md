# MIGU ablations

See the [repository guide](../README.md#ablations) for definitions, all nine
commands, cached inference, semantic-result reuse, and automatic analysis.

```bash
python -m ablations --help
python -m ablations without-semantics --batch --analyze
python -m ablations ray-angle --reuse-result-dir output/migu/<timestamp> --analyze
```

The common inference and fusion code is `migu.pipeline.process_frame`. Geometry
replacements select existing `migu.gesture` modes. Baseline replacements reuse
`baselines.semantic` and `batch_preparation.shared`; their adapters live in
`semantic.py`. `owlv2.py` preserves the original `main_new` query prompt and
detector protocol, with separately configured OWL and SAM endpoints.
