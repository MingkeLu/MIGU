# Reference baseline results

Each method contains 100 original frame results from its approved snapshot.
There are 99 frames with evaluable object ground truth per method.

| Method | Snapshot | Accuracy |
| --- | --- | --- |
| geometry | 20260904_211355_288487 | 54.55% |
| POMDP | 20260904_211412_985657 | 56.57% |
| GIRAF | 20260904_210731_539563 | 55.56% |
| VLM-Grid | 20260904_232949_197535 | 31.31% |
| VLM-PIVOT | 20260904_233025_651138 | 35.35% |
| VLM-SoM | 20260905_130441_477267 | 47.47% |

`metrics_results/` contains recomputed per-frame and aggregate CSV, JSON, and
plots. The five unchanged baselines were checked against the original metrics
implementation using the same prepared inputs; all numeric metrics matched.

VLM-SoM ranks the original main run's `semantic_probability` values only.
Gesture/fusion scores and posterior arrays are excluded. The original fusion
pipeline latency is preserved as `source_fusion_processing_time_seconds`, but
semantic-only latency is unknown and is excluded from processing-time metrics.
`provenance.json` records the original method names and source snapshots.

Grid/PIVOT frame directories also include their original prepared
`perception.npz` masks, so semantic replacement ablations can reuse each
snapshot without depending on an external preparation cache.

Recompute using matching prepared RGB-D inputs:

```bash
python -m baselines.metrics results/baselines \
  --input-root batch_preparation/captures/input
```
