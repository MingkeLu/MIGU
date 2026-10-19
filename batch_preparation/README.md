# Shared baseline preparation

Run `python -m batch_preparation` from the MIGU repository root. This uses
`baselines/config.yaml`, including its existing OWL+SAM endpoint and per-group
segmentation vocabularies. All baselines share this module for live preparation
and reuse its cached files with `--batch`.

```text
batch_preparation/captures/input/<group>/
  key_rgb/                 normalized RGB images
  key_depth/               millimetre depth images
  key_frames/              MediaPipe hand landmarks and fitted rays
  perception/              candidate masks, labels, scores, and boxes
  ground_truth_masks/      largest mask containing each annotated point
  visualization/           optional segmentation and hand overlays
  camera_intrinsics.txt
  instructions.json
  ground_truth.json
  manifest.json            frame timings, annotation status, and failures
```

Use `--input-dir`, `--subdirs`, or `--prepared-dir` to select the source and cache.
Use `--overwrite` to regenerate derived artifacts, `--mode segmentation` to skip
hand detection, and `--visualize` to save overlays. Generated artifacts are
reused by default. A live baseline run prepares the same structure under its
own `_prepared_input/<group>/` directory.
