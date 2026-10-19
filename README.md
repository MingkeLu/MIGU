# MIGU

This repository provides a simplified implementation of the methods described in
the paper [*MIGU: Multimodal Instruction Grounding under Uncertainty for
Manipulation Planning*](https://arxiv.org/pdf/2609.24995).
![Demo](assets/2_framework.png)

MIGU grounds natural-language robot instructions by combining a pointing-gesture
posterior with VLM-proposed, SAM-segmented candidate regions. It accepts either
capture folders or a prepared RGB-D dataset, detects a hand with MediaPipe, fits a
3-D pointing ray, evaluates the configured gesture model, and ranks the candidate
masks with pixel-wise VLM/gesture fusion.

## Installation

Python 3.10 is recommended. Create an isolated environment and install the
package from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

MIGU uses the legacy `mediapipe.solutions.hands` API and therefore pins
MediaPipe 0.10.21.

## Credentials and endpoints

Do not put a VLM API key or base URL in `config.yaml`. Configure both in the
system environment before running MIGU:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
```

`OPENAI_BASE_URL` may be omitted when using the default OpenAI endpoint.

## Perception service

MIGU uses the SAM service from
[`threefruits/perception_services`](https://github.com/threefruits/perception_services).
Install [pixi](https://pixi.sh/), clone that repository beside MIGU (or anywhere on
the SAM host), and start its SAM server:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/threefruits/perception_services.git
cd perception_services
pixi run sam-server
```

The service listens on port `4001` and loads `facebook/sam-vit-huge`. Its first run
creates the pixi environment and downloads the model weights, so startup can take
several minutes. Verify readiness before running MIGU:

```bash
curl http://127.0.0.1:4001/healthz
```

The default `sam.server_url` in `config.yaml` is `http://127.0.0.1:4001`. If the
service runs on another trusted machine, replace it with that host's private URL
and restrict port `4001` with a firewall. The service has no authentication and
returns Python-pickled NumPy masks, so only connect MIGU to a trusted deployment;
do not expose it directly to the public internet.

## Input data
![dataset](assets/8_benchmarks.png8_benchmarks.png)
The default `captures` layout is:

```text
captures/
  group_name/
    frame_id/
      rgb.png
      depth.npz
      metadata.json
      point.json
```

`metadata.json` supplies RGB intrinsics and depth metadata. Each `point.json`
must include the frame instruction in `sentence`; MIGU intentionally has no
fallback instruction.

## Running MIGU

Edit `config.yaml`, then run:

```bash
python3 -m migu --config config.yaml
```

The default output directory is `output/migu/<timestamp>`. Use `--subdirs` to
select capture groups, `--no-visualize` to skip diagnostic images, or
`--output-dir` to override the configured output root.

The public configuration intentionally contains only the stable controls:

- MediaPipe `min_detection_confidence` and `min_tracking_confidence`
- gesture `mode` (default: `gesture_posterior`)
- VLM `model` and `bbox_count`
- all current SAM settings

The ray-fitting and pixel-wise fusion parameters are fixed to the method defaults.
Alternative hand detectors and OWLv2-specific options are not part of this release.

## ROS bag and planning interface

The planning interface uses the same `config.yaml` as the batch pipeline. No
second grounding-tool configuration is maintained. ROS Noetic must be installed
under `/opt/ros/noetic`; the interface decodes standard ROS `Image` messages
directly, so a Python-3.8 `cv_bridge` build is not required by the Python-3.10
MIGU environment. Install the ROS Python extra in the MIGU environment with
`python3 -m pip install -e '.[ros]'`.

The interface connects the following topics:

| Topic | Type | Producer | Consumer |
| --- | --- | --- | --- |
| `/usr_instruction` | `std_msgs/String` | `migu-publish-instruction` | `migu-grounding-node` |
| `/ray_queries` | JSON in `std_msgs/String` | `migu-gesture-monitor` | `migu-grounding-node` |
| `/grounding_results` | JSON in `std_msgs/String` | `migu-grounding-node` | planner or test client |

Download the example recordings from the
[Google Drive bag folder](https://drive.google.com/drive/folders/157PIy2gKYoUREKCnXzNIIsYho62nqnrQ?usp=sharing)
before running the demo. The bag recordings are distributed separately and are
not included in the Git repository. Extract the download if necessary and place
the `.bag` files directly in `bags/` under the MIGU repository root. Ensure the
matching calibration file is available at `bags/camera_intrinsics.txt`, or pass
its location with `--intrinsics /path/to/camera_intrinsics.txt`.

For a bag recording, first start the perception-services SAM server as described
above. One MIGU command then starts roscore when necessary, checks the configured
SAM endpoint, launches the gesture monitor and grounding node, publishes the user
instruction, replays every supplied bag, waits for all detected pointing events,
and shuts down the processes it started:

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

migu-bag-demo bags/*.bag \
  --instruction "pick up this"
```

Bag timestamp gaps are preserved by `rosbag play`.

The gesture monitor reads MediaPipe's `min_detection_confidence` and
`min_tracking_confidence` directly from `config.yaml`. It identifies a pointing
episode from a geometrically straight index finger and a stabilized finger speed,
saves the synchronized RGB-D trigger frame, and includes the camera intrinsics and
3-D hand joints in `/ray_queries`. Pass another calibration file with
`--intrinsics`; the bundled bag command defaults to
`bags/camera_intrinsics.txt`.

Each run is stored below `output/planning_interface/<timestamp>/`:

```text
grounding_inputs/   synchronized trigger RGB-D snapshots
results/            one grounding JSON record per pointing event
logs/               ROS, monitor, publisher, and bag logs
ros_home/            isolated ROS runtime state
```

The MIGU components may also be launched separately with `migu-grounding-node`,
`migu-gesture-monitor`, and `migu-publish-instruction`; SAM remains managed by
the perception-services repository.

## Baselines
[Baselines](assets/7_comparison_plot.pdf)
The public baseline implementations share `baselines/config.yaml`, the MIGU
dataset adapters, and one preparation module. Common fields use the same names
as MIGU, including MediaPipe confidence settings and `vlm.model`. Each method's
algorithm parameters remain configurable under `methods.<method>`. VLM
credentials and endpoints are read from `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

| Public method | Original implementation | Inference |
| --- | --- | --- |
| `geometry` | geometric_based | Geometric object likelihood |
| `POMDP` | LEGS | Language and pointing likelihood fusion |
| `GIRAF` | GIRAF | Semantic filtering and 3-D ray distance |
| `VLM-Grid` | Semantic Grounding-Grid | Grid point votes mapped to masks |
| `VLM-PIVOT` | Semantic Grounding PIVOT | Iterative point votes mapped to masks |
| `VLM-SoM` | main semantic branch | Confidence-weighted Set-of-Marks selections |

Baseline preparation uses both services from `threefruits/perception_services`:
OWLv2 at `segmentation.owl_server_url` (default `http://127.0.0.1:4000`) and SAM
at `segmentation.sam_server_url` (default `http://127.0.0.1:4001`). Start them in
separate terminals from that repository:

```bash
pixi run owl-server
pixi run sam-server
```

Check them with `/healthz` on ports `4000` and `4001`. Configure per-group label
lists in `segmentation.vocabulary`, or pass `--vocabulary cup bottle box` to
override them. Both services must be reachable for preparation and live inference;
prepared batch inference does not call them.

Prepare the shared RGB-D images, hand rays, segmentation masks, annotations, and
preparation timing records once:

```bash
python -m batch_preparation
# Optionally select groups or a different cache location:
python -m batch_preparation --subdirs pantrytable_pick \
  --prepared-dir /path/to/prepared/input
```

The default cache is `batch_preparation/captures/input/<group>/`. Preparation
reuses existing artifacts; `--overwrite` rebuilds generated rays and masks.
`--mode segmentation` skips hand detection, and `--visualize` saves combined
segmentation/hand overlays. Use the same `--prepared-dir` in consumers when
overriding the cache location.

All six methods support both modes. Without `--batch`, the shared preparation
module runs live and stores its inputs inside the new run. With `--batch`, the
method reuses the prepared masks and rays:

```bash
python -m baselines geometry --analyze
python -m baselines geometry --batch --analyze
python -m baselines POMDP --batch --analyze
python -m baselines GIRAF --batch --analyze
python -m baselines VLM-Grid --batch --analyze
python -m baselines VLM-PIVOT --batch --analyze
python -m baselines VLM-SoM --batch --analyze
```

Run these commands from the MIGU repository root after installation. Shared
options include `--config`, `--input-dir`, `--subdirs`, `--instruction`,
`--output-dir`, `--visualize`, and `--no-visualize`. Dataset instructions come from
each capture's `point.json` sentence or normalized `instructions.json`.
Geometry, POMDP, and GIRAF require a valid fitted hand ray. The three VLM-only
baselines do not require a detected hand. For offline checks, POMDP supports
`--language-provider hashing` and GIRAF supports `--semantic-provider lexical`.

Grid, PIVOT, and SoM also accept a single RGB image with an explicit instruction.
Provide `--group` to choose its configured segmentation vocabulary:

```bash
python -m baselines VLM-Grid /path/to/image.png \
  --instruction "pick up this" --group pantrytable_pick
python -m baselines VLM-PIVOT /path/to/image.png \
  --instruction "pick up this" --vocabulary cup bottle box
```

Both single-image and batch Grid/PIVOT inference map point votes directly to
segmentation masks. They do not add Gaussian smoothing. Single RGB images have
no ground-truth or depth annotations, so accuracy and 3-D localization metrics
are unavailable for those inputs.

New runs are written to `output/baselines/<method>/<timestamp>/`. `--analyze`
(also accepted as `--analyse`) writes CSV, JSON, and plots under that run's
`analysis/`, including accuracy, top-k accuracy, NLL, Brier score, 2-D/3-D
localization errors, and timing statistics. Cached preparation times are added
when present. GIRAF is deterministic, so its top-k accuracies equal top-1 and
its NLL/Brier scores are unavailable.

Analyze the latest run of every method after completing all runs:

```bash
python -m baselines.metrics output/baselines \
  --input-root batch_preparation/captures/input
```

For a live run, use its `<run>/_prepared_input` as `--input-root` and select the
method with `--method`. Use `--run <timestamp>` to select a specific run.

### Reference results

The latest approved reference snapshots and recomputed metrics are stored in
`results/baselines/`, separately from new runs. `provenance.json` records their
original method names and timestamps. VLM-SoM is extracted from the semantic
scores of the approved `main/20260905_130441_477267` fusion run; no gesture or
fusion scores are used to rank its objects. Semantic-only latency was not
recorded in that run, so its original fusion timings are retained separately
and excluded from the VLM-SoM timing metrics.

To recompute the reference statistics after preparing the matching dataset:

```bash
python -m baselines.metrics results/baselines \
  --input-root batch_preparation/captures/input
```

## Ablations
| Variant | Acc. ↑ | Top-2 ↑ | Deviation ↓ | NLL ↓ | Brier ↓ |
|:--------|-------:|-------:|------------:|------:|--------:|
| w/o semantics | 51% | 76% | 0.0869 | 1.37 | 0.49 |
| w/o geometry | 46% | 64% | 0.1746 | 1.76 | 0.57 |
| Geo. → ray-distance | 76% | 89% | 0.0805 | 2.91 | 0.40 |
| Geo. → ray-angle | 58% | 83% | 0.1441 | 1.63 | 0.51 |
| Geo. → distance-threshold | 74% | 90% | 0.0851 | 2.88 | 0.36 |
| Sem. → VLM+detector | 60% | 76% | 0.0984 | 5.08 | 0.47 |
| Sem. → VLM-SoM | 77% | 85% | 0.0907 | 3.49 | 0.41 |
| Sem. → VLM-Grid | 40% | 45% | 0.1287 | 17.66 | 1.02 |
| Sem. → VLM-PIVOT | 42% | 49% | 0.1237 | 14.67 | 0.93 |
| **MIGU (full)** | **83%** | **95%** | **0.0592** | **1.17** | **0.25** |
The ablation commands reuse MIGU's inference, geometry models, pixel-wise fusion,
and result analysis. They read the regular `config.yaml`; no ablation settings
are added to it. Replacement semantic methods reuse the implementations and
method-specific settings in `baselines/config.yaml`.

| Command variant | Changed component |
| --- | --- |
| `without-semantics` | Detect all candidates in the baseline vocabulary with OWLv2+SAM, ignore the instruction and detection confidence, and rank masks by their maximum MIGU gesture posterior |
| `without-geometry` | Skip hand detection and the geometric posterior; rank the normal VLM+SAM candidates by VLM confidence |
| `ray-distance` | Use the original `distance_cone` geometric posterior |
| `ray-angle` | Use the original `angle_error_gaussian` geometric posterior |
| `distance-threshold` | Use the original `dist_threshold` geometric posterior |
| `semantic-owlv2` | Generate detector-friendly descriptions with the VLM, detect them with OWLv2, then run SAM |
| `semantic-som` | Replace the semantic branch with the main-derived VLM-SoM baseline |
| `semantic-grid` | Replace the semantic branch with VLM-Grid mask votes |
| `semantic-pivot` | Replace the semantic branch with VLM-PIVOT mask votes |

Prepare the matching ground-truth masks and reusable baseline inputs first:

```bash
python -m batch_preparation
```

Run any variant from the MIGU repository root:

```bash
python -m ablations without-semantics --analyze
python -m ablations without-geometry --analyze
python -m ablations ray-distance --analyze
python -m ablations ray-angle --analyze
python -m ablations distance-threshold --analyze
python -m ablations semantic-owlv2 --analyze
python -m ablations semantic-som --analyze
python -m ablations semantic-grid --analyze
python -m ablations semantic-pivot --analyze
```

The geometry replacements preserve the original `main_new` implementations and
defaults: the distance cone uses `sigma_0=0.005 m` and
`variance=sigma_0**2 + 0.0025 * forward_distance`; the angular Gaussian uses a
standard deviation of `1 rad`; and the distance threshold uses a forward
cylinder with radius `0.20 m`. The rest of the MIGU scoring pipeline is retained.
Fusion uses each mask's maximum confidence-weighted pixel posterior.

`without-semantics` includes every detection in the configured per-group
baseline vocabulary; it does not use VLM proposals or filter candidates using
the instruction. Its candidate confidence is fixed to one. The vocabulary is
not a guarantee of detecting every physical object in a scene.
`without-geometry` preserves the original semantic prompt, including any visual
evidence it uses, while disabling explicit 3-D gesture computation.

The OWLv2 semantic replacement preserves the original separate query-generation
prompt and detector behavior. Detection scores sort/cap candidates; their fusion
confidence defaults to `1.0`, as in `main_new`. OWL uses
`segmentation.owl_server_url` from the baseline config, while SAM uses
`sam.server_url` from the MIGU config. Override detector settings explicitly:

```bash
python -m ablations semantic-owlv2 \
  --owlv2-server-url http://127.0.0.1:4000 \
  --owlv2-bbox-conf-threshold 0.15 --owlv2-confidence 1.0
```

### Reusing prepared data and semantic results

All variants accept `--batch` to reuse normalized RGB-D inputs and hand rays from
the baseline cache. `without-semantics`, SoM, Grid, and PIVOT also reuse that
cache's candidate masks. Semantic inference still runs unless its results are
explicitly reused. For example:

```bash
python -m ablations without-semantics --batch --analyze
python -m ablations semantic-grid --batch --analyze
```

To compare geometry models against identical normal MIGU candidates, reuse a
completed MIGU run's VLM+SAM artifacts:

```bash
python -m ablations ray-distance --reuse-result-dir output/migu/<timestamp> --analyze
python -m ablations ray-angle --reuse-result-dir output/migu/<timestamp> --analyze
python -m ablations distance-threshold --reuse-result-dir output/migu/<timestamp> --analyze
python -m ablations without-geometry --reuse-result-dir output/migu/<timestamp> --analyze
```

The three baseline semantic replacements also accept their baseline run roots:

```bash
python -m ablations semantic-som \
  --reuse-result-dir results/baselines/VLM-SoM/20260905_130441_477267 --analyze
python -m ablations semantic-grid \
  --reuse-result-dir results/baselines/VLM-Grid/20260904_232949_197535 --analyze
python -m ablations semantic-pivot \
  --reuse-result-dir results/baselines/VLM-PIVOT/20260904_233025_651138 --analyze
```

New baseline results and the migrated reference snapshots include their masks
locally. For other legacy Grid/PIVOT snapshots without bundled masks, provide
the matching original cache with `--prepared-dir /path/to/prepared/input`. Do not
re-segment a snapshot with a different model/configuration: its indexed semantic
probabilities must refer to the original candidate masks.

Shared options include `--input-dir`, `--subdirs`, `--config`, `--baseline-config`,
`--prepared-dir`, `--output-dir`, and `--no-visualize`. Outputs are separated under
`output/ablations/<variant>/<timestamp>/` and contain the normal MIGU artifacts.
Replacement semantic records and perception bundles are saved in each frame's
`semantic_source/` directory. `--analyze` (or `--analyse`) uses MIGU's ground-truth
mask metrics and writes `ground_truth_analysis/` in the run directory. It
automatically uses the prepared cache's ground-truth masks when available;
`--analysis-input-dir` selects another annotated root. Analysis requires these
masks even when inference runs without `--batch`.

Without `--analyze`, a live ablation does not require a preparation pass.
Reused semantic predictions are not re-timed; latency measurements from such
runs should not be interpreted as end-to-end semantic inference benchmarks.
For shared OWL+SAM candidate generation, the `sam` timing field includes the
combined candidate preparation call.

## Result analysis

When the dataset also contains
`<group>/ground_truth_masks/<frame_id>.npz`, run grounding and accuracy analysis
in one command:

```bash
python3 -m migu --analyse --analysis-input-dir /path/to/prepared/dataset
```

The report is written to `<run>/ground_truth_analysis` and includes per-frame CSV
metrics, `summary.json`, accuracy, top-k accuracy, NLL, Brier score, localization
distances, and timing statistics. An existing run can
also be analyzed directly:

```bash
migu-result-analysis output/migu/<timestamp> --input-dir /path/to/prepared/dataset
```

## Reusing intermediate results for tests

Reuse controls are command-line-only and never appear in `config.yaml`. To rerun
gesture grounding while reusing VLM boxes and SAM masks from a complete run:

```bash
python3 -m migu \
  --reuse-result-type vlm+seg \
  --reuse-result-dir output/migu/<timestamp>
```

Reuse is strict: MIGU validates that every selected frame has the requested
artifact before processing begins.
