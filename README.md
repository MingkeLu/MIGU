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

## Self-hosting SAM

MIGU does not depend on a project-operated SAM endpoint. Install the optional
server dependencies on the machine that will run segmentation. Add the `ros`
extra when the online ROS interface is also needed:

```bash
python3 -m pip install -e '.[sam-server,ros]'
```

Start the service locally:

```bash
migu-sam-server --host 127.0.0.1 --port 4001
```

The first launch downloads the default full
[`facebook/sam-vit-huge`](https://huggingface.co/facebook/sam-vit-huge) weights,
which occupy approximately 2.6 GB in the local Hugging Face cache. The service
automatically uses CUDA when available and otherwise uses the CPU. For constrained
machines, explicitly select the smaller Base model or another device:

```bash
migu-sam-server --device cuda --model facebook/sam-vit-huge
```

Keep `--host 127.0.0.1` when MIGU and SAM run on the same machine. For a separate
trusted machine, bind to `0.0.0.0`, restrict port 4001 with a firewall, and set
`sam.server_url` in `config.yaml` to that machine's private URL. The bundled server
does not provide authentication and should not be exposed directly to the public
internet.

Verify that the service is ready before starting MIGU:

```bash
curl http://127.0.0.1:4001/health
```

## Input data

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
MIGU environment.

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

For a bag recording, one command starts roscore when necessary, starts the local
SAM service, launches the gesture monitor and grounding node, publishes the user
instruction, replays every supplied bag, waits for all detected pointing events,
and shuts down the processes it started:

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

migu-bag-demo bags/*.bag \
  --instruction "pick up this"
```

The default local model is the full `facebook/sam-vit-huge`. For constrained
testing, explicitly select `--sam-model facebook/sam-vit-base`, or force a device
through `--sam-device`. The first execution may take longer while model weights
are downloaded. Bag timestamp gaps are preserved by `rosbag play`.

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
logs/               ROS, SAM, monitor, publisher, and bag logs
ros_home/            isolated ROS runtime state
```

The components may also be launched separately with
`migu-sam-server`, `migu-grounding-node`, `migu-gesture-monitor`, and
`migu-publish-instruction`.

## Result analysis

When the dataset also contains
`<group>/ground_truth_masks/<frame_id>.npz`, run grounding and accuracy analysis
in one command:

```bash
python3 -m migu --analyse --analysis-input-dir /path/to/prepared/dataset
```

The report is written to `<run>/ground_truth_analysis` and includes per-frame CSV
metrics, `summary.json`, accuracy, top-k accuracy, NLL, Brier score, localization
distances, timing statistics, and diagnostic distributions. An existing run can
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
