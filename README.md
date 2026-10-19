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
![dataset](assets/8_benchmarks.png)
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

## Evaluation and experiments

Detailed instructions and reference results are collected in
[**Baselines, ablations, and result analysis**](EXPERIMENTS.md).
