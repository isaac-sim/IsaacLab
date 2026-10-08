# Cosmos camera integration

`isaaclab_experimental.cosmos` connects camera modifiers to a resident Cosmos
Sim-Transfer model. Start Cosmos first, then run Isaac Lab with a camera configured
to connect to its endpoint. Cosmos uses its own Python environment; training uses
the existing Isaac Lab environment and command.

## Start the service

Set up the Cosmos Framework checkout and its `.venv` with the framework's
dependencies, and provide a complete Cosmos Sim-Transfer checkpoint. From the
Isaac Lab checkout, substitute your checkout and checkpoint paths:

```bash
uv run isaaclab cosmos start \
  --framework-root /path/to/cosmos-framework \
  --checkpoint /path/to/Cosmos3-Nano-Sim-Transfer
```

The command runs `isaaclab_experimental.cosmos.worker` with the framework's
`.venv/bin/python` in the foreground. The worker loads the model on `cuda:0`, then
opens `tcp://127.0.0.1:5555`. No camera connection is needed to load the weights.
Compiled inference is enabled by default. Add `--warmup` to generate a disposable
33-frame session before opening the endpoint, or `--no-compile` for eager inference.
Warmup uses a `(480, 832)` canvas; other canvases or prompts can still require
compilation on their first use.

In another terminal, check readiness:

```bash
uv run isaaclab cosmos status
```

A successful response contains `"ready": true` and the model's capabilities.
The endpoint becomes available after loading and any requested warmup. An early
status request can therefore fail to connect; retry it after startup completes.
`--device cuda:1` selects another CUDA device, and `--endpoint
tcp://127.0.0.1:5556` selects another endpoint. Pass the same endpoint to the camera
configuration and to `status` or `stop`.

## Run the Shadow Hand camera task

After `status` reports readiness, run the registered task with its Cosmos preset:

```bash
uv run isaaclab train \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --num_envs 1 \
  presets=cosmos
```

Add `--max_iterations 1` for a short training smoke run. Recording also requires
the `video` extra:

```bash
uv run --extra video isaaclab train \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --num_envs 1 \
  --video sensor:tiled_camera:rgb \
  --video_length 100 \
  presets=cosmos
```

The recording reads the camera's generated RGB, including held frames between
Cosmos updates.

The preset selects one environment, a `640 x 640` RGB camera, and depth controls
covering `0.1` to `1.5` meters. Its prompt describes a Shadow Hand manipulating a
cube, and it connects to `tcp://127.0.0.1:5555`. The camera captures at 10 Hz of
simulation time. After the initial generated frame, Cosmos updates every four
captures, so fresh generated observations arrive at 2.5 Hz of simulation time.
Generation latency determines the elapsed time needed to run those captures.
The task keeps its 10-second episode length, within the 201-frame Cosmos budget.
The preset requires `scene.lazy_sensor_update=True` and synchronous rendering so
each image remains aligned with its simulation state. With OVRTX, set
`scene.tiled_camera.renderer_cfg.async_rendering=False`.

The Cosmos preset trains its feature extractor from generated RGB. While an image
is held, its supervised cube-pose target is held too; both refresh at captures
1, 5, 9, and so on, and resets invalidate the cached target. Published
feature-extractor checkpoints for the default `120 x 120`, seven-channel camera
are incompatible with this `640 x 640`, three-channel input, so the preset disables
that pretrained fallback. For playback, use a policy checkpoint from a Cosmos
training run and keep its corresponding `cnn_*.pth` feature-extractor checkpoint
in the run's log directory.

```bash
uv run isaaclab play \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --checkpoint /path/to/cosmos-run/model_100.pt \
  --num_envs 1 \
  presets=cosmos
```

Without `presets=cosmos`, the task uses its existing camera configuration.

## Configure another camera task

Add a Cosmos chain to the task's camera configuration. For depth conditioning:

```python
from isaaclab.sensors import CameraCfg
from isaaclab_experimental.cosmos import CosmosModelCfg, depth_processor

input_name, modifiers = depth_processor(
    CosmosModelCfg(
        endpoint="tcp://127.0.0.1:5555",
        modality="depth",
        prompt="A robot arm manipulating an object on a workbench.",
        max_episode_frames=201,
    ),
    near=0.2,
    far=12.0,
    seed=42,
)

camera_cfg = CameraCfg(
    # Keep the task's prim_path, spawn configuration, and camera pose here.
    ...,
    height=480,
    width=832,
    data_types=["rgb"],
    modifiers={input_name: modifiers},
)
```

The camera requests `distance_to_image_plane` from the renderer because the chain
uses it as input. The depth modifier converts metric depth to three-channel uint8
controls: near hits are white and far hits are black. Choose `near` and `far` for
the scene. The Cosmos modifier sends these controls to the service and publishes
the generated images as `camera.data.output["rgb"]`. Observation terms can read
`camera.data.output["rgb"].torch` as usual.

Run the task with its normal Isaac Lab training command after applying this camera
configuration. Starting the service alone does not enable Cosmos for a task, and
the task must already use camera observations. The training entry point does not
need a Cosmos demo runner or model initialization code.

The service imports the private Cosmos Framework implementation only when loading
the model. The adapter does not require the Cosmos cookbook or test-kit fixtures.
Keep the framework's heavyweight dependencies in its environment; do not replace
Isaac Lab's Torch installation to start this service.

## Runtime and resets

The camera's post-processing chain is the integration boundary.
`CosmosTransferModifier` is implemented in `cosmos_modifier.py`, with its
`CosmosTransferModifierCfg` in `cosmos_modifier_cfg.py`. It uses the shared image
transfer modifier for queuing, resets, and publication, and validates Cosmos's
required frame cadence before opening a stream. `CosmosModelCfg` creates an
endpoint client implementing the generic image transfer model contract.
The client opens a generation session on its first control chunk. Requests carry
JSON metadata and binary uint8 image arrays over TCP; generated arrays return to
the control tensor's original device.

Each episode generates one initial frame, then updates in chunks of four captured
frames. The latest generated image stays visible between updates. Inference is
synchronous, so a generation request adds latency to the camera update and can
slow training. Repeated observation reads of one capture do not send more frames.

A full camera reset clears its visible image and queued controls, advances the
deterministic seed, and starts fresh generation state on the next chunk. The
service retains the loaded weights. Closing a camera closes its generation
session; it does not stop Cosmos. Failed requests are surfaced to the caller and
are not silently retried against an already advanced episode.

This initial integration supports one camera view and one active generation
session per service. Supported `(height, width)` canvases are `(480, 832)`,
`(544, 736)`, `(640, 640)`, `(736, 544)`, and `(832, 480)`. An episode's frame budget
must be `1 + 4*k`, with a maximum of 201 generated frames. Reset before exhausting
that budget. Batched environments and independent partial resets are not supported.

Other control recipes are `edge_processor`, `regional_edge_processor`, and
`segmentation_processor`, paired with modalities `"edge"` or `"seg"`. Edge
extraction needs OpenCV in the Isaac Lab environment; the depth and segmentation
recipes do not. Segmentation recipes require uncolorized semantic IDs and a fixed
palette, while regional edges also require the camera's segmentation output and
explicit foreground IDs. See the helper docstrings for their camera requirements.

## Stop the service

```bash
uv run isaaclab cosmos stop
```

This explicitly shuts down the resident model and releases its resources. You can
also interrupt the foreground `start` command. `status` remains available while a
camera session is active.

See [Image transfer for camera images](image_transfer.md) for the underlying model
contract, camera scheduling, and optional PPISP processing.
