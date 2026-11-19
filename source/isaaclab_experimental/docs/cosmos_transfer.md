# Cosmos Transfer camera integration

`isaaclab_experimental.cosmos` puts Cosmos3-Nano-Sim-Transfer on Isaac Lab cameras. The camera's tensor processors
send a control image, depth by default, to a Cosmos service running in its own environment, and publish the image
Cosmos generates as the camera's `rgb`. Training uses the existing Isaac Lab environment and commands.

## Demos

Start with a demo to see Cosmos Transfer run from setup to result. Each demo installs and starts the service as in
[Serve Cosmos](#serve-cosmos) and puts Cosmos on its task's camera with `cosmos_camera`, as in
[Put Cosmos on a camera task](#put-cosmos-on-a-camera-task).

| Demo | What it shows |
| --- | --- |
| [Shadow Hand](cosmos_transfer_shadow_hand.md) | Training an RL policy on Cosmos images, with the Direct and Manager-based Shadow Hand camera tasks |
| [Apple pick-and-place](cosmos_transfer_apple_pick_place.md) | Running a trained GR00T policy on a Unitree G1 with Cosmos images, on Isaac Sim, OVRTX or Newton, with a different appearance and seed per environment |

The rest of this page is for your own camera task: serving Cosmos, configuring the camera, and running with several
environments, cameras, and GPUs.

## Serve Cosmos

The service runs in a Cosmos Framework environment, separate from Isaac Lab's, and keeps the model loaded between
camera sessions. It uses Cosmos Framework for inference and the standalone `isaaclab-cosmos` package, the Cosmos
directory of the Isaac Lab checkout, for the camera-session protocol. It does not need the `isaaclab` or
`isaaclab_experimental` distributions, Isaac Sim, or a running simulation.

Cosmos Framework also provides a native
[Ray serving interface](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md#q-how-do-i-run-online-inference-with-ray).
That service accepts HTTP generation requests and returns output files, so it cannot be a camera endpoint. The
server below supplies the incremental camera interface, over a Unix socket on the same machine or TCP from another
machine.

### Install the server

Prepare a Cosmos Framework checkout and a supported CUDA environment using its
[official setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md).
The service was tested with Framework commit `001fe1835d3b9a987bd29c521a4bd2fa6b454066`;
check it out before `uv sync --locked`, which installs the dependencies that commit locks.
The CUDA 13.0 group used below requires Python 3.13 and Torch 2.10; the Isaac Lab environment uses Python 3.12 and
Torch 2.11. Framework owns its Torch, CUDA, and compiled model dependencies.

In the server terminal, set these variables to your absolute paths. The server commands on this page use them; set
them again when opening a new terminal.

```bash
export ISAAC_LAB_ROOT=/absolute/path/to/IsaacLab
export COSMOS_FRAMEWORK_ROOT=/absolute/path/to/cosmos-framework
export COSMOS_CHECKPOINT=/absolute/path/to/Cosmos3-Nano-Sim-Transfer
```

Create the Framework environment, install **only the Cosmos directory** into it, and download the
[Cosmos3-Nano-Sim-Transfer checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer):

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv sync --locked --all-extras --group=cu130-train
uv run --no-sync uv pip install --editable \
  "$ISAAC_LAB_ROOT/source/isaaclab_experimental/isaaclab_experimental/cosmos"
uv run --no-sync hf download nvidia/Cosmos3-Nano-Sim-Transfer --local-dir "$COSMOS_CHECKPOINT"
```

`uv run` uses the Framework environment without shell activation. The editable install provides
`isaaclab-cosmos-server` from your checkout; keep the checkout in place. Use `--no-sync` on later Framework commands
to preserve the separately installed package; if you run `uv sync` again, repeat the install.

The checkpoint directory must contain the weights and complete model configuration metadata, including a
`config.json` with a `model` section for an unregistered HF export, with model classes and configuration aliases
that the Framework revision understands. An `Unknown Cosmos target alias` or `Unknown Cosmos type alias` error means
the checkpoint needs compatible metadata or a matching Framework revision. Installing the service package does not
convert the checkpoint.

### Start the server

Follow Framework's
[library-path setup](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#environment-variables)
to avoid conflicting host libraries, then start the server. For an eager inference launch:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
export LD_LIBRARY_PATH=
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT" --no-compile --warmup
```

Wait for `Cosmos ready at unix:///tmp/isaaclab-cosmos-<uid>.sock`, then leave this terminal running. The server
listens on that Unix socket, which only your user can open; it opens no network port. The worker reserves the
endpoint before loading the model once on `cuda:0`, so a conflicting endpoint fails immediately, and accepts
requests after model loading and warmup.

- `--no-compile` selects eager inference. Omit it to enable the compiled CUDA-graph path, which can take additional
  time on its first use.
- `--warmup` runs a disposable one-view session through the full history window (`1 + 4 * --kv-window` frames, 121
  by default) before reporting readiness. No camera has connected yet, so it warms the recipe's default
  `(480, 832)` canvas. With compiled inference, a session with another canvas, view count, or prompt compiles again
  on its first step, while the warmup's compiled state keeps its GPU memory. Use `--warmup` for one camera on the
  `(480, 832)` canvas; otherwise omit it to accept requests after model loading.
- `--kv-window N` and `--attention-sink M` set the generation history the model attends to, in latent frames (four
  video frames each). The defaults, 30 and 3, follow the Sim-Transfer recipe. A shorter window is faster and needs
  less memory, especially with several environments, but remembers less of each episode.
- `--device cuda:1` runs the model on another GPU; see [Cosmos on another GPU](#cosmos-on-another-gpu).
- `--endpoint unix:///path/to/socket` or `--endpoint tcp://127.0.0.1:5556` serves another endpoint, for example TCP
  for Isaac Lab on another machine; see [Endpoints and transports](#endpoints-and-transports).

The server sets no episode length or view count: each camera session requests its own frame budget (see
[Episode frame budget](#episode-frame-budget)) and number of views (see [Several environments](#several-environments)).
The equivalent module entrypoint is `uv run --no-sync python -m isaaclab_experimental.cosmos.server.worker` with the
same arguments. Press Ctrl+C in the server terminal to stop the service and release the model.

### Connect Isaac Lab

In a second terminal, use the normal Isaac Lab environment and checkout. The camera client is included in
`isaaclab_experimental`; it does not need Cosmos Framework or a separate `isaaclab-cosmos` install. Check readiness:

```bash
cd /absolute/path/to/IsaacLab
uv run python -m isaaclab_experimental.cosmos.client.cli status
```

A ready service reports `"ready": true` and the model's capabilities. Check that `"session_active": false` before
connecting: the service runs one active camera session, so close another camera or Isaac Lab process first. The
default camera endpoint is the server's default: on Linux the Unix socket `/tmp/isaaclab-cosmos-<uid>.sock`, and
`tcp://127.0.0.1:5555` on Windows. For another endpoint, pass the same `--endpoint` to `status` and set the camera's
`CosmosModelCfg.endpoint` to it.

Resetting or closing a camera clears its generation history while keeping the model loaded. Closing Isaac Lab
leaves the service running for the next command.

### Client and server packages

The implementation has three boundaries:

| Package or module | Responsibility | Environment |
| --- | --- | --- |
| `isaaclab_experimental.cosmos.client` | Camera controls, frame queues, requests, resets, and RGB publication | Isaac Lab |
| `isaaclab_experimental.cosmos.server` | Checkpoint loading, inference, serving, and process startup | Cosmos Framework |
| `isaaclab_experimental.cosmos._protocol` | Shared socket message format and endpoints | Both |

The server imports the shared protocol and its inference adapter without the full `isaaclab` or
`isaaclab_experimental` distributions, and starting it does not import the camera client. The client uses Isaac
Lab's Torch and NumPy dependencies without importing Cosmos Framework or loading model weights. The top-level
`isaaclab_experimental.cosmos` package also exports the camera client API for existing task configurations.

## Put Cosmos on a camera task

Cosmos goes on a camera through the task's configuration: replace the task's camera with `cosmos_camera`. The
[Shadow Hand demo](cosmos_transfer_shadow_hand.md)'s tasks are built this way, and their options are ordinary Isaac
Lab overrides of `env.cosmos` (see [the task](cosmos_transfer_shadow_hand.md#the-task)).

`cosmos_camera` takes the task's camera that publishes only `rgb`, renders it at a Cosmos canvas with the same aspect
ratio and view, generates with Cosmos, and scales the result back to the camera's size, so the policy's observation
keeps its shape. Cosmos generates the task's environments together, so set `env_cfg.scene.num_envs` (or
`--num_envs`) to what the GPU holds, usually one or two (see "Several environments"). The task's episode length and
capture rate determine the frame budget, which `episode_frame_budget` computes; capture timing is preserved. Start
the service before the task.

```python
from isaaclab_experimental.cosmos import CosmosModelCfg, cosmos_camera, episode_frame_budget

env_cfg.scene.num_envs = 1
camera = env_cfg.scene.base_camera
model = CosmosModelCfg(
    prompt="A KUKA arm with an Allegro hand turning a cube in a bright lab.",
    # The initial capture plus the episode's captures at the camera's rate, rounded up to 1 + 4*k.
    max_episode_frames=episode_frame_budget(env_cfg, camera.update_period)[2],
)
env_cfg.scene.base_camera = cosmos_camera(camera, model, near=0.6, far=1.9)
```

| Setting | Meaning |
|---|---|
| `CosmosModelCfg.prompt` | Scene description: text, or a list for a different prompt per episode (one environment) or per environment (several) |
| `near`, `far` | Depth shown white and black [m] (defaults 0.1 and 2.0) |
| `control` | Other control processors instead of depth; see "Other controls" |
| `CosmosModelCfg.endpoint` | Service endpoint, `unix:///path` or `tcp://host:port` (default: the service's default) |
| `CosmosModelCfg.transport` | `"auto"`, `"cuda_ipc"` or `"socket"`: how images move to the service; see "Endpoints and transports" (default `"auto"`) |

Choose `near` and `far` to cover the scene: the Kuka Allegro base camera sees the cube,
hand and table between about 0.7 m and 1.8 m.

Requirements:

- The environment count is limited by GPU memory and the transport. Agent configurations with
  fixed minibatch sizes larger than the rollout (some rl_games configurations) need adjusting; RSL-RL works as
  configured.
- The camera must publish only `rgb`, use a pinhole lens, and have no existing tensor processors. Rendering must be
  synchronous and the scene must use lazy sensor updates (`scene.lazy_sensor_update=True`), so each image matches
  its simulation state.
- The camera renders at the Cosmos canvas, so its `image_shape` and intrinsics describe the canvas, while its `rgb`
  output keeps the original size. Direct tasks that size their observation space from the camera configuration,
  such as `Isaac-Cartpole-Camera-Direct`, are not supported.
- Without a `prompt`, Cosmos runs without a scene description.

Tasks that train an image encoder from a pretrained checkpoint may need their own configuration, as the Shadow Hand
Cosmos tasks do (see [the task](cosmos_transfer_shadow_hand.md#the-task)), because the encoder sees new images.

## Depth control and prompts

The service follows the Sim-Transfer recipe of the Cosmos cookbook: 480p canvases, four distilled denoising steps,
guidance 1.0, and a 30-latent history window with three attention sinks. Isaac Lab sends the camera's depth as the
control, mapped to white (near) through black (far).

The prompt is used as given; like the recipe, the service does not append control instructions to it. The recipe
uses detailed scene descriptions (a short sentence works, but a paragraph describing the scene, materials,
lighting, and camera usually follows the control more closely).

## Configure another camera task

`cosmos_camera` is the simplest way: it chooses the canvas, keeps the view and the published size, and builds the
tensor processors. To write them by hand instead, add them to the task's camera configuration. For depth
conditioning:

```python
import math

from isaaclab.sensors import CameraCfg
from isaaclab_experimental.cosmos.client import CosmosModelCfg, depth_processor

processors = depth_processor(
    CosmosModelCfg(
        prompt="A robot arm manipulating an object on a workbench.",
        # The initial capture plus the episode's captures (here 10 s at 30 per second), rounded up to 1 + 4*k.
        max_episode_frames=1 + 4 * math.ceil(10 * 30 / 4),
    ),
    near=0.2,
    far=12.0,
)

camera_cfg = CameraCfg(
    # Keep the task's prim_path, spawn configuration, and camera pose here.
    ...,
    height=480,
    width=832,
    data_types=["rgb"],
    tensor_processors=processors,
)
```

The camera requests `distance_to_image_plane` from the renderer because the depth
processor reads it. The depth processor converts metric depth to three-channel uint8
controls: near hits are white and far hits are black. Choose `near` and `far` for
the scene. The Cosmos processor sends these controls to the service and publishes
the generated images as `camera.data.output["rgb"]`. Observation terms can read
`camera.data.output["rgb"].torch` as usual.

Run the task with its normal Isaac Lab training command after applying this camera
configuration. Starting the service alone does not enable Cosmos for a task, and
the task must already use camera observations. The training entry point does not
need a Cosmos demo runner or model initialization code.

### Other controls

The checkpoint accepts four kinds of control: depth, segmentation, edge and blur, one kind per session. The service is
not told which kind it receives, so any processors that write the `control` buffer work: uint8 images of shape
`(N, H, W, 3)` at the canvas size. Pass them to `cosmos_camera` as `control`:

```python
import torch

from isaaclab_experimental.cosmos import CosmosModelCfg, cosmos_camera, episode_frame_budget
from isaaclab_experimental.image_transfer import FunctionProcessorCfg


def segmentation_to_control(data: torch.Tensor) -> torch.Tensor:
    """Colorized segmentation of shape (N, H, W, 4) to three-channel controls."""
    return data[..., :3].contiguous()


camera_cfg = cosmos_camera(
    task_camera_cfg,
    CosmosModelCfg(
        prompt="A robot arm manipulating an object on a workbench.",
        max_episode_frames=episode_frame_budget(env_cfg, task_camera_cfg.update_period)[2],
    ),
    control=[
        FunctionProcessorCfg(
            function=segmentation_to_control, inputs=["semantic_segmentation"], outputs=["control"]
        )
    ],
)
```

The camera then requests `semantic_segmentation` from the renderer instead of depth, colorized as
`colorize_semantic_segmentation` does by default. A control computed from the rendered image, such as an edge or blur
control, reads `rgb` in the same way: the renderer then produces `rgb` for the control, and the `rgb` the camera
publishes is the generated one. Without `control`, the camera uses the depth control above.

Isaac Lab provides the depth control, and depth is the control the demos were run with. Check the generated images
of another control on your scene before training on them.

### A different prompt per episode

`prompt` takes one string or a list. With a list, episode `k` of a camera stream uses
`prompt[k % len(prompt)]`, for visual randomization: the session opens with the first prompt, and each
episode reset sends the next one to the service, which rebuilds the text conditioning for the new episode.
The model stays loaded; only the episode's conditioning and generation state change. A reset before the
episode's first update keeps the prompt, so the environment's initial reset right after its first capture does
not skip the first prompt. Rebuilding the conditioning adds time to that reset, and with compiled inference a
prompt's first use can trigger compilation, so prefer a short list of prompts. A compiled server keeps what it
compiled for earlier sessions, and Torch limits how often it recompiles: after several sessions with new prompts or
view counts, a session can fail with `recompile_limit reached`. Restart the server then.

```python
CosmosModelCfg(
    prompt=[
        "A robotic Shadow Hand turning a red cube in a bright warehouse with metal shelves.",
        "A robotic Shadow Hand turning a wooden cube on a kitchen counter in warm evening light.",
        "A robotic Shadow Hand turning a blue cube in a dim laboratory with fluorescent lights.",
    ],
)
```

A prompt can change only at an episode reset.

Prompts can also come from files. `prompt` is always text; file paths go in `prompt_files`, so a prompt is never
mistaken for a path:

```bash
uv run isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct --rl_library rsl_rl --num_envs 2 \
  env.cosmos.prompt_files="['prompts/*.json']"
```

Each file is one prompt and patterns expand in sorted order, so two matching files give each of the two environments
its own prompt. The files replace `prompt`. A `.txt` file is used without leading and trailing whitespace. A `.json` file is a
structured caption, the format the model was trained on: it is parsed and serialized once into the prompt string,
keeping its structure rather than flattening it into prose. The service does not upsample prompts, so prepared
captions reach the model unchanged. In code, `read_prompt_files([...])` returns the same prompts for `CosmosModelCfg.prompt`.

## Runtime and resets

Cosmos requires synchronous camera rendering. Its processor rejects `env_mask=None` before queuing controls
or generating images; asynchronous cameras supply `None`. This also protects cameras configured directly
or changed by later overrides. Use `renderer_cfg.async_rendering=False` for OVRTX. Direct processor callers
must supply a boolean Warp mask identifying the captured views, including an all-true mask when all views are fresh.

The camera's tensor processors are the integration boundary.
`CosmosTransferProcessor` is implemented in `client/cosmos_processor.py`, with its
`CosmosTransferProcessorCfg` in `client/cosmos_processor_cfg.py`. It uses the shared image
transfer processor for queuing, resets, and publication, and validates Cosmos's
required frame cadence before opening a stream. `CosmosModelCfg` creates an
endpoint client implementing the generic image transfer model contract.
The client opens a generation session on its first control chunk. Requests carry
JSON metadata over the endpoint and, with the `socket` transport, binary uint8 image arrays; generated arrays return to
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

The service runs one active generation session at a time; a session serves one camera view per environment
(see "Several environments"). Supported `(height, width)` canvases are `(480, 832)`,
`(544, 736)`, `(640, 640)`, `(736, 544)`, and `(832, 480)`. An episode's frame budget
must be `1 + 4*k`. Reset before exhausting that budget.

## Single-frame generation

By default, Cosmos generates autoregressively, as described above: each update continues the episode's generated
images, so the camera holds an image until the next chunk and shows black after a reset until its episode starts.
A policy that needs no temporal consistency, such as one that observes once per action chunk, can generate each
image on its own instead, with these `CosmosModelCfg` settings (for example `env.cosmos.generation_mode=single_frame`
on a Cosmos task):

| Setting | Effect |
|---|---|
| `generation_mode="single_frame"` | Each generation makes one image per environment from the current capture's depth alone, with no history, so images neither repeat between updates nor turn black after resets |
| `decision_interval=N` | Generate at the first of every `N` captures of the camera, for example once per 16-step action chunk; the other captures publish the rendered `rgb`. Resets do not restart the count |
| `augmentation_probability=p` | Generate at a decision point with probability `p`, one draw for all environments; otherwise publish the rendered `rgb` |

- With a prompt list, environment `v` uses `prompt[(v + k) % len(prompt)]` in its episode `k`, also with several
  environments.
- Each image is a one-frame episode, so the frame budget does not apply. When a capture can skip generation, the
  camera also renders `rgb`; validating the environment configuration adds it to the Cosmos processor's inputs.
- Cameras sharing a session need autoregressive generation.

A learner that shares the GPU with Cosmos can free the weights between rollouts. `offload()` on the camera's
`CosmosModel`, `SimulationContext.instance().get_or_create_backend(env_cfg.cosmos)` in a Cosmos task, moves them to
host memory, and `activate()` moves them back before the next capture; the single-frame session stays open. For an
idle service, `python -m isaaclab_experimental.cosmos.client.cli offload` and `activate` do the same. Each change
moves the weights between host and GPU memory, and the first generations after `activate()` compile again.

## Several environments

One session can generate the cameras of several environments as one batch. Each environment keeps its own
prompt, seed, and generation history, and resets on its own: a resetting environment starts its new episode with
one frame while the others continue with four. More environments share each transformer step, so throughput per
environment rises with GPU size, while each environment needs GPU memory for its own history.

Start the service in compiled mode (resetting one environment while the others continue needs the compiled
runtime). Omit `--warmup`: it compiles one `832 x 480` view and keeps that memory, while a batch of another size
or canvas compiles again on its first step.

```bash
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT"
```

Keep the default history of 30 latent frames for quality. If the environments do not fit beside Isaac Sim, run the
service on a GPU of its own with `--device cuda:1`.

A camera with Cosmos processors requests one view per environment when its session opens. The server uses that
count without a separate view setting. A compiled server compiles again for each new count, and Torch limits how
often it recompiles: a server that has served one count can fail on another with `recompile_limit reached`. Restart
the server when the number of views changes. Available GPU memory
limits the batch: allocation can fail during generation as history grows, so a successful connection alone does
not establish that a full episode fits.

Ask for the environments with `--num_envs`, and give one prompt per environment as a list:

```bash
uv run isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct --rl_library rsl_rl --num_envs 2 \
  env.cosmos.prompt="['A close-up overhead view of a robotic Shadow Hand turning a red cube on a wooden workbench.', 'A close-up overhead view of a robotic Shadow Hand turning a blue cube in a bright white lab.']"
```

For a task without a Cosmos version, set `env_cfg.scene.num_envs = 2` and give `CosmosModelCfg.prompt` a list.

- With several environments, environment `v` uses `prompt[v % len(prompt)]` for all its episodes; the batch keeps
  each environment's prompt across its resets. With one environment, a prompt list changes per episode, and with
  [single-frame generation](#single-frame-generation) it does for each environment.
- Environment `v` starts from seed `seed + v` of the camera's `CosmosTransferProcessorCfg`, 0 by default, or from
  `seed[v]` when `seed` is a list with one seed per environment. Each reset advances an environment's seed by the
  number of environments.
- An environment that resets mid-chunk starts its new episode at the next chunk with its newest capture, so all
  environments keep one four-frame cadence; it shows black until then.
- A camera slower than the environment step captures each environment on its own schedule after independent
  resets; Cosmos receives a frame only for the environments a capture covers, so each environment's frames and
  budget follow its own captures.
- Independent resets of several environments need compiled inference and per-view VAE caches in the Framework
  runtime. A service without them stops the run with an error at the first independent reset.
- With the socket transport, one step's image message carries at most 16 environments.
- Each environment's history (30 latent frames) takes GPU memory. On a 48 GB GPU that Isaac Sim shares, one
  environment fits; for more, give the service its own GPU (`--device cuda:1`) or a larger one, or let
  [several services](#several-gpus) split the environments across GPUs.

### Record several environments

`--video sensor:<camera>:rgb` records environment 0 only. The Newton GL streaming view tiles the camera's `rgb`
output, the generated images, across environments; record it to see every environment side by side:

```bash
uv run --extra video isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct --rl_library rsl_rl --num_envs 2 \
  --max_iterations 20 \
  --viz newton_gl --video viz:newton_gl:streaming_view --video_length 600 \
  env.cosmos.prompt="['A close-up overhead view of a robotic Shadow Hand turning a red cube on a wooden workbench.', 'A close-up overhead view of a robotic Shadow Hand turning a blue cube in a bright white lab.']"
```

`--video_length 600` records 600 steps, one 10-second episode. The run prints the clip directory when it starts.
`--viz newton_gl` keeps the visualizer running every step; a visualizer added only for the recording does not
advance during headless training, and its streaming view records the first frame throughout. On a machine without
a screen, such as over SSH, run `unset DISPLAY` first: with `DISPLAY` set, the headless visualizer cannot create
its OpenGL context.

Each reset shows black until the environment's next chunk. For a clip without mid-episode resets, add
`env.fall_dist=1000.0`: a dropped cube keeps falling, but within a 10-second episode it falls less than 500 m, so it
does not end the episode. Training also starts environments at random points of their episodes; add
`agent.init_at_random_ep_len=false` so they all run until the time limit and reset together.

## Several cameras

Several cameras, such as a base and a wrist camera, can share one session: each capture generates all their
environments as one batch. This is set up in a task's configuration code; no registered task provides it yet. Name
the cameras in `CosmosModelCfg.cameras`, and give each camera that model, one canvas and its scene name: each camera
receives its own copy of the settings, and the cameras share a session only while the copies match.

```python
from isaaclab_experimental.cosmos import CosmosModelCfg, cosmos_camera, episode_frame_budget

names = ("base_camera", "wrist_camera")
model = CosmosModelCfg(
    cameras=names,
    prompt=[
        "A KUKA arm with an Allegro hand turning a cube in a bright lab, seen from the side.",
        "A close-up of an Allegro hand turning a cube in a bright lab, seen from its wrist.",
    ],
    max_episode_frames=episode_frame_budget(env_cfg, env_cfg.scene.base_camera.update_period)[2],
)
for name in names:
    # One canvas (height, width) for the session; square cameras use 640 x 640.
    setattr(env_cfg.scene, name, cosmos_camera(getattr(env_cfg.scene, name), model, canvas=(640, 640), name=name))
```

- Camera `k`'s environment `v` is view `k * num_envs + v`, with the cameras in the order given. A prompt list assigns `prompt[view % len(prompt)]`, so the example's prompts go to the base and
  the wrist camera.
- Each camera keeps its own output size. One session has one canvas, so pass the same `canvas` for every camera;
  each camera's view is the canvas's center crop, scaled back to its size.
- The cameras need the same update period, so they capture together. Each camera must be read at every capture,
  as observation terms do; a camera that misses a shared capture stops the session with an error.
- An environment's cameras share its seed and reset together. Cosmos generates each view on its own, so the images
  of different cameras need not agree, for example on colors and materials.
- The batch holds `num_cameras * num_envs` views, which need GPU memory like as many environments. With several
  environments, the service needs independent resets (see [Several environments](#several-environments));
  [several services](#several-gpus) split the views as they split environments.

## Endpoints and transports

The **endpoint** is how Isaac Lab reaches the service; it carries the session's messages:

| Endpoint | Where it works |
|---|---|
| `unix:///path` (default on Linux, `/tmp/isaaclab-cosmos-<uid>.sock`) | Same machine. Only your user can open the socket; no network port is opened. |
| `tcp://host:port` | Also from another machine, for example Isaac Lab on Windows. |

The **transport** is how control images go to the service, and generated images come back:

| Transport | Data path | Where it works |
|---|---|---|
| `cuda_ipc` | Shared GPU buffers, opened once per session, ordered by interprocess CUDA events. Image transfer between the camera and service needs no host copies. Each step sends only a small message on the endpoint. | Service on the same Linux machine, on the camera's GPU or on another GPU with direct access to it; see [Cosmos on another GPU](#cosmos-on-another-gpu). |
| `socket` | Images go through host memory in the endpoint's messages. | Any endpoint. |

`CosmosModelCfg.transport` defaults to `auto`: CUDA IPC when the service reports it, runs on this machine (the same
boot ID), sees the camera's GPU, and uses that GPU or a GPU with direct access to it, the socket otherwise.
`cuda_ipc` fails when it is unavailable instead of falling back. `status` lists the service's transports and the
GPUs it can see. With the defaults on one Linux
machine, image transport uses shared GPU buffers and the messages use the Unix socket, so no TCP is involved. CUDA
IPC avoids the socket transport's 16-view protocol limit; available GPU memory still limits the batch size.

Buffer ownership and release order are documented with `SharedChannel` in `cosmos/_cuda_ipc.py`.

For Isaac Lab on another machine, start the service with `--endpoint tcp://127.0.0.1:5555` and forward that port,
for example with `ssh -L 5555:localhost:5555 gpu-host`, or keep the Unix socket and forward it with
`ssh -L 5555:/tmp/isaaclab-cosmos-<uid>.sock gpu-host`. Then set the camera's endpoint to `tcp://127.0.0.1:5555`;
the protocol has no authentication, so do not open a public interface. `auto` sees from the boot ID that a tunneled
service runs on another machine and uses the socket.

### Cosmos on another GPU

The service can run on another GPU of the same machine, which leaves the model's memory off the simulator's GPU.
CUDA IPC then works when the two GPUs can access each other's memory directly, over NVLink or PCIe;
`nvidia-smi topo -m` shows how they connect. Each process creates the buffer it writes on its own GPU, and the
other process copies it between the GPUs without host memory. Without direct access, `auto` uses the socket and
`cuda_ipc` fails. MIG instances have no direct access to each other. If the service cannot see the camera's GPU,
the session is rejected even with `auto`; keep both GPUs visible to both processes.

Both processes must see both GPUs, so do not narrow `CUDA_VISIBLE_DEVICES` to one GPU; select the model's GPU with
`--device` instead:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT" --device cuda:1
```

Isaac Lab keeps its own device, `cuda:0` by default. Set `env.cosmos.transport=cuda_ipc` (or
`CosmosModelCfg.transport="cuda_ipc"` in a configuration) to require GPU-to-GPU transfer instead of a socket fallback. Each process also creates a CUDA
context on the other GPU, which takes some memory there.

### Several GPUs

Several services can share one camera's environments, for example one service per GPU. Start each service in its
own terminal with its own GPU and endpoint; each loads its own copy of the model:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT" --device cuda:1 \
  --endpoint unix:///tmp/isaaclab-cosmos-gpu1.sock
```

Start the second with `--device cuda:2 --endpoint unix:///tmp/isaaclab-cosmos-gpu2.sock`. Then override the camera's
endpoint with the list of both:

```bash
uv run isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct --rl_library rsl_rl --num_envs 4 \
  env.cosmos.endpoint="['unix:///tmp/isaaclab-cosmos-gpu1.sock', 'unix:///tmp/isaaclab-cosmos-gpu2.sock']" \
  env.cosmos.transport=cuda_ipc
```

The environments are split into contiguous shares of near-equal size, one per service in the order given:
environments 0-1 on the first service and 2-3 on the second here. Each environment stays on its service, which
keeps its generation history, and keeps its prompt: environment `v` uses `prompt[v % len(prompt)]` on any service.
Every step sends each service its share before waiting for any reply, so the services generate at the same time,
and a step lasts as long as the slowest service.

There must be at least one environment per service, and a service with several environments must run compiled.
If one service fails, the camera closes every session. In a configuration, set `CosmosModelCfg.endpoint` to the
list of endpoints.

## Episode frame budget

Each session requests a frame budget, `1 + 4*k` frames per episode; the server sets no limit of its own. The
service tells the model the episode's duration from this budget. The Shadow Hand Cosmos tasks derive it with
`episode_frame_budget` from the task's episode duration and actual capture cadence, including the initial capture
and rounding up to `1 + 4*k`. Hand-written camera processors must set `CosmosModelCfg.max_episode_frames` explicitly. Reset the environment
before that budget is exhausted; the service rejects further generation rather than silently resetting the episode.

Startup logs show the budget and the captures and generated updates per second of simulation time. The model uses
a sliding history window; longer episodes still need quality and memory validation on the intended workload.

The service also tells the model the images' frame rate, `CosmosModelCfg.fps`, 30 by default as in the Sim-Transfer
recipe. A task whose prompt and settings were calibrated at another rate sets it; it need not equal the capture rate.

## Image transfer

Cosmos is built on `isaaclab_experimental.image_transfer`, which connects an application-owned image-generation
model to cameras through tensor processors. It does not load models, download assets, or select a network
transport. `ImageTransferProcessorCfg` takes the model as its `backend`, and `FunctionProcessorCfg` with
`depth_to_control` prepares the control:

```python
from isaaclab_experimental.image_transfer import FunctionProcessorCfg, ImageTransferProcessorCfg, depth_to_control

wrist_camera = CameraCfg(
    ...,
    data_types=["rgb"],
    tensor_processors=[
        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 0.2, "far": 12.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        ),
        ImageTransferProcessorCfg(backend=MyModelCfg(), initial_frames=1, update_frames=4),
    ],
)
```

- **Model contract:** `backend` is a `BackendCfg` whose `class_type` builds an `ImageTransferModel`. Equal
  configurations share one model through `SimulationContext.get_or_create_backend`, so the model holds only shared
  resources such as weights; without a simulation context, the processor builds and owns the model. Each processor
  opens its own `ImageTransferStream` with `open_stream(num_views, seeds)`, which holds the temporal state of its
  views. `ImageTransferStream.step(controls, reset_rows, seeds)` receives one uint8 THWC three-channel control
  sequence per view and returns uint8 sRGB images with the same sequence lengths and image size, with no implicit
  crop or resize. The controls are valid until `step` returns, since the processor reuses their storage, so a stream
  that keeps a chunk copies it. A stream may return None instead and deliver the images later through a callback, as
  cameras sharing a Cosmos session do. Calls run on the current Torch stream, and returned tensors must be ready on
  it. `close()` must be
  idempotent; the camera closes its processors, and with them their streams, when it is released.
- **Scheduling:** `initial_frames` and `update_frames` set the first and later chunk sizes, both one by default. The
  output holds the last generated image until the next chunk, so larger chunks add observation age as well as
  latency. Every view needs a complete chunk before a chunk is generated, and `max_pending_frames` bounds each
  view's queue. The camera passes `env_mask` to the processor, which queues controls only for the views captured in
  each call.
- **Resets:** resetting a view clears its queued controls and its visible image. Its next chunk uses the initial
  size, and `step` receives the view index and a new deterministic seed: view `i` starts from `seed + i`, or from
  `seed[i]` when `seed` lists one seed per view, and each reset advances it by the number of views, modulo `2**31`.
  The stream must discard all episode state for those views. An exception or invalid result marks the processor as
  failed; recreate the camera instead of retrying against a stream that may already have advanced.
- **Controls:** `depth_to_control` maps metric depth to near-white, far-black controls; missing, nonfinite, and
  nonpositive depth becomes black. Any processor that produces uint8 `(N, H, W, 3)` controls under the image
  transfer processor's first input can take its place.

The image transfer tests run the processors on CPU with small stand-in models and need no model or service:

```bash
uv run python -m pytest source/isaaclab_experimental/test/sensors/test_image_transfer.py
```
