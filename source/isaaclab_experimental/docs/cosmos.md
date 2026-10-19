# Cosmos camera integration

`isaaclab_experimental.cosmos` connects camera modifiers to a resident Cosmos
Sim-Transfer service. Run Isaac Lab with a camera configured to connect to the
service's endpoint. Training uses the existing Isaac Lab environment and command.

## Connect to a running service

Start a compatible streaming service using the separate
[Cosmos service setup guide](cosmos_service.md), or use an endpoint provided by the
service operator. The default camera endpoint is the server's default: on Linux the Unix socket
`/tmp/isaaclab-cosmos-<uid>.sock`, which only your user can open, and `tcp://127.0.0.1:5555` on Windows.
The public Framework's Ray HTTP server has a different interface and cannot be
used as this camera endpoint.

In a second terminal, use the normal Isaac Lab environment and checkout. The camera
client is included in `isaaclab_experimental`; it does not require Cosmos Framework
or a separate `isaaclab-cosmos` install in the training environment.

Check readiness:

```bash
cd /absolute/path/to/IsaacLab
uv run isaaclab cosmos status
```

A successful response contains `"ready": true` and the model's capabilities. Check
that `"session_active": false` before connecting: this service supports one active
camera session. Wait for the service to finish loading and warming up before
starting training.
For a different endpoint, pass `--endpoint unix:///path/to/socket` or `--endpoint tcp://127.0.0.1:5556` to
`status` and set the camera's `CosmosModelCfg.endpoint` to the same address.

## Put Cosmos on a camera task

Cosmos goes on a camera through the task's configuration. The Shadow Hand tasks have a `cosmos` preset (see
"Run the Shadow Hand camera task"), and its options are ordinary Isaac Lab overrides of `env.cosmos`.
For any other compatible camera task, call `apply_cosmos(env_cfg, prompt=...)` on its configuration in code, or build a
preset with `cosmos_camera` (see "Configure another camera task").

`apply_cosmos` finds the task's camera that publishes only `rgb`, renders it at a Cosmos canvas with the same aspect
ratio and view, generates with Cosmos, and scales the result back to the camera's size, so the policy's observation
keeps its shape. It uses one environment unless `num_envs` asks for more (see "Several environments"). The task's
episode length and capture rate determine the requested frame budget; capture timing is preserved. Start the service
first: `apply_cosmos` checks that it is running and idle before the simulation is created.

```python
from isaaclab_experimental.cosmos import apply_cosmos

apply_cosmos(env_cfg, prompt="A KUKA arm with an Allegro hand turning a cube in a bright lab.", near=0.6, far=1.9)
```

| Argument | Meaning |
|---|---|
| `prompt` | Scene description: text, or a list for a different prompt per episode (one environment) or per environment (several) |
| `camera` | Scene camera to use, when the task has several rgb cameras |
| `near`, `far` | Depth shown white and black [m] (defaults 0.1 and 2.0) |
| `endpoint` | Service endpoint, `unix:///path` or `tcp://host:port` (default: the service's default) |
| `transport` | `"auto"`, `"cuda_ipc"` or `"socket"`: how images move to the service; see "Endpoints and transports" (default `"auto"`) |
| `num_envs` | Environments generated together as one batch |

Choose `near` and `far` to cover the scene: the Kuka Allegro base camera sees the cube,
hand and table between about 0.7 m and 1.8 m.

Limits of `apply_cosmos`:

- One environment unless `num_envs` sets more, subject to GPU memory and transport limits. Agent configurations with
  fixed minibatch sizes larger than the rollout (some rl_games configurations) need adjusting; RSL-RL works as
  configured.
- The camera must publish only `rgb`, use a pinhole lens, and have no existing modifiers. Rendering must be
  synchronous and the scene must use lazy sensor updates.
- The camera renders at the Cosmos canvas, so its `image_shape` and intrinsics describe the canvas, while its `rgb`
  output keeps the original size. Direct tasks that size their observation space from the camera configuration,
  such as `Isaac-Cartpole-Camera-Direct`, are not supported; Isaac Lab warns for Direct tasks.
- Without a `prompt`, Cosmos runs without a scene description and Isaac Lab warns.

Tasks that train an image encoder from a pretrained checkpoint may need their own preset, as the Shadow Hand tasks
below do, because the encoder sees new images.

## Depth control and prompts

The service follows the Sim-Transfer recipe of the Cosmos cookbook: 480p canvases, four distilled denoising steps,
guidance 1.0, and a 30-latent history window with three attention sinks. Isaac Lab sends the camera's depth as the
control, mapped to white (near) through black (far).

The prompt is used as given; like the recipe, the service does not append control instructions to it. The recipe
uses detailed scene descriptions (a short sentence works, but a paragraph describing the scene, materials,
lighting, and camera usually follows the control more closely).

## Run the Shadow Hand camera task

The Direct task `Isaac-Reorient-Cube-Shadow-Camera-Direct` and the Manager-based task
`Isaac-Reorient-Cube-Shadow-Camera` both offer `presets=cosmos`, with the same Cosmos camera.

Both presets use one environment unless `--num_envs` sets more; see [Several environments](#several-environments).

After `status` reports readiness, run a task with its Cosmos preset:

```bash
uv run isaaclab train \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --num_envs 1 \
  --max_iterations 100 \
  presets=cosmos
```

This runs 100 training iterations. Use `--max_iterations 1` for a short training
smoke run. Recording also requires
the `video` extra:

```bash
uv run --extra video isaaclab train \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --num_envs 1 \
  --max_iterations 100 \
  --video sensor:tiled_camera:rgb \
  --video_length 100 \
  presets=cosmos
```

The recording reads the camera's generated RGB, including held frames between
Cosmos updates.

The preset selects a `640 x 640` RGB camera and depth controls
covering `0.1` to `1.5` meters. Its prompt describes a Shadow Hand manipulating a
cube, and it connects to the default endpoint. Like the task's other cameras, it
captures every environment step, 60 times per second of simulation time. After the
initial generated frame, Cosmos updates every four captures, so fresh generated
observations arrive at 15 Hz of simulation time. Generation latency determines the
elapsed time needed to run those captures.
The task keeps its 10-second episode length and requests 601 frames: one capture per step, plus the initial capture.
For fewer Cosmos requests, capture less often, for example `env.scene.tiled_camera.update_period=0.1` (10 Hz,
101 frames); generated images then change less often.
The preset requires `scene.lazy_sensor_update=True` and synchronous rendering so
each image remains aligned with its simulation state. With OVRTX, set
`env.scene.tiled_camera.renderer_cfg.async_rendering=False` on the command line.

The Cosmos preset trains its feature extractor from generated RGB. While an image
is held, its supervised cube-pose target is held too: each environment's target is
the cube pose of the step its image appeared, kept until that image changes. An
environment showing black after a reset is left out of the loss until its next image. Published
feature-extractor checkpoints for the default `120 x 120`, seven-channel camera
are incompatible with this `640 x 640`, three-channel input, so the preset disables
that pretrained fallback. For playback, use a policy checkpoint from a Cosmos
training run and keep its corresponding `cnn_*.pth` feature-extractor checkpoint
in the run's log directory.

```bash
uv run isaaclab play \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --checkpoint /path/to/cosmos-run/model_99.pt \
  --num_envs 1 \
  presets=cosmos
```

Without a Cosmos preset, the task uses its existing camera configuration.

The preset's Cosmos settings are `env.cosmos`, the camera's own `CosmosModelCfg`, and they are set with ordinary
Isaac Lab overrides:

| Option | Override |
|---|---|
| Prompt | `env.cosmos.prompt=<text>` |
| Prompt per environment | `env.cosmos.prompt="['<text>', '<text>']"` |
| Service endpoint | `env.cosmos.endpoint=<url>` |
| Transport | `env.cosmos.transport=socket` |
| Depth range [m] | `env.scene.tiled_camera.modifiers.distance_to_image_plane.0.params.near=0.2` (and `far`) |

For example, `isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Direct presets=cosmos
env.cosmos.prompt="A Shadow Hand turning a red cube."`. The depth range belongs to the step that prepares the
control, item `0` of the camera's depth chain. Omitted options preserve the preset's values, and the overrides keep its
image size, modifier chain, and feature-extractor supervision.

## Client and server packages

The implementation has three boundaries:

| Package or module | Responsibility | Environment |
| --- | --- | --- |
| `isaaclab_experimental.cosmos.client` | Camera controls, frame queues, requests, resets, and RGB publication | Isaac Lab |
| `isaaclab_experimental.cosmos.server` | Checkpoint loading, inference, serving, and process startup | Cosmos Framework |
| `isaaclab_experimental.cosmos._protocol` | Shared socket message format and endpoints | Both |

The standalone `isaaclab-cosmos` distribution installs only the Cosmos directory
into the Framework environment. The server imports the shared protocol and its
inference adapter without requiring the full `isaaclab` or `isaaclab_experimental`
distributions. Follow [Cosmos service setup](cosmos_service.md) to install it.

The client uses Isaac Lab's Torch and NumPy dependencies without importing Cosmos
Framework or loading model weights. The top-level `isaaclab_experimental.cosmos`
package also exports the camera client API for existing task configurations.

## Configure another camera task

`cosmos_camera` is the simplest way: it chooses the canvas, keeps the view and the published size, and builds the
chain. To write the chain by hand instead, add it to the task's camera configuration. For depth conditioning:

```python
import math

from isaaclab.sensors import CameraCfg
from isaaclab_experimental.cosmos.client import CosmosModelCfg, depth_processor

input_name, modifiers = depth_processor(
    CosmosModelCfg(
        prompt="A robot arm manipulating an object on a workbench.",
        # The initial capture plus the episode's captures (here 10 s at 30 per second), rounded up to 1 + 4*k.
        max_episode_frames=1 + 4 * math.ceil(10 * 30 / 4),
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

### A different prompt per episode

`prompt` takes one string or a list. With a list, episode `k` of a camera stream uses
`prompt[k % len(prompt)]`, for visual randomization: the session opens with the first prompt, and each
episode reset sends the next one to the service, which rebuilds the text conditioning for the new episode.
The model stays loaded; only the episode's conditioning and generation state change. A reset before the
episode's first update keeps the prompt, so the environment's initial reset right after its first capture does
not skip the first prompt. Rebuilding the conditioning adds time to that reset, and with compiled inference a
prompt's first use can trigger compilation, so prefer a short list of prompts.

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

## Runtime and resets

The camera's post-processing chain is the integration boundary.
`CosmosTransferModifier` is implemented in `client/cosmos_modifier.py`, with its
`CosmosTransferModifierCfg` in `client/cosmos_modifier_cfg.py`. It uses the shared image
transfer modifier for queuing, resets, and publication, and validates Cosmos's
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

## Several environments

One session can generate the cameras of several environments as one batch. Each environment keeps its own
prompt, seed, and generation history, and resets on its own: a resetting environment starts its new episode with
one frame while the others continue with four. More environments share each transformer step, so throughput per
environment rises with GPU size, while each environment needs GPU memory for its own history.

Start the service in compiled mode (resetting one environment while the others continue needs the compiled
runtime). Omit `--warmup`: it compiles one `832 x 480` view and keeps that memory, while a batch of another size
or canvas compiles again on its first step.

```bash
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT" --kv-window 4
```

`--kv-window 4` fits two `640 x 640` environments on a 48 GB GPU that Isaac Sim shares; larger GPUs can keep the
default window of 30 latent frames, which remembers more of each episode.

A camera with a Cosmos chain requests one view per environment when its session opens. The server uses that
count without a separate view setting. A compiled server compiles again for each new count, and Torch limits how
often it recompiles: a server that has served one count can fail on another with `recompile_limit reached`. Restart
the server when the number of views changes. Available GPU memory
limits the batch: allocation can fail during generation as history grows, so a successful connection alone does
not establish that a full episode fits.

Ask for the environments with `--num_envs`, and give one prompt per environment as a list:

```bash
uv run isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Direct --rl_library rsl_rl --num_envs 2 \
  presets=cosmos \
  env.cosmos.prompt="['A close-up overhead view of a robotic Shadow Hand turning a red cube on a wooden workbench.', 'A close-up overhead view of a robotic Shadow Hand turning a blue cube in a bright white lab.']"
```

For a task without a preset, call `apply_cosmos(env_cfg, prompt=[...], num_envs=2)`.

- With several environments, environment `v` uses `prompt[v % len(prompt)]` for all its episodes; the batch keeps
  each environment's prompt across its resets. With one environment, a prompt list changes per episode.
- An environment that resets mid-chunk starts its new episode at the next chunk with its newest capture, so all
  environments keep one four-frame cadence; it shows black until then.
- A camera slower than the environment step captures each environment on its own schedule after independent
  resets; Cosmos receives a frame only for the environments a capture covers, so each environment's frames and
  budget follow its own captures.
- Independent resets of several environments need compiled inference and per-view VAE caches in the Framework
  runtime. `apply_cosmos` checks this before training; with a preset, a service without them stops the run with an
  error at the first independent reset.
- With the socket transport, one step's image message carries at most 16 environments.
- Each environment's history window (`--kv-window`, default 30 latent frames) takes GPU memory. On a 48 GB GPU,
  one environment fits with the default window. Two `640 x 640` Shadow Hand environments ran with `--kv-window 4`
  and no `--warmup` while Isaac Sim shared the GPU. Larger GPUs fit more environments, and
  [several services](#several-gpus) can split the environments across GPUs.
- Reference (RTX PRO 6000 Blackwell MIG 2g.48gb, compiled, 640 x 640, `--kv-window 8`): one environment 917 ms per
  step, two 1596 ms (1.15x the throughput), with 39.7 GiB peak memory for two.

### Record several environments

`--video sensor:<camera>:rgb` records environment 0 only. The Newton GL streaming view tiles the camera's `rgb`
output, the generated images, across environments; record it to see every environment side by side:

```bash
uv run --extra video isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Direct --rl_library rsl_rl --num_envs 2 \
  --max_iterations 20 \
  --viz newton_gl --video viz:newton_gl:streaming_view --video_length 600 \
  presets=cosmos \
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
boot ID), and uses the camera's GPU or a GPU with direct access to it, the socket otherwise. `cuda_ipc` fails when
it is unavailable instead of falling back. `status` lists the service's transports. With the defaults on one Linux
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

Isaac Lab keeps its own device, `cuda:0` by default. Set `env.cosmos.transport=cuda_ipc` (or `transport="cuda_ipc"`
in `apply_cosmos`) to require GPU-to-GPU transfer instead of a socket fallback. Each process also creates a CUDA
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
uv run isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Direct --rl_library rsl_rl --num_envs 4 \
  presets=cosmos \
  env.cosmos.endpoint="['unix:///tmp/isaaclab-cosmos-gpu1.sock', 'unix:///tmp/isaaclab-cosmos-gpu2.sock']" \
  env.cosmos.transport=cuda_ipc
```

The environments are split into contiguous shares of near-equal size, one per service in the order given:
environments 0-1 on the first service and 2-3 on the second here. Each environment stays on its service, which
keeps its generation history, and keeps its prompt: environment `v` uses `prompt[v % len(prompt)]` on any service.
Every step sends each service its share before waiting for any reply, so the services generate at the same time,
and a step lasts as long as the slowest service.

`apply_cosmos` checks that every service is running and idle and that a service with several environments runs
compiled; there must be at least one environment per service. If one service fails, the camera closes every
session. In a configuration, set `CosmosModelCfg.endpoint` to the list of endpoints.

## Episode frame budget

Each session requests a frame budget, `1 + 4*k` frames per episode; the server sets no limit of its own. The
service tells the model the episode's duration from this budget. `apply_cosmos` and the Shadow Hand presets derive it
from the task's episode duration and actual capture cadence, including the initial capture and rounding up to
`1 + 4*k`. Low-level camera chains must set `CosmosModelCfg.max_episode_frames` explicitly. Reset the environment
before that budget is exhausted; the service rejects further generation rather than silently resetting the episode.

Startup logs show the budget and the captures and generated updates per second of simulation time. The model uses
a sliding history window; longer episodes still need quality and memory validation on the intended workload.

See [Image transfer for camera images](image_transfer.md) for the underlying model
contract, camera scheduling, and optional PPISP processing. Service shutdown and
environment setup are covered in [Cosmos service setup](cosmos_service.md).
