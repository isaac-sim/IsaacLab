# Isaac Lab: Neural simulators

`isaaclab_neural` provides a learned world-model physics backend and a renderer for manager-based tasks. The current
backend accepts joint-position targets and predicts RGB video. It reads joint metadata from the robot's USD asset
and runs without a physics engine. Tasks access the results through the standard Articulation and Camera APIs.

The manager owns joint state: the recorded position at reset, then the last commanded position target. The model
predicts visual observations only. Joint state describes the commanded trajectory, without simulated tracking
error, contact response, velocity or effort dynamics.

```text
Policy / action queue -> Task action term -> Articulation target buffer
                                                   | shared with manager
                                            WorldModelManager
                                                   | one call per chunk
                                               WorldModel
                                                   | RGB chunks per view
Task observations <- Camera <- WorldModelRenderer <- FrameBuffer
```

## Components

| Component | Responsibility |
| --- | --- |
| `WorldModelCfg` | Physics configuration with the model config, reset manifest, split, seed and record grouping. |
| `WorldModelManager` | Physics backend (`backend_name = "neural"). Owns shared joint buffers, collects targets and publishes model frames. |
| `WorldModel` / `ModelCfg` | Pure model interface and configuration selecting its concrete implementation. |
| `Articulation` | Exposes shared joint positions and targets, names and position limits from USD, and the configured default pose. |
| `WorldModelRenderer` / `WorldModelRendererCfg` | Routes one named model view into a Camera's RGB output and frame history. |
| `FrameBuffer` / `FrameReader` | Device-side frames per view and independent read progress per camera. |
| `reset_from_records` | Event term that selects records and restores selected environments' state and images. |
| `check_chunk_timing` / `at_chunk_boundary` | Validate environment timing and restrict termination results to chunk boundaries. |

## Configure a task

Use ordinary `ArticulationCfg` and `CameraCfg` declarations, with these backend-specific choices:

* Select `WorldModelCfg(model=..., reset_manifest=...)` for `sim.physics` and `WorldModelRendererCfg(...)` for each
  camera's `renderer_cfg`. A task can expose these through its existing `physics=` and `renderer=` presets.
* Replace physics joint-reset events with `reset_from_records`. Events that reset root poses or other physical
  properties are unsupported; a record reset restores the matching joint state and image together.
* Set each camera's `frame_history_length` to `spec.chunk_length`, and its height and width to `spec.image_size`.
  History is required by this renderer even when the task uses only the latest RGB image.
* Set `sim.dt = 1 / spec.fps`. Use a `decimation` that divides the chunk length and an episode containing a whole
  number of chunks. Wrap task-specific termination functions with `at_chunk_boundary`.
* Keep reset records and model joint inputs in the robot's joint order. Movable revolute and prismatic joints come
  from the first robot instance in USD traversal order, or are reordered by `ArticulationCfg.joint_ordering`. Fixed
  joints are excluded. The manager checks record joint counts; matching counts do not verify joint order.

For example, a task's event configuration can declare:

```python
from isaaclab.managers import EventTermCfg
from isaaclab_neural.mdp import reset_from_records

reset = EventTermCfg(func=reset_from_records, mode="reset")
```

The backend supports one articulation containing the vectorized environments. A second independent articulation
in the same initialization is rejected. `spec.max_envs`, when set, also limits the number of parallel environments.
A full simulation reset permits rebinding and closes the previous model before replacing it.

Joint-position observation and action terms can use the standard APIs. Terms requiring body or root poses,
Jacobians, velocities, forces or physical properties are unsupported. Joint velocity writes during reset are
ignored; velocity and effort control targets raise `NotImplementedError`. Reusing a physics task therefore requires
checking its events and terms, not just changing the backend preset.

## Episode lifecycle

1. **Initialize.** The articulation reads joint metadata and binds the manager. The manager constructs the model,
   reads `spec()`, checks the environment limit and backend timestep, allocates shared joint and per-view frame
   buffers, and loads the selected split's reset images onto the simulation device. Cameras bind readers to those
   frame buffers. The reset event validates environment-step and episode timing when its term is constructed.
2. **Reset selected environments.** `Articulation.reset()` clears their position targets to the current joint state.
   The `reset_from_records` event then selects records, writes their joint positions into both shared state and
   target buffers, and publishes their original RGB images immediately. History repeats each start image until a
   generated chunk arrives. The model is marked for restart, but is not called yet.
3. **Collect targets.** Each backend step stores one joint target and copies it into joint state. The first step of
   a chunk also saves its starting joint state. Cameras keep their last published images during these steps.
4. **Generate.** On the last backend step of a chunk, the manager calls `model.reset()` for environments marked
   since the previous generation, then calls `model.step()` once with all environments' target trajectories and
   starting states. It publishes the returned views; Camera readers copy them on their next update/read.
5. **Evaluate and continue.** Task rewards can use the latest image or the full chunk history. Terminations take
   effect at chunk boundaries. Only selected environments restart; other model histories continue.

The configured `default_joint_pos` stays fixed across episodes. A recorded reset state can differ from that pose,
so `joint_pos_rel` remains relative to the configured default. Record randomization supplies initial-state
variation; applying a separate random joint offset can make state inconsistent with the record image.

Records contain `image`, `state`, `prompt` and optional `split` (default `"train"`). Relative image paths resolve
against the manifest directory. A single-view record uses `"image": "start.png"`; a multi-view record uses
`"image": {"front": "front.png", "left": "left.png"}`, with one correctly sized image per model view.

Training draws records randomly with a seeded Torch generator; other splits visit them sequentially. Consecutive
groups of `envs_per_reset_record` environments share a draw within a reset call, for group-relative algorithms such
as GRPO. The group size must divide the environment count. Each environment receives its own model seed; sharing a
record does not require sharing generated history. Torch's random sequence differs from NumPy's for the same seed.

## Timing and action sequences

For chunk length `T`, model frame rate `f` and environment decimation `d`:

| Quantity | Value |
| --- | --- |
| Backend step duration | `1 / f` |
| Environment step duration | `d / f` |
| Duration advanced per model generation | `T / f` |
| Environment steps per chunk | `T / d`, requiring `T % d == 0` |
| Episode alignment | `(max_episode_length * d) % T == 0` |

These are simulation-time intervals, independent of training or inference wall-clock speed. The current contract
uses one joint target per predicted frame; it does not declare separate action and observation sampling rates.
The whole generated chunk arrives together. The prediction horizon and the interval advanced between model calls
are both `T / f`.

If a policy produces 16 different actions at 30 Hz and the model consumes 16 targets per chunk, use `decimation=1`
and submit one queued action per environment step:

```text
Latest image -> policy inference -> [A0, A1, ..., A15]
                                    | one action per environment step
                              manager collects 16 joint targets
                                    |
                              model generates 16 frames
                                    |
                              latest image for next policy inference
```

The policy wrapper owns that action queue. The manager buffers the resulting joint targets, not raw policy action
chunks. With `JointPositionAction`, increasing `decimation` reapplies the same processed target for several backend
steps; it does not execute the next action from a policy's sequence. Calling a policy again inside a chunk also
does not supply fresh RGB: only joint state and other task-maintained inputs can change before generation.

For a policy loop consuming one action sequence per model generation, the sequence must cover `T / d` environment
steps. Validate this in the policy integration; the backend does not inspect RLinf's action-chunk configuration or
check that alignment at startup.

All environments share one chunk clock. An actual reset inside a chunk is recorded on the device and raises at
the next boundary, before model reset or generation; an all-false reset mask does not cause this error. Deferring
the check avoids a host read during reset. It does not support mid-chunk episode restarts.

`reset_from_records` checks episode timing during term construction. `at_chunk_boundary` also checks timing on its
first call for an environment. Use it on a termination function, for example:

```python
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_neural.mdp import at_chunk_boundary

done = DoneTerm(func=at_chunk_boundary(my_condition), params={...})
```

The condition is evaluated every environment step, but its result is masked between boundaries. A persistent
condition is reported at the next boundary; a condition true only between boundaries can be missed. A frame-history
tracker can instead remember a success/failure streak and report it when that chunk arrives. `is_chunk_boundary()`
describes the last backend step: it stays true across reset and becomes false on the first step of the next chunk.

## Model contract

Implement `WorldModel`, subclass `ModelCfg` and set `class_type`. Construction is
`class_type(cfg, num_envs, device)`. The base class has no constructor or stored state. Keep construction and
`spec()` cheap; load weights on the first model reset or generation call.

Let `N` be the environment count, `J` the joint count, `T` the chunk length and `K` the number of environments being
reset. Joint input tensors are `float32`; values are meters for prismatic joints and radians for revolute joints.

| Method | Contract |
| --- | --- |
| `spec()` | Return `chunk_length`, `fps`, `image_size=(H, W)`, `views` and optional `max_envs`. `None` means no declared environment limit. |
| `reset(env_ids, images, prompts, seeds)` | Initialize selected environments immediately before their next generation. IDs, prompts and seeds are host lists; images are `{view: tensor}` of shape `(K, H, W, 3)`, `uint8`, on the simulation device. Return `None`. |
| `step(joint_targets, start_joint_state)` | Accept device Torch tensors `(N, T, J)` and `(N, J)`; return `{view: tensor}` with RGB tensors `(N, T, H, W, 3)`, `uint8`, on that device. Return generated frames only, excluding a conditioning prefix. |
| `close()` | Release model resources, including any worker or session the implementation owns. |

The manager reset interface accepts a device boolean `env_mask` of shape `(N,)`. The model reset interface receives
the selected IDs as a host list. Record sampling, state updates and image publication stay on the device; reset
metadata is read on the host at a chunk boundary to select prompts and model seeds. The prompt is passed to the
model; exposing it as a policy observation is the task integration's responsibility.

Target and starting-state buffers are reused for subsequent chunks. An adapter retaining either input must copy
it. Adapters can convert controls or communicate with another process internally, while honoring the device tensor
contract at the manager boundary. The Cosmos3 adapter below uses a separate inference service.

Each model step returns all named views for the same environments, time interval and frame positions. The adapter
owns that alignment. The manager checks view names and publishes the views together; the renderer does not infer
cross-view timing or enforce visual consistency. Each camera selects its view with `WorldModelRendererCfg(view=...)`;
`view=None` selects the sole view of a single-view model. Readers skip unchanged publications and update selected
environments after partial resets. `provides_frame_history` is declared on the renderer configuration.

The output dictionary names **views**, not observation modalities. Outputs currently support RGB/RGBA; depth,
normals and segmentation need additional contracts and renderer support. The current model interface takes joint
targets and produces no actions or joint state. Direct end-effector commands, separate action/observation rates,
and joint state supplied by another dynamics provider require further integration or a custom physics manager and
renderer; they are not established by the existing model interface or tests.

## Integration scope

Model-specific inference modes, kinematics and control encoding belong in concrete model adapters or their task
configuration. Reset datasets, policy prompt observations, classifier rewards and streak tracking belong with the
task. Keep these dependencies separate from the generic backend.

Camera outputs can be consumed by application recording code. Standard play/eval video recording through
`PhysicsManager.video_capture_backend()` is not implemented here; the inherited hook returns `None`. Task
integrations, policy wrappers and checkpoint quality need their own validation alongside this backend.

## Cosmos3 inference service

Isaac Lab runs `Cosmos3Model`, a client of a separate process in the Cosmos Framework environment. The service owns
weights and generation history. Closing a client releases its session, while the service keeps the weights loaded.
Only one client session can own the model at a time; that session may contain several vectorized environments.

```text
Isaac Lab: manager -> Cosmos3Model -> task joint mapping -> normalized controls
                                     | socket request
Cosmos environment: resident model <- service -> generated RGB chunk
                                     | CUDA IPC or socket reply
Isaac Lab: manager frame buffers <--- client device tensors
```

The standalone service package provides the socket protocol, CUDA IPC primitives and endpoint lifecycle used by
the client.

### Start the service

Use Cosmos Framework revision `3a84cd3b67f60520f523e184fa7b82011bc75cb4` with a compatible checkpoint. In the
Framework environment, install the lightweight service package from this checkout:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv pip install --python .venv/bin/python --editable \
  "$ISAAC_LAB_ROOT/source/isaaclab_experimental/isaaclab_experimental/cosmos"
uv run --no-sync isaaclab-cosmos-world-model-server \
  --checkpoint "$COSMOS3_CHECKPOINT" --mode window --device cuda:0
```

For a DCP checkpoint, also pass `--config-file /path/to/saved-config.yaml`. `--output-dir` selects the Framework
artifact directory. `--endpoint unix:///path/to/service.sock` or `--endpoint tcp://host:port` overrides the default;
use the same endpoint in the client. Wait for `Cosmos ready at ...` before starting the task.

The standalone package adds only NumPy; Torch, torchvision and Cosmos Framework come from the Framework environment.
The service CLI can display help without importing those inference dependencies. TCP has no authentication; use a
trusted connection or tunnel when the service runs on another machine.

### Configure the task's client

```python
from isaaclab_neural.models.cosmos3 import Cosmos3Cfg
from isaaclab_neural.physics import WorldModelCfg

physics = WorldModelCfg(
    model=Cosmos3Cfg(joint_mapping=my_robot_mapping_cfg),
    reset_manifest="/path/to/reset-records.json",
)
```

`my_robot_mapping_cfg` is a task-defined `JointMappingCfg`. Its implementation is constructed as `class_type(cfg)`
and implements `encode(joint_targets, start_joint_state)`: `(T, J)` targets and `(J,)` starting joint positions in
meters/radians become `(T, 59)` normalized `unified_v1` controls. Robot kinematics and normalization remain with the
task. Recorded normalized controls can instead be replayed through `Cosmos3Model.step_controls()` after reset.

| Configuration location | Settings |
| --- | --- |
| Isaac Lab `Cosmos3Cfg` | Endpoint, request timeout, transport and task joint mapping. |
| Service CLI | Checkpoint, saved config, mode, device, endpoint and artifact directory. |
| Service `Cosmos3Settings` | Also sets image size, chunk length, FPS and sampling parameters when starting programmatically. |

The client reads dimensions and environment limits from the running service. Defaults are 16 generated frames,
30 FPS and RGB at 480 x 640. Cosmos3 currently predicts the single view `camera`; the generic backend's multi-view
support does not make this adapter a multi-view model.

- `window` conditions on the reset image, then on the previous generated chunk's last frame. It serves environments
  sequentially within each chunk. Use this mode with Stage 0 checkpoints.
- `session` uses Cosmos's native causal session and requires a compatible distilled checkpoint. It serves one
  environment and preserves the native session's internal history.

`transport="auto"` selects CUDA IPC on Linux when the client can directly access the local service's GPU, and
otherwise uses socket transport. `"socket"` and `"cuda_ipc"` force the choice; forcing unavailable IPC raises an
error. Reset images and action controls pass through host memory and the socket in both modes. IPC is used for
returned image chunks, which the client copies into its own device tensors. The client records a CUDA event after
each copy and waits for it before the next request or closing the mapping, so the service cannot reuse or free the
buffer while the copy is pending, including when successive calls use different CUDA streams.
Socket payloads are capped at 256 MiB per message (about 18 environments at the default output size; allow headroom).
Each step waits for the generated chunk; running a service does not add asynchronous policy execution.

The backend publishes the record's original image immediately on reset and defers the model reset to the
next chunk boundary. The client reset returns no replacement image. In native `session` mode the model's internal
conditioning comes from its initial decoded observation, following the Framework API.
The native session owns its temporary start image and removes it when reset, closed or initialization fails.

## Tests

The backend tests use a deterministic model and local USD assets. Service tests use a fake inference model with
real client/server requests, so neither suite requires weights or a simulation app. Run CPU tests:

```bash
ISAACLAB_TEST_DEVICES=1 uv run python -m pytest source/isaaclab_neural/test -q
```

On a CUDA-capable host, include GPU cases:

```bash
ISAACLAB_TEST_DEVICES=11 uv run python -m pytest source/isaaclab_neural/test -q
```

Coverage includes chunked targets, shared articulation state, record sampling, partial resets, independent camera
readers and multiple RGB views, timing checks, model environment limits and CUDA synchronization checks. It does
not validate model quality, depth or end-effector interfaces, hybrid dynamics, or a complete training task. Service
tests cover socket requests, partial reset, client ownership, failed-session recovery and cross-process CUDA IPC;
the IPC case requires a CUDA-capable Linux host. These tests do not run checkpoint inference.
