# Isaac Lab: Neural simulators

`isaaclab_neural` lets a learned **world model** replace the physics engine. A task keeps its ordinary articulation
and cameras; the world model supplies the joint state and the camera frames through the same interfaces.

```text
Policy / trainer  ->  Task  ->  Isaac Lab  ->  NeuralManager  ->  WorldModel
```

The world model only generates images. The manager keeps each environment's joint state itself (the reset record's
state at reset, then the last joint target), so joint observations and actions behave as on a physics backend.

## Components

| Component | Role |
| --- | --- |
| `NeuralCfg` | `PhysicsCfg` selected as the task's `physics=` preset. Holds the model configuration and the reset records. |
| `NeuralManager` | Physics backend (`backend_name = "neural"`). Buffers one joint target per step, calls the model once per chunk, and keeps the predicted frames of each view. |
| `WorldModel` | Abstract model: `spec()`, `reset()`, `step()`, `close()`. |
| `NeuralRendererCfg` | Selected as a camera's `renderer_cfg`, with the model view the camera shows. Copies that view's frames into the camera, including its frame history. |
| `Articulation` | Virtual articulation. Shares the manager's joint state and targets, so the task's joint observation and action terms work unchanged. Its reset starts an episode from a reset record. Everything that needs a body raises. |
| `at_chunk_boundary` | Makes a termination function fire only where environments can reset. |
| `Cosmos3Cfg`, `Cosmos3Model` | Client of the Cosmos3 world-model service (`isaaclab_neural.models.cosmos3`). |

## Using a task's configuration

The task declares an `ArticulationCfg` and a `CameraCfg` as it would for a physics backend, and selects the world
model with `NeuralCfg(model_cfg=..., reset_manifest=...)` and `NeuralRendererCfg()`. The camera needs
`frame_history_length` equal to the model's chunk length and the model's image size.

* **Views**: a model predicts one or more named camera views (`WorldModelSpec.views`). Each world-model camera shows
  one of them, set with `NeuralRendererCfg(view=...)`; for a model with one view, `view` can stay `None`. Several
  cameras of a multi-view model each select their own view.
* **Joints and limits** come from the spawned USD asset (a URDF spawner converts to USD): the movable joints in
  traversal order, or in the order of `ArticulationCfg.joint_ordering`, which must match the reset records. Joint
  limits are read from the USD, and `soft_joint_pos_limit_factor` applies as usual.
* **Observations and actions**: `joint_pos` terms read the joint state and `JointPositionAction` writes the target.
  Velocity and effort targets, poses, forces and physical properties raise `NotImplementedError`.
* **Resets**: `Articulation.reset`, which the scene calls before the reset events, samples a reset record (`image`,
  `state`, `prompt`, `split` in the JSON manifest) and starts the episode from it: the record's start image becomes the
  camera's frames (`image` is a path, or `{view: path}` for a model with several views), and its joint state the joint
  state and the default joint position. The model restarts the environment from the record's image and prompt at the end
  of the next chunk, before it predicts that chunk. The standard `reset_joints_by_offset` and `reset_joints_by_scale`
  events therefore start from the record; a non-zero range moves the joints away from the start image the model sees, so
  randomize through the records instead. `reset_scene_to_default`, the default reset event, also resets the root pose,
  which the world model does not have, so a task replaces the default event configuration. Training samples records at
  random; other splits visit them in order. `envs_per_reset_record` consecutive environments share a record (default 1);
  a group-relative algorithm such as GRPO sets it to its group size so that a group compares rollouts of one scene.
* **Device**: joint state, targets, frames and the start images of the split's records stay on the simulation device.
  Resets and the steps inside a chunk read nothing back to the host. The step that ends a chunk, which waits for the
  model anyway, reads which environments reset, since the records' prompts are host strings.

## Timing

A model predicts `chunk_length` frames from `chunk_length` joint targets. Use `sim.dt = 1 / fps` and `decimation = 1`:
one environment step is one frame, and the model runs on the last step of a chunk. The frames of a chunk arrive
together, so a reward on them is paid in the step that completes the chunk.

Isaac Lab resets an environment in any step in which a termination fires, but the model can only restart between
chunks, so a reset in the middle of a chunk raises when the chunk ends.

* Make `episode_length_s` a multiple of the chunk duration, so that the time-out falls on a boundary.
* A termination you add must fire only on boundaries. Wrap its function with `at_chunk_boundary`:

  ```python
  done = DoneTerm(func=at_chunk_boundary(my_condition), params={...})
  ```

  The condition is judged on the step that completes a chunk. A condition that stays true, such as a success or a
  failure, is caught at the next boundary, at most one chunk late; one that holds only briefly inside a chunk is
  missed.
* Each environment resets at its own next boundary; it does not wait for the others.

## Add a world model

Implement the `WorldModel` interface, subclass `ModelCfg` and set its `class_type`. The manager builds the model as
`class_type(cfg, num_envs, device)` with the simulation device, and the model keeps what it needs of those:

| Method | Responsibility |
| --- | --- |
| `spec()` | Chunk length, frame rate, image size, the largest number of environments and the names of the views. |
| `reset(env_ids, images, prompts, seeds)` | Restart the selected environments (host lists) from their start images, `{view: (K, H, W, 3)}`. Called right before `step()`, for the environments that reset since the last chunk. |
| `step(joint_targets, start_joint_state)` | Joint targets `(N, T, J)` and the state before the chunk `(N, J)` to frames `{view: (N, T, H, W, 3)}`. |
| `close()` | Release resources. |

Frames are `uint8` torch tensors on the device, and joint targets arrive there. The targets buffer is reused for the
next chunk, so a model that keeps it copies it. Construction is cheap; load the weights on the first call.

## Cosmos3

Cosmos3 runs in its own process, in the Cosmos Framework's environment, so Isaac Lab does not install the Framework.
`Cosmos3Model` is the client: it reads the chunk length, frame rate and image size from the service, sends start
images and controls, and receives frames. The service is the world-model counterpart of the Cosmos Transfer service
in [`cosmos_transfer.md`](../../isaaclab_experimental/docs/cosmos_transfer.md) and uses the same transport.

In a server terminal, create the Framework environment at commit `3a84cd3b67f60520f523e184fa7b82011bc75cb4` of
[cosmos-framework](https://github.com/NVIDIA/cosmos-framework), install the Cosmos service package into it, and start
the service with a `unified_v1` checkpoint:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv sync --locked --all-extras --group=cu130-train
uv run --no-sync uv pip install --editable "$ISAAC_LAB_ROOT/source/isaaclab_experimental/isaaclab_experimental/cosmos"
export LD_LIBRARY_PATH=
uv run --no-sync isaaclab-cosmos-world-model-server --checkpoint "$COSMOS3_CHECKPOINT"
```

Wait for `Cosmos ready at ...`, then run the task in the Isaac Lab environment. A task selects the model with
`NeuralCfg(model_cfg=Cosmos3Cfg(joint_mapping=...))`. `Cosmos3Model.step` converts joint targets to 59 normalized
controls with the task's `JointMappingCfg`, so the robot description and the normalization stay with the task.

The service's `--mode window` (default) serves any number of environments, one after the other; `--mode session`
serves one. With the service on the same machine and a GPU this process can access directly, chunks stay on the GPU
(CUDA IPC, Linux); otherwise they go through the socket, which carries about 16 environments of 16 frames at
480 x 640 per message. `Cosmos3Cfg.transport` forces either one.

### The medical-pad task

`Isaac-Cosmos3-Medical-Pad-Unfold-v0` is a manager-based task on this backend. It uses `JointPositionAction` over the
58 recorded H2 joints, a `joint_pos` observation and the head camera's image, and has no reset event. Its reward and its
success and failure terminations read one `ClassifierTracker`, which judges each generated chunk frame by frame:

* An episode **succeeds** (**fails**) at the first frame that completes 30 consecutive frames of the classifier's best
  (worst) class. The streaks continue across chunks and start again with each episode.
* The **reward** is the sum of the classifier's per-frame progress up to and including that frame; later frames
  belong to an episode that has already ended. The term's weight is the frame rate, which cancels the reward manager's
  `step_dt` scaling.
* The **terminations** report the outcome when the chunk completes. Episodes also time out after 30 chunks.

The URDF import needs the `importers` extra. The task fixes the 17 joints that the reset records do not cover, so the
articulation's joints are exactly the 58 recorded ones.

Prepare its checkpoints and start the service from the Isaac Lab checkout, with the Framework environment:

```bash
cd "$ISAAC_LAB_ROOT"
uv run --project "$COSMOS_FRAMEWORK_ROOT" --no-sync python scripts/tools/cosmos/prepare_medical_pad.py --download
export LD_LIBRARY_PATH=
uv run --project "$COSMOS_FRAMEWORK_ROOT" --no-sync isaaclab-cosmos-world-model-server \
    --checkpoint .pretrained_checkpoints/medical_pad_unfold/world_model
```

`prepare_medical_pad.py` checks the model files listed in the task's
[`artifacts.json`](../../isaaclab_tasks/isaaclab_tasks/contrib/medical_pad_unfold/assets/artifacts.json) and
points the world model's config at them; it also copies the Framework's action statistics, which the task's joint
mapping reads in the Isaac Lab environment. Then, in the Isaac Lab environment, replay recorded controls into
`generated.mp4`, or run the task:

```bash
uv sync --extra importers --extra video
uv run python scripts/tools/cosmos/cosmos3_replay.py --output outputs/cosmos3/replay
uv run python scripts/environments/random_agent.py --task Isaac-Cosmos3-Medical-Pad-Unfold-v0 --num_envs 2
```

Train a PPO policy from the head camera's image and the joint positions with `rsl_rl`. Window mode generates the
environments one after the other, about 3 s per environment per chunk of 16 steps, so start with a few environments
and iterations:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cosmos3-Medical-Pad-Unfold-v0 --num_envs 2 --max_iterations 3
```

## Tests

```bash
uv run python -m pytest source/isaaclab_neural/test -q
```

The tests use a fake model: no weights, GPU or simulation app.
