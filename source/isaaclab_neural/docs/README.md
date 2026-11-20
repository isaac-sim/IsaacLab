# Isaac Lab: Neural simulators

`isaaclab_neural` lets a learned **world model** replace the physics engine. A task keeps its ordinary articulation
and cameras; the world model supplies the joint state and the camera frames through the same interfaces.

```text
Policy / trainer  ->  Task  ->  Isaac Lab  ->  WorldModelManager  ->  WorldModel
```

The world model only generates images. The manager keeps each environment's joint state itself (the reset record's
state at reset, then the last joint target), so joint observations and actions behave as on a physics backend.

## Components

| Component | Role |
| --- | --- |
| `WorldModelCfg` | `PhysicsCfg` selected as the task's `physics=` preset. Holds the model configuration and the reset records. |
| `WorldModelManager` | Physics backend (`backend_name = "neural"`). Buffers one joint target per step, calls the model once per chunk, and keeps the predicted frames of each view. |
| `WorldModel` | Abstract model: `spec()`, `reset()`, `step()`, `close()`. |
| `WorldModelRendererCfg` | Selected as a camera's `renderer_cfg`, with the model view the camera shows. Copies that view's frames into the camera, including its frame history. |
| `Articulation` | Virtual articulation. Shares the manager's joint state and targets, so the task's joint observation and action terms work unchanged. Its reset starts an episode from a reset record. Everything that needs a body raises. |
| `at_chunk_boundary` | Makes a termination function fire only where environments can reset. |

## Using a task's configuration

The task declares an `ArticulationCfg` and a `CameraCfg` as it would for a physics backend, and selects the world
model with `WorldModelCfg(model=..., reset_manifest=...)` and `WorldModelRendererCfg()`. The camera needs
`frame_history_length` equal to the model's chunk length and the model's image size.

* **Views**: a model predicts one or more named camera views (`WorldModelSpec.views`). Each world-model camera shows
  one of them, set with `WorldModelRendererCfg(view=...)`; for a model with one view, `view` can stay `None`. Several
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

## Tests

```bash
uv run python -m pytest source/isaaclab_neural/test -q
```

The tests use a fake model: no weights, GPU or simulation app.
