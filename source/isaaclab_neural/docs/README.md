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
| `WorldModelManager` | Physics backend (`backend_name = "neural"`). Buffers one joint target per step and calls the model once per chunk. |
| `WorldModel` | Abstract model: `spec()`, `reset()`, `step()`, `close()`. |
| `WorldModelRendererCfg` | Selected as a camera's `renderer_cfg`. Copies the generated frames into the camera, including its frame history. |
| `FrameBuffer`, `FrameReader` | Keep one device-side frame buffer per model view and independent read progress per camera. |
| `Articulation` | Virtual articulation. Reads and writes the manager's joint state, so the task's joint observation and action terms work unchanged. Everything that needs a body raises. |
| `reset_from_records` | Reset event that starts an episode from a reset record, in place of `reset_joints_by_*`. |
| `at_chunk_boundary` | Makes a termination function fire only where environments can reset. |

## Using a task's configuration

The task declares an `ArticulationCfg` and a `CameraCfg` as it would for a physics backend, and selects the world
model with `WorldModelCfg(model=..., reset_manifest=...)` and `WorldModelRendererCfg()`. The camera needs
`frame_history_length` equal to the model's chunk length.

The model names its views in `WorldModelSpec.views`. Each camera selects one with
`WorldModelRendererCfg(view="front")`; `view=None` selects the only view of a single-view model. A model step returns
all views of the same chunk together. Cameras read the latest published frames through their own `FrameReader`,
skipping unchanged steps without copying frames. The renderer's `provides_frame_history` capability is declared in
its configuration, so the camera leaves that history to the renderer.

* **Joints and limits** come from the spawned USD asset (a URDF spawner converts to USD): the movable joints in
  traversal order, or in the order of `ArticulationCfg.joint_ordering`, which must match the reset records. Joint
  limits are read from the USD, and `soft_joint_pos_limit_factor` applies as usual.
* **Observations and actions**: the articulation shares the manager's joint-state and target tensors on the simulation
  device, so `joint_pos` terms read the state and `JointPositionAction` writes the target without per-step host copies.
  Velocity and effort targets, poses, forces and physical properties raise `NotImplementedError`.
* **Resets**: `Articulation.reset` only clears the command buffers, as for a physics backend. The reset event
  `reset_from_records` samples a reset record (`image`, `state`, `prompt`, `split` in the JSON manifest), immediately
  publishes its original image and joint state, and schedules the model to restart from its image and prompt just
  before generating the next chunk. The configured `default_joint_pos` stays unchanged. Training samples records at random;
  other splits visit them in order. `envs_per_reset_record` consecutive environments share a record (default 1); a group-relative algorithm such as GRPO sets it to its group size so that a group compares rollouts of one scene.

  Record sampling and reset state/image updates run on the simulation device. Only at the next chunk generation
  does the manager read reset metadata on the host to select prompts and model seeds. The sampler uses a seeded
  Torch generator; its random sequence differs from the previous NumPy sampler, including with the same seed.

## Timing

A model predicts `chunk_length` frames from `chunk_length` joint targets. Use `sim.dt = 1 / fps`; `decimation` must
divide `chunk_length`. One environment step advances `decimation` model frames, and the model runs on the last
backend step of a chunk. The frames of a chunk arrive together, so a reward on them is paid in the environment step
that completes the chunk. For a policy that produces a sequence of actions at the model's FPS, keep `decimation = 1`
and submit one queued action per environment step; the policy network can run once per action sequence.

Isaac Lab resets an environment in any step in which a termination fires, but the model can only restart between
chunks. A reset in the middle of a chunk is recorded on the device and raises when the chunk ends, before model
reset or generation. An all-false reset mask does not trigger this error. Deferring the check avoids reading the
device during reset; it does not make mid-chunk episode resets supported.

* `reset_from_records` uses `check_chunk_timing` when the environment is created to check that decimation divides
  the chunk length and that `max_episode_length * decimation` is a multiple of `chunk_length`, so the time-out falls
  on a boundary. `at_chunk_boundary` also checks timing on its first call for each environment.
* A termination you add must fire only on boundaries. Wrap its function with `at_chunk_boundary`:

  ```python
  done = DoneTerm(func=at_chunk_boundary(my_condition), params={...})
  ```

  The condition is judged on the step that completes a chunk. A condition that stays true, such as a success or a
  failure, is caught at the next boundary, at most one chunk late; one that holds only briefly inside a chunk is
  missed.
* Environments share one chunk clock. At a boundary, only the selected environments reset; the others continue.

## Add a world model

Implement `WorldModel`, subclass `ModelCfg`, and set the configuration's `class_type`. The manager constructs the
implementation as `class_type(cfg, num_envs, device)`. `WorldModel` is a pure interface with no constructor or stored
state; each implementation keeps the configuration and resources it needs.

| Method | Responsibility |
| --- | --- |
| `spec()` | Chunk length, frame rate, image size and the largest number of environments. |
| `reset(env_ids, images, prompts, seeds)` | Initialize selected environments from a `{view: tensor}` dictionary of start images `(K, H, W, 3)` on the simulation device. IDs, prompts and seeds are host lists. Return nothing. |
| `step(joint_targets, start_joint_state)` | Joint targets `(N, T, J)` and the state before the chunk `(N, J)` to `{view: frames}`; each frame tensor is `(N, T, H, W, 3)`, `uint8`, on the simulation device. |
| `close()` | Release resources. |

Construction is cheap; load the weights on the first `reset()`.

Joint targets accumulate on the simulation device. At a chunk boundary the manager first resets any environments
that restarted, then passes device tensors to `WorldModel.step` and publishes the returned view tensors on that
device. A model adapter can convert controls to NumPy internally if needed. Reset images are loaded once during
`bind` and must match the model's image size. A single-view record keeps `"image": "start.png"`; a multi-view record
uses `"image": {"front": "front.png", "left": "left.png", "right": "right.png"}` with one image per model view.

## Tests

```bash
uv run python -m pytest source/isaaclab_neural/test -q
```

The tests use a fake model: no weights, GPU or simulation app.
