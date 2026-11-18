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
| `Articulation` | Virtual articulation. Reads and writes the manager's joint state, so the task's joint observation and action terms work unchanged. Everything that needs a body raises. |
| `reset_from_records` | Reset event that starts an episode from a reset record, in place of `reset_joints_by_*`. |
| `at_chunk_boundary` | Makes a termination function fire only where environments can reset. |
| `Cosmos3Cfg`, `Cosmos3Model` | Cosmos3 `unified_v1` implementation of `WorldModel` (`isaaclab_neural.models.cosmos3`). |

## Using a task's configuration

The task declares an `ArticulationCfg` and a `CameraCfg` as it would for a physics backend, and selects the world
model with `WorldModelCfg(model=..., reset_manifest=...)` and `WorldModelRendererCfg()`. The camera needs
`frame_history_length` equal to the model's chunk length.

* **Joints and limits** come from the spawned USD asset (a URDF spawner converts to USD): the movable joints in
  traversal order, or in the order of `ArticulationCfg.joint_ordering`, which must match the reset records. Joint
  limits are read from the USD, and `soft_joint_pos_limit_factor` applies as usual.
* **Observations and actions**: `joint_pos` terms read the joint state and `JointPositionAction` writes the target.
  Velocity and effort targets, poses, forces and physical properties raise `NotImplementedError`.
* **Resets**: `Articulation.reset` only clears the command buffers, as for a physics backend. The reset event
  `reset_from_records` samples a reset record (`image`, `state`, `prompt`, `split` in the JSON manifest), resets the
  model from the record's image and prompt, and writes the record's joint state. Training samples records at random;
  other splits visit them in order. `envs_per_reset_record` consecutive environments share a record (default 1); a group-relative algorithm such as GRPO sets it to its group size so that a group compares rollouts of one scene.

## Timing

A model predicts `chunk_length` frames from `chunk_length` joint targets. Use `sim.dt = 1 / fps` and `decimation = 1`:
one environment step is one frame, and the model runs on the last step of a chunk. The frames of a chunk arrive
together, so a reward on them is paid in the step that completes the chunk.

Isaac Lab resets an environment in any step in which a termination fires, but the model can only restart between
chunks, so a reset in the middle of a chunk raises.

* `reset_from_records` checks when the environment is created that one step is one frame and that `episode_length_s`
  is a multiple of the chunk duration, so that the time-out falls on a boundary.
* A termination you add must fire only on boundaries. Wrap its function with `at_chunk_boundary`:

  ```python
  done = DoneTerm(func=at_chunk_boundary(my_condition), params={...})
  ```

  The condition is judged on the step that completes a chunk. A condition that stays true, such as a success or a
  failure, is caught at the next boundary, at most one chunk late; one that holds only briefly inside a chunk is
  missed.
* Each environment resets at its own next boundary; it does not wait for the others.

## Add a world model

Subclass `ModelCfg` and `WorldModel`, and set the configuration's `class_type`:

| Method | Responsibility |
| --- | --- |
| `spec()` | Chunk length, frame rate, image size and the largest number of environments. |
| `reset(env_ids, images, prompts, seeds)` | Reset the selected environments; return their first frames `(K, H, W, 3)`. |
| `step(joint_targets, start_joint_state)` | Joint targets `(N, T, J)` and the state before the chunk `(N, J)` to frames `(N, T, H, W, 3)`. |
| `close()` | Release resources. |

Construction is cheap; load the weights on the first `reset()`.

## Cosmos3

```bash
uv sync --extra cosmos3 --extra importers
uv run python scripts/tools/cosmos/prepare_medical_pad.py --download
uv run python scripts/tools/cosmos/cosmos3_replay.py \
    --checkpoint .pretrained_checkpoints/medical_pad_unfold/world_model --output outputs/cosmos3/replay
```

`prepare_medical_pad.py` checks the model files listed in the task's
[`artifacts.json`](../../isaaclab_tasks/isaaclab_tasks/contrib/medical_pad_unfold/assets/artifacts.json) and
points the world model's config at them. The replay script drives `Cosmos3Model` with recorded controls and writes
`generated.mp4`.

`Cosmos3Model.step` converts joint targets to 59 normalized controls with the task's `JointMappingCfg`, so the robot
description and the normalization stay with the task. `window` mode serves any number of environments, one after the
other; `session` mode serves one.

### The medical-pad task

`Isaac-Cosmos3-Medical-Pad-Unfold-v0` is a manager-based task on this backend. It uses `JointPositionAction` over the
58 recorded H2 joints, a `joint_pos` observation, the head camera's image and `reset_from_records`. Its reward and its
success and failure terminations read one `ClassifierTracker`, which judges each generated chunk frame by frame:

* An episode **succeeds** (**fails**) at the first frame that completes 30 consecutive frames of the classifier's best
  (worst) class. The streaks continue across chunks and start again with each episode.
* The **reward** is the sum of the classifier's per-frame progress up to and including that frame; later frames
  belong to an episode that has already ended. The term's weight is the frame rate, which cancels the reward manager's
  `step_dt` scaling.
* The **terminations** report the outcome when the chunk completes. Episodes also time out after 30 chunks.

The URDF import needs the `importers` extra. The task fixes the 17 joints that the reset records do not cover, so the
articulation's joints are exactly the 58 recorded ones.

## Tests

```bash
uv run python -m pytest source/isaaclab_neural/test -q
```

The tests use a fake model: no weights, GPU or simulation app.
