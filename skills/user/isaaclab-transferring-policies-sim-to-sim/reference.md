# Sim-To-Sim Policy Transfer Reference

This reference follows the sections in the [sim-to-sim how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst).

## Task Readiness And Checkpoint Compatibility

The same registered task should retain one MDP. Resolve each backend's `PresetCfg`; use backend
alternatives for intentional physics, asset, and control differences without changing
checkpoint-facing MDP terms. If an MDP-term preset differs, restore one contract or treat it as a
different task and retrain. Ensure the task trains in both engines, then compare:

| Contract | Required equality |
| --- | --- |
| Actions | term order, ordered targets, width, type, scale, offset, clipping |
| Observations | group/term order, width, history, units, frames, clipping, corruption |
| Policy state | normalization, recurrent state and reset, commands, actor inputs |
| Timing | `dt`, decimation, policy period, action hold; Newton substeps may differ internally |
| Mechanism | ordered bodies/joints, active DOFs, mimic/equality coupling |
| Episode | resets, commands, rewards, terminations, horizon, success |

## Mimic-Joint Action Nuance

Both backends preserve `panda_finger_joint1` and `panda_finger_joint2`, but their drive graphs differ:

- Newton imports the mimic relation as a constraint, lowers it to `mjEQ_JOINT`, drives `panda_finger_joint1`, and moves `panda_finger_joint2` through the equality.
- PhysX creates a native two-way mimic constraint, but the follower remains driveable.
- If a wildcard actuator gives both fingers nonzero stiffness and damping, one logical command written to both targets produces two PhysX PD-drive contributions versus one in MJWarp.
- Drive only `panda_finger_joint1`. Keep `panda_finger_joint2` in the articulation but set its stiffness and damping to zero.

Action width and order still belong to the exact checkpoint contract and must match across both task variants.

## Transferring Control Behavior

Match nominal actuator response before policy tuning:

- distinguish `velocity_limit` from `velocity_limit_sim`;
- use per-joint effort, stiffness, damping, friction, and armature;
- preserve `dt * decimation` and action hold;
- keep targets away from hard stops; and
- monitor saturation and consecutive action sign changes.

Use enough damping to prevent saturated bang-bang control. Armature bounds the acceleration produced by impulses in low-inertia MJWarp coordinates. Retune damping after increasing armature.

## Introducing Domain Randomization

| Family | Required nuance |
| --- | --- |
| Robot/object friction | Newton currently uses one coefficient; PhysX static/dynamic values and buckets do not map one-to-one. |
| Object mass/inertia | Keep inertia positive and physically consistent; decide whether mass changes recompute inertia. |
| Joint gains/friction | Cover actuator and solver-response uncertainty. |
| Joint armature | Use positive, physically supported ranges and coherent coupled mechanisms. |
| Gravity | Progress to full nominal gravity and evaluate there. |
| Actuator response | Randomize behavior such as gripper closing speed. |
| Reset pose/geometry | Recheck collision-valid resets for every geometry. |
| Observation noise | Preserve tensor shape/order and use plausible sensor noise. |

Use plausible ranges around a corrected nominal model. Use curriculum only when the final distribution blocks learning, promote to final difficulty, and retain deterministic nominal evaluation.

## Validate The Full Matrix

| Training | Deployment | Label |
| --- | --- | --- |
| PhysX | PhysX | PP |
| PhysX | Newton MJWarp | PN |
| Newton MJWarp | Newton MJWarp | NN |
| Newton MJWarp | PhysX | NP |

Generic command pattern:

```bash
uv run isaaclab train --rl_library rsl_rl --task TRAIN_TASK physics=isaacsim_physx
uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/physx_checkpoint.pt physics=isaacsim_physx
uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/physx_checkpoint.pt physics=newton_mjwarp

uv run isaaclab train --rl_library rsl_rl --task TRAIN_TASK physics=newton_mjwarp
uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/newton_checkpoint.pt physics=newton_mjwarp
uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/newton_checkpoint.pt physics=isaacsim_physx
```

### Run The Franka Lift Transfer

For Franka, use `Isaac-Lift-Franka` to train and infer. The play entry point applies the task's `play_mode` overrides and disables Franka gripper-closing-speed randomization automatically. Use the exact PP, PN, NN, and NP commands from the how-to.
