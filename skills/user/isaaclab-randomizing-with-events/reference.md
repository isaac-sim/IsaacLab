# Event Randomization Reference

## Contents

- Direct and manager-based usage
- Event mode selection
- Backend compatibility
- Timing limitations
- Current workflow
- Old patterns
- Validation checklist

## Direct and Manager-Based Usage

Event randomization is not limited to manager-based environments. Direct RL and direct multi-agent RL configs also expose an `events` field that is handled by `EventManager`.

Use the same event-term structure in both workflows:

| Workflow | Where events are configured |
| --- | --- |
| Direct RL | Add an event config class to the direct task config's `events` field. |
| Direct MARL | Add an event config class to the direct multi-agent task config's `events` field. |
| Manager-based | Add an event config class to the manager-based env config's `events` field. |

In direct workflows, observations and rewards remain direct methods; only randomization is routed through event terms.

## Event Mode Selection

Use this default mapping:

| Desired behavior | Event mode |
| --- | --- |
| Author USD-level properties before simulation starts | `prestartup` |
| Randomize once after simulation starts | `startup` |
| Randomize at episode reset | `reset` |
| Apply repeated disturbances during an episode | `interval` |

## Backend Compatibility

Check each event function against the current implementation before assuming the same behavior on PhysX and Newton. Some event functions are backend-neutral through asset APIs, while others dispatch to backend-specific implementations.

Important examples from `source/isaaclab/isaaclab/envs/mdp/events.py`:

| Event concern | PhysX behavior | Newton behavior |
| --- | --- | --- |
| Rigid body material randomization | Bucket-based static friction, dynamic friction, and restitution; uses CPU tensors and the PhysX tensor API. | Continuous per-shape friction and restitution; Newton uses one friction coefficient, so `dynamic_friction_range` and `num_buckets` are ignored. |
| Rigid body CoM randomization | Writes full CoM pose data. | Writes position-only CoM data; runtime CoM changes may have different stability implications. |
| Collider offset randomization | Uses PhysX rest/contact offsets. | Maps to Newton margin/gap concepts. |

For multi-backend environments, use `PresetCfg` event configs. Keep PhysX-only terms in the PhysX preset and provide a Newton-compatible preset when an event is unsupported or has different parameters.

## Timing Limitations

Some randomizations are only valid before simulation starts. Use `prestartup` for USD-stage or topology-level authoring that must happen before physics views and simulation buffers are created. Do not move those changes to `reset` or `interval` just to get per-episode variation.

Use `startup` for one-time setup after simulation starts. Use `reset` only for state or parameter changes that the environment and backend support after initialization. Use `interval` for runtime disturbances such as pushes.

If the desired property cannot be changed after startup, choose between pre-generating variants, cloning separate authored assets, or documenting that the randomization is fixed per run.

## Current Workflow

Define randomization as event terms in the environment configuration. Each term should identify the workflow, target scene entity, backend, mode, distribution, range, and timing limitation.

Keep randomization ranges conservative at first. Expand ranges after reset and rollout smoke tests pass.

## Old Patterns

Avoid hiding domain randomization in environment constructors, training scripts, or reward functions. Those patterns make the randomized behavior hard to inspect and hard to validate.

## Validation Checklist

- Each randomized property has a target scene entity.
- The plan states whether the task is direct or manager-based.
- Each randomized property has an explicit backend compatibility note.
- Event modes match the intended timing.
- Prestartup-only changes are not placed in reset or interval events.
- CPU/GPU expectations match the event implementation.
- Distribution ranges use the expected units.
- Repeated resets produce valid states.
- Interval disturbances do not break tensor shapes or device placement.
