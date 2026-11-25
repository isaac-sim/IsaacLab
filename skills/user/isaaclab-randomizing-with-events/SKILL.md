---
name: isaaclab-randomizing-with-events
description: Implements Isaac Lab domain randomization with event terms and success-driven ADR curricula. Use when randomizing physics or observations, for example initial joint positions each episode, random pushes during training, or friction and restitution on PhysX and Newton, when porting reset randomization, or when configuring Automatic or Adaptive Domain Randomization (ADR).
license: Apache-2.0
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
  tags: [domain-randomization, events, curriculum, adr]
audience: user
status: stable
owners:
  - isaaclab-maintainers
---

# Randomizing With Events

## When To Use

Use this skill when a user wants to add domain randomization to an Isaac Lab task through event terms. Use ADR when policy success should change the randomization range.

Do not use this skill for unrelated curriculum, command sampling, or reward shaping unless those changes interact with randomization.

## Workflow

1. Identify what should vary: assets, physics properties, observations, initial state, external disturbances, or rendering.
2. Identify the task workflow: direct (`DirectRLEnv` or `DirectMARLEnv`) or manager-based. Both can use `EventManager` through an `events` config.
3. Identify the active target backend: PhysX, Newton, or both through `PresetCfg`.
4. Check whether the randomization function has backend-specific behavior or unsupported backends in [events](../../../source/isaaclab/isaaclab/envs/mdp/events.py).
5. Choose the event mode:
   - Use prestartup events for USD-level properties that must be authored before simulation starts.
   - Use startup events for one-time setup randomization after simulation starts.
   - Use reset events for per-episode randomization.
   - Use interval events for repeated disturbances during an episode.
6. Check timing limitations before editing. Some USD-stage or topology-level changes are prestartup-only and cannot be safely moved to reset or interval events.
7. Define event terms in the environment configuration. In direct workflows, assign the event config to the task config's `events` field. In manager-based workflows, assign it to the manager-based env config's `events` field.
8. Use backend-specific `PresetCfg` event configs when PhysX and Newton need different terms.
9. Scope each term to the correct scene entities.
10. Use one clear distribution and range for each randomized quantity.
11. Validate with a small number of environments and repeated resets on each backend.
12. Expand ranges only after the baseline randomized task is stable.
13. For manager-based ADR, validate the final range as a fixed event, then use a task-owned scheduler and `modify_term_cfg` to interpolate it; see the [gravity example](examples.md#success-driven-adr).

## Examples

Representative requests this skill handles (worked patterns in [examples.md](examples.md)):

- "Randomize initial joint positions every episode."
- "Apply random pushes to the robot during training."
- "Randomize friction and restitution for both PhysX and Newton."

## Validation

Use the plan-validate-execute loop:

1. List each randomized property, target entity, backend, event mode, distribution, range, and timing limitation.
2. Check the list against the environment config before editing.
3. Check CPU/GPU expectations in the implementation. Some PhysX paths use CPU tensors while Newton paths may operate on the environment device.
4. Run a small reset or rollout smoke test for every targeted backend.
5. Fix shape, device, backend, and entity-name errors before scaling.
6. For ADR, test the exact endpoints and confirm policy success changes the next reset's range within its bounds.

For skill changes, run the repository skill validator; the exact `check` command is in the [Skills README](../../README.md#authoring-rules).

## Maintenance

Keep this skill synchronized with [event manager](../../../source/isaaclab/isaaclab/managers/event_manager.py), [direct rl env](../../../source/isaaclab/isaaclab/envs/direct_rl_env.py), [direct marl env](../../../source/isaaclab/isaaclab/envs/direct_marl_env.py), the direct and manager-based environment tutorials, and the managers API docs. For ADR, follow the curriculum guide and Core Lift example.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Event manager source](../../../source/isaaclab/isaaclab/managers/event_manager.py)
- [Direct workflow randomization tutorial](../../../docs/source/how-to/create_direct_rl_env.rst)
- [Manager-based event terms tutorial](../../../docs/source/how-to/create_manager_base_env.rst)
- [Curriculum utilities guide](../../../docs/source/how-to/curriculums.rst)
- [Core Lift ADR config](../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/adr_curriculum.py)
- [Managers API](../../../docs/source/api/lab/isaaclab.managers.rst)
