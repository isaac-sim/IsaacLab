---
name: isaaclab-randomizing-with-events
description: Implements Isaac Lab domain randomization with event terms in direct and manager-based workflows. Use when adding domain randomization, randomizing physics or observations, porting reset randomization, or configuring event-based variation.
audience: user
status: stable
owners:
  - isaaclab-maintainers
---

# Randomizing With Events

## When To Use

Use this skill when a user wants to add domain randomization to an Isaac Lab task through event terms.

Do not use this skill for unrelated curriculum, command sampling, or reward shaping unless those changes interact with randomization.

## Workflow

1. Identify what should vary: assets, physics properties, observations, initial state, external disturbances, or rendering.
2. Identify the task workflow: direct (`DirectRLEnv` or `DirectMARLEnv`) or manager-based. Both can use `EventManager` through an `events` config.
3. Identify the active target backend: PhysX, Newton, or both through `PresetCfg`.
4. Check whether the randomization function has backend-specific behavior or unsupported backends in `source/isaaclab/isaaclab/envs/mdp/events.py`.
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

## Validation

Use the plan-validate-execute loop:

1. List each randomized property, target entity, backend, event mode, distribution, range, and timing limitation.
2. Check the list against the environment config before editing.
3. Check CPU/GPU expectations in the implementation. Some PhysX paths use CPU tensors while Newton paths may operate on the environment device.
4. Run a small reset or rollout smoke test for every targeted backend.
5. Fix shape, device, backend, and entity-name errors before scaling.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `source/isaaclab/isaaclab/managers/event_manager.py`, `source/isaaclab/isaaclab/envs/direct_rl_env.py`, `source/isaaclab/isaaclab/envs/direct_marl_env.py`, the direct and manager-based environment tutorials, and the managers API docs. If event-term behavior or mode semantics change, update the official docs or examples first and keep this skill focused on selecting the right workflow.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Event manager source](../../../source/isaaclab/isaaclab/managers/event_manager.py)
- [Direct workflow randomization tutorial](../../../docs/source/tutorials/03_envs/create_direct_rl_env.rst)
- [Manager-based event terms tutorial](../../../docs/source/tutorials/03_envs/create_manager_base_env.rst)
- [Managers API](../../../docs/source/api/lab/isaaclab.managers.rst)
