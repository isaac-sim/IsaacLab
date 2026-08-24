---
name: isaaclab-randomizing-with-events
description: Implements fixed and adaptive Isaac Lab domain randomization with event and curriculum terms. Use when randomizing physics, observations, or resets; configuring Automatic or Adaptive Domain Randomization (ADR); expanding ranges from policy success; or porting randomization into direct or manager-based tasks.
audience: user
status: stable
owners:
  - isaaclab-maintainers
---

# Randomizing With Events

## When To Use

Use this skill for fixed event-based randomization or success-driven Automatic/Adaptive Domain Randomization (ADR).

Do not use this skill for unrelated curricula, command sampling, or reward shaping.

## Workflow

1. Identify the property, scene entity, physical units, and target backend.
2. Choose the default:
   - Use a fixed event range when training should sample the same distribution throughout.
   - Use ADR when the range should grow or shrink from a stable policy-success signal.
3. Identify the workflow. Direct and manager-based tasks both support event terms, but the maintained `CurriculumManager` ADR pattern is manager-based.
4. Choose the event mode:
   - Use prestartup events for USD-level properties that must be authored before simulation starts.
   - Use startup events for one-time setup randomization after simulation starts.
   - Use reset events for per-episode randomization.
   - Use interval events for repeated disturbances during an episode.
5. Read the event implementation before editing. Check backend behavior and whether the property can change after startup.
6. Implement and validate the intended final range as a fixed event first. Use backend-specific `PresetCfg` terms when PhysX and Newton differ.
7. For ADR, follow the [gravity example](examples.md#success-driven-adr):
   - Keep the randomization in its event term.
   - Add a task-owned difficulty scheduler driven by terminal success; do not import another task's private scheduler.
   - Put the scheduler before dependent terms in `CurriculumCfg`.
   - Use `mdp.modify_term_cfg` to interpolate each event or manager-term parameter from an easy initial value to the validated final value.
   - Assign both `events` and `curriculum` on the manager-based environment config.
8. Validate with a small number of environments and repeated resets before scaling.

## Validation

Use the plan-validate-execute loop:

1. Confirm the final fixed randomization range resets and rolls out cleanly on every target backend.
2. For ADR, check initial and maximum difficulty produce the exact initial and final parameter values.
3. Check success promotes difficulty, failure follows the chosen demotion policy, and difficulty remains clamped.
4. Confirm the scheduler runs before interpolation terms and the updated range is consumed by the following reset event.
5. Log difficulty and current bounds; fix shape, device, backend, and entity-name errors before scaling.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with the event and curriculum managers, the curriculum guide, the direct and manager-based environment tutorials, and the Core Lift ADR example. Update maintained docs or source examples before copying new API details into this skill.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Event manager source](../../../source/isaaclab/isaaclab/managers/event_manager.py)
- [Direct workflow randomization tutorial](../../../docs/source/tutorials/03_envs/create_direct_rl_env.rst)
- [Manager-based event terms tutorial](../../../docs/source/tutorials/03_envs/create_manager_base_env.rst)
- [Curriculum utilities guide](../../../docs/source/how-to/curriculums.rst)
- [Core Lift ADR config](../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/adr_curriculum.py)
- [Core Lift ADR terms](../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/mdp/curriculums.py)
- [Managers API](../../../docs/source/api/lab/isaaclab.managers.rst)
