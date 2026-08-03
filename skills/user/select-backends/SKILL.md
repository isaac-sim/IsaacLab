---
name: isaaclab-selecting-backends
description: Selects and validates Isaac Lab physics and rendering backends. Use when choosing PhysX or Newton, adding backend presets, comparing backend behavior, or debugging backend-specific simulation, randomization, sensor, or renderer issues.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Selecting Backends

## When To Use

Use this skill when a user needs to choose, configure, compare, or debug Isaac Lab physical backends or renderer-related behavior.

Do not use this skill to duplicate backend reference material. Link to the multi-backend docs, schema cfg docs, and source examples for current configuration details. If the user is converting or validating a specific USD asset for Newton, use `isaaclab-preparing-assets-for-newton`.

## Workflow

1. Identify the target backend: PhysX, Newton, or a task that must support both through presets.
2. Read the multi-backend architecture and schema cfg docs before editing backend configs.
3. Start with the backend that best matches the source task or current maintained example. Use PhysX first when matching Isaac Gym behavior.
4. Add backend presets only after the task runs on one backend.
5. Map simulation parameters through public cfg schemas instead of copying old simulator-specific keys. Import universal schema fragments and base cfgs from `isaaclab.sim.schemas`, PhysX-specific cfgs from `isaaclab_physx.sim.schemas`, and Newton or MuJoCo cfgs from `isaaclab_newton.sim.schemas`.
6. Check backend support for sensors, randomization events, terrain, contacts, and actuators before enabling them.
7. Separate backend-specific differences using `PresetCfg` or existing preset helpers rather than runtime conditionals scattered through task code.
8. Use suffixless task names in backend smoke-test and training commands.
9. Validate each backend with a small reset/step rollout before training.
10. Document intentional behavior differences, especially around contacts, randomization timing, CPU/GPU data paths, and renderer requirements.

## Validation

Use this checklist:

1. Run a small reset/step smoke test on the primary backend.
2. If adding another backend, run the same smoke test there before training.
3. Compare observation shape, action shape, reset behavior, and contact behavior.
4. Check randomization events for backend-specific support and device assumptions.
5. Run short training only after both backends pass smoke tests.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/overview/core-concepts/multi_backend_architecture.rst`, `docs/source/overview/core-concepts/schema_cfgs.rst`, backend installation docs, and backend-aware task examples under `source/isaaclab_tasks/isaaclab_tasks/`. If backend docs are incomplete, improve the docs rather than expanding this skill into a backend reference.

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [Prepare assets for Newton skill](../prepare-assets-for-newton/SKILL.md)
- [Multi-backend architecture](../../../docs/source/overview/core-concepts/multi_backend_architecture.rst)
- [Schema cfgs](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [PhysX installation](../../../docs/source/overview/core-concepts/physical-backends/physx/installation.rst)
- [Newton installation](../../../docs/source/overview/core-concepts/physical-backends/newton/installation.rst)
- [Task examples](../../../source/isaaclab_tasks/isaaclab_tasks)
