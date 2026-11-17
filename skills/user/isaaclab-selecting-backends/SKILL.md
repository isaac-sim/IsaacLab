---
name: isaaclab-selecting-backends
description: Selects and validates Isaac Lab physics and rendering backends. Use when choosing PhysX, Newton, or OvPhysX, deciding which backend to start a port on, adding Newton support to a task that already works on PhysX, running a quick check that a task runs on Newton before training, adding backend presets, comparing backend behavior, or debugging backend-specific simulation, randomization, sensor, or renderer issues.
license: Apache-2.0
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
  tags: [physx, newton, backends, rendering]
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Selecting Backends

## When To Use

Use this skill when a user needs to choose, configure, compare, or debug Isaac Lab physical backends or renderer-related behavior.

Do not use this skill to duplicate backend reference material. Link to the backend architecture, backend and preset selector, schema cfg docs, and source examples for current configuration details. If the user is converting or validating a specific USD asset for Newton, use `isaaclab-preparing-assets-for-newton`.

## Workflow

1. Identify the target backend: PhysX, Newton, OvPhysX, or a task that must
   support multiple backends through presets.
2. Read the physics-backends concept, backend architecture, backend and preset
   selector, and schema cfg docs before editing backend configs.
3. Start with the backend that best matches the source task or current maintained example. Use PhysX first when matching Isaac Gym behavior.
4. Add backend presets only after the task runs on one backend.
5. Map simulation parameters through public cfg schemas instead of copying old simulator-specific keys. Import universal schema fragments and base cfgs from the `schemas` module under `sim` in `isaaclab`, PhysX-specific cfgs from the same module in `isaaclab_physx`, and Newton or MuJoCo cfgs from the same module in `isaaclab_newton`.
6. Check backend support for sensors, randomization events, terrain, contacts, and actuators before enabling them.
7. Separate backend-specific differences using `PresetCfg` or existing preset helpers rather than runtime conditionals scattered through task code.
8. Use suffixless task names in backend smoke-test and training commands.
9. Validate each backend with a small reset/step rollout before training.
10. For backend architecture changes, resolve shared native runtimes through
    `SimulationContext.get_or_create_backend(...)` instead of copying state between consumers.
11. Document intentional behavior differences, especially around contacts, randomization timing, CPU/GPU data paths, and renderer requirements.

## Examples

Representative requests this skill handles (worked patterns in [examples.md](examples.md)):

- "I am porting an Isaac Gym task. Should I start with PhysX or Newton?"
- "My task works on PhysX. Add Newton support too."
- "Give me a command to check whether my Ant task runs on Newton before training."

## Validation

Use this checklist:

1. Run a small reset/step smoke test on the primary backend.
2. Run the same smoke test on each selected backend before training.
3. Compare observation shape, action shape, reset behavior, and contact behavior.
4. Check randomization events for backend-specific support and device assumptions.
5. Run short training only after each selected backend passes its smoke test.

For skill changes, run the repository skill validator; the exact `check` command is in the [Skills README](../../README.md#authoring-rules).

## Maintenance

Keep this skill synchronized with [physics backends](../../../docs/source/concepts/physics_backends.rst), [backends and presets](../../../docs/source/concepts/backends_and_presets.rst), [schema cfgs](../../../docs/source/concepts/schema_cfgs.rst), and backend-aware task examples under isaaclab tasks package. If backend docs are incomplete, improve the docs rather than expanding this skill into a backend reference.

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [Prepare assets for Newton skill](../isaaclab-preparing-assets-for-newton/SKILL.md)
- [Physics backends](../../../docs/source/concepts/physics_backends.rst)
- [Backends and presets](../../../docs/source/concepts/backends_and_presets.rst)
- [Renderers, including asynchronous rendering](../../../docs/source/concepts/renderers.rst)
- [Schema cfgs](../../../docs/source/concepts/schema_cfgs.rst)
- [Installation](../../../docs/source/isaaclab/installation.rst)
- [Task examples](../../../source/isaaclab_tasks/isaaclab_tasks)
