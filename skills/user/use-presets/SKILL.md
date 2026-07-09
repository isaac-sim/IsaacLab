---
name: isaaclab-using-presets
description: Defines and uses Isaac Lab preset configurations for multi-backend tasks. Use when adding PhysX/Newton variants, renderer variants, domain presets, or deciding whether a task needs PresetCfg.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Using Presets

## When To Use

Use this skill when a user needs to define, select, or debug Isaac Lab `PresetCfg` variants for environments, physics backends, renderers, sensors, events, or task-specific configuration options.

Do not use presets for simple one-backend tasks with no meaningful configuration variants. Prefer a plain config until the task needs a real selectable alternative.

## Workflow

1. Identify whether the task needs variants. Common reasons are physics backend differences, renderer differences, camera data types, event randomization differences, or play/train variants.
2. If there is only one supported behavior, keep the config simple and do not add `PresetCfg`.
3. If variants are needed, choose the selector category:
   - Use a `PhysicsCfg(PresetCfg)` field for physics backend variants selected by `physics=NAME`.
   - Use a renderer preset for renderer variants selected by `renderer=NAME`.
   - Use task/domain presets for environment-specific variants selected by `presets=NAME[,NAME,...]`.
4. Define a `default` variant. Add explicit named variants such as `physx`, `newton_mjwarp`, `newton_kamino`, `ovphysx`, `rgb`, or `depth` only when the task supports them.
5. Assign the preset wrapper to the owning environment config field, for example `sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())`.
6. Keep backend-specific values inside preset classes rather than scattering runtime conditionals through task logic.
7. Use suffixless task names in commands.
8. List available preset names before using them in commands.
9. Smoke-test every preset with a small random-agent rollout before training.

## Validation

Use this checklist:

1. Confirm a plain config would not be sufficient.
2. Confirm every preset variant has a clear reason to exist.
3. Confirm `default` points to a valid config.
4. Confirm selector names are discoverable through the task's preset help or environment list.
5. Run a small reset/step smoke test for every new preset.
6. Run training only after preset-specific shape, device, backend, and renderer behavior is stable.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py`, `source/isaaclab_tasks/isaaclab_tasks/utils/preset_cli.py`, `source/isaaclab_tasks/isaaclab_tasks/utils/preset_target.py`, the environment catalog, and maintained preset examples under `source/isaaclab_tasks/isaaclab_tasks/`. If preset selector behavior changes, update the source docs or examples first and keep this skill as a routing checklist.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Preset utility source](../../../source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py)
- [Preset CLI source](../../../source/isaaclab_tasks/isaaclab_tasks/utils/preset_cli.py)
- [Preset targets source](../../../source/isaaclab_tasks/isaaclab_tasks/utils/preset_target.py)
- [Environment catalog](../../../docs/source/overview/environments.rst)
- [Quickstart preset details](../../../docs/source/setup/quickstart_details.rst)
