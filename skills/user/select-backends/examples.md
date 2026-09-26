# Backend Selection Examples

## List Available Presets

Start by checking which preset selectors the task exposes:

```bash
uv run python scripts/environments/list_envs.py --show_presets
```

Preset selectors are typed tokens such as `physics=NAME`, `renderer=NAME`, and `presets=NAME`.

## Smoke Test A Backend

Use a small random-agent rollout before training:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Ant --num_envs 4 physics=physx
```

For Newton, use the physics preset name exposed by `list_envs.py` for that task, for example:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Ant --num_envs 4 physics=newton_mjwarp
```

If the preset name is not listed, do not guess. Add or update the task's backend presets first.

## Train After Smoke Tests

Once reset/step behavior is stable on the selected backend:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Ant physics=physx
```

Repeat the same small smoke test on every backend before comparing training curves.

## Common Decision Points

- Use PhysX first when preserving Isaac Gym behavior.
- Use Newton when the task specifically targets kit-less or Warp-native workflows.
- Use OvPhysX when evaluating the experimental kit-less PhysX path and the task
  explicitly exposes an OvPhysX preset.
- Use backend presets for solver, contact, material, sensor, and renderer differences.
- Do not copy PhysX parameters directly into Newton configs without checking schema docs.
- For USD physics authoring, follow the selected backend's public schema cfgs
  and generated API documentation. OvPhysX reads scene-level solver settings
  primarily from the USD `PhysicsScene` prim.
