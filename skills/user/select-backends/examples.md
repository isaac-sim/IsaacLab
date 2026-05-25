# Backend Selection Examples

## List Available Presets

Start by checking which preset selectors the task exposes:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py --show_presets
```

Preset selectors are typed tokens such as `physics=NAME`, `renderer=NAME`, and `presets=NAME`.

## Smoke Test A Backend

Use a small random-agent rollout before training:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Ant-v0 --num_envs 4 physics=physx
```

For Newton, use the physics preset name exposed by `list_envs.py` for that task, for example:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Ant-v0 --num_envs 4 physics=newton_mjwarp
```

If the preset name is not listed, do not guess. Add or update the task's backend presets first.

## Train After Smoke Tests

Once reset/step behavior is stable on the selected backend:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-v0 --headless physics=physx
```

Repeat the same small smoke test on every backend before comparing training curves.

## Common Decision Points

- Use PhysX first when preserving Isaac Gym behavior.
- Use Newton when the task specifically targets kit-less or Warp-native workflows.
- Use backend presets for solver, contact, material, sensor, and renderer differences.
- Do not copy PhysX parameters directly into Newton configs without checking schema docs.
