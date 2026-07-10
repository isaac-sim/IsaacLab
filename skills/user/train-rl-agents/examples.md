# RL Training Examples

## Quick Commands

Use these as starting points, then confirm the task's registered agent config exists.

Training runs headless by default; omit any visualizer flag for fastest training. The legacy `--headless` flag is deprecated. To watch a run, pass `--viz kit` (or `--viz rerun,newton,viser`); use `--viz none` to force-disable configured visualizers. Use suffixless task names, for example `Isaac-Cartpole` instead of `Isaac-Cartpole-v0`.

RSL-RL:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --run_name ppo
```

RL-Games direct Cartpole:

```bash
uv run isaaclab train --rl_library rl_games --task Isaac-Cartpole-Direct
```

Stable Baselines 3:

```bash
uv run isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64
```

SKRL:

```bash
uv run isaaclab train --rl_library skrl --task Isaac-Cartpole
```

## Before Training

Always run a small random-action check first:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole --num_envs 8
```

For visual observations or camera tasks, lower `--num_envs` and confirm renderer and sensor support before scaling. Do not add `--enable_cameras` unless the current task or docs explicitly require it.

## After Training

TensorBoard example:

```bash
uv run --with tensorboard python -m tensorboard.main --logdir logs/rsl_rl/cartpole
```

Play example:

```bash
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole --checkpoint logs/rsl_rl/cartpole/RUN_NAME/model_100.pt --viz kit
```

Resume example:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --resume --load_run RUN_NAME --checkpoint model_100.pt
```

## Config Lookup

Agent configs live near the task implementation, for example:

- `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/agents/`

Do not mix framework formats: RSL-RL configs are Python classes, while RL-Games, SKRL, and SB3 commonly use YAML or framework-specific config files.
