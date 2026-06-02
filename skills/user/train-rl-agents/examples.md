# RL Training Examples

## Quick Commands

Use these as starting points, then confirm the task's registered agent config exists.

Training runs headless by default; omit any visualizer flag for fastest training. The legacy `--headless` flag is deprecated. To watch a run, pass `--viz kit` (or `--viz rerun,newton,viser`); use `--viz none` to force-disable configured visualizers.

RSL-RL:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --run_name ppo
```

RL-Games direct Cartpole:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Cartpole-Direct-v0
```

Stable Baselines 3:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64
```

SKRL:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-v0
```

## Before Training

Always run a small random-action check first:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 8
```

For visual observations or camera tasks, lower `--num_envs` and confirm renderer and sensor support before scaling.

## After Training

TensorBoard example:

```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/cartpole
```

Play example:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-v0 --checkpoint logs/rsl_rl/cartpole/RUN_NAME/model_100.pt --viz kit
```

Resume example:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --resume --load_run RUN_NAME --checkpoint model_100.pt
```

## Config Lookup

Agent configs live near the task implementation, for example:

- `source/isaaclab_tasks/isaaclab_tasks/core/manager_cartpole/agents/`
- `source/isaaclab_tasks/isaaclab_tasks/core/direct_cartpole/agents/`

Do not mix framework formats: RSL-RL configs are Python classes, while RL-Games, SKRL, and SB3 commonly use YAML or framework-specific config files.
