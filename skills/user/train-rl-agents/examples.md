# RL Training Examples

## Quick Commands

Use these as starting points, then confirm the task's registered agent config exists.

RSL-RL:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless --run_name ppo
```

RL-Games direct Cartpole:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Cartpole-Direct-v0 --headless
```

Stable Baselines 3:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64
```

SKRL:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-v0 --headless
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
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-v0 --use_last_checkpoint --viz kit
```

## Config Lookup

Agent configs live near the task implementation, for example:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/`
- `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/agents/`

Do not mix framework formats: RSL-RL configs are Python classes, while RL-Games, SKRL, and SB3 commonly use YAML or framework-specific config files.
