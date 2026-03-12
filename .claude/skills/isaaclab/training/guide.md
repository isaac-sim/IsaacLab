# Isaac Lab RL Training

For full training documentation, refer to:

```
docs/source/tutorials/03_envs/run_rl_training.rst
docs/source/tutorials/03_envs/configuring_rl_training.rst
```

## Quick Reference

```bash
# SB3
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task <TASK> --headless

# RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK> --headless

# SKRL
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task <TASK> --headless

# RL-Games
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task <TASK> --headless
```

Add `--num_envs N` to control parallelism (default varies by task, use 256+ for
large GPUs — there is no hard upper limit).

## Evaluate a Trained Policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/<framework>/play.py --task <TASK> --num_envs 10
```

Play scripts auto-find the latest checkpoint in `logs/<framework>/<task>/`.

## Monitor with TensorBoard

```bash
./isaaclab.sh -p -m tensorboard.main --logdir=logs/
# Then open http://localhost:6006
```

## Common CLI Arguments

| Arg | Description |
|-----|-------------|
| `--task` | Task ID (e.g., `Isaac-Cartpole-v0`) |
| `--num_envs` | Parallel envs (256+ for large GPUs) |
| `--headless` | No viewport rendering |
| `--seed` | Random seed |
| `--max_iterations` | Training iterations |
| `--checkpoint` | Resume from checkpoint path |

## Framework Selection

| Framework | Best For |
|-----------|---------|
| **RSL-RL** | Locomotion, legged robots, fast convergence |
| **SB3** | Prototyping, familiar API |
| **SKRL** | Research, JAX support, flexibility |
| **RL-Games** | High performance |

## GPU Memory

- Use `--headless` to skip viewport rendering
- Reduce `--num_envs` if you hit CUDA OOM
- JAX users: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid pre-allocating all GPU memory

## Per-Framework Details

See the framework-specific reference files in this directory:
- [sb3-reference.md](sb3-reference.md)
- [rsl-rl-reference.md](rsl-rl-reference.md)
- [rl-games-reference.md](rl-games-reference.md)
- [hyperparameters.md](hyperparameters.md)
