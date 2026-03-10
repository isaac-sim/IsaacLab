# Isaac Lab RL Training

## Train a Policy

```bash
ISAACLAB_ROOT=$(git rev-parse --show-toplevel)
PYTHON="$ISAACLAB_ROOT/_isaac_sim/python.sh"

# SB3 (simple, good for prototyping)
$PYTHON scripts/reinforcement_learning/sb3/train.py \
  --task <TASK> --headless

# RSL-RL (fastest convergence, best for robotics)
$PYTHON scripts/reinforcement_learning/rsl_rl/train.py \
  --task <TASK> --headless

# SKRL (flexible, YAML configs, JAX support)
$PYTHON scripts/reinforcement_learning/skrl/train.py \
  --task <TASK> --headless

# RL-Games (high performance)
$PYTHON scripts/reinforcement_learning/rl_games/train.py \
  --task <TASK> --headless
```

Add `--num_envs N` to control parallelism (default: 4096, use 64 for small GPUs).

## Evaluate a Trained Policy

```bash
$PYTHON scripts/reinforcement_learning/sb3/play.py --task <TASK> --num_envs 10
$PYTHON scripts/reinforcement_learning/rsl_rl/play.py --task <TASK> --num_envs 10
$PYTHON scripts/reinforcement_learning/skrl/play.py --task <TASK> --num_envs 10
```

Play scripts auto-find the latest checkpoint in `logs/<framework>/<task>/`.

**Note**: Play scripts run in an **infinite loop** (continuous simulation). In headless
mode, use Ctrl-C to stop. There is no built-in `--max_steps` or `--episodes` flag.

## Monitor with TensorBoard

TensorBoard is installed inside Kit Python but NOT on the system PATH. Launch via:
```bash
# Via Kit Python (always works)
$PYTHON -m tensorboard.main --logdir=logs/
# Then open http://localhost:6006

# Or install globally first: pip install tensorboard
# Then: tensorboard --logdir=logs/
```

The "TensorFlow installation not found" warning is safe to ignore.

### Key Metrics by Framework

Metric names differ across frameworks. Here are the ones to watch:

| What to Monitor | SB3 | RSL-RL | SKRL | RL-Games |
|----------------|-----|--------|------|----------|
| Mean reward | `rollout/ep_rew_mean` | `Train/mean_reward` | `Reward / Total reward (mean)` | `rewards/iter` |
| Episode length | `rollout/ep_len_mean` | `Train/mean_episode_length` | `Episode length (mean)` | `episode_lengths/iter` |
| Policy loss | `train/policy_gradient_loss` | `Loss/surrogate` | `Loss / Policy loss` | `losses/a_loss` |
| Value loss | `train/value_loss` | `Loss/value_function` | `Loss / Value loss` | `losses/c_loss` |
| Learning rate | `train/learning_rate` | `Loss/learning_rate` | `Learning / Learning rate` | `info/last_lr` |

## Common CLI Arguments

| Arg | Description | Default |
|-----|-------------|---------|
| `--task` | Task ID (e.g., `Isaac-Cartpole-v0`) | Required |
| `--num_envs` | Parallel envs | 4096 |
| `--headless` | No viewport | False |
| `--seed` | Random seed | 42 |
| `--max_iterations` | Training iterations | Config |
| `--checkpoint` | Resume path | None |
| `--video` | Record videos | False |

## Framework Selection

| Framework | Best For | Config | On-Policy | Terminal Output |
|-----------|---------|--------|-----------|-----------------|
| **RSL-RL** | Locomotion, legged robots | Python class | PPO | Detailed terminal output |
| **SB3** | Prototyping, familiar API | YAML | PPO (also SAC, TD3) | Periodic terminal summaries |
| **SKRL** | Research, JAX, flexibility | YAML | PPO | **Progress bar only** (no terminal metrics) |
| **RL-Games** | High performance | YAML | PPO (A2C continuous) | FPS + epoch counter |
| **RLinf** | VLA fine-tuning (GR00T) | YAML (Hydra) | PPO / Actor-Critic | N/A |

**RLinf** is specialized for fine-tuning Vision-Language-Action models (e.g., GR00T)
with FSDP distributed training. It requires a pretrained VLA checkpoint and multi-GPU
setup. See `scripts/reinforcement_learning/rlinf/README.md` for details.

## Where Are the Loss Functions?

Loss functions are defined in each RL framework's library code, **not** in Isaac Lab.
Isaac Lab provides environment wrappers only — all loss computation is delegated to
the framework. Locations (in Kit Python site-packages):

| Framework | File | Method | Key Lines |
|-----------|------|--------|-----------|
| SB3 | `stable_baselines3/ppo/ppo.py` | `train()` | 220-256 |
| RSL-RL | `rsl_rl/algorithms/ppo.py` | `update()` | 294-313 |
| SKRL | `skrl/agents/torch/ppo/ppo.py` | `_compute_loss()` | 469-495 |
| RL-Games | `rl_games/algos_torch/a2c_continuous.py` | `calc_gradients()` | 138-153 |

All four implement clipped PPO surrogate loss + value loss + entropy bonus.

## Log Structure

Logs go to `logs/<framework>/<task>/<timestamp>/`:

**RSL-RL**: `logs/rsl_rl/{experiment}/{timestamp}/`
- `model_0.pt`, `model_50.pt`, ... `model_149.pt` (every `save_interval`)
- `events.out.tfevents.*` (TensorBoard)
- `params/env.yaml`, `params/agent.yaml`
- `git/IsaacLab.diff` (git state at training time)

**SB3**: `logs/sb3/{task}/{timestamp}/`
- `model.zip` (final), `model_*_steps.zip` (every 64K steps)
- `PPO_1/events.out.tfevents.*` (TensorBoard, note subfolder)
- `params/env.yaml`, `params/agent.yaml`
- `command.txt` (exact CLI command used)

**SKRL**: `logs/skrl/{experiment}/{timestamp}_ppo_torch/`
- `checkpoints/agent_*.pt`, `checkpoints/best_agent.pt`
- `events.out.tfevents.*` (TensorBoard)
- `params/env.yaml`, `params/agent.yaml`

**RL-Games**: `logs/rl_games/{experiment}/{timestamp}/`
- `nn/cartpole.pth` (best), `nn/last_*_ep_*_rew_*.pth` (periodic)
- `summaries/events.out.tfevents.*` (TensorBoard, note subfolder)
- `params/` (if present)

## Detailed References

- [sb3-reference.md](sb3-reference.md) - Stable Baselines3 deep-dive
- [rsl-rl-reference.md](rsl-rl-reference.md) - RSL-RL & SKRL deep-dive
- [rl-games-reference.md](rl-games-reference.md) - RL-Games deep-dive
- [hyperparameters.md](hyperparameters.md) - Tuning guide across all frameworks
