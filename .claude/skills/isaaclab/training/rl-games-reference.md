# RL-Games Reference

## Overview
- **Best for**: High performance training, large-scale environments
- **Algorithm**: `a2c_continuous` (PPO variant with continuous actions)
- **Wrapper**: `RlGamesVecEnvWrapper` (from `isaaclab_rl.rl_games`)
- **Config format**: YAML files
- **Config entry point**: `rl_games_cfg_entry_point`

## CartPole Config

**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/rl_games_ppo_cfg.yaml`

Key hyperparameters:
```yaml
params:
  algo:
    name: a2c_continuous       # PPO for continuous action spaces
  model:
    name: continuous_a2c_logstd
  network:
    name: actor_critic
    separate: False            # Shared network for actor and critic
    mlp:
      units: [32, 32]
      activation: elu
  config:
    ppo: True
    gamma: 0.99
    tau: 0.95                  # GAE lambda
    learning_rate: 3e-4
    lr_schedule: adaptive      # Adapts based on KL divergence
    kl_threshold: 0.008
    e_clip: 0.2                # PPO clip range
    horizon_length: 16         # Rollout steps
    minibatch_size: 8192
    mini_epochs: 8             # PPO epochs per update
    max_epochs: 150            # Total training epochs
    save_frequency: 25         # Checkpoint every 25 epochs
    entropy_coef: 0.0
    critic_coef: 4             # Value loss multiplier
    clip_value: True           # Clip value function updates
    bounds_loss_coef: 0.0001   # Action bounds regularization
```

**Additional CartPole configs** (for visual/feature variants):
- `rl_games_camera_ppo_cfg.yaml` — camera-based observations
- `rl_games_feature_ppo_cfg.yaml` — feature-based observations

## Train Command
```bash
$PYTHON scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Cartpole-v0 --num_envs 4096 --headless
```

## Play Command
```bash
$PYTHON scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Cartpole-v0 --num_envs 10
```

**Note**: Play runs in an infinite loop. Use Ctrl-C to stop in headless mode.

## Log Structure
```
logs/rl_games/{task_name}/{timestamp}/
├── params/env.yaml, agent.yaml
├── nn/{task_name}.pth         # Checkpoints (every save_frequency epochs)
└── summaries/                 # TensorBoard event files
```

## Resume Training
```bash
$PYTHON scripts/reinforcement_learning/rl_games/train.py \
  --task <TASK> --checkpoint <path/to/model.pth>
```

## TensorBoard Metric Names (RL-Games-specific)

| Metric | TensorBoard Tag |
|--------|----------------|
| Mean reward | `rewards/iter` |
| Episode length | `episode_lengths/iter` |
| Actor (policy) loss | `losses/a_loss` |
| Critic (value) loss | `losses/c_loss` |
| Entropy | `info/entropy` |
| KL divergence | `info/kl` |
| Learning rate | `info/last_lr` |

## Where Is the Loss Function?

RL-Games' loss is in the library, not in Isaac Lab:

**Main file**: `rl_games/algos_torch/a2c_continuous.py` (in Kit Python site-packages)
**Method**: `calc_gradients()`, lines ~138-153

**Helper functions**: `rl_games/common/common_losses.py`
- `actor_loss()` (lines 39-47): Clipped PPO surrogate
- `critic_loss()` (lines 10-19): Optionally clipped value loss

```python
# Combined loss (a2c_continuous.py:153)
loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
```

## Key Differences from Other Frameworks

- Uses `max_epochs` (iterations), not `n_timesteps`
- `critic_coef: 4` is much higher than SB3's `vf_coef: 1.0`
- `bounds_loss_coef` adds action-space bounds regularization (unique to RL-Games)
- `kl_threshold: 0.008` (vs RSL-RL's `desired_kl: 0.01`)
- Config uses nested YAML (`params.config.*`) not flat keys
