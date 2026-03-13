# Hyperparameter Tuning Guide

## Quick Reference: CartPole Defaults Across Frameworks

| Parameter | SB3 | RSL-RL | SKRL | RL-Games |
|-----------|-----|--------|------|----------|
| Network | [32, 32] | [32, 32] | [32, 32] | [32, 32] |
| Activation | ELU | ELU | ELU | ELU |
| Learning rate | 3e-4 | 1e-3 | 3e-4 | 3e-4 |
| Clip range | 0.2 | 0.2 | 0.2 | 0.2 |
| Gamma | 0.99 | 0.99 | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 | 0.95 | 0.95 |
| Entropy coef | 0.01 | 0.005 | 0.0 | 0.0 |
| Rollout steps | 16 | 16 | 16 | 16 |
| Mini-epochs | 20 | 5 | 8 | 8 |
| Total training | 1M steps | 150 iters | 2400 steps | 150 epochs |
| LR schedule | Fixed | Adaptive (KL) | KLAdaptiveLR | Adaptive (KL) |

## Tuning Guidelines

### Learning Rate
- **Start**: 3e-4 (SB3/SKRL) or 1e-3 (RSL-RL)
- **If learning stalls**: Reduce by 3-10x
- **If unstable/diverging**: Reduce by 10x
- **Adaptive schedule**: RSL-RL `schedule="adaptive"`, SKRL `KLAdaptiveLR`

### Network Architecture
- **Simple tasks** (CartPole): `[32, 32]` sufficient
- **Medium tasks** (locomotion): `[128, 64]` or `[256, 128]`
- **Complex tasks** (dexterous manipulation): `[512, 256, 128]`
- **Activation**: ELU works well for robotics (default across all frameworks)

### Batch Size / Mini-batches
- Default: `num_envs * rollout_steps` for total batch
- SB3: `batch_size: 4096` (explicit)
- RSL-RL: `num_mini_batches: 4` (splits total batch)
- RL-Games: `minibatch_size: 8192` (explicit)
- Larger batches = more stable but slower per update

### Entropy Coefficient
- **More exploration**: 0.01-0.05
- **More exploitation**: 0.001-0.005
- **SKRL default**: 0.0 (relies on initial log std for exploration)

### Discount Factor (gamma)
- **Short-horizon tasks**: 0.95-0.99
- **Long-horizon tasks**: 0.995-0.999
- **CartPole**: 0.99 (5-second episodes)

### Clip Range
- Default 0.2 works for most tasks
- Reduce if training is unstable (try 0.1)
- Increase if updates are too conservative (try 0.3)

### Number of Environments
- More envs = more data per update = more stable training
- Limited by GPU memory
- Small GPUs (GB10, RTX 2070): 64-256 for SB3, 1024-4096 for RSL-RL
- Large GPUs (RTX 3090+, A100): 4096-8192

## TensorBoard Metrics to Monitor

**Important**: Metric names differ across frameworks. Use the table below.

Launch TensorBoard via Kit Python (it is NOT on the system PATH):
```bash
$PYTHON -m tensorboard.main --logdir=logs/
```

### Metric Names by Framework

| What | SB3 | RSL-RL | SKRL | RL-Games |
|------|-----|--------|------|----------|
| Reward | `rollout/ep_rew_mean` | `Train/mean_reward` | `Reward / Total reward (mean)` | `rewards/iter` |
| Ep length | `rollout/ep_len_mean` | `Train/mean_episode_length` | `Episode length (mean)` | `episode_lengths/iter` |
| Policy loss | `train/policy_gradient_loss` | `Loss/surrogate` | `Loss / Policy loss` | `losses/a_loss` |
| Value loss | `train/value_loss` | `Loss/value_function` | `Loss / Value loss` | `losses/c_loss` |
| Entropy | `train/entropy_loss` | `Loss/entropy` | `Loss / Entropy loss` | `info/entropy` |
| KL divergence | `train/approx_kl` | (via desired_kl) | (via KLAdaptiveLR) | `info/kl` |
| Learning rate | `train/learning_rate` | `Loss/learning_rate` | `Learning / Learning rate` | `info/last_lr` |

### What to Look For

| Metric | Healthy Trend |
|--------|--------------|
| Reward | Should steadily increase |
| Episode length | Should increase (for survival tasks) |
| Policy loss | Should decrease and stabilize |
| Value loss | Should decrease |
| Entropy | Should slowly decrease (not crash to zero) |
| KL divergence | Should stay near desired_kl (~0.01) |

## Agent Config Locations

All located at: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/`
- `sb3_ppo_cfg.yaml`
- `rsl_rl_ppo_cfg.py`
- `skrl_ppo_cfg.yaml`
- `rl_games_ppo_cfg.yaml`

For other tasks, look in: `source/isaaclab_tasks/isaaclab_tasks/<type>/<category>/<task>/agents/`
