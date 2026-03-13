# RSL-RL Reference

## Overview
- **Best for**: Locomotion, legged robots, on-policy training
- **Runner**: `OnPolicyRunner` (from `rsl_rl.runners`)
- **Wrapper**: `RslRlVecEnvWrapper` (from `isaaclab_rl.rsl_rl`)
- **Config format**: Python class (`RslRlOnPolicyRunnerCfg`)
- **Config entry point**: `rsl_rl_cfg_entry_point`

## CartPole Config
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/rsl_rl_ppo_cfg.py`

Key hyperparameters:
```
max_iterations: 150         # Total iterations (much fewer than SB3)
num_steps_per_env: 16       # Rollout length
learning_rate: 1.0e-3
actor_hidden_dims: [32, 32]
critic_hidden_dims: [32, 32]
activation: "elu"
clip_param: 0.2
entropy_coef: 0.005
num_learning_epochs: 5
num_mini_batches: 4
gamma: 0.99
lam: 0.95
desired_kl: 0.01
schedule: "adaptive"        # LR adapts based on KL divergence
save_interval: 50           # Checkpoint every 50 iterations
```

## Train Command
```bash
$PYTHON scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 --num_envs 4096 --headless
```

CartPole converges in ~150 iterations (~2-5 minutes).

## Play Command
```bash
$PYTHON scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Cartpole-v0 --num_envs 10
```

## Log Structure
```
logs/rsl_rl/{experiment_name}/{timestamp}/
├── params/env.yaml, agent.yaml
├── model_{iteration}.pt    # Periodic checkpoints
└── model_{final}.pt        # Final model
```

## Resume Training
RSL-RL uses `--resume` flag (reads from config):
```bash
$PYTHON scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 --resume
```

## Key Difference from SB3
- RSL-RL counts in **iterations** (each = rollout + update), not timesteps
- Much faster convergence for locomotion tasks
- Uses TF32 precision by default for speed
- Supports `OnPolicyRunner` and `DistillationRunner`

## TensorBoard Metric Names (RSL-RL-specific)

| Metric | TensorBoard Tag |
|--------|----------------|
| Mean reward | `Train/mean_reward` |
| Episode length | `Train/mean_episode_length` |
| Surrogate (policy) loss | `Loss/surrogate` |
| Value function loss | `Loss/value_function` |
| Entropy loss | `Loss/entropy` |
| Learning rate | `Loss/learning_rate` |
| Action noise std | `Policy/mean_noise_std` |
| Per-reward-term | `Episode_Reward/<term_name>` |
| Termination reasons | `Episode_Termination/<term_name>` |

## Where Is the Loss Function?

RSL-RL's PPO loss is in the library, not in Isaac Lab:

**File**: `rsl_rl/algorithms/ppo.py` (in Kit Python site-packages)
**Method**: `update()`, lines ~294-313

```python
# Surrogate loss (clipped PPO)
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
surrogate = -advantages_batch * ratio
surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

# Value loss (optionally clipped)
value_loss = (returns_batch - value_batch).pow(2).mean()

# Combined
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()
```

## Symmetry Augmentation

RSL-RL supports data augmentation via symmetry. CartPole includes a variant:

**File**: `agents/rsl_rl_ppo_cfg.py` — `CartpolePPORunnerWithSymmetryCfg`

This adds `RslRlSymmetryCfg(use_data_augmentation=True)` to double the effective
training data by mirroring observations/actions through a symmetry function.

## SKRL Reference

**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/skrl_ppo_cfg.yaml`

```yaml
models:
  separate: False
  policy:
    class: GaussianMixin
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32]
        activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32]
        activations: elu
    output: ONE
agent:
  class: PPO
  rollouts: 16
  learning_epochs: 8
  mini_batches: 8
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 3.0e-04
  learning_rate_scheduler: KLAdaptiveLR
  ratio_clip: 0.2
  value_clip: 0.2
trainer:
  class: SequentialTrainer
  timesteps: 2400
```

```bash
$PYTHON scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Cartpole-v0 --num_envs 4096 --headless
```
