# Stable Baselines3 (SB3) Reference

## Overview
- **Best for**: Prototyping, familiar API, extensive documentation
- **Agent**: `PPO` (also supports SAC, TD3 for off-policy)
- **Wrapper**: `Sb3VecEnvWrapper` (from `isaaclab_rl.sb3`)
- **Config format**: YAML files
- **Config entry point**: `sb3_cfg_entry_point`

## CartPole Config
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/sb3_ppo_cfg.yaml`

```yaml
seed: 42
n_timesteps: 1000000       # Total training steps
policy: 'MlpPolicy'
n_steps: 16                # Steps per env per update
batch_size: 4096
n_epochs: 20               # PPO epochs per update
learning_rate: 3e-4
clip_range: 0.2
gamma: 0.99                # Discount factor
gae_lambda: 0.95           # GAE lambda
ent_coef: 0.01             # Entropy coefficient
vf_coef: 1.0               # Value function coefficient
max_grad_norm: 1.0
policy_kwargs:
  activation_fn: 'nn.ELU'
  net_arch: [32, 32]       # Hidden layer sizes
  squash_output: False
device: "cuda:0"
```

## Train Command
```bash
$PYTHON scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-Cartpole-v0 --num_envs 64 --headless
```

## Play Command
```bash
# Auto-find latest model.zip
$PYTHON scripts/reinforcement_learning/sb3/play.py \
  --task Isaac-Cartpole-v0 --num_envs 10

# Specific checkpoint
$PYTHON scripts/reinforcement_learning/sb3/play.py \
  --task Isaac-Cartpole-v0 --checkpoint <path/to/model.zip> --num_envs 10

# Use last intermediate checkpoint (not final)
$PYTHON scripts/reinforcement_learning/sb3/play.py \
  --task Isaac-Cartpole-v0 --use_last_checkpoint --num_envs 10
```

## Log Structure
```
logs/sb3/{task_name}/{timestamp}/
├── params/env.yaml, agent.yaml
├── command.txt
├── model_{N}_steps.zip     # Every 1000 steps (CheckpointCallback)
├── model.zip               # Final model
├── model_vecnormalize.pkl  # If VecNormalize used
├── PPO_1/events.out.tfevents.*  # TensorBoard
└── videos/                 # If --video
```

## Resume Training
```bash
$PYTHON scripts/reinforcement_learning/sb3/train.py \
  --task <TASK> --checkpoint <path/to/model.zip>
```

## Training Pipeline (what train.py does)
1. Parse CLI args + Hydra overrides
2. Launch AppLauncher (boots Isaac Sim)
3. `@hydra_task_config` loads env_cfg and agent_cfg from registry
4. Create gym env: `gym.make(task, cfg=env_cfg)`
5. Wrap with `Sb3VecEnvWrapper`
6. Create PPO agent
7. Train with CheckpointCallback (saves every 1000 steps)
8. Save final model.zip

## TensorBoard Metric Names (SB3-specific)

| Metric | TensorBoard Tag |
|--------|----------------|
| Mean reward | `rollout/ep_rew_mean` |
| Episode length | `rollout/ep_len_mean` |
| Policy loss | `train/policy_gradient_loss` |
| Value loss | `train/value_loss` |
| Entropy loss | `train/entropy_loss` |
| Approx KL | `train/approx_kl` |
| Clip fraction | `train/clip_fraction` |
| Learning rate | `train/learning_rate` |
| FPS | `time/fps` |

## Where Is the Loss Function?

SB3's PPO loss is in the library, not in Isaac Lab:

**File**: `stable_baselines3/ppo/ppo.py` (in Kit Python site-packages)
**Method**: `train()`, lines ~220-256

```python
# Policy loss (clipped surrogate)
policy_loss_1 = advantages * ratio
policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

# Value loss
value_loss = F.mse_loss(rollout_data.returns, values_pred)

# Combined
loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
```
