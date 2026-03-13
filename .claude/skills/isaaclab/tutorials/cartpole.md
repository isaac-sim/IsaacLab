# CartPole Tutorial

The CartPole problem: a cart moves left/right to balance a pole vertically.
This is the "Hello World" of Isaac Lab RL training.

## Quick Start

```bash
ISAACLAB_ROOT=$(git rev-parse --show-toplevel)
PYTHON="$ISAACLAB_ROOT/_isaac_sim/python.sh"

# Train with SB3 (simple, ~30 min on small GPU)
$PYTHON scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-Cartpole-v0 --num_envs 64 --headless

# Or train with RSL-RL (faster, ~2-5 min)
$PYTHON scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 --num_envs 4096 --headless
```

## Step-by-Step Walkthrough

### 1. Verify Environment
```bash
$PYTHON -c "from isaaclab.app import AppLauncher; print('OK')"
```

### 2. Choose a Framework and Train
- **RSL-RL**: 150 iterations, `--num_envs 4096` (~37s on GB10, fastest)
- **SKRL**: 2400 timesteps, `--num_envs 4096` (~55s on GB10)
- **RL-Games**: 150 epochs, `--num_envs 4096` (~61s on GB10)
- **SB3**: 1M timesteps, `--num_envs 64` for small GPUs (~103s on GB10)

**Note**: SKRL only shows a progress bar during training — no reward metrics are
printed to the terminal. You must check TensorBoard to see if training converged.

### 3. Monitor with TensorBoard

TensorBoard is inside Kit Python, NOT on the system PATH. Launch via:
```bash
$PYTHON -m tensorboard.main --logdir=logs/
# Open http://localhost:6006
```

The "TensorFlow installation not found" warning is safe to ignore.

**Key metrics to watch** (names differ per framework):
- SB3: `rollout/ep_rew_mean` (should increase toward ~4.9)
- RSL-RL: `Train/mean_reward` (should increase toward ~4.9)

See [hyperparameters.md](../training/hyperparameters.md) for the full metric name table.

### 4. Play Trained Policy
```bash
$PYTHON scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Cartpole-v0 --num_envs 10
```

**Note**: Play scripts run in an **infinite loop** (continuous visualization).
In headless mode, use Ctrl-C to stop. There is no `--max_steps` flag.

### 5. Understand the Code
See [code-walkthrough.md](code-walkthrough.md) for detailed analysis of:
- Robot asset configuration
- Reward function design
- Observation and action spaces
- Termination conditions

### 6. Experiment
See [experiments.md](experiments.md) for customization ideas:
- Modify reward weights
- Change episode length
- Add domain randomization
- Use camera observations

## The CartPole MDP

| Component | Details |
|-----------|---------|
| **Joints** | `slider_to_cart` (linear), `cart_to_pole` (revolute) |
| **Action** | 1D force on cart (scale=100.0) |
| **Observations** | Joint positions (2) + joint velocities (2) = 4D |
| **Episode** | 5 seconds max, 60 Hz control (300 steps) |
| **Termination** | Timeout OR cart leaves [-3, 3] |

## Key Files

| File | Purpose |
|------|---------|
| `source/isaaclab_assets/.../robots/cartpole.py` | Robot asset config |
| `source/isaaclab_tasks/.../classic/cartpole/cartpole_env_cfg.py` | Environment config |
| `source/isaaclab_tasks/.../classic/cartpole/__init__.py` | Gym registration |
| `source/isaaclab_tasks/.../classic/cartpole/agents/` | RL agent configs |

## Available CartPole Variants

| Task ID | Description | Status |
|---------|-------------|--------|
| `Isaac-Cartpole-v0` | Standard (manager-based) | Works (all 4 frameworks verified) |
| `Isaac-Cartpole-Direct-v0` | Direct implementation | **Broken on develop branch** (see below) |
| `Isaac-Cartpole-RGB-v0` | With RGB camera observations | Works (RL-Games only) |
| `Isaac-Cartpole-Depth-v0` | With depth camera observations | Works (RL-Games only) |
| `Isaac-Cartpole-RGB-ResNet18-v0` | ResNet18 feature extraction | Works (RL-Games only) |
| `Isaac-Cartpole-RGB-TheiaTiny-v0` | Theia-Tiny Transformer features | Works (RL-Games only) |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` | Direct + RGB camera | Untested |
| `Isaac-Cartpole-Depth-Camera-Direct-v0` | Direct + depth camera | Untested |

### Known Bug: Direct CartPole (develop branch)

`Isaac-Cartpole-Direct-v0` crashes with `RuntimeError: Cannot cast dtypes of unequal
byte size` in `write_root_pose_to_sim()` during `_reset_idx()`. This is a warp dtype
mismatch bug on the develop branch (Isaac Sim 6.0). It affects ALL RL frameworks.
Use `Isaac-Cartpole-v0` (manager-based) instead — it is functionally identical and
works correctly.
