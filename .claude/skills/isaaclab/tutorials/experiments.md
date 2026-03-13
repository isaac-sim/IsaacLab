# CartPole Experiments & Customization

## Experiment 1: Modify Reward Weights

Edit `source/isaaclab_tasks/.../classic/cartpole/cartpole_env_cfg.py`:

```python
# More aggressive balancing
alive = RewTerm(func=mdp.is_alive, weight=2.0)        # Double survival reward
pole_pos = RewTerm(..., weight=-2.0, ...)              # Stricter angle penalty
```

Re-train and compare TensorBoard curves:
```bash
$PYTHON -m tensorboard.main --logdir=logs/
```

## Experiment 2: Longer Episodes

```python
def __post_init__(self):
    self.episode_length_s = 10  # 10 seconds instead of 5
```

The agent must balance longer, which may require more training iterations.

## Experiment 3: Bigger Network

In SB3 config (`agents/sb3_ppo_cfg.yaml`):
```yaml
policy_kwargs:
  net_arch: [64, 64]  # Larger hidden layers
```

In SKRL config (`agents/skrl_ppo_cfg.yaml`):
```yaml
models:
  policy:
    network:
      - name: net
        layers: [64, 64]
```

## Experiment 4: Add Domain Randomization

In the `EventCfg` section:
```python
@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),    # Wider initial range
            "velocity_range": (-0.5, 0.5),
        },
    )
    # NEW: Randomize pole mass during training
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.5, 1.5),  # 50%-150% of original
        },
    )
```

## Experiment 5: Camera Observations

Use a visual variant:
```bash
$PYTHON scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-Cartpole-RGB-v0 --num_envs 16 --headless
```

This uses RGB camera images instead of joint states. Requires more training and smaller `num_envs`.

## Experiment 6: Direct Environment

> **Known Bug (develop branch)**: `Isaac-Cartpole-Direct-v0` crashes with a warp
> dtype error (`RuntimeError: Cannot cast dtypes of unequal byte size`) in
> `write_root_pose_to_sim()`. This affects ALL RL frameworks on the develop branch
> (Isaac Sim 6.0). Skip this experiment until the bug is fixed, or use the `main`
> branch where it may work.

Try the monolithic implementation (when working):
```bash
$PYTHON scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-Cartpole-Direct-v0 --num_envs 64 --headless
```

Compare convergence with the manager-based version.

## Experiment 7: Create External Project

Follow the NVIDIA tutorial's external project approach:
```bash
./isaaclab.sh --new
# Select: External, ~/Cartpole, Cartpole, Manager-based, skrl, PPO

cd ~/Cartpole
$PYTHON -m pip install -e source/Cartpole
$PYTHON scripts/list_envs.py  # Verify Template-Cartpole-v0
$PYTHON scripts/skrl/train.py --task=Template-Cartpole-v0
```

This creates an isolated project you can modify without touching the Isaac Lab repo.
