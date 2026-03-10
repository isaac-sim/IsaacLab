# Environment Builder

## Environment Config Structure

Every manager-based environment needs:

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg(num_envs=4096, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()           # Reset randomization
    # Optional: commands, curriculum, recorders

    def __post_init__(self):
        self.decimation = 2           # Physics steps per action
        self.episode_length_s = 5.0   # Max episode duration
        self.sim.dt = 1/120           # Physics timestep (120 Hz)
        self.sim.render_interval = self.decimation
```

## Quick: Create a New Task

1. Use the template generator: `./isaaclab.sh --new`
2. Or manually create files following [templates.md](templates.md)
3. Register with gymnasium (see below)
4. Train: `$PYTHON scripts/reinforcement_learning/sb3/train.py --task MyTask-v0`

## Reward Design

```python
@configclass
class RewardsCfg:
    # Always include these two:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # Task reward (primary objective):
    task_reward = RewTerm(
        func=mdp.joint_pos_target_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint"]), "target": 0.0},
    )

    # Shaping rewards (secondary, smaller weights):
    regularization = RewTerm(func=mdp.joint_vel_l1, weight=-0.01, ...)
```

**Rules:**
- Positive weight = encourage, Negative weight = penalize
- Task reward weight > shaping weight > regularization weight
- All functions return per-env tensors: shape `(num_envs,)`

## Gymnasium Registration

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

## Detailed References

- [templates.md](templates.md) - Full code templates for new environments
- [mdp-catalog.md](mdp-catalog.md) - Complete catalog of built-in MDP functions

## Reference: CartPole as Example

| File | Purpose |
|------|---------|
| `source/isaaclab_assets/.../robots/cartpole.py` | Robot asset |
| `source/isaaclab_tasks/.../classic/cartpole/cartpole_env_cfg.py` | Env config |
| `source/isaaclab_tasks/.../classic/cartpole/__init__.py` | Registration |
| `source/isaaclab_tasks/.../classic/cartpole/agents/` | Agent configs |
