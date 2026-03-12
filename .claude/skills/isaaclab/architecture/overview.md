# Isaac Lab Architecture

## Stack: OpenUSD -> PhysicsBackend -> Isaac Lab

- **Isaac Sim**: Physics (PhysX/Newton), RTX rendering, asset management
- **Isaac Lab**: Robot Learning framework that can be used for RL, IL, data generation and much more

## Two Environment Patterns

### Manager-Based (Recommended)

Modular. Separate config classes for each MDP component.

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg(num_envs=4096, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1/120
```

**Best for**: Most tasks, collaboration, experimentation with rewards/observations.

### Direct (Monolithic)

All logic in one class. Override specific methods.

```python
class MyEnv(DirectRLEnv):
    def _setup_scene(self): ...
    def _pre_physics_step(self, actions): ...
    def _apply_action(self): ...
    def _get_observations(self) -> dict: ...
    def _get_rewards(self) -> Tensor: ...
    def _get_dones(self) -> tuple[Tensor, Tensor]: ...
    def _reset_idx(self, env_ids): ...
```

**Best for**: Simple environments, maximum control, multi-agent (DirectMARLEnv).

For detailed comparison, see [patterns-comparison.md](patterns-comparison.md).

## MDP Term Types (Manager-Based)

| Manager | Config | Signature |
|---------|--------|-----------|
| Observation | `ObsTerm(func=...)` | `func(env) -> Tensor` |
| Action | `JointEffortActionCfg(...)` | Built-in terms |
| Reward | `RewTerm(func=..., weight=...)` | `func(env, **params) -> Tensor` |
| Termination | `DoneTerm(func=...)` | `func(env, **params) -> BoolTensor` |
| Event | `EventTerm(func=..., mode=...)` | `func(env, env_ids, **params)` |

Built-in MDP functions: `isaaclab.envs.mdp.*`

## Key Source Locations

| Component | Path |
|-----------|------|
| Core framework | `source/isaaclab/isaaclab/` |
| MDP functions | `source/isaaclab/isaaclab/envs/mdp/` |
| Managers | `source/isaaclab/isaaclab/managers/` |
| Robot assets | `source/isaaclab_assets/isaaclab_assets/robots/` |
| Task environments | `source/isaaclab_tasks/isaaclab_tasks/` |
| RL wrappers | `source/isaaclab_rl/isaaclab_rl/` |
| Controllers | `source/isaaclab/isaaclab/controllers/` |

## For More Detail

- [patterns-comparison.md](patterns-comparison.md) - Manager-based vs Direct deep-dive
- [sensors-actuators.md](sensors-actuators.md) - Cameras, IMU, actuator types, visualizers
