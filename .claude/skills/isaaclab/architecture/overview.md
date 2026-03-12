# Isaac Lab Architecture

## Multi-Backend Architecture (Lab 3.0)

Isaac Lab uses a **factory pattern** for physics and rendering. The core `isaaclab`
package defines abstract base classes; concrete implementations live in backend packages.

```
isaaclab (core)          — Abstract base classes, managers, MDP, utils
  ├── isaaclab_physx     — PhysX physics backend + Isaac RTX renderer
  ├── isaaclab_newton    — Newton/MuJoCo-Warp physics backend + Warp renderer
  └── isaaclab_ov        — Omniverse RTX (OVRTX) renderer
```

When you write `from isaaclab.assets import Articulation`, the factory detects the
active backend (from `SimulationContext.physics_manager`) and returns the correct
implementation (e.g., `isaaclab_physx.assets.Articulation` or
`isaaclab_newton.assets.Articulation`). Your code doesn't change between backends.

For details on backends and renderers, see [backends.md](backends.md).
For the migration guide: `docs/source/migration/migrating_to_isaaclab_3-0.rst`

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
| Physics base classes | `source/isaaclab/isaaclab/physics/` |
| MDP functions | `source/isaaclab/isaaclab/envs/mdp/` |
| Managers | `source/isaaclab/isaaclab/managers/` |
| PhysX backend | `source/isaaclab_physx/` |
| Newton backend | `source/isaaclab_newton/` |
| OV renderers | `source/isaaclab_ov/` |
| Visualizers | `source/isaaclab_visualizers/` |
| Robot assets | `source/isaaclab_assets/isaaclab_assets/robots/` |
| Task environments | `source/isaaclab_tasks/isaaclab_tasks/` |
| RL wrappers | `source/isaaclab_rl/isaaclab_rl/` |
| Controllers | `source/isaaclab/isaaclab/controllers/` |

## For More Detail

- [backends.md](backends.md) - Physics backends, renderers, and visualizers
- [patterns-comparison.md](patterns-comparison.md) - Manager-based vs Direct deep-dive
- [sensors-actuators.md](sensors-actuators.md) - Cameras, IMU, actuator types
- `docs/source/refs/reference_architecture/` - Official reference architecture
