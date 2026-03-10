# MDP Function Catalog

All built-in functions are in `isaaclab.envs.mdp`.

## Observation Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `joint_pos_rel` | Tensor | Joint positions relative to default |
| `joint_vel_rel` | Tensor | Joint velocities relative to default |
| `joint_pos` | Tensor | Absolute joint positions |
| `joint_vel` | Tensor | Absolute joint velocities |
| `root_pos_w` | Tensor | Root body position in world frame |
| `root_quat_w` | Tensor | Root body quaternion (wxyz) in world frame |
| `root_lin_vel_w` | Tensor | Root linear velocity in world frame |
| `root_ang_vel_w` | Tensor | Root angular velocity in world frame |
| `root_lin_vel_b` | Tensor | Root linear velocity in body frame |
| `root_ang_vel_b` | Tensor | Root angular velocity in body frame |
| `body_projected_gravity_b` | Tensor | Gravity vector in body frame |
| `last_action` | Tensor | Previous action |

## Reward Functions

| Function | Params | Description |
|----------|--------|-------------|
| `is_alive` | none | +1 per alive step |
| `is_terminated` | none | Flag on termination |
| `joint_pos_target_l2` | `target`, `asset_cfg` | L2 penalty vs target angle |
| `joint_vel_l1` | `asset_cfg` | L1 velocity penalty |
| `joint_vel_l2` | `asset_cfg` | L2 velocity penalty |
| `action_rate_l2` | none | Penalize action changes (smooth) |
| `applied_torque_limits` | none | Penalize reaching torque limits |
| `body_lin_vel_l2` | `asset_cfg` | L2 penalty on body linear velocity |

## Termination Functions

| Function | Params | Description |
|----------|--------|-------------|
| `time_out` | none | Episode exceeds `episode_length_s` |
| `joint_pos_out_of_manual_limit` | `bounds`, `asset_cfg` | Joint exceeds range |
| `root_height_below_minimum` | `minimum_height`, `asset_cfg` | Root fell below height |

## Event Functions (Randomization)

| Function | Mode | Params | Description |
|----------|------|--------|-------------|
| `reset_scene_to_default` | reset | none | Reset to initial state |
| `reset_joints_by_offset` | reset | `position_range`, `velocity_range` | Randomize joints |
| `reset_root_state_uniform` | reset | `pose_range`, `velocity_range` | Randomize root state |
| `randomize_rigid_body_mass` | interval | `mass_distribution_params`, `asset_cfg` | Mass randomization |

**Event modes:**
- `reset`: Runs when environments reset
- `interval`: Runs periodically (use `interval_range_s`)
- `startup`: Runs once at environment creation

## Action Types

| Config | Control | Params |
|--------|---------|--------|
| `JointEffortActionCfg` | Force/torque | `asset_name`, `joint_names`, `scale` |
| `JointPositionActionCfg` | Position target | `asset_name`, `joint_names`, `scale` |
| `JointVelocityActionCfg` | Velocity target | `asset_name`, `joint_names`, `scale` |
| `DifferentialInverseKinematicsActionCfg` | Cartesian IK | `asset_name`, `body_name`, controller config |

## Custom Reward Function Template

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

def my_reward(
    env: ManagerBasedRLEnv,
    target: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Custom reward: penalize deviation from target."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)
```

Usage in config:
```python
my_term = RewTerm(
    func=my_reward, weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]), "target": 0.0},
)
```
