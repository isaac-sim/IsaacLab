# Supplemental Migration Checks

## Contents

- Current-source checks
- Prototype-skill checks
- Documentation gaps

## Current-Source Checks

Use these checks to route investigation, not as standalone migration docs:

| Symptom or old pattern | Current source of truth |
| --- | --- |
| Task names include the old Gym version suffix, such as `-v0` | Current task docs and the environment catalog; use suffixless task names in examples |
| `--headless` launch behavior changed | `docs/source/migration/migrating_to_isaaclab_3-0.rst` and `source/isaaclab/isaaclab/app/app_launcher.py` |
| Camera examples require `--enable_cameras` by default | Current sensor, renderer, and visualization docs; do not add the flag unless the task or docs explicitly require it |
| Backend-specific physics or schema cfgs | `docs/source/overview/core-concepts/multi_backend_architecture.rst` and `docs/source/overview/core-concepts/schema_cfgs.rst` |
| Imports of PhysX/Newton schema cfgs from `isaaclab.sim.schemas` | Move backend-specific imports to `isaaclab_physx.sim.schemas` or `isaaclab_newton.sim.schemas`; core forwarding shims are deprecated |
| Spawner schema overrides that need multiple namespaces in one slot | Prefer schema fragments such as `UsdPhysicsDriveCfg`, `PhysxJointCfg`, `NewtonCollisionCfg`, or `MujocoJointCfg` instead of forcing one legacy property cfg to carry every backend attribute |
| Quaternion order changed from WXYZ to XYZW | `docs/source/migration/migrating_to_isaaclab_3-0.rst` and `scripts/tools/find_quaternions.py` |
| Asset or sensor data no longer behaves like plain tensors | `ProxyArray` sections in `docs/source/migration/migrating_to_isaaclab_3-0.rst` |
| `root_physx_view` or object API warnings | asset view sections in `docs/source/migration/migrating_to_isaaclab_3-0.rst` |
| RSL-RL config compatibility errors | `source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py` |
| Pretrained checkpoint helper import path | `source/isaaclab_rl/isaaclab_rl/utils/pretrained_checkpoint.py` |

## Prototype-Skill Checks

The external prototype migration skill called out useful search terms. Before using any fix from that prototype, verify against current docs/source:

- `AdditiveUniformNoiseCfg`
- `SimulationCfg.physics`
- `PhysxCfg`
- `isaaclab.sim.schemas`
- `PhysxRigidBodyPropertiesCfg`
- `NewtonCollisionPropertiesCfg`
- `UsdPhysicsDriveCfg`
- `PhysxJointCfg`
- `MujocoJointCfg`
- `asset.data.*.detach()`
- `root_physx_view`
- `get_published_pretrained_checkpoint`
- `noise_std_type`
- `--viz`
- `--enable_cameras`
- `-v0`

## Documentation Gaps

If one of these checks reveals a real migration issue that is missing from the official migration guide, add the migration note to `docs/source/migration/migrating_to_isaaclab_3-0.rst` and keep this file as a pointer.
