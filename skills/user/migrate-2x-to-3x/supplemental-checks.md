# Supplemental Migration Checks

## Contents

- Current-source checks
- Prototype-skill checks
- Documentation gaps

## Current-Source Checks

Use these checks to route investigation, not as standalone migration docs:

| Symptom or old pattern | Current source of truth |
| --- | --- |
| `--headless` launch behavior changed | `docs/source/migration/migrating_to_isaaclab_3-0.rst` and `source/isaaclab/isaaclab/app/app_launcher.py` |
| Backend-specific physics or schema cfgs | `docs/source/overview/core-concepts/multi_backend_architecture.rst` and `docs/source/overview/core-concepts/schema_cfgs.rst` |
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
- `asset.data.*.detach()`
- `root_physx_view`
- `get_published_pretrained_checkpoint`
- `noise_std_type`
- `--viz`

## Documentation Gaps

If one of these checks reveals a real migration issue that is missing from the official migration guide, add the migration note to `docs/source/migration/migrating_to_isaaclab_3-0.rst` and keep this file as a pointer.
