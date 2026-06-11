# Refactor and cleanup core lift tasks

## Summary

Consolidates the rigid and soft Franka lifting tasks into a single
`isaaclab_tasks.core.lift` package, eliminating the duplicated
`lift_franka_soft` task and its parallel `mdp` module. Soft and cloth variants
now live as configs alongside the rigid one and share a common MDP package.

## Changes

### Package consolidation (breaking)
- Moved `isaaclab_tasks.core.lift_franka_soft` under
  `isaaclab_tasks.core.lift.config.franka_soft`, next to the rigid `franka`
  config.
- Merged the deformable MDP terms (observations, rewards, terminations) into the
  shared `isaaclab_tasks.core.lift.mdp` package instead of a separate per-variant
  `mdp`. The old `lift_franka_soft/mdp` module is removed.

### Environment IDs (breaking)
- Dropped the `-v0` version suffix from all lift Gym environment IDs:
  - `Isaac-Lift-Cube-Franka-v0` → `Isaac-Lift-Cube-Franka`
  - `Isaac-Lift-Cube-Franka-Play-v0` → `Isaac-Lift-Cube-Franka-Play`
  - `Isaac-Lift-Soft-Franka-v0` → `Isaac-Lift-Soft-Franka`
  - `Isaac-Lift-Cloth-Franka-v0` → `Isaac-Lift-Cloth-Franka`

### Fixes
- Renamed the `Isaac-Lift-Cloth-Franka` physics preset from the misspelled
  `newton_mjwarp_vdb` to `newton_mjwarp_vbd`, matching the soft-body task and the
  underlying VBD solver.
- The cloth `RewardsCfg`, which duplicated the soft task's rewards verbatim, now
  inherits them instead of redefining.

### Misc
- Updated imports, docs, benchmarks, and tests to reference the new module paths
  and environment IDs.
- Minor docstring/comment cleanup in `lift_env_cfg.py`.

## Migration

```python
# Before
from isaaclab_tasks.core.lift_franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
# After
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg

# Deformable MDP terms now come from the shared package
from isaaclab_tasks.core.lift.mdp import deformable_lifted
```

Update any `gym.make(...)` / `--task` calls to use the unversioned environment
IDs listed above.
