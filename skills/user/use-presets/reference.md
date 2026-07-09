# Preset System Reference

## Contents

- When presets are useful
- When presets are unnecessary
- Selector types
- Definition pattern
- Validation checklist

## When Presets Are Useful

Use `PresetCfg` when a field has multiple meaningful named variants that users should select from the command line or through task registration.

Good candidates:

- Physics backend settings, such as PhysX versus Newton MJWarp.
- Renderer settings, such as Isaac RTX, Newton Warp, or OVRTX renderers.
- Camera data types, such as RGB, depth, albedo, or segmentation.
- Backend-specific event or sensor configs.
- Backend-specific schema cfg choices when PhysX, Newton, or MuJoCo require different USD physics attributes.
- Domain variants where one task supports multiple authored modes.

## When Presets Are Unnecessary

Do not add presets when:

- The task supports only one backend or renderer.
- The difference is a one-off training override.
- A plain config field is clearer.
- The variant is not exposed or tested.

Prefer the simple config first. Add presets after the task has at least two tested variants.

## Selector Types

Isaac Lab preset-aware entry points recognize three selector forms:

| Selector | Purpose |
| --- | --- |
| `physics=NAME` | Selects variants whose values are physics config objects. |
| `renderer=NAME` | Selects variants whose values are renderer config objects. |
| `presets=NAME[,NAME,...]` | Applies domain-specific variants or broadcasts preset names across matching preset fields. |

From the Isaac Lab checkout, use `uv run python scripts/environments/list_envs.py --show_presets` to inspect available names before guessing.

## Definition Pattern

Import paths:

```python
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
```

Pattern:

```python
@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg()
    physx = default
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())


@configclass
class MyEnvCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
```

For multi-backend tasks, keep backend-specific solver values in the preset wrapper. Do not branch on backend names inside step, reward, or reset logic unless behavior truly cannot be represented as config.

For schema presets, import universal fragments and base cfgs from `isaaclab.sim.schemas`, PhysX-specific cfgs from `isaaclab_physx.sim.schemas`, and Newton or MuJoCo cfgs from `isaaclab_newton.sim.schemas`.

## Validation Checklist

- The `default` variant is valid.
- Every named variant is tested.
- Selector names match existing conventions such as `physx`, `newton_mjwarp`, `newton_kamino`, `ovphysx`, `rgb`, and `depth`.
- A small random-agent rollout succeeds for each variant.
- Training commands include only preset names that the task exposes.
- Backend-specific schema, sensor, or event variants are kept inside preset classes rather than hidden in scattered runtime conditionals.
