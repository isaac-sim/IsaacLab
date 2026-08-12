# Preset Examples

## Contents

- Simplified config without presets
- Physics presets
- Domain presets
- Combined presets
- Existing source examples

## Simplified Config Without Presets

Use this when the environment has one supported physics setup and no user-selectable variants.

```python
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg


class MySimpleEnvCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
```

This is enough when the task only supports PhysX and there are no renderer, sensor, event, or domain variants to expose.

## Physics Presets

Use `PresetCfg` when the same task supports multiple physics backends.
The example below applies when the task's established default is PhysX. Preserve
an explicit Newton or other backend default when adding more variants.

```python
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=120, nconmax=15),
        num_substeps=1,
    )


@configclass
class MyMultiBackendEnvCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
```

Command examples:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Ant --num_envs 4
uv run python scripts/environments/random_agent.py --task Isaac-Ant --num_envs 4 physics=physx
uv run python scripts/environments/random_agent.py --task Isaac-Ant --num_envs 4 physics=newton_mjwarp
```

## Domain Presets

Use domain presets for environment-specific variants such as camera output type.

```python
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg


@configclass
class CameraTaskCfg(PresetCfg):
    @configclass
    class BaseCfg(DirectRLEnvCfg):
        observation_space = [100, 100, 3]

    default = BaseCfg()
    rgb = default
    depth = BaseCfg(observation_space=[100, 100, 1])
```

Command examples:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera-Direct --num_envs 4 presets=rgb
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera-Direct --num_envs 4 presets=depth
```

## Combined Presets

For camera tasks that expose physics, renderer, and data-type variants, combine selectors:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera-Direct --num_envs 4 physics=isaacsim_physx renderer=isaacsim_rtx presets=rgb
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Camera-Direct --num_envs 4 physics=newton_mjwarp renderer=newton_renderer presets=depth
```

Before using a name, list the task's exposed presets:

```bash
uv run python scripts/environments/list_envs.py --show_presets
```

## Existing Source Examples

Inspect these maintained examples before adding new preset patterns:

- `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/flat_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/go1/flat_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_camera_env_cfg.py`
