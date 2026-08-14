# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True, limit_cpu_threads=1)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest

from isaaclab.physics import PhysicsCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import collect_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


_PHYSICS_PRESET_NAMES = ("newton_mjwarp", "physx", "isaacsim_physx")


def _core_physics_params() -> list:
    """Return core task/backend pairs for explicitly supported physics presets."""
    params = []
    for task_param in setup_environment(multi_agent=False, tier="core"):
        task_name = getattr(task_param, "values", (task_param,))[0]
        marks = getattr(task_param, "marks", ())
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        physics_preset_groups = collect_presets(env_cfg).values()
        for physics_preset_name in _PHYSICS_PRESET_NAMES:
            if any(
                physics_preset_name in preset_group and isinstance(preset_group[physics_preset_name], PhysicsCfg)
                for preset_group in physics_preset_groups
            ):
                params.append(
                    pytest.param(
                        task_name,
                        physics_preset_name,
                        id=f"{task_name}-{physics_preset_name}",
                        marks=marks,
                    )
                )
    return params


@pytest.mark.parametrize("task_name, physics_preset_name", _core_physics_params())
@pytest.mark.parametrize("num_envs, device", [(2, "cuda"), (1, "cuda")])
@pytest.mark.isaacsim_ci
def test_environments(task_name, physics_preset_name, num_envs, device):
    # run environments without stage in memory
    _run_environments(
        task_name, device, num_envs, create_stage_in_memory=False, physics_preset_name=physics_preset_name
    )
