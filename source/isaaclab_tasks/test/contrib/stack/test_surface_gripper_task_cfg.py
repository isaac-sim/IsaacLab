# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_SURFACE_GRIPPER_TASKS = [
    "IsaacContrib-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow",
    "IsaacContrib-Stack-Cube-UR10-Long-Suction-IK-Rel",
    "IsaacContrib-Stack-Cube-UR10-Short-Suction-IK-Rel",
]


@pytest.mark.parametrize("task_name", _SURFACE_GRIPPER_TASKS)
def test_surface_gripper_tasks_default_to_cpu(task_name: str) -> None:
    """Surface-gripper tasks should select their only supported simulation device."""
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    assert env_cfg.sim.device == "cpu"
    env_cfg.validate()


@pytest.mark.parametrize("task_name", _SURFACE_GRIPPER_TASKS)
def test_surface_gripper_tasks_reject_gpu_override(task_name: str) -> None:
    """An explicit unsupported GPU device should fail during config validation."""
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.sim.device = "cuda:0"

    with pytest.raises(ValueError, match="only supported on the CPU simulation device"):
        env_cfg.validate()
