# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim before importing task modules."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Run each dashboard-validated FeatherPGS task."""

import pytest

import isaaclab_tasks  # noqa: F401

from env_test_utils import _run_environments  # isort: skip


_TASKS = [
    "Isaac-Reach-Franka",
    "Isaac-Reach-UR10",
    "Isaac-Cartpole",
    "Isaac-Cartpole-Direct",
    "Isaac-Ant",
    "Isaac-Ant-Direct",
    "Isaac-Reorient-Cube-Allegro-Direct",
    "Isaac-Reorient-Cube-Shadow-Direct",
    "IsaacContrib-Velocity-Flat-AnymalB",
    "IsaacContrib-Velocity-Rough-AnymalB",
    "IsaacContrib-Velocity-Flat-AnymalC",
    "IsaacContrib-Velocity-Rough-AnymalC",
    "Isaac-Velocity-Flat-AnymalD",
    "Isaac-Velocity-Rough-AnymalD",
    "IsaacContrib-Velocity-Flat-UnitreeA1",
    "IsaacContrib-Velocity-Rough-UnitreeA1",
    "IsaacContrib-Velocity-Flat-UnitreeGo1",
    "IsaacContrib-Velocity-Rough-UnitreeGo1",
    "Isaac-Velocity-Flat-UnitreeGo2",
    "Isaac-Velocity-Rough-UnitreeGo2",
    "Isaac-Velocity-Flat-Cassie",
    "Isaac-Velocity-Rough-Cassie",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Rough-G1",
    "Isaac-Velocity-Flat-H1",
    "Isaac-Velocity-Rough-H1",
    "Isaac-Humanoid",
    "Isaac-Humanoid-Direct",
    "Isaac-Open-Drawer-Franka",
]


@pytest.mark.parametrize("task_name", _TASKS)
@pytest.mark.newton_ci
def test_environments_feather_pgs(task_name: str) -> None:
    _run_environments(
        task_name,
        "cuda",
        2,
        num_steps=2,
        physics_preset_name="feather_pgs",
        create_stage_in_memory=False,
    )
