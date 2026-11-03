# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Pollen task discovery must preserve the original MicroDuck environments."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401


@pytest.mark.unit
def test_pollen_tasks_do_not_replace_original_microduck_tasks():
    for terrain in ("Flat", "Rough"):
        builtin = gym.spec(f"IsaacContrib-Velocity-{terrain}-MicroDuck")
        assert builtin.kwargs["env_cfg_entry_point"] == (
            f"isaaclab_tasks.contrib.microduck.{terrain.lower()}_env_cfg:MicroDuckVelocity{terrain}EnvCfg"
        )

    project_tasks = {
        spec.id
        for spec in gym.registry.values()
        if spec.kwargs.get("env_cfg_entry_point", "").startswith("isaaclab_tasks.contrib.microduck.")
        and spec.id not in {"IsaacContrib-Velocity-Flat-MicroDuck", "IsaacContrib-Velocity-Rough-MicroDuck"}
    }
    assert project_tasks == {
        "IsaacContrib-Velocity-Flat-MicroDuck-Pollen",
        "IsaacContrib-Velocity-Rough-MicroDuck-Pollen",
        "IsaacContrib-Velocity-Flat-MicroDuck-Rollers",
        "IsaacContrib-Velocity-Swizzle-MicroDuck",
        "IsaacContrib-StandUp-Flat-MicroDuck",
        "IsaacContrib-StandUp-Rough-MicroDuck",
        "IsaacContrib-SitStand-Flat-MicroDuck",
        "IsaacContrib-SitStand-Rough-MicroDuck",
        "IsaacContrib-Roulade-Flat-MicroDuck",
        "IsaacContrib-BallKick-Flat-MicroDuck",
        "IsaacContrib-VelStand-Flat-MicroDuck",
        "IsaacContrib-VelStand-Rough-MicroDuck",
        "IsaacContrib-GroundPick-Flat-MicroDuck",
        "IsaacContrib-GroundPick-Rough-MicroDuck",
        "IsaacContrib-RollerCrouch-Flat-MicroDuck",
        "IsaacContrib-RollerStandUp-Flat-MicroDuck",
        "IsaacContrib-Spin-Flat-MicroDuck",
        "IsaacContrib-RollerSlope-Flat-MicroDuck",
    }
