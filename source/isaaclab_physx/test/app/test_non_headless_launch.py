# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script checks that the ``isaaclab.python.kit`` experience launches and steps the simulation without hanging.
"""

import pytest

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(physics="isaacsim_physx", experience="isaaclab.python.kit")

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.integration


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


def run_simulator(
    sim: sim_utils.SimulationContext,
) -> int:
    """Run the simulator and return the number of steps taken."""

    count = 0

    # Simulate physics
    while sim.is_running() and count < 100:
        # perform step
        sim.step()
        count += 1
    return count


@pytest.mark.isaacsim_ci
def test_full_experience_launch_steps():
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(physics=PhysxCfg(), dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=1, env_spacing=2.0)
    InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # the app must keep running for every requested step instead of stopping early
    assert run_simulator(sim) == 100
