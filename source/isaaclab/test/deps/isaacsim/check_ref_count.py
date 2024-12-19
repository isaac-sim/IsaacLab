# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates reference count of the robot view in Isaac Sim.

When we make a class instance, the reference count of the class instance should always be 1.
However, in this script, the reference count of the robot view is 2 after the class is created.
This causes a memory leak in the Isaac Sim simulator and the robot view is not garbage collected.

The issue is observed with torch 2.2 and Isaac Sim 4.0. It works fine with torch 2.0.1 and Isaac Sim 2023.1.
It can be resolved by uncommenting the line that creates a dummy tensor in the main function.

To reproduce the issue, run this script and check the reference count of the robot view.

For more details, please check: https://github.com/isaac-sim/IsaacLab/issues/639
"""

"""Launch Isaac Sim Simulator first."""


import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from isaacsim import SimulationApp

# launch omniverse app
simulation_app = SimulationApp({"headless": True})

"""Rest everything follows."""

import ctypes
import gc
import torch  # noqa: F401

import omni.log

try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.carb import set_carb_setting

# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    omni.log.error(msg)
    raise RuntimeError(msg)


ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the `Isaac/IsaacLab` directory on the NVIDIA Nucleus Server."""


"""
Classes
"""


class AnymalArticulation:
    """Anymal articulation class."""

    def __init__(self):
        """Initialize the Anymal articulation class."""
        # resolve asset
        usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        # add asset
        print("Loading robot from: ", usd_path)
        prim_utils.create_prim("/World/Robot", usd_path=usd_path, translation=(0.0, 0.0, 0.6))

        # Resolve robot prim paths
        root_prim_path = "/World/Robot/base"
        # Setup robot
        self.view = Articulation(root_prim_path, name="ANYMAL")

    def __del__(self):
        """Delete the Anymal articulation class."""
        print("Deleting the Anymal view.")
        self.view = None

    def initialize(self):
        """Initialize the Anymal view."""
        self.view.initialize()


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")

    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create a dummy tensor for testing
    # Uncommenting the following line will yield a reference count of 1 for the robot (as desired)
    # dummy_tensor = torch.zeros(1, device="cuda:0")

    # Robot
    robot = AnymalArticulation()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    # Play the simulator
    sim.reset()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    robot.initialize()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    # Stop the simulator
    sim.stop()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    # Clean up
    sim.clear()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
