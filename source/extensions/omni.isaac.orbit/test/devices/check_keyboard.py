# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""
This script shows how to use a teleoperation device with Isaac Sim.

The teleoperation device is a keyboard device that allows the user to control the robot.
It is possible to add additional callbacks to it for user-defined operations.
"""

"""Launch Isaac Sim Simulator first."""


from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import traceback

import carb
from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.orbit.devices import Se3Keyboard


def print_cb():
    """Dummy callback function executed when the key 'L' is pressed."""
    print("Print callback")


def quit_cb():
    """Dummy callback function executed when the key 'ESC' is pressed."""
    print("Quit callback")
    simulation_app.close()


def main():
    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01)

    # Create teleoperation interface
    teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
    # Add teleoperation callbacks
    # available key buttons: https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.KeyboardInput
    teleop_interface.add_callback("L", print_cb)
    teleop_interface.add_callback("ESCAPE", quit_cb)

    print("Press 'L' to print a message. Press 'ESC' to quit.")

    # Check that boundedness of articulation is correct
    if ctypes.c_long.from_address(id(teleop_interface)).value != 1:
        raise RuntimeError("Teleoperation interface is not bounded to a single instance.")

    # Reset interface internals
    teleop_interface.reset()

    # Play simulation
    sim.reset()

    # Simulate
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue
        # get keyboard command
        delta_pose, gripper_command = teleop_interface.advance()
        # print command
        if gripper_command:
            print(f"Gripper command: {gripper_command}")
        # step simulation
        sim.step()
        # check if simulator is stopped
        if sim.is_stopped():
            break


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
