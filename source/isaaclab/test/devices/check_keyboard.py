# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use a teleoperation device with Isaac Sim.

The teleoperation device is a keyboard device that allows the user to control the robot.
It is possible to add additional callbacks to it for user-defined operations. Keyboard input needs the Kit
window, so run it with ``--viz kit``.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Check the keyboard teleoperation device.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import sys

from isaaclab.sim import SimulationCfg, SimulationContext


def print_cb():
    """Dummy callback function executed when the key 'L' is pressed."""
    print("Print callback")


def quit_cb():
    """Dummy callback function executed when the key 'ESC' is pressed."""
    print("Quit callback")
    SimulationContext.instance().stop()


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        # the keyboard device uses Kit's input interface, so import it once Kit is running
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        sim = SimulationContext(sim_cfg)

        # Create teleoperation interface
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        # Add teleoperation callbacks
        # available key buttons: https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.KeyboardInput
        teleop_interface.add_callback("L", print_cb)
        teleop_interface.add_callback("ESCAPE", quit_cb)

        print("Press 'L' to print a message. Press 'ESC' to quit.")

        # Check that the framework doesn't hold excessive strong references.
        if sys.getrefcount(teleop_interface) >= 10:
            raise RuntimeError("Possible reference leak for teleoperation interface.")

        # Reset interface internals
        teleop_interface.reset()

        # Play simulation
        sim.reset()

        # Simulate
        while sim.is_running():
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
    main()
