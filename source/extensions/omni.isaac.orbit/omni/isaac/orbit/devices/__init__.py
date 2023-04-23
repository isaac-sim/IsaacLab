# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module providing interfaces to different teleoperation devices.

Currently, the module supports three categories of devices:

* Keyboard: Standard keyboard with WASD and arrow keys.
* Spacemouse: 3D mouse with 6 degrees of freedom.
* Gamepad: Gamepad with 2D two joysticks and buttons. Example: Xbox controller.

All device interfaces inherit from the :class:`DeviceBase` class, which provides a
common interface for all devices. The device interface reads the input data when
the :meth:`advance()` method is called. It also provides the function :meth:`add_callback()`
to add user-defined callback functions to be called when a particular input is pressed from
the peripheral device.

Example usage showing the keyboard interface:
    .. code-block:: python

        from omni.isaac.kit import SimulationApp

        # launch the simulator
        simulation_app = SimulationApp({"headless": False})

        from omni.isaac.core.simulation_context import SimulationContext

        from omni.isaac.orbit.devices import Se3Keyboard


        def print_cb():
            print("Print callback")


        def quit_cb():
            print("Quit callback")
            simulation_app.close()


        # Load kit helper
        sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01)

        # Create teleoperation interface
        teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
        # Add teleoperation callbacks
        teleop_interface.add_callback("L", print_cb)
        teleop_interface.add_callback("ESCAPE", quit_cb)

        # print instructions
        print(teleoperation_interface)
        print("Press 'L' to print a message. Press 'ESC' to quit.")

        # Reset interface internals
        teleop_interface.reset()

        # Play simulation
        sim.reset()

        # Run simulation
        while simulation_app.is_running():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # step simulation
            sim.step()
            # check if simulator is stopped
            if sim.is_stopped():
                break

"""

from .gamepad import Se2Gamepad, Se3Gamepad
from .keyboard import Se2Keyboard, Se3Keyboard
from .spacemouse import Se2SpaceMouse, Se3SpaceMouse

__all__ = ["Se2Keyboard", "Se3Keyboard", "Se2SpaceMouse", "Se3SpaceMouse", "Se2Gamepad", "Se3Gamepad"]
