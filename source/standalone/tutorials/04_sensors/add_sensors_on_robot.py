# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/04_sensors/add_sensors_on_robot.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config.anymal import ANYMAL_C_CFG
from omni.isaac.orbit.sensors import (
    Camera,
    CameraCfg,
    ContactSensor,
    ContactSensorCfg,
    RayCaster,
    RayCasterCfg,
    patterns,
)


def design_scene() -> tuple[Articulation, tuple[tuple[Camera], tuple[RayCaster], tuple[ContactSensor]]]:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- robot
    anymal_c_cfg = ANYMAL_C_CFG
    anymal_c_cfg.spawn.func("/World/Anymal_c/Robot_1", anymal_c_cfg.spawn, translation=(1.5, -1.5, 0.65))
    anymal_c_cfg.spawn.func("/World/Anymal_c/Robot_2", anymal_c_cfg.spawn, translation=(1.5, -0.5, 0.65))
    anymal_c = Articulation(anymal_c_cfg.replace(prim_path="/World/Anymal_c/Robot.*"))

    # adds different sensors
    sensors = add_sensors()

    return anymal_c, sensors


def add_sensors() -> tuple[tuple[Camera], tuple[RayCaster], tuple[ContactSensor]]:
    """Adds sensors to the robot."""
    # -- usd camera
    camera_cfg = CameraCfg(
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    cameras = (
        camera_cfg.class_type(camera_cfg.replace(prim_path="/World/Anymal_c/Robot_1/base/front_cam")),
        camera_cfg.class_type(camera_cfg.replace(prim_path="/World/Anymal_c/Robot_2/base/front_cam")),
    )
    # -- height scanner
    height_scanner_cfg = RayCasterCfg(
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )
    height_scanner = (
        height_scanner_cfg.class_type(height_scanner_cfg.replace(prim_path="/World/Anymal_c/Robot_1/base")),
        height_scanner_cfg.class_type(height_scanner_cfg.replace(prim_path="/World/Anymal_c/Robot_2/base")),
    )
    # -- contact sensors
    contact_forces_cfg = ContactSensorCfg(history_length=3, track_pose=True)
    contact_forces = (
        contact_forces_cfg.class_type(contact_forces_cfg.replace(prim_path="/World/Anymal_c/Robot_1/.*")),
        contact_forces_cfg.class_type(contact_forces_cfg.replace(prim_path="/World/Anymal_c/Robot_2/.*")),
    )

    return cameras, height_scanner, contact_forces


def run_simulator(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    sensors: tuple[tuple[Camera], tuple[RayCaster], tuple[ContactSensor]],
):
    """Run the simulator."""

    # unpack sensors
    cameras, height_scanner, contact_forces = sensors

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # apply default actions to the quadrupedal robots
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)
        for camera in cameras:
            camera.update(dt=sim_dt)
        for height_sensor in height_scanner:
            height_sensor.update(dt=sim_dt)
        for contact_sensor in contact_forces:
            contact_sensor.update(dt=sim_dt)

        # update camera data and print information
        print(cameras[0])
        print("Received shape of rgb   image: ", cameras[0].data.output["rgb"].shape)
        print("Received shape of depth image: ", cameras[0].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(height_scanner[0])
        print("Received max height value: ", torch.max(height_scanner[0].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(contact_forces[0])
        print("Received max contact force of: ", torch.max(contact_forces[0].data.net_forces_w).item())


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    robot, sensors = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, robot, sensors)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
