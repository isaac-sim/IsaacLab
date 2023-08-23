# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the contact sensor sensor in Orbit.

.. code-block:: bash

    ./orbit.sh -p source/extensions/omni.isaac.orbit/test/sensors/test_contact_sensor.py --num_robots 2
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Contact Sensor Test Script")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_robots", type=int, default=64, help="Number of robots to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.compat.utils.kit as kit_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config.anymal import ANYMAL_C_CFG
from omni.isaac.orbit.sensors.contact_sensor import ContactSensor, ContactSensorCfg

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Enable flatcache which avoids passing data over to USD structure
    # this speeds up the read-write operation of GPU buffers
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")
    # Clone the scene
    num_envs = args_cli.num_robots
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    _ = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
    # Design props
    design_scene()
    # Spawn things into the scene
    robot_cfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True
    robot = Articulation(cfg=robot_cfg)
    # Contact sensor
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_SHANK", debug_vis=False if args_cli.headless else True
    )
    contact_sensor = ContactSensor(cfg=contact_sensor_cfg)
    # filter collisions within each environment instance
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )

    # Play the simulator
    sim.reset()
    # print info
    print(contact_sensor)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    decimation = 4
    physics_dt = sim.get_physics_dt()
    sim_dt = decimation * physics_dt
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
        # perform 4 steps
        for _ in range(decimation):
            # apply actions
            robot.set_joint_position_target(robot.data.default_joint_pos)
            # write commands to sim
            robot.write_data_to_sim()
            # perform step
            sim.step(render=not args_cli.headless)
            # fetch data
            robot.update(physics_dt)
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update the buffers
        if sim.is_playing():
            contact_sensor.update(sim_dt, force_recompute=True)
            if count % 100 == 0:
                print("Sim-time: ", sim_time)
                print("Number of contacts: ", torch.count_nonzero(contact_sensor.data.current_air_time == 0.0).item())
                print("-" * 80)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
