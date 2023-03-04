# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""

import scipy.spatial.transform as tf
import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.objects.rigid import RigidObject, RigidObjectCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.math import convert_quat, quat_mul, random_yaw_orientation, sample_cylinder

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
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
    """Imports all legged robots supported in Orbit and applies zero actions."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cpu")
    # Set main camera
    set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # design props
    design_scene()
    # add YCB objects
    ycb_usd_paths = {
        "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    }
    for key, usd_path in ycb_usd_paths.items():
        translation = torch.rand(3).tolist()
        prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation)

    # Setup rigid object
    cfg = RigidObjectCfg()
    # -- usd path
    cfg.meta_info.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
    # -- rotate the object to align with the ground plane
    cfg.init_state.rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90, 90, 0), degrees=True).as_quat(), to="wxyz")

    # Create rigid object handler
    rigid_object = RigidObject(cfg)

    # Spawn rigid object
    # note: Spawning object like this will apply rigid object properties and physics material configurations.
    rigid_object.spawn("/World/Objects/crackerBox2")

    # Play the simulator
    sim.reset()
    # Initialize handles
    # note: We desire view over all the objects in the scene.
    rigid_object.initialize("/World/Objects/.*")

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
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
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = rigid_object.get_default_root_state()
            # -- position
            root_state[:, :3] = sample_cylinder(
                radius=0.5, h_range=(0.15, 0.25), size=rigid_object.count, device=rigid_object.device
            )
            # -- orientation: apply yaw rotation
            root_state[:, 3:7] = quat_mul(
                random_yaw_orientation(rigid_object.count, rigid_object.device), root_state[:, 3:7]
            )
            # -- set root state
            rigid_object.set_root_state(root_state)
            # reset buffers
            rigid_object.reset_buffers()
            print(">>>>>>>> Reset!")
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            rigid_object.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
