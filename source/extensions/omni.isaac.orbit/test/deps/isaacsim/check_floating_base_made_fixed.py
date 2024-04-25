# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to make a floating robot fixed in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script shows the issue in Isaac Sim with making a floating robot fixed."
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
parser.add_argument("--fix-base", action="store_true", help="Whether to fix the base of the robot.")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": args_cli.headless})

"""Rest everything follows."""

import torch

import carb
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from pxr import Sdf, UsdPhysics
import omni.physx
import omni.kit.commands
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.world import World

# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    carb.log_error(msg)
    raise RuntimeError(msg)


ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""

ISAAC_ORBIT_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac/Samples/Orbit"
"""Path to the `Isaac/Samples/Orbit` directory on the NVIDIA Nucleus Server."""


"""
Main
"""


def main():
    """Spawns the ANYmal robot and makes it fixed."""
    # -- Robot
    # resolve asset
    usd_path = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    root_prim_path = "/World/Robot/base"
    # add asset
    print("Loading robot from: ", usd_path)
    prim_utils.create_prim(
        "/World/Robot",
        usd_path=usd_path,
        translation=(0.0, 0.0, 0.6),
    )
    # create fixed joint
    if args_cli.fix_base:
        stage = stage_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath("/World/Robot/base")

        # create fixed joint
        omni.kit.commands.execute(
            "CreateJointCommand", stage=stage, joint_type="Fixed", from_prim=None, to_prim=root_prim
        )
        # joint_prim = UsdPhysics.FixedJoint.Define(stage, "/World/Robot/rootJoint")
        # joint_prim.GetJointEnabledAttr().Set(True)
        # # joint_prim.CreateBody0Rel().SetTargets([])
        # joint_prim.CreateBody1Rel().SetTargets([root_prim_path])

    # Load kit helper
    world = World(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cpu")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(world._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Spawn things into stage
    # Ground-plane
    world.scene.add_default_ground_plane(prim_path="/World/defaultGroundPlane", z_position=0.0)
    # Lights-1
    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
    # Lights-2
    prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))

   

    # Setup robot
    robot_view = ArticulationView(root_prim_path, name="ANYMAL")
    world.scene.add(robot_view)
    # Play the simulator
    world.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    # actions = torch.zeros(robot.count, robot.num_actions, device=robot.device)

    init_root_pos_w, init_root_quat_w = robot_view.get_world_poses()
    # Define simulation stepping
    sim_dt = world.get_physics_dt()
    # episode counter
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if world.is_stopped():
            break
        # If simulation is paused, then skip.
        if not world.is_playing():
            world.step(render=False)
            continue
        # do reset
        if count % 20 == 0:
            # reset
            sim_time = 0.0
            count = 0
            # reset root state
            root_pos_w = init_root_pos_w.clone()
            root_pos_w[:, :2] += torch.rand_like(root_pos_w[:, :2]) * 0.5
            robot_view.set_world_poses(root_pos_w, init_root_quat_w)
            # print if it is fixed base
            print("Fixed base: ", robot_view._physics_view.shared_metatype.fixed_base)
            print("Moving base to: ", root_pos_w[0].cpu().numpy())
            print("-" * 50)

        # apply random joint actions
        actions = torch.rand_like(robot_view.get_joint_positions()) * 0.001
        robot_view.set_joint_efforts(actions)
        # perform step
        world.step()
        # update sim-time
        sim_time += sim_dt
        count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
