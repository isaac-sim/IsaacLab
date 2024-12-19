# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to make a floating robot fixed in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""


import argparse
import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from isaacsim import SimulationApp

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

import isaacsim.core.utils.nucleus as nucleus_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.log
import omni.physx
from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.carb import set_carb_setting
from isaacsim.core.utils.viewports import set_camera_view
from pxr import PhysxSchema, UsdPhysics

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
Main
"""


def main():
    """Spawns the ANYmal robot and makes it fixed."""
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
    # -- Robot
    # resolve asset
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
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
        # get all necessary information
        stage = stage_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(root_prim_path)
        parent_prim = root_prim.GetParent()

        # here we assume that the root prim is a rigid body
        # there is no clear way to deal with situation where the root prim is not a rigid body but has articulation api
        # in that case, it is unclear how to get the link to the first link in the tree
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("The root prim does not have the RigidBodyAPI applied.")

        # create fixed joint
        omni.kit.commands.execute(
            "CreateJointCommand",
            stage=stage,
            joint_type="Fixed",
            from_prim=None,
            to_prim=root_prim,
        )

        # move the root to the parent if this is a rigid body
        # having a fixed joint on a rigid body makes physx treat it as a part of the maximal coordinate tree
        # if we put to joint on the parent, physx parser treats it as a fixed base articulation
        # get parent prim
        parent_prim = root_prim.GetParent()
        # apply api to parent
        UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
        PhysxSchema.PhysxArticulationAPI.Apply(parent_prim)

        # copy the attributes
        # -- usd attributes
        root_usd_articulation_api = UsdPhysics.ArticulationRootAPI(root_prim)
        for attr_name in root_usd_articulation_api.GetSchemaAttributeNames():
            attr = root_prim.GetAttribute(attr_name)
            parent_prim.GetAttribute(attr_name).Set(attr.Get())
        # -- physx attributes
        root_physx_articulation_api = PhysxSchema.PhysxArticulationAPI(root_prim)
        for attr_name in root_physx_articulation_api.GetSchemaAttributeNames():
            attr = root_prim.GetAttribute(attr_name)
            parent_prim.GetAttribute(attr_name).Set(attr.Get())

        # remove api from root
        root_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        root_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        # rename root path to parent path
        root_prim_path = parent_prim.GetPath().pathString

    # Setup robot
    robot_view = Articulation(root_prim_path, name="ANYMAL")
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
