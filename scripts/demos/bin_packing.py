# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn different number of objects in multiple environments.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/bin_packing.py --num_envs 2048

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Demo on spawning different number of objects in multiple bin packing environments."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene Configuration
##

POSE_RANGE = {"roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)}
VELOCITY_RANGE = {"roll": (-0.76, 0.76), "pitch": (-0.76, 0.76), "yaw": (-0.76, 0.76)}


RANDOM_YCB_RIGID_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            ),
            sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            ),
            sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            ),
            sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            ),
            # note: the placeholder, this allows the effect of having less objects in some env ids
            sim_utils.SphereCfg(
                radius=0.1,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visible=False
            ),
        ],
        random_choice=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(2.0, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )

    # object collection
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "Object_A_Layer1": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_A_Layer1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.035, -0.06, 0.2)),
            ),
            "Object_B_Layer1": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_B_Layer1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.035, 0.06, 0.2)),
            ),
            "Object_C_Layer1": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_C_Layer1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.035, 0.06, 0.2)),
            ),
            "Object_D_Layer1": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_D_Layer1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.035, -0.06, 0.2)),
            ),
            "Object_A_Layer2": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_A_Layer2",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.035, -0.06, 0.4)),
            ),
            "Object_B_Layer2": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_B_Layer2",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.035, 0.06, 0.4)),
            ),
            "Object_C_Layer2": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_C_Layer2",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.035, 0.06, 0.4)),
            ),
            "Object_D_Layer2": RANDOM_YCB_RIGID_OBJECT_CFG.replace(
                prim_path="/World/envs/env_.*/Object_D_Layer2",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.035, -0.06, 0.4)),
            ),
        }
    )


def reset_object_collections(scene: InteractiveScene, view_ids: torch.Tensor):
    if len(view_ids) == 0:
        return
    rigid_object_collection: RigidObjectCollection = scene["object_collection"]
    default_state_w = rigid_object_collection.data.default_object_state.clone()
    default_state_w[..., :3] = default_state_w[..., :3] + scene.env_origins.unsqueeze(1)
    default_state_w_view = rigid_object_collection.reshape_data_to_view(default_state_w)[view_ids]
    range_list = [POSE_RANGE.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=scene.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(view_ids), 6), device=scene.device)

    positions = default_state_w_view[:, :3] + samples[..., 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(samples[..., 3], samples[..., 4], samples[..., 5])
    orientations = math_utils.quat_mul(default_state_w_view[:, 3:7], orientations_delta)
    # velocities
    range_list = [VELOCITY_RANGE.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=scene.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(view_ids), 6), device=scene.device)

    velocities = default_state_w_view[:, 7:13] + samples
    new_poses = torch.concat((positions, orientations), dim=-1)

    new_poses[..., 3:] = math_utils.convert_quat(new_poses[..., 3:], to="xyzw")
    rigid_object_collection.root_physx_view.set_transforms(new_poses, indices=view_ids.view(-1, 1))
    rigid_object_collection.root_physx_view.set_velocities(velocities, indices=view_ids.view(-1, 1))



##
# Simulation Loop
##


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    rigid_object: RigidObject = scene["object"]
    rigid_object_collection: RigidObjectCollection = scene["object_collection"]
    view_indices = torch.arange(scene.num_envs * rigid_object_collection.num_objects, device=scene.device)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 250 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # object
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            rigid_object.write_root_pose_to_sim(root_state[:, :7])
            rigid_object.write_root_velocity_to_sim(root_state[:, 7:])
            # object collection
            reset_object_collections(scene, view_indices)
            scene.reset()
            print("[INFO]: Resetting scene state...")

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        object_pos_b = rigid_object_collection.data.object_pos_w - scene.env_origins.unsqueeze(1)
        object_pos_b_view = rigid_object_collection.reshape_data_to_view(object_pos_b)
        inbound_mask = (-1.0 < object_pos_b_view[:, 0]) & (object_pos_b_view[:, 0] < 1.0)
        inbound_mask &= (-1.0 < object_pos_b_view[:, 1]) & (object_pos_b_view[:, 1] < 1.0)
        reset_object_collections(scene, view_indices[~inbound_mask])
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
