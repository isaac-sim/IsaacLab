# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--robot", type=str, default="allegro_hand", help="Robot type to use.", choices=["allegro_hand", "anymal_d"]
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if args_cli.robot == "allegro_hand":
    robot_cfg = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0, -0.1, 0.3)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/thumb_link_.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/index_link.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                target_prim_expr="{ENV_REGEX_NS}/Robot/middle_link_.*/visuals_xform"
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/ring_link_.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/palm_link/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/allegro_mount/visuals_xform"),
        ],
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.005, size=(0.4, 0.4), direction=(0, 0, -1)),
        track_mesh_transforms=True,
        debug_vis=not args_cli.headless,
    )

elif args_cli.robot == "anymal_d":
    robot_cfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0, -0.1, 0.3)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/LF_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/RF_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/LH_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/RH_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr="{ENV_REGEX_NS}/Robot/base/visuals"),
        ],
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(2.5, 2.5), direction=(0, 0, -1)),
        track_mesh_transforms=True,
        debug_vis=not args_cli.headless,
    )
else:
    raise ValueError(f"Unknown robot type: {args_cli.robot}")


@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            scale=(1, 1, 1),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = robot_cfg
    # ray caster
    ray_caster = ray_caster_cfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    triggered = True
    countdown = 42

    # Simulate physics
    while simulation_app.is_running():

        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos + 5 * (
            torch.rand_like(scene["robot"].data.default_joint_pos) - 0.5
        )
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        # print("-------------------------------")
        # print(scene["ray_caster"])
        # print("Ray cast hit results: ", scene["ray_caster"].data.ray_hits_w)

        if not triggered:
            if countdown > 0:
                countdown -= 1
                continue
            data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()
            np.save("cast_data.npy", data)
            triggered = True
        else:
            continue


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
