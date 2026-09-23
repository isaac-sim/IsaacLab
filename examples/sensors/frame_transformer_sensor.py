# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track poses between a robot's base, feet, and end-effector frames."""

import argparse
from typing import TYPE_CHECKING, cast

import torch

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

parser = argparse.ArgumentParser(description="Example on using the frame transformer sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--log_interval", type=int, default=100, help="Steps between compact sensor summaries.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx"],
    help="Physics backend.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.log_interval < 1:
    parser.error("--log_interval must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")


@configclass
class FrameTransformerSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Rigid Object
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(1, 1, 1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=100.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5, 0, 0.5)),
    )

    specific_transforms = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT"),
        ],
        debug_vis=True,
    )

    cube_transform = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Cube")],
        debug_vis=False,
    )

    robot_transforms = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")],
        debug_vis=False,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: "InteractiveScene") -> None:
    """Run the simulator."""
    sim_dt = sim.get_physics_dt()
    count = 0

    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or count < args_cli.max_steps):
        if count % 500 == 0:
            root_pose = scene["robot"].data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = scene["robot"].data.default_root_vel.torch.clone()
            scene["robot"].write_root_velocity_to_sim_index(root_velocity=root_vel)
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.torch.clone(),
                scene["robot"].data.default_joint_vel.torch.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
            scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")
        targets = scene["robot"].data.default_joint_pos.torch
        scene["robot"].set_joint_position_target_index(target=targets)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        if count % args_cli.log_interval == 0:
            feet = scene["specific_transforms"].data.target_pos_source.torch[0]
            cube = scene["cube_transform"].data.target_pos_source.torch[0, 0]
            print(f"[INFO] step={count} feet_pos_b={feet.tolist()} cube_pos_b={cube.tolist()}")


def main() -> None:
    """Run the frame-transformer example."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
        scene_cfg = FrameTransformerSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        scene_class = cast(type["InteractiveScene"], scene_cfg.class_type)
        scene = scene_class(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
