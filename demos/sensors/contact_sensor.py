# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect contact forces measured on a robot and a falling cube."""

import argparse
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Example on using the contact sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--log_interval", type=int, default=100, help="Steps between compact sensor summaries.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--physics",
    default="newton_mjwarp",
    choices=["isaacsim_physx", "newton_mjwarp"],
    help="Physics backend.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.log_interval < 1:
    parser.error("--log_interval must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
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
            size=(0.5, 0.5, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=100.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=sim_utils.UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.05)),
    )

    contact_forces_LF = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        track_friction_forces=args_cli.physics == "newton_mjwarp",
    )

    contact_forces_RF = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        track_friction_forces=args_cli.physics == "newton_mjwarp",
    )

    contact_forces_H = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*H_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_friction_forces=args_cli.physics == "newton_mjwarp",
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
            summaries = []
            for name in ("contact_forces_LF", "contact_forces_RF", "contact_forces_H"):
                force = scene[name].data.net_normal_forces_w
                if force is None:
                    raise RuntimeError(f"Contact sensor {name!r} did not produce normal forces.")
                summaries.append(f"{name}={force.torch.norm(dim=-1).mean().item():.3f} N")
            print(f"[INFO] step={count} mean contact force: " + ", ".join(summaries))


def main() -> None:
    """Run the contact-sensor demo."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(3.5, 3.5, 3.5), target=(0.0, 0.0, 0.0))
        scene_cfg = ContactSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        scene_class = cast(type["InteractiveScene"], scene_cfg.class_type)
        scene = scene_class(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
