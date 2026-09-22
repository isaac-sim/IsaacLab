# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Showcase a small robot zoo in one scene.

.. code-block:: bash

    uvx isaaclab demo zoo
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Showcase several robot families in one scene.")
parser.add_argument(
    "--physics", default="newton_mjwarp", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort: skip
from isaaclab_assets.robots.anymal import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, ANYMAL_D_CFG  # isort: skip
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG  # isort: skip
from isaaclab_assets.robots.shadow_hand import (  # isort: skip
    JOINT_NAMES as SHADOW_HAND_JOINT_NAMES,
    SHADOW_HAND_NEWTON_CFG,
    SHADOW_HAND_PHYSX_CFG,
)
from isaaclab_assets.robots.unitree import G1_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots import UR10e_CFG  # isort: skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.scene import InteractiveScene


_RIGID_PROPS = {
    "rigid_props": sim_utils.UsdPhysicsRigidBodyCfg(),
    "mass_props": sim_utils.MassCfg(mass=0.5),
    "collision_props": sim_utils.UsdPhysicsCollisionCfg(),
}
_HAND_CFG = SHADOW_HAND_NEWTON_CFG if args_cli.physics == "newton_mjwarp" else SHADOW_HAND_PHYSX_CFG


def _prop_cfg(spawn: sim_utils.RigidObjectSpawnerCfg, position: tuple[float, float, float]) -> RigidObjectCfg:
    """Create a dynamic prop configuration."""
    return RigidObjectCfg(
        prim_path="",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


@configclass
class ZooSceneCfg(InteractiveSceneCfg):
    """Configuration for the robot zoo."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )

    arm: ArticulationCfg = UR10e_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Arm",
        init_state=UR10e_CFG.init_state.replace(pos=(-2.2, 1.4, 0.0)),
    )
    biped: ArticulationCfg = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Biped",
        init_state=G1_CFG.init_state.replace(pos=(0.0, 1.5, 0.74)),
    )
    quadruped: ArticulationCfg = ANYMAL_D_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Quadruped",
        init_state=ANYMAL_D_CFG.init_state.replace(pos=(2.2, 1.4, 0.6)),
        actuators={"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG},
    )
    hand: ArticulationCfg = _HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Hand",
        init_state=_HAND_CFG.init_state.replace(pos=(-1.4, -1.3, 0.5)),
    )
    drone: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Drone",
        init_state=CRAZYFLIE_CFG.init_state.replace(pos=(1.5, -1.4, 1.3)),
    )

    props: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cube": _prop_cfg(
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.95)),
                    **_RIGID_PROPS,
                ),
                (0.0, -0.6, 2.0),
            ).replace(prim_path="{ENV_REGEX_NS}/Props/Cube"),
            "sphere": _prop_cfg(
                sim_utils.SphereCfg(
                    radius=0.18,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15)),
                    **_RIGID_PROPS,
                ),
                (0.4, -0.6, 2.5),
            ).replace(prim_path="{ENV_REGEX_NS}/Props/Sphere"),
            "cylinder": _prop_cfg(
                sim_utils.CylinderCfg(
                    radius=0.16,
                    height=0.4,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.8, 0.25)),
                    **_RIGID_PROPS,
                ),
                (-0.4, -0.6, 3.0),
            ).replace(prim_path="{ENV_REGEX_NS}/Props/Cylinder"),
        }
    )


def _reset_scene(scene: InteractiveScene) -> None:
    """Restore every dynamic asset to its configured state."""
    for robot in scene.articulations.values():
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())

    props: RigidObjectCollection = scene["props"]
    body_pose = props.data.default_body_pose.torch.clone()
    body_pose[..., :3] += scene.env_origins.unsqueeze(1)
    props.write_body_pose_to_sim_index(body_poses=body_pose)
    props.write_body_com_velocity_to_sim_index(body_velocities=props.data.default_body_vel.torch.clone())
    scene.reset()


def _set_joint_targets(
    robot: Articulation,
    joint_ids: list[int],
    phase: torch.Tensor,
    time: float,
    amplitude: float,
    frequency: float,
) -> None:
    """Apply a smooth deterministic joint trajectory."""
    target = robot.data.default_joint_pos.torch.clone()
    target[:, joint_ids] += amplitude * math.sin(frequency * time) * torch.cos(phase)
    limits = robot.data.soft_joint_pos_limits.torch
    robot.actuators.target_command.set_position_index(value=target.clamp(limits[..., 0], limits[..., 1]))


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Animate the zoo until the viewer closes or the step limit is reached."""
    arm: Articulation = scene["arm"]
    biped: Articulation = scene["biped"]
    quadruped: Articulation = scene["quadruped"]
    hand: Articulation = scene["hand"]
    drone: Articulation = scene["drone"]
    prop_body_ids = drone.find_bodies("m.*_prop")[0]
    drone_mass = drone.data.body_mass.torch[0].sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    forces = torch.zeros(drone.num_instances, len(prop_body_ids), 3, device=sim.device)
    torques = torch.zeros_like(forces)
    motions = [
        (arm, list(range(arm.num_joints)), 0.16, 0.8),
        (biped, biped.find_joints(".*_(shoulder|elbow)_.*")[0], 0.18, 1.0),
        (quadruped, list(range(quadruped.num_joints)), 0.08, 1.5),
        (hand, hand.find_joints(SHADOW_HAND_JOINT_NAMES, preserve_order=True)[0], 0.3, 1.2),
    ]
    phases = [
        torch.linspace(0.0, 2.0 * math.pi, len(joint_ids) + 1, device=robot.device)[:-1]
        for robot, joint_ids, _, _ in motions
    ]

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
        if step_count % 800 == 0:
            _reset_scene(scene)
            sim_time = 0.0

        for (robot, joint_ids, amplitude, frequency), phase in zip(motions, phases):
            _set_joint_targets(robot, joint_ids, phase, sim_time, amplitude, frequency)

        forces[..., 2] = drone_mass * gravity * (1.0 + 0.03 * math.sin(1.5 * sim_time)) / len(prop_body_ids)
        drone.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=prop_body_ids,
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt
        step_count += 1


def main() -> None:
    """Launch the robot zoo showcase."""
    torch.manual_seed(42)
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.solver_cfg.njmax = 300
            physics_cfg.solver_cfg.nconmax = 200
            physics_cfg.solver_cfg.ls_iterations = 40
            physics_cfg.solver_cfg.cone = "elliptic"
            physics_cfg.solver_cfg.impratio = 10.0
            physics_cfg.solver_cfg.ls_parallel = False
            physics_cfg.solver_cfg.update_data_interval = 2
            physics_cfg.solver_cfg.ccd_iterations = 50
            physics_cfg.num_substeps = 2
            physics_cfg.debug_mode = False

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=(6.0, -7.5, 4.5), target=(0.0, 0.0, 0.7))
        scene_cfg = ZooSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True)
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        print("[INFO]: Robot zoo ready.")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
