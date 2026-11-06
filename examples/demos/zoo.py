# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Showcase twenty robot models from the Isaac Lab asset library in one scene.

Humanoids, quadrupeds, manipulators, dexterous hands, and MicroDuck perform gentle
joint motions while a Crazyflie hovers beside the display stands.

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
parser.add_argument("--num_envs", type=int, default=1, help="Number of zoo environments to spawn.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

import torch
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, instantiate, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort: skip
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, ANYMAL_D_CFG  # isort: skip
from isaaclab_assets.robots.flexiv import FLEXIV_RIZON4S_CFG  # isort: skip
from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG, KINOVA_JACO2_N7S300_CFG  # isort: skip
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG  # isort: skip
from isaaclab_assets.robots.microduck import MICRODUCK_CFG  # isort: skip
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG  # isort: skip
from isaaclab_assets.robots.sawyer import SAWYER_CFG  # isort: skip
from isaaclab_assets.robots.shadow_hand import (  # isort: skip
    JOINT_NAMES as SHADOW_HAND_JOINT_NAMES,
    SHADOW_HAND_NEWTON_CFG,
    SHADOW_HAND_PHYSX_CFG,
)
from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip
from isaaclab_assets.robots.unitree import G1_CFG, H1_CFG, UNITREE_A1_CFG, UNITREE_GO2_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots import UR10_CFG, UR10e_CFG  # isort: skip

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.scene import InteractiveScene


_RIGID_PROPS = {
    "rigid_props": sim_utils.UsdPhysicsRigidBodyCfg(),
    "mass_props": sim_utils.MassCfg(mass=0.5),
    "collision_props": sim_utils.UsdPhysicsCollisionCfg(),
}
_HAND_CFG = SHADOW_HAND_NEWTON_CFG if args_cli.physics == "newton_mjwarp" else SHADOW_HAND_PHYSX_CFG

# Each entry declares the asset, root position, animated joints, amplitude [rad], and frequency [rad/s].
# The taller models occupy the back row; small models are raised on foreground stands.
_ROBOTS = {
    "h1": (H1_CFG, (-4.4, 3.0, 1.05), ".*_(shoulder|elbow).*", 0.18, 1.0),
    "g1": (G1_CFG, (-2.2, 3.0, 0.74), ".*_(shoulder|elbow)_.*", 0.18, 1.1),
    "fourier": (GR1T2_CFG, (0.0, 3.0, 0.95), "(head|left_shoulder|right_shoulder|left_elbow|right_elbow).*", 0.15, 0.9),
    "ur10": (UR10_CFG, (2.2, 3.0, 0.45), ".*", 0.14, 0.8),
    "kuka_allegro": (KUKA_ALLEGRO_CFG, (4.4, 3.0, 0.45), "iiwa7_joint_[1-7]", 0.12, 0.8),
    "anymal": (
        replace(ANYMAL_D_CFG, actuators={"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG}),
        (-4.4, 0.9, 0.6),
        ".*HAA",
        0.08,
        1.5,
    ),
    "spot": (SPOT_CFG, (-2.2, 0.9, 0.5), ".*_h[xy]", 0.06, 1.3),
    "go2": (UNITREE_GO2_CFG, (0.0, 0.9, 0.4), ".*_(hip|thigh)_joint", 0.06, 1.4),
    "a1": (UNITREE_A1_CFG, (2.2, 0.9, 0.42), ".*_(hip|thigh)_joint", 0.06, 1.6),
    "franka": (FRANKA_PANDA_CFG, (4.4, 0.9, 0.45), "panda_joint[1-7]", 0.14, 0.9),
    "ur10e": (UR10e_CFG, (-4.4, -1.2, 0.45), ".*", 0.16, 0.8),
    "sawyer": (SAWYER_CFG, (-2.2, -1.2, 0.45), "right_j[0-6]", 0.14, 0.9),
    "kinova": (KINOVA_GEN3_N7_CFG, (0.0, -1.2, 0.45), "joint_[1-7]", 0.14, 1.0),
    "flexiv": (FLEXIV_RIZON4S_CFG, (2.2, -1.2, 0.45), "joint[1-7]", 0.12, 0.8),
    "jaco": (KINOVA_JACO2_N7S300_CFG, (4.4, -1.2, 0.45), "j2n7s300_joint_[1-7]", 0.14, 1.1),
    "shadow_hand": (_HAND_CFG, (-4.4, -3.3, 1.15), SHADOW_HAND_JOINT_NAMES, 0.3, 1.2),
    "allegro_hand": (ALLEGRO_HAND_CFG, (-2.2, -3.3, 1.15), ".*_joint_[1-3]", 0.25, 1.1),
    "microduck": (
        replace(
            MICRODUCK_CFG,
            actuators={
                # A separate PD group replaces the asset's authored servo prims on both backends.
                "display": IdealPDActuatorCfg(
                    joint_names_expr=["^(?!passive_).*"],
                    actuator_effort_limit=0.5,
                    stiffness=2.0,
                    damping=0.1,
                )
            },
        ),
        (0.0, -3.3, 0.925),
        "head_(yaw|roll)",
        0.25,
        1.2,
    ),
    "so101": (SO101_CFG, (2.2, -3.3, 0.8), "(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)", 0.14, 1.0),
    "drone": (CRAZYFLIE_CFG, (4.4, -3.3, 1.3), None, 0.0, 0.0),
}


def _spawn_fixed_spot(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Anchor Spot's physical body beneath its articulation container for display."""
    from isaaclab.sim.schemas.schemas import create_world_fixed_joint

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    for root in sim_utils.find_matching_prims(prim_path, stage=stage):
        create_world_fixed_joint(root.GetChild("body"), stage)
    return prim


@configclass
class ZooSceneCfg(InteractiveSceneCfg):
    """Configuration for the robot zoo."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )

    props: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Props/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.95)),
                    **_RIGID_PROPS,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -4.9, 1.0)),
            ),
            "sphere": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Props/Sphere",
                spawn=sim_utils.SphereCfg(
                    radius=0.18,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15)),
                    **_RIGID_PROPS,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -4.9, 1.4)),
            ),
            "cylinder": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Props/Cylinder",
                spawn=sim_utils.CylinderCfg(
                    radius=0.16,
                    height=0.4,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.8, 0.25)),
                    **_RIGID_PROPS,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, -4.9, 1.8)),
            ),
        }
    )

    def __post_init__(self) -> None:
        """Arrange independent robot configurations and their display stands."""
        super().__post_init__()
        for name, (asset_cfg, position, _, _, _) in _ROBOTS.items():
            robot = replace(asset_cfg, prim_path=f"{{ENV_REGEX_NS}}/{name}")
            robot.init_state.pos = position
            robot.spawn.activate_contact_sensors = False
            if name == "spot":
                robot.spawn.func = _spawn_fixed_spot
            elif name != "drone":
                robot.spawn.fix_root_link = True
            if name != "drone":
                robot.spawn.articulation_props = [
                    PhysxArticulationCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                    ),
                    NewtonArticulationCfg(self_collision_enabled=False),
                ]
            if name == "franka":
                robot.spawn.variants["Physics"] = "mujoco" if args_cli.physics == "newton_mjwarp" else "physx"
            elif name == "so101":
                robot.spawn.variants["Physics"] = "physics" if args_cli.physics == "newton_mjwarp" else "physx"
            elif name == "fourier":
                # Use conservative gains for the fixed-base humanoid display.
                for group in ("head", "trunk", "legs", "left-arm", "right-arm"):
                    robot.actuators[group] = replace(
                        robot.actuators[group],
                        stiffness=150.0,
                        damping=5.0,
                        joint_effort_limit=100.0,
                        armature=0.1,
                    )
                # Keep the coupled distal finger joints passive.
                for hand in ("left-hand", "right-hand"):
                    robot.actuators[hand] = replace(
                        robot.actuators[hand],
                        stiffness={".*proximal.*": 3.0, ".*(intermediate|distal).*": 0.0},
                        damping={".*proximal.*": 0.1, ".*(intermediate|distal).*": 0.0},
                        joint_effort_limit=0.5,
                        armature=2.0e-4,
                    )
            # Account for the authored body and wrist transforms when fixing these display bases.
            if name == "h1":
                robot.init_state.pos = (position[0], position[1], position[2] - 1.1)
            elif name == "shadow_hand":
                robot.init_state.rot = (0.5, -0.5, 0.5, 0.5)
            elif name == "allegro_hand":
                robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
            setattr(self, name, robot)

            pedestal_height = position[2] - asset_cfg.init_state.pos[2]
            # The hand defaults are suspended; their upright wrists sit directly on the stands.
            if name in ("shadow_hand", "allegro_hand"):
                pedestal_height = position[2]
            if name != "drone" and pedestal_height > 0.0:
                setattr(
                    self,
                    f"{name}_stand",
                    AssetBaseCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{name}_stand",
                        spawn=sim_utils.CylinderCfg(
                            radius=0.5,
                            height=pedestal_height,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.08, 0.11, 0.15), metallic=0.3, roughness=0.45
                            ),
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(position[0], position[1], pedestal_height / 2.0)),
                    ),
                )


def _reset_scene(scene: InteractiveScene) -> None:
    """Restore every dynamic asset to its configured state."""
    for robot in scene.articulations.values():
        if not robot.is_fixed_base:
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
    drone: Articulation = scene["drone"]
    prop_body_ids = drone.find_bodies("m.*_prop")[0]
    drone_mass = drone.data.body_mass.torch[0].sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    forces = torch.zeros(drone.num_instances, len(prop_body_ids), 3, device=sim.device)
    torques = torch.zeros_like(forces)
    motions = []
    for name, (_, _, joints, amplitude, frequency) in _ROBOTS.items():
        if joints is not None:
            robot = scene[name]
            joint_ids = robot.find_joints(joints, preserve_order=True)[0]
            motions.append((robot, joint_ids, amplitude, frequency))
    phases = [
        torch.linspace(0.0, 2.0 * math.pi, len(joint_ids) + 1, device=robot.device)[:-1]
        for robot, joint_ids, _, _ in motions
    ]

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    while sim.is_running() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
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
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        visualizer_cfgs=[NewtonGLVisualizerCfg(exposure=0.8)],
        default_visualizer_cfg=VisualizerCfg(focal_length=36.0),
    )
    with launch_simulation(sim_cfg, args_cli) as physics_cfg:
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.solver_cfg.njmax = 2048
            physics_cfg.solver_cfg.nconmax = 512
            physics_cfg.solver_cfg.ls_iterations = 40
            physics_cfg.solver_cfg.cone = "elliptic"
            physics_cfg.solver_cfg.impratio = 10.0
            physics_cfg.solver_cfg.ls_parallel = False
            physics_cfg.solver_cfg.update_data_interval = 2
            physics_cfg.solver_cfg.ccd_iterations = 50
            physics_cfg.num_substeps = 2
            physics_cfg.debug_mode = False

        sim = sim_utils.SimulationContext(sim_cfg)
        camera_scale = math.ceil(math.sqrt(args_cli.num_envs))
        sim.set_camera_view(
            eye=(11.5 * camera_scale, -15.5 * camera_scale, 10.0 * camera_scale), target=(0.0, -0.2, 1.0)
        )
        scene_cfg = ZooSceneCfg(num_envs=args_cli.num_envs, env_spacing=14.0)
        scene = instantiate(scene_cfg)
        sim.reset()
        print(f"[INFO]: Robot zoo ready. Showcasing {len(_ROBOTS)} robot models.")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
