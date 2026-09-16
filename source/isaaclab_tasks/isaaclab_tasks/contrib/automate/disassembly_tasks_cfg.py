# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import config_field
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/AutoMate"

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@dataclass
class FixedAssetCfg:
    usd_path: str = config_field("")
    diameter: float = config_field(0.0)
    height: float = config_field(0.0)
    base_height: float = config_field(0.0)  # Used to compute held asset CoM.
    friction: float = config_field(0.75)
    mass: float = config_field(0.05)


@dataclass
class HeldAssetCfg:
    usd_path: str = config_field("")
    diameter: float = config_field(0.0)  # Used for gripper width.
    height: float = config_field(0.0)
    friction: float = config_field(0.75)
    mass: float = config_field(0.05)


@dataclass
class RobotCfg:
    robot_usd: str = config_field("")
    franka_fingerpad_length: float = config_field(0.017608)
    friction: float = config_field(0.75)


@dataclass
class DisassemblyTask:
    robot_cfg: RobotCfg = config_field(RobotCfg())
    name: str = config_field("")
    duration_s: Any = config_field(5.0)

    fixed_asset_cfg: FixedAssetCfg = config_field(FixedAssetCfg())
    held_asset_cfg: HeldAssetCfg = config_field(HeldAssetCfg())
    asset_size: float = config_field(0.0)

    # palm_to_finger_dist: float = 0.1034
    palm_to_finger_dist: float = config_field(0.1134)

    # Robot
    hand_init_pos: list = config_field([0.0, 0.0, 0.015])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = config_field([0.02, 0.02, 0.01])
    hand_init_orn: list = config_field([3.1416, 0, 2.356])
    hand_init_orn_noise: list = config_field([0.0, 0.0, 1.57])

    # Action
    unidirectional_rot: bool = config_field(False)

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = config_field([0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = config_field(0.0)
    fixed_asset_init_orn_range_deg: float = config_field(10.0)

    num_point_robot_traj: int = config_field(10)  # number of waypoints included in the end-effector trajectory


@dataclass
class Peg8mm(HeldAssetCfg):
    usd_path: Any = config_field("plug.usd")
    obj_path: Any = config_field("plug.obj")
    diameter: Any = config_field(0.007986)
    height: Any = config_field(0.050)
    mass: Any = config_field(0.019)


@dataclass
class Hole8mm(FixedAssetCfg):
    usd_path: Any = config_field("socket.usd")
    obj_path: Any = config_field("socket.obj")
    diameter: Any = config_field(0.0081)
    height: Any = config_field(0.050896)
    base_height: Any = config_field(0.0)


@dataclass
class Extraction(DisassemblyTask):
    name: Any = config_field("extraction")

    assembly_id: Any = config_field("00015")
    assembly_dir: Any = config_field(f"{ASSET_DIR}/{assembly_id}/")
    disassembly_dir: Any = config_field("disassembly_dir")
    num_log_traj: Any = config_field(100)

    fixed_asset_cfg: Any = config_field(Hole8mm())
    held_asset_cfg: Any = config_field(Peg8mm())
    asset_size: Any = config_field(8.0)
    duration_s: Any = config_field(10.0)

    plug_grasp_json: Any = config_field(f"{ASSET_DIR}/plug_grasps.json")
    disassembly_dist_json: Any = config_field(f"{ASSET_DIR}/disassembly_dist.json")

    move_gripper_sim_steps: Any = config_field(64)
    disassemble_sim_steps: Any = config_field(64)

    # Robot
    hand_init_pos: list = config_field([0.0, 0.0, 0.047])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = config_field([0.02, 0.02, 0.01])
    hand_init_orn: list = config_field([3.1416, 0.0, 0.0])
    hand_init_orn_noise: list = config_field([0.0, 0.0, 0.785])
    hand_width_max: float = config_field(0.080)  # maximum opening width of gripper

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = config_field([0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = config_field(0.0)
    fixed_asset_init_orn_range_deg: float = config_field(10.0)
    fixed_asset_z_offset: float = config_field(0.1435)

    fingertip_centered_pos_initial: list = config_field(
        [
            0.0,
            0.0,
            0.2,
        ]
    )  # initial position of midpoint between fingertips above table
    fingertip_centered_rot_initial: list = config_field([3.141593, 0.0, 0.0])  # initial rotation of fingertips (Euler)
    gripper_rand_pos_noise: list = config_field([0.05, 0.05, 0.05])
    gripper_rand_rot_noise: list = config_field([0.174533, 0.174533, 0.174533])  # +-10 deg for roll/pitch/yaw
    gripper_rand_z_offset: float = config_field(0.05)

    fixed_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            # fixed_asset: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{assembly_dir}{fixed_asset_cfg.usd_path}",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                # init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )
    )
    # held_asset: ArticulationCfg = ArticulationCfg(
    held_asset: RigidObjectCfg = config_field(
        RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{assembly_dir}{held_asset_cfg.usd_path}",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            # init_state=ArticulationCfg.InitialStateCfg(
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1),
                rot=(0.0, 0.0, 0.0, 1.0),
                # joint_pos={},
                # joint_vel={}
            ),
            # actuators={}
        )
    )
