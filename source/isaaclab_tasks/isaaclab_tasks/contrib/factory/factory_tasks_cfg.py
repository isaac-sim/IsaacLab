# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import config_field
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


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
class FactoryTask:
    robot_cfg: RobotCfg = config_field(RobotCfg())
    name: str = config_field("")
    duration_s: Any = config_field(5.0)

    fixed_asset_cfg: FixedAssetCfg = config_field(FixedAssetCfg())
    held_asset_cfg: HeldAssetCfg = config_field(HeldAssetCfg())
    asset_size: float = config_field(0.0)

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
    fixed_asset_init_orn_range_deg: float = config_field(360.0)

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = config_field([0.0, 0.006, 0.003])  # noise level of the held asset in gripper
    held_asset_rot_init: float = config_field(-90.0)

    # Reward
    ee_success_yaw: float = config_field(0.0)  # nut_thread task only.
    action_penalty_ee_scale: float = config_field(0.0)
    action_grad_penalty_scale: float = config_field(0.0)
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = config_field(4)
    keypoint_scale: float = config_field(0.15)
    keypoint_coef_baseline: list = config_field([5, 4])  # General movement towards fixed object.
    keypoint_coef_coarse: list = config_field([50, 2])  # Movement to align the assets.
    keypoint_coef_fine: list = config_field([100, 0])  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = config_field(0.04)
    engage_threshold: float = config_field(0.9)


@dataclass
class Peg8mm(HeldAssetCfg):
    usd_path: Any = config_field(f"{ASSET_DIR}/factory_peg_8mm.usd")
    diameter: Any = config_field(0.007986)
    height: Any = config_field(0.050)
    mass: Any = config_field(0.019)


@dataclass
class Hole8mm(FixedAssetCfg):
    usd_path: Any = config_field(f"{ASSET_DIR}/factory_hole_8mm.usd")
    diameter: Any = config_field(0.0081)
    height: Any = config_field(0.025)
    base_height: Any = config_field(0.0)


@dataclass
class PegInsert(FactoryTask):
    name: Any = config_field("peg_insert")
    fixed_asset_cfg: Any = config_field(Hole8mm())
    held_asset_cfg: Any = config_field(Peg8mm())
    asset_size: Any = config_field(8.0)
    duration_s: Any = config_field(10.0)

    # Robot
    hand_init_pos: list = config_field([0.0, 0.0, 0.047])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = config_field([0.02, 0.02, 0.01])
    hand_init_orn: list = config_field([3.1416, 0.0, 0.0])
    hand_init_orn_noise: list = config_field([0.0, 0.0, 0.785])

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = config_field([0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = config_field(0.0)
    fixed_asset_init_orn_range_deg: float = config_field(360.0)

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = config_field([0.003, 0.0, 0.003])  # noise level of the held asset in gripper
    held_asset_rot_init: float = config_field(0.0)

    # Rewards
    keypoint_coef_baseline: list = config_field([5, 4])
    keypoint_coef_coarse: list = config_field([50, 2])
    keypoint_coef_fine: list = config_field([100, 0])
    # Fraction of socket height.
    success_threshold: float = config_field(0.04)
    engage_threshold: float = config_field(0.9)

    fixed_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=fixed_asset_cfg.usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
    held_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=held_asset_cfg.usd_path,
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
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )


@dataclass
class GearBase(FixedAssetCfg):
    usd_path: Any = config_field(f"{ASSET_DIR}/factory_gear_base.usd")
    height: Any = config_field(0.02)
    base_height: Any = config_field(0.005)
    small_gear_base_offset: Any = config_field([5.075e-2, 0.0, 0.0])
    medium_gear_base_offset: Any = config_field([2.025e-2, 0.0, 0.0])
    large_gear_base_offset: Any = config_field([-3.025e-2, 0.0, 0.0])


@dataclass
class MediumGear(HeldAssetCfg):
    usd_path: Any = config_field(f"{ASSET_DIR}/factory_gear_medium.usd")
    diameter: Any = config_field(0.03)  # Used for gripper width.
    height: float = config_field(0.03)
    mass: Any = config_field(0.012)


@dataclass
class GearMesh(FactoryTask):
    name: Any = config_field("gear_mesh")
    fixed_asset_cfg: Any = config_field(GearBase())
    held_asset_cfg: Any = config_field(MediumGear())
    duration_s: Any = config_field(20.0)

    small_gear_usd: Any = config_field(f"{ASSET_DIR}/factory_gear_small.usd")
    large_gear_usd: Any = config_field(f"{ASSET_DIR}/factory_gear_large.usd")

    small_gear_cfg: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/SmallGearAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=small_gear_usd,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )

    large_gear_cfg: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/LargeGearAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=large_gear_usd,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )

    # Gears Asset
    add_flanking_gears: Any = config_field(True)
    add_flanking_gears_prob: Any = config_field(1.0)

    # Robot
    hand_init_pos: list = config_field([0.0, 0.0, 0.035])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = config_field([0.02, 0.02, 0.01])
    hand_init_orn: list = config_field([3.1416, 0, 0.0])
    hand_init_orn_noise: list = config_field([0.0, 0.0, 0.785])

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = config_field([0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = config_field(0.0)
    fixed_asset_init_orn_range_deg: float = config_field(15.0)

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = config_field([0.003, 0.0, 0.003])  # noise level of the held asset in gripper
    held_asset_rot_init: float = config_field(-90.0)

    keypoint_coef_baseline: list = config_field([5, 4])
    keypoint_coef_coarse: list = config_field([50, 2])
    keypoint_coef_fine: list = config_field([100, 0])
    # Fraction of gear peg height.
    success_threshold: float = config_field(0.05)
    engage_threshold: float = config_field(0.9)

    fixed_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=fixed_asset_cfg.usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
    held_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=held_asset_cfg.usd_path,
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
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )


@dataclass
class NutM16(HeldAssetCfg):
    usd_path: Any = config_field(f"{ASSET_DIR}/factory_nut_m16.usd")
    diameter: Any = config_field(0.024)
    height: Any = config_field(0.01)
    mass: Any = config_field(0.03)
    friction: Any = config_field(0.01)  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25


@dataclass
class BoltM16(FixedAssetCfg):
    usd_path: Any = config_field(f"{ASSET_DIR}/factory_bolt_m16.usd")
    diameter: Any = config_field(0.024)
    height: Any = config_field(0.025)
    base_height: Any = config_field(0.01)
    thread_pitch: Any = config_field(0.002)


@dataclass
class NutThread(FactoryTask):
    name: Any = config_field("nut_thread")
    fixed_asset_cfg: Any = config_field(BoltM16())
    held_asset_cfg: Any = config_field(NutM16())
    asset_size: Any = config_field(16.0)
    duration_s: Any = config_field(30.0)

    # Robot
    hand_init_pos: list = config_field([0.0, 0.0, 0.015])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = config_field([0.02, 0.02, 0.01])
    hand_init_orn: list = config_field([3.1416, 0.0, 1.83])
    hand_init_orn_noise: list = config_field([0.0, 0.0, 0.26])

    # Action
    unidirectional_rot: bool = config_field(True)

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = config_field([0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = config_field(120.0)
    fixed_asset_init_orn_range_deg: float = config_field(30.0)

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = config_field([0.0, 0.003, 0.003])  # noise level of the held asset in gripper
    held_asset_rot_init: float = config_field(-90.0)

    # Reward.
    ee_success_yaw: Any = config_field(0.0)
    keypoint_coef_baseline: list = config_field([100, 2])
    keypoint_coef_coarse: list = config_field([500, 2])  # 100, 2
    keypoint_coef_fine: list = config_field([1500, 0])  # 500, 0
    # Fraction of thread-height.
    success_threshold: float = config_field(0.375)
    engage_threshold: float = config_field(0.5)
    keypoint_scale: float = config_field(0.05)

    fixed_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=fixed_asset_cfg.usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
    held_asset: ArticulationCfg = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=held_asset_cfg.usd_path,
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
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
