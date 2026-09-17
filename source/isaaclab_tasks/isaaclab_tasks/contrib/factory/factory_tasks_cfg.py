# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@dataclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@dataclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@dataclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@dataclass
class FactoryTask:
    robot_cfg: RobotCfg = field(default_factory=RobotCfg)
    name: str = ""
    duration_s: Any = 5.0

    fixed_asset_cfg: FixedAssetCfg = field(default_factory=FixedAssetCfg)
    held_asset_cfg: HeldAssetCfg = field(default_factory=HeldAssetCfg)
    asset_size: float = 0.0

    # Robot
    hand_init_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.015])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = field(default_factory=lambda: [0.02, 0.02, 0.01])
    hand_init_orn: list = field(default_factory=lambda: [3.1416, 0, 2.356])
    hand_init_orn_noise: list = field(default_factory=lambda: [0.0, 0.0, 1.57])

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = field(default_factory=lambda: [0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = field(
        default_factory=lambda: [0.0, 0.006, 0.003]
    )  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_thread task only.
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = field(default_factory=lambda: [5, 4])  # General movement towards fixed object.
    keypoint_coef_coarse: list = field(default_factory=lambda: [50, 2])  # Movement to align the assets.
    keypoint_coef_fine: list = field(
        default_factory=lambda: [100, 0]
    )  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9


@dataclass
class Peg8mm(HeldAssetCfg):
    usd_path: Any = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter: Any = 0.007986
    height: Any = 0.050
    mass: Any = 0.019


@dataclass
class Hole8mm(FixedAssetCfg):
    usd_path: Any = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter: Any = 0.0081
    height: Any = 0.025
    base_height: Any = 0.0


@dataclass
class PegInsert(FactoryTask):
    name: Any = "peg_insert"
    fixed_asset_cfg: Any = field(default_factory=Hole8mm)
    held_asset_cfg: Any = field(default_factory=Peg8mm)
    asset_size: Any = 8.0
    duration_s: Any = 10.0

    # Robot
    hand_init_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.047])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = field(default_factory=lambda: [0.02, 0.02, 0.01])
    hand_init_orn: list = field(default_factory=lambda: [3.1416, 0.0, 0.0])
    hand_init_orn_noise: list = field(default_factory=lambda: [0.0, 0.0, 0.785])

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = field(default_factory=lambda: [0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = field(
        default_factory=lambda: [0.003, 0.0, 0.003]
    )  # noise level of the held asset in gripper
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = field(default_factory=lambda: [5, 4])
    keypoint_coef_coarse: list = field(default_factory=lambda: [50, 2])
    keypoint_coef_fine: list = field(default_factory=lambda: [100, 0])
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=Hole8mm().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=Hole8mm().mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
    held_asset: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=Peg8mm().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=Peg8mm().mass),
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
    usd_path: Any = f"{ASSET_DIR}/factory_gear_base.usd"
    height: Any = 0.02
    base_height: Any = 0.005
    small_gear_base_offset: Any = field(default_factory=lambda: [5.075e-2, 0.0, 0.0])
    medium_gear_base_offset: Any = field(default_factory=lambda: [2.025e-2, 0.0, 0.0])
    large_gear_base_offset: Any = field(default_factory=lambda: [-3.025e-2, 0.0, 0.0])


@dataclass
class MediumGear(HeldAssetCfg):
    usd_path: Any = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter: Any = 0.03  # Used for gripper width.
    height: float = 0.03
    mass: Any = 0.012


@dataclass
class GearMesh(FactoryTask):
    name: Any = "gear_mesh"
    fixed_asset_cfg: Any = field(default_factory=GearBase)
    held_asset_cfg: Any = field(default_factory=MediumGear)
    duration_s: Any = 20.0

    small_gear_usd: Any = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd: Any = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/SmallGearAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSET_DIR}/factory_gear_small.usd",
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

    large_gear_cfg: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/LargeGearAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSET_DIR}/factory_gear_large.usd",
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
    add_flanking_gears: Any = True
    add_flanking_gears_prob: Any = 1.0

    # Robot
    hand_init_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.035])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = field(default_factory=lambda: [0.02, 0.02, 0.01])
    hand_init_orn: list = field(default_factory=lambda: [3.1416, 0, 0.0])
    hand_init_orn_noise: list = field(default_factory=lambda: [0.0, 0.0, 0.785])

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = field(default_factory=lambda: [0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 15.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = field(
        default_factory=lambda: [0.003, 0.0, 0.003]
    )  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    keypoint_coef_baseline: list = field(default_factory=lambda: [5, 4])
    keypoint_coef_coarse: list = field(default_factory=lambda: [50, 2])
    keypoint_coef_fine: list = field(default_factory=lambda: [100, 0])
    # Fraction of gear peg height.
    success_threshold: float = 0.05
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=GearBase().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=GearBase().mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
    held_asset: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=MediumGear().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=MediumGear().mass),
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
    usd_path: Any = f"{ASSET_DIR}/factory_nut_m16.usd"
    diameter: Any = 0.024
    height: Any = 0.01
    mass: Any = 0.03
    friction: Any = 0.01  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25


@dataclass
class BoltM16(FixedAssetCfg):
    usd_path: Any = f"{ASSET_DIR}/factory_bolt_m16.usd"
    diameter: Any = 0.024
    height: Any = 0.025
    base_height: Any = 0.01
    thread_pitch: Any = 0.002


@dataclass
class NutThread(FactoryTask):
    name: Any = "nut_thread"
    fixed_asset_cfg: Any = field(default_factory=BoltM16)
    held_asset_cfg: Any = field(default_factory=NutM16)
    asset_size: Any = 16.0
    duration_s: Any = 30.0

    # Robot
    hand_init_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.015])  # Relative to fixed asset tip.
    hand_init_pos_noise: list = field(default_factory=lambda: [0.02, 0.02, 0.01])
    hand_init_orn: list = field(default_factory=lambda: [3.1416, 0.0, 1.83])
    hand_init_orn_noise: list = field(default_factory=lambda: [0.0, 0.0, 0.26])

    # Action
    unidirectional_rot: bool = True

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = field(default_factory=lambda: [0.05, 0.05, 0.05])
    fixed_asset_init_orn_deg: float = 120.0
    fixed_asset_init_orn_range_deg: float = 30.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = field(
        default_factory=lambda: [0.0, 0.003, 0.003]
    )  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    # Reward.
    ee_success_yaw: Any = 0.0
    keypoint_coef_baseline: list = field(default_factory=lambda: [100, 2])
    keypoint_coef_coarse: list = field(default_factory=lambda: [500, 2])  # 100, 2
    keypoint_coef_fine: list = field(default_factory=lambda: [1500, 0])  # 500, 0
    # Fraction of thread-height.
    success_threshold: float = 0.375
    engage_threshold: float = 0.5
    keypoint_scale: float = 0.05

    fixed_asset: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=BoltM16().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=BoltM16().mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
    held_asset: ArticulationCfg = field(
        default_factory=lambda: ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=NutM16().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=NutM16().mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
    )
