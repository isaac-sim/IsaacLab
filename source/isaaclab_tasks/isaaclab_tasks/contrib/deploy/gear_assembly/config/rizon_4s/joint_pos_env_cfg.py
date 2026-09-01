# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math

import torch
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.mdp as mdp
import isaaclab_tasks.contrib.deploy.mdp.events as gear_assembly_events
from isaaclab_tasks.contrib.deploy.gear_assembly.gear_assembly_env_cfg import (
    _NEWTON_GEAR_NUM_ENVS,
    _NEWTON_GEAR_OFFSETS,
    _PHYSX_GEAR_OFFSETS,
    GearAssemblyEnvCfg,
)
from isaaclab_tasks.contrib.deploy.mdp.noise_models import (
    ResetSampledConstantNoiseModelCfg,
    ResetSampledQuaternionNoiseModelCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

##
# Pre-defined configs
##
from . import events as rizon_events

from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG  # isort: skip


##
# Gripper-specific helper functions
##


_GRAV_GRIPPER_MIMIC_GEARING = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": -1.0,
    "right_outer_finger_joint": -1.0,
}


def set_finger_joint_pos_grav(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
    joint_name_to_idx: dict[str, int] | None = None,
):
    """Set Grav gripper joint positions by joint name.

    Args:
        joint_pos: Joint positions [rad].
        reset_ind_joint_pos: Row indices into ``joint_pos``.
        finger_joints: Gripper joint indices retained for setter compatibility.
        finger_joint_position: Main finger-joint target [rad].
        joint_name_to_idx: Mapping from joint names to simulation indices.
    """
    if joint_name_to_idx is None:
        raise ValueError("set_finger_joint_pos_grav requires 'joint_name_to_idx'")

    missing = [name for name in _GRAV_GRIPPER_MIMIC_GEARING if name not in joint_name_to_idx]
    if missing:
        raise ValueError(f"Grav gripper joints not found on robot: {missing}")

    for idx in reset_ind_joint_pos:
        for joint_name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items():
            joint_pos[idx, joint_name_to_idx[joint_name]] = gearing * finger_joint_position


_NEWTON_PIN_UNSELECTED_GEARS_EVENT = EventTerm(
    func=rizon_events.pin_unselected_gears_to_shafts,
    mode="interval",
    interval_range_s=(0.0, 0.0),
    params={"gear_offsets": _NEWTON_GEAR_OFFSETS, "seated_gear_z_offset": 0.0075},
)


def _gear_friction_range(*, newton_default: bool = False) -> PresetCfg:
    return preset(
        default=(3.0, 3.0) if newton_default else (0.75, 0.75),
        physx=(0.75, 0.75),
        physx_sdf=(0.75, 0.75),
        newton_mjwarp=(3.0, 3.0),
        newton_sdf=(3.0, 3.0),
        newton_hydroelastic=(3.0, 3.0),
    )


def _backend_preset(physx_value: object, newton_value: object, *, newton_default: bool = False) -> PresetCfg:
    """Create a backend preset with one value for all Newton variants."""
    return preset(
        default=newton_value if newton_default else physx_value,
        physx=physx_value,
        physx_sdf=physx_value,
        newton_mjwarp=newton_value,
        newton_sdf=newton_value,
        newton_hydroelastic=newton_value,
    )


def _gear_asset_frame_preset(
    legacy_value: object, centered_value: object, *, newton_default: bool = False
) -> PresetCfg:
    """Create a preset that selects centered gear frames across physics backends."""
    return preset(
        default=centered_value if newton_default else legacy_value,
        physx=legacy_value,
        physx_sdf=centered_value,
        newton_mjwarp=centered_value,
        newton_sdf=centered_value,
        newton_hydroelastic=centered_value,
    )


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    small_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
            "static_friction_range": _gear_friction_range(),
            "dynamic_friction_range": _gear_friction_range(),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    medium_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_medium", body_names=".*"),
            "static_friction_range": _gear_friction_range(),
            "dynamic_friction_range": _gear_friction_range(),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    large_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_large", body_names=".*"),
            "static_friction_range": _gear_friction_range(),
            "dynamic_friction_range": _gear_friction_range(),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    gear_base_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_base", body_names=".*"),
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_gear_type = EventTerm(
        func=gear_assembly_events.randomize_gear_type,
        mode="reset",
        params={"gear_types": ["gear_small", "gear_medium", "gear_large"]},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_gears_and_base_pose = EventTerm(
        func=gear_assembly_events.randomize_gears_and_base_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],
                "y": [-0.25, 0.25],
                "z": [-0.1, 0.1],
                "roll": [-math.pi / 90, math.pi / 90],  # 2 degree
                "pitch": [-math.pi / 90, math.pi / 90],  # 2 degree
                "yaw": [-math.pi / 6, math.pi / 6],  # 30 degree
            },
            "gear_pos_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0575, 0.0775],
            },
            "velocity_range": {},
            "gear_offsets": preset(
                default=None,
                physx=None,
                physx_sdf=_NEWTON_GEAR_OFFSETS,
                newton_mjwarp=_NEWTON_GEAR_OFFSETS,
                newton_sdf=_NEWTON_GEAR_OFFSETS,
                newton_hydroelastic=_NEWTON_GEAR_OFFSETS,
            ),
            "seated_gear_z_offset": preset(
                default=0.0,
                physx=0.0,
                physx_sdf=0.0075,
                newton_mjwarp=0.0075,
                newton_sdf=0.0075,
                newton_hydroelastic=0.0075,
            ),
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=gear_assembly_events.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
        },
    )

    pin_unselected_gears_to_shafts = preset(
        default=None,
        physx=None,
        physx_sdf=None,
        newton_mjwarp=_NEWTON_PIN_UNSELECTED_GEARS_EVENT,
        newton_sdf=_NEWTON_PIN_UNSELECTED_GEARS_EVENT,
        newton_hydroelastic=_NEWTON_PIN_UNSELECTED_GEARS_EVENT,
    )


@configclass
class Rizon4sGearAssemblyEnvCfg(GearAssemblyEnvCfg):
    """Configure Flexiv Rizon 4s with the Grav gripper for gear assembly."""

    _newton_default = False

    ee_grasp_weight_ramp_steps: int = 512_000

    def _select_backend(self, physx_value: object, newton_value: object) -> PresetCfg:
        """Create a backend preset using this task variant's fallback."""
        return _backend_preset(physx_value, newton_value, newton_default=self._newton_default)

    def _select_asset_frame(self, legacy_value: object, centered_value: object) -> PresetCfg:
        """Create an asset-frame preset using this task variant's fallback."""
        return _gear_asset_frame_preset(legacy_value, centered_value, newton_default=self._newton_default)

    def __post_init__(self):
        super().__post_init__()
        self.gear_offsets = self._select_asset_frame(_PHYSX_GEAR_OFFSETS, _NEWTON_GEAR_OFFSETS)

        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.end_effector_body_name = "link7"
        self.num_arm_joints = len(arm_joint_names)
        self.grasp_rot_offset = [-0.707, 0.707, 0.0, 0.0]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav

        self.observations.policy.gear_shaft_pos.noise = ResetSampledConstantNoiseModelCfg(
            noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
        )
        self.observations.policy.gear_shaft_quat.noise = ResetSampledQuaternionNoiseModelCfg(
            roll_range=(-0.03491, 0.03491),
            pitch_range=(-0.03491, 0.03491),
            yaw_range=(-0.03491, 0.03491),
        )
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = arm_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = arm_joint_names

        self.events = EventCfg()
        for material_term_name in (
            "small_gear_physics_material",
            "medium_gear_physics_material",
            "large_gear_physics_material",
        ):
            material_params = getattr(self.events, material_term_name).params
            material_params["static_friction_range"] = _gear_friction_range(newton_default=self._newton_default)
            material_params["dynamic_friction_range"] = _gear_friction_range(newton_default=self._newton_default)
        reset_pose_params = self.events.randomize_gears_and_base_pose.params
        reset_pose_params["gear_offsets"] = self._select_backend(None, _NEWTON_GEAR_OFFSETS)
        reset_pose_params["seated_gear_z_offset"] = self._select_backend(0.0, 0.0075)
        self.events.pin_unselected_gears_to_shafts = self._select_backend(None, _NEWTON_PIN_UNSELECTED_GEARS_EVENT)
        self.terminations.gear_orientation_exceeded.params["roll_threshold_deg"] = 15.0
        self.terminations.gear_orientation_exceeded.params["pitch_threshold_deg"] = 15.0
        self.terminations.gear_orientation_exceeded.params["yaw_threshold_deg"] = 180.0

        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joint_names,
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )

        physx_robot_init = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": -0.698,
                "joint3": 0.0,
                "joint4": 1.571,
                "joint5": 0.0,
                "joint6": 0.698,
                "joint7": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        )
        newton_robot_init = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.050265,
                "joint2": -0.372105,
                "joint3": 0.111177,
                "joint4": 2.276781,
                "joint5": -0.083078,
                "joint6": 1.074427,
                "joint7": 0.230907,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.scene.robot = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
                # Newton currently cannot exclude only the robot from gravity. Actuator gravity
                # compensation below holds the arm while preserving gravity for the gears.
                joint_drive_props=self._select_backend(
                    None, sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)
                ),
                rigid_props=PhysxRigidBodyPropertiesCfg(
                    disable_gravity=self._select_backend(True, False),
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=self._select_backend(physx_robot_init, newton_robot_init),
        )

        # Use backend-specific arm actuator gains.
        for actuator_name, physx_stiffness, physx_damping, newton_stiffness, newton_damping in (
            ("shoulder", 1320.0, 72.0, 6000.0, 108.5),
            ("elbow", 600.0, 35.0, 4200.0, 90.7),
            ("wrist", 216.0, 29.0, 1500.0, 54.2),
        ):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.stiffness = self._select_backend(physx_stiffness, newton_stiffness)
            actuator.damping = self._select_backend(physx_damping, newton_damping)

        physx_gripper_drive = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e1,
            friction=0.0,
            armature=0.0,
        )
        newton_gripper_drive = physx_gripper_drive.replace(
            effort_limit_sim=200.0,
            velocity_limit_sim=2.0,
            armature=0.1,
        )
        self.scene.robot.actuators["gripper_drive"] = self._select_backend(physx_gripper_drive, newton_gripper_drive)

        physx_gripper_passive = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        )
        newton_gripper_passive = physx_gripper_passive.replace(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=20.0,
            stiffness=2e3,
            damping=10.0,
            armature=0.05,
        )
        self.scene.robot.actuators["gripper_passive"] = self._select_backend(
            physx_gripper_passive, newton_gripper_passive
        )

        base_rot = (0.0, 0.0, 0.70711, -0.70711)
        physx_asset_state = RigidObjectCfg.InitialStateCfg(pos=(0.481, -0.073, 0.071), rot=base_rot)
        newton_base_pos = (0.481, -0.073, -0.005)
        self.scene.factory_gear_base.init_state = self._select_asset_frame(
            physx_asset_state,
            RigidObjectCfg.InitialStateCfg(pos=newton_base_pos, rot=base_rot),
        )
        for gear_name, asset in (
            ("gear_small", self.scene.factory_gear_small),
            ("gear_medium", self.scene.factory_gear_medium),
            ("gear_large", self.scene.factory_gear_large),
        ):
            newton_gear_pos = (
                newton_base_pos[0] - _NEWTON_GEAR_OFFSETS[gear_name][0],
                newton_base_pos[1],
                newton_base_pos[2],
            )
            asset.init_state = self._select_asset_frame(
                physx_asset_state,
                RigidObjectCfg.InitialStateCfg(pos=newton_gear_pos, rot=base_rot),
            )

        physx_grasp_offsets = {name: [0.0, -offset[0], -0.35] for name, offset in _PHYSX_GEAR_OFFSETS.items()}
        newton_grasp_offsets = {
            "gear_small": [0.0, 0.0, -0.026],
            "gear_medium": [0.0, 0.0, -0.026],
            "gear_large": [0.0, 0.0, -0.026],
        }
        self.gear_offsets_grasp = self._select_asset_frame(physx_grasp_offsets, newton_grasp_offsets)
        self.grasp_center_body_names = self._select_asset_frame(None, ("left_finger_tip", "right_finger_tip"))
        self.hand_grasp_width = self._select_asset_frame(
            {"gear_small": 0.05, "gear_medium": 0.2, "gear_large": 0.28},
            {"gear_small": 0.01, "gear_medium": 0.2, "gear_large": 0.28},
        )
        self.hand_close_width = self._select_asset_frame(
            {"gear_small": 0.0, "gear_medium": 0.139626, "gear_large": 0.139626},
            {"gear_small": 0.01, "gear_medium": 0.139626, "gear_large": 0.139626},
        )
        self.ee_grasp_weight_ramp_start = self._select_backend(0.0, 0.2)
        self.ee_grasp_weight_ramp_steps = self._select_backend(512_000, 250_000)

        grasp_event_params = self.events.set_robot_to_grasp_pose.params
        grasp_event_params["gear_offsets_grasp"] = self.gear_offsets_grasp
        grasp_event_params["end_effector_body_name"] = self.end_effector_body_name
        grasp_event_params["num_arm_joints"] = self.num_arm_joints
        grasp_event_params["grasp_rot_offset"] = self.grasp_rot_offset
        grasp_event_params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        grasp_event_params["grasp_center_body_names"] = self.grasp_center_body_names

        grasp_reward_params = {
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "keypoint_scale": 0.15,
            "ee_grasp_threshold": 0.0,
            "weight_ramp_start": self.ee_grasp_weight_ramp_start,
            "weight_ramp_steps": self.ee_grasp_weight_ramp_steps,
            "end_effector_body_name": self.end_effector_body_name,
            "grasp_rot_offset": self.grasp_rot_offset,
            "gear_offsets_grasp": self.gear_offsets_grasp,
            "grasp_center_body_names": self.grasp_center_body_names,
        }
        self.rewards.end_effector_grasp_keypoint_tracking = RewTerm(
            func=mdp.keypoint_ee_grasp_error,
            weight=-0.5,
            params=grasp_reward_params,
        )
        self.rewards.end_effector_grasp_keypoint_tracking_exp = RewTerm(
            func=mdp.keypoint_ee_grasp_error_exp,
            weight=0.5,
            params={
                **grasp_reward_params,
                "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
                "kp_use_sum_of_exps": False,
            },
        )

        drop_params = self.terminations.gear_dropped.params
        drop_params["gear_offsets_grasp"] = self.gear_offsets_grasp
        drop_params["grasp_center_body_names"] = self.grasp_center_body_names
        drop_params["end_effector_body_name"] = self.end_effector_body_name
        drop_params["grasp_rot_offset"] = self.grasp_rot_offset
        self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset

        newton_scene = copy.deepcopy(self.scene)
        newton_scene.num_envs = _NEWTON_GEAR_NUM_ENVS
        self.scene = preset(
            default=newton_scene if self._newton_default else self.scene,
            physx=self.scene,
            physx_sdf=self.scene,
            newton_mjwarp=newton_scene,
            newton_sdf=newton_scene,
            newton_hydroelastic=newton_scene,
        )
