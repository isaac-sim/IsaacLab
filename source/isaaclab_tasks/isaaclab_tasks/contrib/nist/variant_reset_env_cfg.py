# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset composition for the homogeneous Factory scene."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg

from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from . import mdp
from .assembly_profile_cfg import UniformPoseNoiseCfg
from .factory_presets import EndEffectorBodyCfg, GripperGraspOffsetCfg, GripperJointNamesCfg, IKJointNamesCfg
from .utils import (
    BetaSamplingStrategyCfg,
    SamplerCfg,
)

RESET_ASSETS = ["nistboard", "fixed_asset", "held_asset", "robot"]

START_RANDOM = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_robot_joint": EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "reset_held_asset": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (0.0, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.015, 0.2),
                        "roll": (-1.57, 1.57),
                        "pitch": (-1.57, 1.57),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("held_asset"),
                },
            ),
            "reset_end_effector": EventTerm(
                func=mdp.reset_variant_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": "held_grasp_middle",
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "robot_ik_body_offset": GripperGraspOffsetCfg(),
                    "upright_gripper": True,
                    "ik_iterations": (1, 30),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_variant_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg()),
                },
            ),
        }
    },
)


START_ASSEMBLED = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_robot_joint": EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "reset_held_asset": EventTerm(
                func=mdp.reset_variant_held_asset_on_fixed_asset,
                mode="reset",
                params={
                    # Stop at the entry: past it the part is no longer assembled, and that stretch
                    # is what ``START_NEAR_ASSEMBLED`` below covers.
                    "assembly_fraction_range": (0.0, 1.0),
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                },
            ),
            "reset_end_effector": EventTerm(
                func=mdp.reset_variant_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": "held_grasp_middle",
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (-0.1, 0.1),
                        "pitch": (-1.0, 1.0),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "robot_ik_body_offset": GripperGraspOffsetCfg(),
                    # The part is already seated, so the gripper has to arrive where it was sent or
                    # it closes on nothing. Solve it out, and report the ones that still missed.
                    "ik_iterations": (25, 35),
                    "pose_tolerance": (0.001, 0.05),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_variant_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg()),
                },
            ),
        }
    },
)


# Identical to START_ASSEMBLED but drawn from just past the entry rather than along the seated path,
# with pose noise around it: the held asset's insertion point sits at the mouth of the fixed asset,
# aligned and a hair short of going in.
START_NEAR_ASSEMBLED = START_ASSEMBLED.copy()
START_NEAR_ASSEMBLED.params["terms"]["reset_held_asset"].params.update(
    assembly_fraction_range=(1.0, 1.1),
    pose_noise=UniformPoseNoiseCfg(
        x=(-0.002, 0.002),
        y=(-0.002, 0.002),
        z=(0.0, 0.01),
        roll=(-0.3, 0.3),
        pitch=(-0.3, 0.3),
        yaw=(-0.5, 0.5),
    ),
)
# Clear of the hole, the part has nothing under it, so the gripper has to be the thing holding it.
# The flexible angle would leave it anywhere up to wide open, which drops the part on the first step.
START_NEAR_ASSEMBLED.params["terms"]["grasp_held_asset"].params["flexible_angle"] = False

start_near_grasped = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_robot_joint": EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "reset_end_effector": EventTerm(
                func=mdp.reset_variant_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "fixed_asset_offset": "fixed_tip",
                    "pose_range_b": "grasped",
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "robot_ik_body_offset": GripperGraspOffsetCfg(),
                    "ik_iterations": (10, 20),
                },
            ),
            "reset_held_asset": EventTerm(
                func=mdp.reset_variant_held_asset_in_gripper,
                mode="reset",
                params={
                    "holding_body_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                    "held_asset_inhand_range": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "pitch": (-2.0, 2.0),
                    },
                    "gripper_grasp_offset": GripperGraspOffsetCfg(),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_variant_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg()),
                    "flexible_angle": True,
                },
            ),
        }
    },
)

# Identical to start_near_grasped but with a fixed grasp angle.
ASSET_IN_GRIPPER = start_near_grasped.copy()
ASSET_IN_GRIPPER.params["terms"]["grasp_held_asset"].params["flexible_angle"] = False


# Identical to ASSET_IN_GRIPPER but with the gripper sampled close to the fixed asset.
GRASPED_NEAR_GOAL = ASSET_IN_GRIPPER.copy()
GRASPED_NEAR_GOAL.params["terms"]["reset_end_effector"].params["pose_range_b"] = "grasped_centered"


START_PICK = START_RANDOM.copy()
_drop = START_PICK.params["terms"]["reset_held_asset"]
_drop.params["pose_range"]["z"] = (0.03, 0.05)
_SETTLE_HELD_ASSET = EventTerm(
    func=mdp.sample_settled_asset_pose,
    mode="reset",
    params={
        "held_asset_cfg": SceneEntityCfg("held_asset"),
        "drop_term": _drop,
        "scene_assets": RESET_ASSETS,
        "num_steps": 20,
    },
)
START_PICK.params["terms"]["reset_held_asset"] = _SETTLE_HELD_ASSET
START_PICK.params["terms"]["reset_end_effector"].params.update(
    ik_iterations=(25, 35),
    pose_tolerance=(0.001, 0.05),
)
START_PICK.params["terms"]["reset_end_effector"].params["pose_range_b"]["z"] = (0.00, 0.01)
START_PICK.params["terms"]["reset_end_effector"].params["pose_range_b"]["yaw"] = (-0.3, 0.3)
START_PICK.params["terms"]["grasp_held_asset"].params["flexible_angle"] = False


SCENE_RESET = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "select_assembly": EventTerm(
                func=mdp.select_assembly_variant,
                mode="reset",
                params={},
            ),
            "reset_robot": EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "reset_fixed_asset": EventTerm(
                func=mdp.reset_variant_fixed_asset_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (0.075, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
                },
            ),
            # Board follows the placed fixed asset.
            "reset_board": EventTerm(
                func=mdp.reset_variant_board_under_fixed_asset,
                mode="reset",
                params={},
            ),
            "reset_strategies": EventTerm(
                func=mdp.PreparedTermChoice,
                mode="reset",
                params={
                    "terms": {
                        "start_random": START_RANDOM,
                        "start_assembled": START_ASSEMBLED,
                        "start_near_assembled": START_NEAR_ASSEMBLED,
                        "start_pick": START_PICK,
                        "grasped_near_goal": GRASPED_NEAR_GOAL,
                        "start_grasped": ASSET_IN_GRIPPER,
                        "start_near_grasped": start_near_grasped,
                    },
                },
            ),
        }
    },
)


VARIANT_ACCUMULATOR_RESET = EventTerm(
    func=mdp.variant_reset_accumulator,
    mode="reset",
    params={
        "reset_assets": RESET_ASSETS,
        "acceptance_conditions": {
            "object_collision_free": mdp.CollisionAnalyzerCfg(
                num_points=256,
                max_dist=0.5,
                min_dist=-0.0005,
                asset_cfg=SceneEntityCfg("held_asset"),
                obstacle_cfgs=[
                    SceneEntityCfg("fixed_asset"),
                    SceneEntityCfg("robot"),
                    SceneEntityCfg("nistboard"),
                    SceneEntityCfg("table"),
                ],
            ),
            "robot_collision_free": mdp.CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.002,
                asset_cfg=SceneEntityCfg("robot", body_names="panda_link[2-7]|panda_hand|panda_(left|right)finger"),
                obstacle_cfgs=[
                    SceneEntityCfg("table"),
                    SceneEntityCfg("nistboard"),
                    SceneEntityCfg("fixed_asset"),
                ],
            ),
            "object_in_bound": ManagerTermBaseCfg(
                func=mdp.in_bound,
                params={
                    "asset_cfg": SceneEntityCfg("held_asset"),
                    "in_bound_range": {"x": (0.05, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
                },
            ),
        },
        "state_table_size": 32760,
        "state_tag_term": "reset_strategies",
        "variant_context": "assembly_variants",
        "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=50),
        "sampling": SamplerCfg(
            strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)],
            eps=1e-4,
        ),
        "reset_term": SCENE_RESET,
        "report": True,
    },
)
