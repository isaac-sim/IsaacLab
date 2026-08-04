# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg
from isaaclab_tasks.core.multi_task.utils import (
    BetaSamplingStrategyCfg,
    SamplerCfg,
)
from isaaclab_tasks.utils import preset

from . import mdp
from .factory_presets import (
    EndEffectorBodyCfg,
    FactoryAssemblyProfileCfg,
    FixedAssetMapCfg,
    FixedAssetTipCfg,
    GraspedPoseRangeCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    HeldAssetAlignOffsetCfg,
    HeldAssetGraspDiameterCfg,
    HeldAssetGraspMiddleCfg,
    HeldAssetGraspPointCfg,
    IKJointNamesCfg,
)

GRIPPER_GRASP_ASSET_IN_AIR = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_asset_in_air": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.15, 0.5),
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
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "ik_iterations": (5, 30),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg(
                        "robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                },
            ),
        }
    },
)


ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_held_asset_on_fixed_asset": EventTerm(
                func=mdp.reset_held_asset_on_fixed_asset,
                mode="reset",
                params={
                    "assembly_profile": FactoryAssemblyProfileCfg(),
                    "held_asset_align_offset": HeldAssetAlignOffsetCfg(),
                    "assembly_fraction_range": (0.0, 1.1),
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                },
            ),
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-1.0, 1.0),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "ik_iterations": (15, 25),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg(
                        "robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                },
            ),
        }
    },
)

GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_end_effector_around_fixed_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "fixed_asset_offset": FixedAssetTipCfg(),
                    "pose_range_b": GraspedPoseRangeCfg(),
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "ik_iterations": (10, 20),
                },
            ),
            "reset_held_asset_in_hand": EventTerm(
                func=mdp.reset_held_asset_in_gripper,
                mode="reset",
                params={
                    "holding_body_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                    "held_asset_graspable_offset": HeldAssetGraspPointCfg(),
                    "held_asset_inhand_range": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.000, 0.005),
                        "pitch": (-1.0, 1.0),
                    },
                    "gripper_grasp_offset": GripperGraspOffsetCfg(),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg(
                        "robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                    "flexible_angle": False,
                },
            ),
        }
    },
)


SCENE_RESET = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_robot": EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            # Fixed asset first: sample a filled uniform x-y patch (+ full yaw) about its nominal
            # board pose. Deriving the board from a yaw-randomized center used to smear the fixed
            # asset over a ring/donut; placing it directly avoids that. Tune the x-y extent.
            "reset_fixed_asset": EventTerm(
                func=mdp.reset_fixed_asset_uniform,
                mode="reset",
                params={
                    "asset_map": FixedAssetMapCfg(),
                    "pose_range": {"x": (0.075, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
                },
            ),
            # Board follows: seat it (and any extra board assets) under the placed fixed asset.
            "reset_board": EventTerm(
                func=mdp.reset_board_under_fixed_asset,
                mode="reset",
                params={
                    "asset_map": FixedAssetMapCfg(),
                },
            ),
            "reset_strategies": EventTerm(
                func=mdp.TermChoice,
                mode="reset",
                params={
                    "terms": {
                        "grasp_asset_in_air": GRIPPER_GRASP_ASSET_IN_AIR,
                        "start_assembled": ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE,
                        "start_grasped_then_assembled": GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER,
                    },
                    "sampling": SamplerCfg(
                        strategies=[
                            BetaSamplingStrategyCfg(
                                target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates"
                            )
                        ],
                        eps=1e-4,
                    ),
                    "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
                    "report": preset(accumulator=False, choice=True, default=False),
                },
            ),
        }
    },
)


ACCUMULATOR_RESET = EventTerm(
    func=mdp.reset_accumulator,
    mode="reset",
    params={
        "reset_assets": ["nistboard", "fixed_asset", "held_asset", "robot"],
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
                asset_cfg=SceneEntityCfg("robot"),
                obstacle_cfgs=[
                    SceneEntityCfg("table"),
                    SceneEntityCfg("nistboard"),
                    SceneEntityCfg("fixed_asset"),
                ],
            ),
        },
        "state_table_size": 32768,
        "state_tag_names_bind": "list(reset_term.func.terms['reset_strategies'].func.term_partitions.keys())",
        "state_tag_indices_bind": "reset_term.func.terms['reset_strategies'].func.term_samples",
        "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=50),
        "sampling": SamplerCfg(
            strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates")],
            eps=1e-4,
        ),
        "reset_term": SCENE_RESET,
        "report": True,
    },
)
