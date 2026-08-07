# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual reach environment configuration for Pollen Robotics Reachy 2.

Both arms simultaneously track independent end-effector pose targets.

Environments:
* :class:`Reachy2BimanualReachEnvCfg`: Both arms reach to independent targets.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import REACHY2_CFG  # isort: skip


@configclass
class Reachy2BimanualReachEnvCfg(ReachEnvCfg):
    """Reach environment for Reachy 2 — both arms simultaneously.

    Each arm receives an independent uniformly-sampled pose target.
    Action space is 14-DOF (7 right + 7 left arm joints).
    Observation space includes both pose commands (right + left).
    """

    def __post_init__(self):
        super().__post_init__()

        # Reachy 2 is a standing humanoid — no table needed
        self.scene.table = None
        # Ground flush with robot base
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)

        # ── Robot ──────────────────────────────────────────────────────────
        self.scene.robot = REACHY2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "neck_.*": 0.0,
                    "r_shoulder_pitch": 0.0,
                    "r_shoulder_roll": -0.2,
                    "r_elbow_yaw": 0.0,
                    "r_elbow_pitch": -0.5,
                    "r_wrist_roll": 0.0,
                    "r_wrist_pitch": 0.0,
                    "r_wrist_yaw": 0.0,
                    "l_shoulder_pitch": 0.0,
                    "l_shoulder_roll": 0.2,
                    "l_elbow_yaw": 0.0,
                    "l_elbow_pitch": -0.5,
                    "l_wrist_roll": 0.0,
                    "l_wrist_pitch": 0.0,
                    "l_wrist_yaw": 0.0,
                    ".*hand_finger.*": 0.0,
                },
            ),
        )

        # ── Actions — right arm (inherited arm_action → repurpose for right) ──
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "r_shoulder_pitch",
                "r_shoulder_roll",
                "r_elbow_yaw",
                "r_elbow_pitch",
                "r_wrist_roll",
                "r_wrist_pitch",
                "r_wrist_yaw",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # ── Actions — left arm (new term) ──────────────────────────────────
        self.actions.l_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "l_shoulder_pitch",
                "l_shoulder_roll",
                "l_elbow_yaw",
                "l_elbow_pitch",
                "l_wrist_roll",
                "l_wrist_pitch",
                "l_wrist_yaw",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # ── Commands — right EE (reuse inherited ee_pose) ──────────────────
        _r_ee = "r_hand_palm_link"
        self.commands.ee_pose.body_name = _r_ee
        self.commands.ee_pose.ranges.pos_x = (0.3, 0.55)
        self.commands.ee_pose.ranges.pos_y = (-0.4, 0.0)
        self.commands.ee_pose.ranges.pos_z = (0.8, 1.2)
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # ── Commands — left EE (new term) ──────────────────────────────────
        _l_ee = "l_hand_palm_link"
        self.commands.ee_pose_left = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=_l_ee,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            position_success_threshold=0.05,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.3, 0.55),
                pos_y=(0.0, 0.4),
                pos_z=(0.8, 1.2),
                roll=(0.0, 0.0),
                pitch=(math.pi / 2, math.pi / 2),
                yaw=(-3.14, 3.14),
            ),
        )

        # ── Rewards — right EE (reuse inherited reward terms) ──────────────
        _r_cfg = SceneEntityCfg("robot", body_names=[_r_ee])
        self.rewards.end_effector_position_tracking.params["asset_cfg"] = _r_cfg
        self.rewards.end_effector_position_tracking.params["command_name"] = "ee_pose"
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"] = _r_cfg
        self.rewards.end_effector_position_tracking_fine_grained.params["command_name"] = "ee_pose"
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"] = _r_cfg
        self.rewards.end_effector_orientation_tracking.params["command_name"] = "ee_pose"

        # ── Rewards — left EE (new terms, same weights) ────────────────────
        _l_cfg = SceneEntityCfg("robot", body_names=[_l_ee])
        self.rewards.l_end_effector_position_tracking = RewTerm(
            func=mdp.position_command_error,
            weight=-0.2,
            params={"asset_cfg": _l_cfg, "command_name": "ee_pose_left"},
        )
        self.rewards.l_end_effector_position_tracking_fine_grained = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.1,
            params={"asset_cfg": _l_cfg, "std": 0.1, "command_name": "ee_pose_left"},
        )
        self.rewards.l_end_effector_orientation_tracking = RewTerm(
            func=mdp.orientation_command_error,
            weight=-0.1,
            params={"asset_cfg": _l_cfg, "command_name": "ee_pose_left"},
        )

        # ── Observations — add left pose command ───────────────────────────
        # Inherited: joint_pos(27) + joint_vel(27) + pose_command/right(7) + actions(14)
        # Added: pose_command_left(7)
        # Total policy obs: 82
        self.observations.policy.pose_command_left = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose_left"},
        )


@configclass
class Reachy2BimanualReachEnvCfg_PLAY(Reachy2BimanualReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
