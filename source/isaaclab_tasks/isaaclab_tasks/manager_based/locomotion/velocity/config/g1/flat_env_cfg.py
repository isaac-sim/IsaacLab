# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import SymmetryCfg, SymmetryTermCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


g1_joints_names = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
]


# Symmetry joint configurations
# all pitch joints needs to be swapped
symmetry_joint_swap_ids = [
    (g1_joints_names.index("left_hip_pitch_joint"), g1_joints_names.index("right_hip_pitch_joint")),
    (g1_joints_names.index("left_knee_joint"), g1_joints_names.index("right_knee_joint")),
    (g1_joints_names.index("left_shoulder_pitch_joint"), g1_joints_names.index("right_shoulder_pitch_joint")),
    (g1_joints_names.index("left_ankle_pitch_joint"), g1_joints_names.index("right_ankle_pitch_joint")),
    (g1_joints_names.index("left_elbow_pitch_joint"), g1_joints_names.index("right_elbow_pitch_joint")),
]
# all roll and yaw joints needs to be swapped and negated
symmetry_joint_swap_negate_ids = [
    (g1_joints_names.index("left_hip_roll_joint"), g1_joints_names.index("right_hip_roll_joint")),
    (g1_joints_names.index("left_hip_yaw_joint"), g1_joints_names.index("right_hip_yaw_joint")),
    (g1_joints_names.index("left_shoulder_roll_joint"), g1_joints_names.index("right_shoulder_roll_joint")),
    (g1_joints_names.index("left_ankle_roll_joint"), g1_joints_names.index("right_ankle_roll_joint")),
    (g1_joints_names.index("left_shoulder_yaw_joint"), g1_joints_names.index("right_shoulder_yaw_joint")),
]
# non-symmetric (at center) yaw joints needs to be negated
symmetry_joint_negate_ids = [
    g1_joints_names.index("waist_yaw_joint"),
]


@configclass
class G1Symmetry(SymmetryCfg):
    """Configuration for the symmetry of the environment."""

    @configclass
    class ActionsCfg(SymmetryTermCfg):
        swap_terms = {
            "joint_pos": symmetry_joint_swap_ids,
        }
        swap_negate_terms = {
            "joint_pos": symmetry_joint_swap_negate_ids,
        }
        negate_terms = {
            "joint_pos": symmetry_joint_negate_ids,
        }

    @configclass
    class ActorObservationsCfg(SymmetryTermCfg):
        swap_terms = {
            "joint_pos": symmetry_joint_swap_ids,
            "joint_vel": symmetry_joint_swap_ids,
            "actions": symmetry_joint_swap_ids,
        }
        swap_negate_terms = {
            "joint_pos": symmetry_joint_swap_negate_ids,
            "joint_vel": symmetry_joint_swap_negate_ids,
            "actions": symmetry_joint_swap_negate_ids,
        }
        negate_terms = {
            "velocity_commands": [1, 2],
            "base_ang_vel": [0, 2],
            "projected_gravity": [1],
            "joint_pos": symmetry_joint_negate_ids,
            "joint_vel": symmetry_joint_negate_ids,
            "actions": symmetry_joint_negate_ids,
        }

    @configclass
    class CriticObservationsCfg(ActorObservationsCfg):
        negate_terms = {
            "velocity_commands": [1, 2],
            "base_lin_vel": [1],
            "base_ang_vel": [0, 2],
            "projected_gravity": [1],
            "joint_pos": symmetry_joint_negate_ids,
            "joint_vel": symmetry_joint_negate_ids,
            "actions": symmetry_joint_negate_ids,
        }

    actions: ActionsCfg = ActionsCfg()
    actor_observations: ActorObservationsCfg = ActorObservationsCfg()
    critic_observations: CriticObservationsCfg = CriticObservationsCfg()


@configclass
class G1FlatSymmetryEnvCfg(G1FlatEnvCfg):
    # Symmetry loss configuration
    symmetry: G1Symmetry = G1Symmetry()


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
