# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic tasks using the shared G1 controller and matching local motion asset."""

from pathlib import Path

from isaaclab_isaacsim.sim.schemas import PhysxRigidBodyCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils import configclass, replace

from ...mdp.model_based import observations
from ...model_based_env_cfg import ModelBasedMotionTrackingEnvCfg
from .env_cfg import G1_ACTION_SCALE, G1_CFG, G1CommandsCfg

G1_DATA_DIR = Path(ISAACLAB_ROOT) / "data/g1_tracking"

G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip


@configclass
class G1ModelBasedEnvCfg(ModelBasedMotionTrackingEnvCfg):
    scene = replace(
        ModelBasedMotionTrackingEnvCfg().scene,
        robot=replace(
            G1_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(G1_DATA_DIR / "g1_usd/g1_29dof/g1_29dof.usda"),
                variants={"Physics": "mujoco"},
                activate_contact_sensors=False,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
                articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            ),
        ),
    )
    actions = replace(
        ModelBasedMotionTrackingEnvCfg().actions,
        joint_pos=replace(ModelBasedMotionTrackingEnvCfg().actions.joint_pos, scale=G1_ACTION_SCALE),
    )
    commands = replace(
        ModelBasedMotionTrackingEnvCfg().commands,
        motion=replace(
            ModelBasedMotionTrackingEnvCfg().commands.motion,
            motion_dir=str(G1_DATA_DIR / "lafan1_g1"),
            joint_names=G1_JOINT_NAMES,
            tracked_bodies=G1CommandsCfg().motion.body_names,
        ),
    )


@configclass
class G1MaxCoordModelBasedEnvCfg(G1ModelBasedEnvCfg):
    observations = replace(
        G1ModelBasedEnvCfg().observations,
        full_state=replace(
            G1ModelBasedEnvCfg().observations.full_state,
            state=replace(G1ModelBasedEnvCfg().observations.full_state.state, func=observations.maximal_state),
        ),
    )
