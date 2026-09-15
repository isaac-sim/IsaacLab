# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab articulation presets for the Unitree H2 + Sharpa Wave embodiment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.rlinf_assets import ROBOT_ASSET_ROOT

from .metadata import H2_ACTION_JOINT_ORDER, H2_DEFAULT_JOINT_POS, POLICY_58_ORDER

H2_SHARPA_USD_PATH: str = f"{ROBOT_ASSET_ROOT}/h2_with_sharpa/H2_with_sharpa_flat.usd"

# Neutral articulation pose; tasks override it through the preset helpers.
_H2_INIT_POS_NEUTRAL = (-0.95, 0.0, 1.05)
_H2_INIT_ROT_NEUTRAL = (0.0, 0.0, 0.0, 1.0)

# Arm and head PD gains measured on the physical robot.
H2_REAL_ARM_STIFFNESS: dict[str, float] = {
    ".*_shoulder_.*_joint": 150.0,
    ".*_elbow_joint": 150.0,
    ".*_wrist_.*_joint": 50.0,
}
H2_REAL_ARM_DAMPING: dict[str, float] = {
    ".*_shoulder_.*_joint": 10.0,
    ".*_elbow_joint": 10.0,
    ".*_wrist_.*_joint": 3.0,
}
H2_REAL_HEAD_STIFFNESS = 150.0
H2_REAL_HEAD_DAMPING = 10.0

H2_SHARPA_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=H2_SHARPA_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        # Bind a contact material to the whole robot (recurses to all collision
        # meshes incl. fingertips) so hand<->apple friction matches the apple's
        # 0.95, instead of the PhysX default ~0.5 baked into the H2 USD.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.95,
            dynamic_friction=0.95,
            restitution=0.0,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=_H2_INIT_POS_NEUTRAL,
        rot=_H2_INIT_ROT_NEUTRAL,
        joint_pos=H2_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=1000.0,
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=1000.0,
            armature=0.03,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=1e6,
            damping=1e5,
            armature=0.03,
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=H2_REAL_HEAD_STIFFNESS,
            damping=H2_REAL_HEAD_DAMPING,
            armature=0.03,
            friction=0.03,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 220.0,
                ".*_shoulder_roll_joint": 154.0,
                ".*_shoulder_yaw_joint": 154.0,
                ".*_elbow_joint": 154.0,
                ".*_wrist_roll_joint": 154.0,
                ".*_wrist_pitch_joint": 125.0,
                ".*_wrist_yaw_joint": 125.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 28.0,
                ".*_shoulder_roll_joint": 34.0,
                ".*_shoulder_yaw_joint": 34.0,
                ".*_elbow_joint": 34.0,
                ".*_wrist_roll_joint": 34.0,
                ".*_wrist_pitch_joint": 50.0,
                ".*_wrist_yaw_joint": 50.0,
            },
            stiffness=H2_REAL_ARM_STIFFNESS,
            damping=H2_REAL_ARM_DAMPING,
            armature={".*_shoulder_.*": 0.03, ".*_elbow_.*": 0.03, ".*_wrist_.*_joint": 0.03},
            friction=0.03,
        ),
        "hands": IdealPDActuatorCfg(
            joint_names_expr=[".*_thumb_.*", ".*_index_.*", ".*_middle_.*", ".*_ring_.*", ".*_pinky_.*"],
            # Grasp vs tunneling tradeoff (IdealPD, torque capped by effort_limit):
            #   20/2  -> contact wins, but fingers often cannot hold the apple
            #   40/5  -> grasp succeeds more, but fingers tunnel into the apple
            #   28/3  -> middle ground; keep effort_limit=5 so peak force stays bounded
            effort_limit=5.0,
            velocity_limit=16.0,
            stiffness=28.0,
            damping=3.0,
            armature=0.03,
            friction=0.03,
        ),
    },
)


def make_h2_sharpa_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: tuple[float, float, float] = _H2_INIT_POS_NEUTRAL,
    init_rot: tuple[float, float, float, float] = _H2_INIT_ROT_NEUTRAL,
    custom_joint_pos: dict[str, float] | None = None,
    base_config: ArticulationCfg = H2_SHARPA_CFG,
) -> ArticulationCfg:
    """H2 + Sharpa cfg with per-task pose overrides merged onto ``H2_DEFAULT_JOINT_POS``."""
    joint_pos = dict(H2_DEFAULT_JOINT_POS)
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


def h2_body_joint_offsets(custom_joint_pos: dict[str, float] | None = None) -> dict[str, float]:
    """Action offsets that park the joints GR00T does not predict at the robot's default pose.

    The policy emits only ``POLICY_58_ORDER`` (arms and hands); the legs, waist and head entries of
    the action vector are zero-filled by the GR00T action converter. Without an offset those joints
    are driven to 0 rad, which tips the head up from its 0.6 rad default and points the front
    camera at the wall instead of the table.

    Args:
        custom_joint_pos: Per-task overrides merged onto ``H2_DEFAULT_JOINT_POS``, matching the
            ``custom_joint_pos`` passed to :func:`make_h2_sharpa_cfg`.

    Returns:
        Mapping from body joint name to its default position [rad].
    """
    joint_pos = dict(H2_DEFAULT_JOINT_POS)
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return {name: joint_pos[name] for name in H2_ACTION_JOINT_ORDER if name not in POLICY_58_ORDER}


@configclass
class H2RobotPresets:
    """H2 robot presets."""

    @classmethod
    def h2_sharpa_base_fix(
        cls,
        init_pos: tuple[float, float, float] = _H2_INIT_POS_NEUTRAL,
        init_rot: tuple[float, float, float, float] = _H2_INIT_ROT_NEUTRAL,
        custom_joint_pos: dict[str, float] | None = None,
    ) -> ArticulationCfg:
        """H2 + Sharpa Wave, base-fixed."""
        return make_h2_sharpa_cfg(init_pos=init_pos, init_rot=init_rot, custom_joint_pos=custom_joint_pos)
