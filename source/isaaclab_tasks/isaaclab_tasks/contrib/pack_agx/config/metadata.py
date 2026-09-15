# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight Unitree H2 + Sharpa Wave metadata.

Warning: these joint orders / naming conventions are distinct:

* ``POLICY_58_ORDER``: Isaac-style ``left_arm(7), right_arm(7), left_hand(22),
  right_hand(22)`` policy order.
* ``DATASET_JOINT_NAMES_58``: LeRobot feature names — Unitree ``k*`` arms +
  Isaac-style hands (same layout as ``POLICY_58_ORDER``).
* ``ARM_K2JOINT``: map Unitree ``k*`` arm names -> Isaac arm joint names.
* ``H2_ACTION_JOINT_ORDER``: 75-DoF ``JointPositionAction`` input order.
* ``H2_ISAAC_ARTICULATION_ORDER``: runtime PhysX BFS ``robot.data.joint_names``
  order (diverges from ``H2_ACTION_JOINT_ORDER`` starting at index 2).
* ``H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER``: the 44-joint Isaac hand
  sub-order used by PinkIK.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTION_DIM = 58
"""Dataset arm and hand dimensions."""


FULL_ARTICULATION_DOF = 75
"""Full H2 + Sharpa articulation dimensions."""


# Neutral body pose; tasks provide their own initial poses.
H2_DEFAULT_JOINT_POS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": -0.2,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# 44 Sharpa hand joints in Isaac articulation sub-order (left/right interleaved).
H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER: list[str] = [
    "left_index_MCP_FE",
    "left_middle_MCP_FE",
    "left_pinky_CMC",
    "left_ring_MCP_FE",
    "left_thumb_CMC_FE",
    "right_index_MCP_FE",
    "right_middle_MCP_FE",
    "right_pinky_CMC",
    "right_ring_MCP_FE",
    "right_thumb_CMC_FE",
    "left_index_MCP_AA",
    "left_middle_MCP_AA",
    "left_pinky_MCP_FE",
    "left_ring_MCP_AA",
    "left_thumb_CMC_AA",
    "right_index_MCP_AA",
    "right_middle_MCP_AA",
    "right_pinky_MCP_FE",
    "right_ring_MCP_AA",
    "right_thumb_CMC_AA",
    "left_index_PIP",
    "left_middle_PIP",
    "left_pinky_MCP_AA",
    "left_ring_PIP",
    "left_thumb_MCP_FE",
    "right_index_PIP",
    "right_middle_PIP",
    "right_pinky_MCP_AA",
    "right_ring_PIP",
    "right_thumb_MCP_FE",
    "left_index_DIP",
    "left_middle_DIP",
    "left_pinky_PIP",
    "left_ring_DIP",
    "left_thumb_MCP_AA",
    "right_index_DIP",
    "right_middle_DIP",
    "right_pinky_PIP",
    "right_ring_DIP",
    "right_thumb_MCP_AA",
    "left_pinky_DIP",
    "left_thumb_IP",
    "right_pinky_DIP",
    "right_thumb_IP",
]
assert len(H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER) == 44

# 75-D action order. For Isaac BFS data, map through ``robot.data.joint_names``.
H2_ACTION_JOINT_ORDER: list[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
] + H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER
assert len(H2_ACTION_JOINT_ORDER) == FULL_ARTICULATION_DOF

# 58-D policy order (Isaac-style joint names).
POLICY_ARM_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
_HAND_22: list[str] = [
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
]
POLICY_HAND_JOINT_NAMES: list[str] = [f"left_{n}" for n in _HAND_22] + [f"right_{n}" for n in _HAND_22]
POLICY_58_ORDER: list[str] = POLICY_ARM_JOINT_NAMES + POLICY_HAND_JOINT_NAMES
assert len(POLICY_58_ORDER) == ACTION_DIM


# Cross-consumer camera identity; simulation mounts live in ``isaaclab``.
@dataclass(frozen=True)
class CameraSpec:
    """Cross-consumer camera names."""

    name: str  # scene sensor
    video_view: str  # modality view


H2_CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("front_camera", "high"),
    CameraSpec("left_wrist_camera", "left_wrist_view"),
    CameraSpec("right_wrist_camera", "right_wrist_view"),
)
CAMERA_BY_NAME: dict[str, CameraSpec] = {c.name: c for c in H2_CAMERAS}


# LeRobot dataset schema shared by converters and policy data configs.

MODALITY_LANGUAGE_KEYS: list[str] = ["annotation.human.task_description"]

# N1.7 nested-modality keys omit prefixes.
MODALITY_VIDEO_KEYS_BARE: list[str] = ["high", "left_wrist_view", "right_wrist_view"]
assert [c.video_view for c in H2_CAMERAS] == MODALITY_VIDEO_KEYS_BARE, "modality views out of sync with H2_CAMERAS"
MODALITY_STATE_KEYS_BARE: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand"]
MODALITY_ACTION_KEYS_BARE: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand"]

OBSERVATION_DELTA_INDICES: list[int] = [0]
ACTION_HORIZON_N17_INDICES: list[int] = list(range(32))

# Pick-and-place apple task placement constants.
#
# Retry placements for the replay sweep, as XY offsets [m] from the environment's
# configured apple position. Offsets rather than absolute table coordinates, so a
# re-calibrated apple pose or table height needs no edit here.
