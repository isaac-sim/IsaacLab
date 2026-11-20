# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint ordering, wrist calibration, and control shared by the H2 + Sharpa tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import H2_SHARPA_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ARM_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]

POLICY_JOINT_NAMES = ARM_JOINT_NAMES + [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
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
    )
]
"""Absolute policy joint positions: left arm, right arm, left hand, right hand."""

# pnp_apple_sim_new_bg_v1 episode_000027 frame 0, replayed from
# apple_pick_and_place_05142000_filtered episode_000043, in POLICY_JOINT_NAMES
# (left_arm, right_arm, left_hand, right_hand).
# fmt: off
_FRAME0_POLICY_58_POS = (
    # Left arm
    0.2754, 0.0215, -0.0674, -0.0774, 0.0518, -0.1481, -0.026,
    # Right arm
    0.1734, -0.0544, 0.0165, -0.0457, 0.0006, 0.0136, 0.1644,
    # Left hand
    0.2834, 0.0498, -0.4345, 0.2344, 0.7783, -0.1513, -0.0731, 0.0373, 0.1479, -0.154, 0.0019,
    0.0358, 0.1647, -0.1514, 0.1047, 0.0385, 0.1432, 0.1048, -0.1552, 0.1278, 0.0348, 0.0924,
    # Right hand
    -0.0452, 0.0498, 0.3536, 0.0689, 0.0869, -0.1104, 0.1215, 0.0687, 0.1541, -0.1028, 0.1457,
    0.081, 0.0946, -0.1501, 0.1255, 0.0361, 0.089, 0.1171, -0.1064, 0.0177, 0.0611, 0.0884,
)
# fmt: on

H2_TABLETOP_ROBOT_CFG = H2_SHARPA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=H2_SHARPA_CFG.init_state.replace(
        joint_pos={
            **H2_SHARPA_CFG.init_state.joint_pos,
            **dict(zip(POLICY_JOINT_NAMES, _FRAME0_POLICY_58_POS, strict=True)),
            "head_pitch_joint": 0.6,
        },
    ),
)
"""H2 + Sharpa robot in the starting pose shared by both tabletop tasks."""

LEFT_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_hand_C_MC/left_wrist_camera",
    update_period=0.02,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=0.15435,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_nominal_width=640.0,
        fisheye_nominal_height=480.0,
        fisheye_optical_centre_x=319.90624,
        fisheye_optical_centre_y=239.67252,
        fisheye_max_fov=196,
        fisheye_polynomial_a=0.0,
        fisheye_polynomial_b=5.788058552990647e-3,
        fisheye_polynomial_c=1.6881056765838268e-6,
        fisheye_polynomial_d=-4.2228624221772085e-8,
        fisheye_polynomial_e=1.4575948452911756e-10,
        fisheye_polynomial_f=-1.4645006973842839e-13,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.07498591, 0.0007267178, 0.004823103),
        rot=(-0.6486821, 0.6996208, 0.1177079, 0.2754761),
        convention="opengl",
    ),
)
"""Left Sharpa Wave wrist camera, mounted on the hand's CAD bracket."""

RIGHT_WRIST_CAMERA_CFG = LEFT_WRIST_CAMERA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_C_MC/right_wrist_camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.07964737, 0.001976802, 0.01021968),
        rot=(0.6884326, -0.664184, 0.2683462, 0.1136246),
        convention="opengl",
    ),
)
"""Right Sharpa Wave wrist camera, mirrored from the left mount."""


@configclass
class ActionsCfg:
    """Absolute arm and hand position commands with gravity feed-forward on the arms."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_JOINT_NAMES,
        gravity_compensation_joint_names=ARM_JOINT_NAMES,
        use_default_offset=False,
        preserve_order=True,
    )


def apply_camera_response(
    image: torch.Tensor,
    response_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
    response_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Apply the fitted color response to a raw uint8 RGB observation."""
    output_dtype = image.dtype
    image = image.to(torch.float32)
    image = (image / 255.0).clamp(0.0, 1.0)
    image = image - 0.18 * image.pow(3)
    warm_balance = image.new_tensor((1.01, 0.99, 0.96))
    image = (image * warm_balance).clamp_(0.0, 1.0)

    # Match the real camera's sRGB response, white balance, and exposure.
    gamma = image.new_tensor(response_gamma)
    gain = image.new_tensor(response_gain)
    image = image.pow(gamma) * gain

    return (image * 255.0).clamp_(0.0, 255.0).to(output_dtype)


def advance_phase(
    phase: torch.Tensor, hold_counter: torch.Tensor, current: int, condition: torch.Tensor, hold_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance after consecutive matching steps; other phases leave the counter unchanged."""
    hold_counter = torch.where(phase == current, torch.where(condition, hold_counter + 1, 0), hold_counter)
    advance = (phase == current) & (hold_counter >= hold_steps)
    return torch.where(advance, current + 1, phase), torch.where(advance, 0, hold_counter)


def phase_reward(env: ManagerBasedRLEnv, from_phase: int) -> torch.Tensor:
    """Reward one transition, measured by the task-success term before rewards are evaluated."""
    progress = env.termination_manager.get_term_cfg("task_success").func
    return ((progress.previous_phase == from_phase) & (progress.phase > from_phase)).float() / env.step_dt
