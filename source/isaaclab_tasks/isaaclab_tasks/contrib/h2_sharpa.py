# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint ordering, wrist calibration, and control shared by the H2 + Sharpa tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp import JointPositionActionCfg

POLICY_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
] + [
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


class H2GravityCompensatedJointPositionAction(JointPositionAction):
    """Absolute joint-position action with PhysX gravity feed-forward on both arms."""

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._gravity_joint_ids, _ = self._asset.find_joints(
            [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
        )
        self._gravity_columns = (
            torch.as_tensor(self._gravity_joint_ids, device=self.device, dtype=torch.long) + self._asset.num_base_dofs
        )

    def apply_actions(self) -> None:
        if not self._asset.cfg.spawn.rigid_props.disable_gravity:
            gravity = self._asset.data.gravity_compensation_forces[:, self._gravity_columns]
            self._asset.set_joint_effort_target_index(target=gravity, joint_ids=self._gravity_joint_ids)
        super().apply_actions()


def warm_rgb_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
    normalize: bool = False,
    response_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
    response_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Apply the fitted real-camera color response to an RTX RGB image."""
    image = base_mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    if data_type != "rgb":
        return image

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
