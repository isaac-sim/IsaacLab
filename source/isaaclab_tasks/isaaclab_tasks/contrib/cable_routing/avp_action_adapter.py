# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Right-hand AVP control for the Newton-IK YAM cable-routing teleop task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from .yam_frames import YAM_CONTACT_FRAME_OFFSET_POS, YAM_CONTACT_FRAME_OFFSET_QUAT

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class CableRoutingAVPActionAdapter:
    """Map one AVP right hand to the teleop task's native Newton-IK action.

    IsaacTeleop supplies an eight-dimensional command ordered as right-hand pose
    ``(x, y, z, qx, qy, qz, qw)`` in world coordinates followed by one gripper
    command. The teleop-only environment has four manager action terms:

    * six absolute left-arm joints, used only to hold the inactive YAM;
    * one fixed-open left gripper command;
    * one absolute seven-dimensional Newton IK target for the right YAM; and
    * one binary right-gripper command.

    The training environments retain their fourteen-dimensional relative joint
    position action space. Only this interactive teleop variant swaps its right
    arm action term to :class:`~isaaclab_newton.envs.mdp.NewtonInverseKinematicsAction`.
    The adapter therefore does no Jacobian solve and imposes no Cartesian servo
    ramp. On engagement it aligns the hand position with the current YAM pinch
    position. Orientation is already the anatomical YAM grasp-frame orientation
    produced by the task's retargeter and remains absolute: contact ``+Z`` follows
    the human fingers, contact ``+X`` follows index-to-little, and the contact
    ``+X/+Z`` pad plane follows the hand plane. The target is then rebased into
    the right-YAM root.

    Args:
        cfg: Adapter configuration.
        env: Instantiated single-environment cable-routing task.
    """

    RAW_ACTION_DIM = 8
    ENV_ACTION_DIM = 15
    _EXPECTED_ACTION_TERMS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    _EXPECTED_ACTION_TERM_DIMS = (6, 1, 7, 1)

    def __init__(self, cfg: CableRoutingAVPActionAdapterCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self._env = env.unwrapped
        self._device = torch.device(self._env.device)

        if self._env.num_envs != 1:
            raise ValueError(
                "CableRoutingAVPActionAdapter supports exactly one environment, "
                f"but received num_envs={self._env.num_envs}."
            )
        actual_terms = tuple(self._env.action_manager.active_terms)
        actual_dims = tuple(self._env.action_manager.action_term_dim)
        if actual_terms != self._EXPECTED_ACTION_TERMS or actual_dims != self._EXPECTED_ACTION_TERM_DIMS:
            raise ValueError(
                "The right-hand AVP adapter requires teleop action terms "
                f"{self._EXPECTED_ACTION_TERMS} with dimensions {self._EXPECTED_ACTION_TERM_DIMS}; "
                f"received {actual_terms} with dimensions {actual_dims}."
            )

        self._left_robot = self._env.scene[cfg.left_asset_name]
        self._right_robot = self._env.scene[cfg.right_asset_name]
        self._left_joint_ids = self._resolve_arm_joint_ids(self._left_robot, cfg.left_asset_name)

        body_ids, body_names = self._right_robot.find_bodies([cfg.end_effector_body_name], preserve_order=True)
        if len(body_ids) != 1 or body_names != [cfg.end_effector_body_name]:
            raise ValueError(
                f"YAM asset '{cfg.right_asset_name}' did not resolve the unique end-effector body "
                f"'{cfg.end_effector_body_name}'."
            )
        self._right_body_id = body_ids[0]

        self._contact_offset_pos = torch.tensor(cfg.contact_frame_offset_pos, device=self._device).reshape(1, 3)
        self._contact_offset_quat = self._normalized_quaternion(
            torch.tensor(cfg.contact_frame_offset_quat, device=self._device).reshape(1, 4)
        )
        self._workspace_min = torch.tensor(cfg.workspace_min, device=self._device).reshape(1, 3)
        self._workspace_max = torch.tensor(cfg.workspace_max, device=self._device).reshape(1, 3)

        self._left_hold_joint_pos: torch.Tensor
        self._right_hold_pos_b: torch.Tensor
        self._right_hold_quat_b: torch.Tensor
        self._right_tracking_active = False
        self._previous_hand_pos_w: torch.Tensor | None = None
        self._previous_hand_quat_w: torch.Tensor | None = None
        self._hand_reference_pos_w: torch.Tensor | None = None
        self._contact_reference_pos_w: torch.Tensor | None = None
        self._last_gripper_action = cfg.gripper_open_action
        self.reset()

    def reset(self) -> None:
        """Hold the inactive left YAM and reset right-hand tracking state."""
        self._left_hold_joint_pos = self._left_robot.data.joint_pos.torch[:, self._left_joint_ids].clone()
        self._right_hold_pos_b, self._right_hold_quat_b = self._current_right_contact_pose_b()
        self._right_tracking_active = False
        self._previous_hand_pos_w = None
        self._previous_hand_quat_w = None
        self._hand_reference_pos_w = None
        self._contact_reference_pos_w = None
        self._last_gripper_action = self.cfg.gripper_open_action

    def prewarm(self) -> torch.Tensor:
        """Capture the right-arm IK graph at the current physical hold pose.

        Teleoperation deliberately waits for an explicit XR start event before
        stepping physics. Without this warmup, the first tracked hand sample
        would also pay the one-time Warp graph-capture cost. Processing a safe
        hold through the manager and applying only the right-arm term keeps the
        robot stationary while moving that cost into session preparation.
        """
        self._right_hold_pos_b, self._right_hold_quat_b = self._current_right_contact_pose_b()
        hold_action = self._compose_manager_action(
            self._right_hold_pos_b,
            self._right_hold_quat_b,
            self.cfg.gripper_open_action,
        )
        self._env.action_manager.process_action(hold_action.unsqueeze(0))
        self._env.action_manager.get_term("right_arm").apply_actions()
        return hold_action

    def process(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Convert one right-hand IsaacTeleop command to the teleop action layout.

        Args:
            raw_action: Right AVP pose and gripper command, shape ``(8,)`` or
                ``(1, 8)``. Position is in world coordinates [m] and the
                quaternion is in ``(x, y, z, w)`` order.

        Returns:
            Left hold, right Newton IK pose, and gripper commands, shape ``(15,)``.

        Raises:
            ValueError: If :paramref:`raw_action` does not contain exactly eight values.
        """
        raw_action = torch.as_tensor(raw_action, device=self._device, dtype=torch.float32)
        if raw_action.ndim == 2 and raw_action.shape[0] == 1:
            raw_action = raw_action.squeeze(0)
        if raw_action.ndim != 1 or raw_action.shape[0] != self.RAW_ACTION_DIM:
            raise ValueError(
                "CableRoutingAVPActionAdapter expects an 8-D command in shape (8,) or (1, 8); "
                f"received {tuple(raw_action.shape)}."
            )

        right_pos_b, right_quat_b, right_gripper = self._process_right_hand(raw_action[:7], raw_action[7])
        return self._compose_manager_action(right_pos_b, right_quat_b, right_gripper)

    def _compose_manager_action(
        self,
        right_pos_b: torch.Tensor,
        right_quat_b: torch.Tensor,
        right_gripper: float,
    ) -> torch.Tensor:
        """Assemble the fixed manager action layout from right-hand targets."""
        output = torch.empty(self.ENV_ACTION_DIM, device=self._device, dtype=self._left_hold_joint_pos.dtype)
        output[0:6] = self._left_hold_joint_pos.squeeze(0)
        output[6] = self.cfg.gripper_open_action
        output[7:10] = right_pos_b.squeeze(0)
        output[10:14] = right_quat_b.squeeze(0)
        output[14] = right_gripper
        return output

    def _resolve_arm_joint_ids(self, robot: Articulation, asset_name: str) -> list[int]:
        joint_ids, joint_names = robot.find_joints(list(self.cfg.arm_joint_names), preserve_order=True)
        if tuple(joint_names) != self.cfg.arm_joint_names:
            raise ValueError(
                f"YAM asset '{asset_name}' resolved arm joints {tuple(joint_names)}, "
                f"expected {self.cfg.arm_joint_names}."
            )
        return joint_ids

    def _process_right_hand(
        self, hand_pose_w: torch.Tensor, gripper_command: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        hand_pos_w = hand_pose_w[:3].reshape(1, 3)
        hand_quat_w = hand_pose_w[3:7].reshape(1, 4)
        pose_valid = self._pose_is_valid(hand_pos_w, hand_quat_w)
        if pose_valid and self._tracking_jump_detected(hand_pos_w, hand_quat_w):
            pose_valid = False
        if not pose_valid:
            self._release_tracking()
            return self._right_hold_pos_b, self._right_hold_quat_b, self._last_gripper_action

        if torch.isfinite(gripper_command):
            self._last_gripper_action = (
                self.cfg.gripper_open_action
                if float(gripper_command) >= self.cfg.gripper_threshold
                else self.cfg.gripper_close_action
            )

        hand_quat_w = self._normalized_quaternion(hand_quat_w)
        if not self._right_tracking_active:
            contact_pos_w, _ = self._current_right_contact_pose_w()
            self._hand_reference_pos_w = hand_pos_w.clone()
            self._contact_reference_pos_w = contact_pos_w.clone()

        if self._hand_reference_pos_w is None or self._contact_reference_pos_w is None:
            raise RuntimeError("Right-hand engagement references were not initialized.")

        target_pos_w = self._contact_reference_pos_w + (hand_pos_w - self._hand_reference_pos_w)
        # The pipeline quaternion is not a runtime-specific wrist basis. It is
        # the absolute physical YAM contact basis reconstructed from hand-joint
        # geometry, so clutching it would reintroduce the very axis offset this
        # semantic mapping removes.
        target_quat_w = hand_quat_w
        root_pos_w = self._right_robot.data.root_pos_w.torch
        root_quat_w = self._right_robot.data.root_quat_w.torch
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        target_pos_b = self._clamp_target_to_workspace(target_pos_b, root_pos_w, root_quat_w)

        self._right_tracking_active = True
        self._previous_hand_pos_w = hand_pos_w.clone()
        self._previous_hand_quat_w = hand_quat_w.clone()
        self._right_hold_pos_b = target_pos_b.clone()
        self._right_hold_quat_b = target_quat_b.clone()
        return target_pos_b, target_quat_b, self._last_gripper_action

    def _release_tracking(self) -> None:
        if self._right_tracking_active:
            self._right_hold_pos_b, self._right_hold_quat_b = self._current_right_contact_pose_b()
        self._right_tracking_active = False
        self._previous_hand_pos_w = None
        self._previous_hand_quat_w = None
        self._hand_reference_pos_w = None
        self._contact_reference_pos_w = None

    def _current_right_contact_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_pos_w, contact_quat_w = self._current_right_contact_pose_w()
        return math_utils.subtract_frame_transforms(
            self._right_robot.data.root_pos_w.torch,
            self._right_robot.data.root_quat_w.torch,
            contact_pos_w,
            contact_quat_w,
        )

    def _current_right_contact_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the measured physical pinch-frame pose in world coordinates."""
        link_pos_w = self._right_robot.data.body_pos_w.torch[:, self._right_body_id]
        link_quat_w = self._right_robot.data.body_quat_w.torch[:, self._right_body_id]
        return math_utils.combine_frame_transforms(
            link_pos_w,
            link_quat_w,
            self._contact_offset_pos,
            self._contact_offset_quat,
        )

    def _clamp_target_to_workspace(
        self,
        target_pos_b: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        target_pos_w, _ = math_utils.combine_frame_transforms(root_pos_w, root_quat_w, target_pos_b)
        env_origin = self._env.scene.env_origins[0:1]
        target_pos_e = torch.clamp(target_pos_w - env_origin, min=self._workspace_min, max=self._workspace_max)
        target_pos_b, _ = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_pos_e + env_origin,
        )
        return target_pos_b

    def _pose_is_valid(self, position: torch.Tensor, quaternion: torch.Tensor) -> bool:
        if not bool(torch.isfinite(position).all() and torch.isfinite(quaternion).all()):
            return False
        quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
        return bool(
            (quaternion_norm >= self.cfg.min_quaternion_norm).all()
            and (torch.linalg.vector_norm(position, dim=-1) <= self.cfg.max_tracking_position_norm).all()
        )

    def _tracking_jump_detected(self, hand_pos_w: torch.Tensor, hand_quat_w: torch.Tensor) -> bool:
        if self._previous_hand_pos_w is None or self._previous_hand_quat_w is None:
            return False
        position_jump = torch.linalg.vector_norm(hand_pos_w - self._previous_hand_pos_w, dim=-1)
        current_quat = self._normalized_quaternion(hand_quat_w)
        quaternion_dot = torch.sum(current_quat * self._previous_hand_quat_w, dim=-1).abs().clamp(max=1.0)
        rotation_jump = 2.0 * torch.acos(quaternion_dot)
        return bool(
            (position_jump > self.cfg.max_tracking_position_jump).any()
            or (rotation_jump > self.cfg.max_tracking_rotation_jump).any()
        )

    @staticmethod
    def _normalized_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
        return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)


@configclass
class CableRoutingAVPActionAdapterCfg:
    """Configuration for direct right-hand AVP-to-Newton-IK control."""

    class_type: type[CableRoutingAVPActionAdapter] = CableRoutingAVPActionAdapter
    """Adapter implementation instantiated by the teleoperation script."""

    left_asset_name: str = "yam_left"
    """Inactive articulation held at its pose when teleoperation starts."""

    right_asset_name: str = "yam_right"
    """Articulation controlled by the right AVP hand."""

    arm_joint_names: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    """Ordered arm joints used by the left-YAM hold action."""

    end_effector_body_name: str = "link_6"
    """Rigid body that owns the physical fingertip contact frame."""

    contact_frame_offset_pos: tuple[float, float, float] = YAM_CONTACT_FRAME_OFFSET_POS
    """Midpoint of the inner fingertip contact pads in ``link_6`` coordinates [m]."""

    contact_frame_offset_quat: tuple[float, float, float, float] = YAM_CONTACT_FRAME_OFFSET_QUAT
    """Pinch-frame orientation in ``link_6`` coordinates, quaternion ``(x, y, z, w)``."""

    workspace_min: tuple[float, float, float] = (-0.48, -0.40, 0.775)
    """Minimum right-hand target position relative to the environment origin [m]."""

    workspace_max: tuple[float, float, float] = (0.48, 0.40, 1.30)
    """Maximum right-hand target position relative to the environment origin [m]."""

    max_tracking_position_norm: float = 10.0
    """Maximum finite AVP hand-position norm considered valid [m]."""

    max_tracking_position_jump: float = 0.25
    """Maximum accepted hand translation between successive samples [m]."""

    max_tracking_rotation_jump: float = 2.10
    """Maximum accepted hand rotation between successive samples [rad]."""

    min_quaternion_norm: float = 1.0e-4
    """Minimum AVP hand-quaternion norm considered valid."""

    gripper_threshold: float = 0.0
    """Retargeter value at or above which the binary action opens the gripper."""

    gripper_open_action: float = 1.0
    """Positive action expected by :class:`BinaryJointPositionAction` for opening."""

    gripper_close_action: float = -1.0
    """Negative action expected by :class:`BinaryJointPositionAction` for closing."""

    def __post_init__(self) -> None:
        if any(lower >= upper for lower, upper in zip(self.workspace_min, self.workspace_max)):
            raise ValueError(
                "workspace_min must be strictly below workspace_max, "
                f"got {self.workspace_min} and {self.workspace_max}."
            )
        if self.max_tracking_position_jump <= 0.0 or self.max_tracking_rotation_jump <= 0.0:
            raise ValueError("Tracking jump thresholds must be positive.")
        if self.min_quaternion_norm <= 0.0:
            raise ValueError("The quaternion-norm limit must be positive.")
