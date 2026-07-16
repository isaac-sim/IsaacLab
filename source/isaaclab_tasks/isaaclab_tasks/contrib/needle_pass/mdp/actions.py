# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned action terms for the dVRK bimanual 18D action ABI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        DonorReleaseGuardedPairedJawJointPositionActionCfg,
        PairedJawJointPositionActionCfg,
        WorldFrameDifferentialInverseKinematicsActionCfg,
    )

PSM_JAW_JOINT_ORDER = ("psm_tool_gripper1_joint", "psm_tool_gripper2_joint")


def donor_release_is_allowed(phase: torch.Tensor, receiver_grasp: torch.Tensor, co_hold_phase: int) -> torch.Tensor:
    """Return whether a donor opening may proceed in each environment.

    A completed co-hold dwell is necessary but deliberately insufficient on
    its own: the receiver must still have a bilateral, opposed measured
    contact in the latest post-physics sensor sample.  The caller passes the
    live contact result rather than a phase-derived latch so a contact loss
    cannot leave an opening request authorised.

    Args:
        phase: Current hand-off phase per environment.
        receiver_grasp: Whether the receiver currently has bilateral contact.
        co_hold_phase: Integer value of the first phase permitting release.

    Returns:
        Boolean release permission per environment.
    """

    if phase.shape != receiver_grasp.shape:
        raise ValueError("phase and receiver_grasp must have identical batch shapes")
    if receiver_grasp.dtype is not torch.bool:
        raise ValueError("receiver_grasp must be a boolean tensor")
    if not isinstance(co_hold_phase, int):
        raise TypeError("co_hold_phase must be an integer phase value")
    return (phase >= co_hold_phase) & receiver_grasp


def donor_opening_requested(
    jaw_targets: torch.Tensor, hold_targets: torch.Tensor, aperture_threshold_rad: float
) -> torch.Tensor:
    """Identify any donor-jaw opening relative to the held grasp.

    The ordered dVRK joints have opposite signs.  A release therefore moves
    joint one negative or joint two positive *from the measured-compatible
    held target*.  The public ABI exposes two joint targets, so guarding only a
    simultaneous paired command would leave an unsafe one-jaw escape path.
    Deliberate further closing remains an ordinary actuator command.

    Args:
        jaw_targets: Ordered donor-jaw targets [rad], shape ``(N, 2)``.
        hold_targets: Ordered held-grasp targets [rad], shape ``(N, 2)``.
        aperture_threshold_rad: Minimum outward displacement treated as release [rad].

    Returns:
        Boolean release request per environment.
    """

    if jaw_targets.shape != hold_targets.shape or jaw_targets.ndim != 2 or jaw_targets.shape[1] != 2:
        raise ValueError("jaw_targets and hold_targets must both have shape (N, 2)")
    if not torch.isfinite(jaw_targets).all() or not torch.isfinite(hold_targets).all():
        raise ValueError("jaw targets must be finite")
    if not math.isfinite(aperture_threshold_rad) or aperture_threshold_rad < 0.0:
        raise ValueError("aperture threshold must be finite and non-negative")
    return _donor_opening_requested(jaw_targets, hold_targets, aperture_threshold_rad)


def _donor_opening_requested(
    jaw_targets: torch.Tensor, hold_targets: torch.Tensor, aperture_threshold_rad: float
) -> torch.Tensor:
    """Evaluate validated jaw targets without synchronising a CUDA stream."""

    return (jaw_targets[:, 0] < hold_targets[:, 0] - aperture_threshold_rad) | (
        jaw_targets[:, 1] > hold_targets[:, 1] + aperture_threshold_rad
    )


def world_pose_xyzw_to_root_pose_xyzw(
    pose_w_xyzw: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Convert absolute world poses from the public xyzw ABI to root-frame xyzw.

    Each row is converted against its matching live root transform.  The helper
    deliberately accepts batched tensors so differently placed cloned PSMs can
    never accidentally share one root transform.

    Args:
        pose_w_xyzw: World position [m] and xyzw quaternion, shape ``(N, 7)``.
        root_pos_w: Live articulation-root position [m], shape ``(N, 3)``.
        root_quat_w_xyzw: Live articulation-root xyzw quaternion, shape ``(N, 4)``.

    Returns:
        Root-frame position [m] and normalised xyzw quaternion, shape ``(N, 7)``.
    """

    if pose_w_xyzw.ndim != 2 or pose_w_xyzw.shape[1] != 7:
        raise ValueError("pose_w_xyzw must have shape (N, 7)")
    if root_pos_w.shape != pose_w_xyzw[:, :3].shape or root_quat_w_xyzw.shape != pose_w_xyzw[:, 3:].shape:
        raise ValueError("root transforms must match the batched world poses")
    if not torch.isfinite(pose_w_xyzw).all():
        raise ValueError("world-frame IK actions must be finite")

    target_quat_norm = torch.linalg.vector_norm(pose_w_xyzw[:, 3:7], dim=-1, keepdim=True)
    if torch.any(target_quat_norm <= 1.0e-9):
        raise ValueError("world-frame IK action quaternions must be normalisable")
    return _world_pose_xyzw_to_root_pose_xyzw(pose_w_xyzw, root_pos_w, root_quat_w_xyzw)


def _world_pose_xyzw_to_root_pose_xyzw(
    pose_w_xyzw: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Convert a validated world pose without synchronising a CUDA stream."""

    target_quat_xyzw = pose_w_xyzw[:, 3:7]
    target_quat_norm = torch.linalg.vector_norm(target_quat_xyzw, dim=-1, keepdim=True)
    target_quat_xyzw = target_quat_xyzw / target_quat_norm.clamp_min(1.0e-9)
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w_xyzw,
        pose_w_xyzw[:, :3],
        target_quat_xyzw,
    )
    return torch.cat((target_pos_b, target_quat_b), dim=-1)


class WorldFrameDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Absolute IK action that converts the live world target for every solve.

    Input is ``[x, y, z, qx, qy, qz, qw]`` in the shared world/XR frame.
    Immediately before each IK solve, the term reads this articulation's live
    root pose and supplies the controller with a root-frame xyzw target.  A
    command-level hold therefore keeps a target fixed; it does not disable the
    articulation's actuator drives or latch measured joint state.
    """

    cfg: WorldFrameDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: WorldFrameDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        if cfg.scale != 1.0:
            raise ValueError("world-frame absolute IK must use scale=1.0")
        if cfg.controller.command_type != "pose" or cfg.controller.use_relative_mode:
            raise ValueError("world-frame absolute IK requires an absolute pose controller")
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Cache the world target without applying a stale root transform."""

        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"expected world-frame IK actions with shape {tuple(self._raw_actions.shape)}")
        self._raw_actions[:] = actions
        processed_actions = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            processed_actions = torch.clamp(
                processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        quaternion_norm = torch.linalg.vector_norm(processed_actions[:, 3:7], dim=-1, keepdim=True)
        valid = torch.isfinite(processed_actions).all(dim=-1, keepdim=True) & (quaternion_norm > 1.0e-9)
        current_position_w, current_quaternion_w = self._compute_frame_pose_w()
        normalised_quaternion = torch.nan_to_num(
            processed_actions[:, 3:7] / quaternion_norm.clamp_min(1.0e-9),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._processed_actions[:, :3] = torch.where(valid, processed_actions[:, :3], current_position_w)
        self._processed_actions[:, 3:7] = torch.where(valid, normalised_quaternion, current_quaternion_w)

    def apply_actions(self) -> None:
        """Convert against the live root and solve the current articulation."""

        target_pose_b = _world_pose_xyzw_to_root_pose_xyzw(
            self._processed_actions,
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
        )
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        ee_quat_norm = torch.linalg.vector_norm(ee_quat_b, dim=-1, keepdim=True)
        ee_pose_valid = (
            torch.isfinite(ee_pos_b).all(dim=-1, keepdim=True)
            & torch.isfinite(ee_quat_b).all(dim=-1, keepdim=True)
            & (ee_quat_norm > 1.0e-9)
        )
        safe_ee_pos_b = torch.nan_to_num(ee_pos_b, nan=0.0, posinf=0.0, neginf=0.0)
        identity_quaternion = torch.zeros_like(ee_quat_b)
        identity_quaternion[:, 3] = 1.0
        safe_ee_quat_b = torch.where(
            ee_pose_valid,
            torch.nan_to_num(
                ee_quat_b / ee_quat_norm.clamp_min(1.0e-9),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            identity_quaternion,
        )
        self._ik_controller.set_command(target_pose_b, safe_ee_pos_b, safe_ee_quat_b)
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if not self._limits_injected and getattr(self.cfg.controller, "joint_limit_avoidance_gain", 0.0) > 0.0:
            limits = self._asset.data.soft_joint_pos_limits.torch[0, self._joint_ids, :]
            self._ik_controller.set_joint_pos_limits(limits[:, 0].clone(), limits[:, 1].clone())
            self._limits_injected = True
        computed_joint_pos = self._ik_controller.compute(
            safe_ee_pos_b,
            safe_ee_quat_b,
            self._compute_frame_jacobian(),
            joint_pos,
        )
        joint_pos_des = torch.where(ee_pose_valid, computed_joint_pos, joint_pos)
        self._asset.set_joint_position_target_index(target=joint_pos_des, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset selected commands to the live tool pose.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all environments.
        """

        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        current_position_w, current_quaternion_w = self._compute_frame_pose_w()
        self._processed_actions[selected, :3] = current_position_w[selected]
        self._processed_actions[selected, 3:7] = current_quaternion_w[selected]

    def _compute_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live controlled-frame pose directly in world coordinates."""

        frame_position_w = self._asset.data.body_pos_w.torch[:, self._body_idx]
        frame_quaternion_w = self._asset.data.body_quat_w.torch[:, self._body_idx]
        if self.cfg.body_offset is not None:
            frame_position_w, frame_quaternion_w = math_utils.combine_frame_transforms(
                frame_position_w,
                frame_quaternion_w,
                self._offset_pos,
                self._offset_rot,
            )
        return frame_position_w, frame_quaternion_w


class PairedJawJointPositionAction(JointPositionAction):
    """Ordered two-jaw position term with a start-up ABI assertion."""

    cfg: PairedJawJointPositionActionCfg

    def __init__(self, cfg: PairedJawJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if tuple(self._joint_names) != PSM_JAW_JOINT_ORDER:
            raise ValueError(
                f"dVRK jaw action must resolve exactly {list(PSM_JAW_JOINT_ORDER)}, got {self._joint_names}"
            )
        self._last_finite_target = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._last_command_finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply configured transforms and hold the last finite jaw target.

        Args:
            actions: Ordered paired-jaw position commands [rad], shape ``(num_envs, 2)``.
        """

        super().process_actions(actions)
        candidate_finite = torch.isfinite(self._processed_actions).all(dim=-1, keepdim=True)
        self._processed_actions = torch.where(candidate_finite, self._processed_actions, self._last_finite_target)
        self._last_finite_target[:] = self._processed_actions
        self._last_command_finite[:] = candidate_finite.squeeze(-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset selected jaw targets to the configured articulation defaults.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all environments.
        """

        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        default_target = self._asset.data.default_joint_pos.torch[selected][:, self._joint_ids]
        self._processed_actions[selected] = default_target
        self._last_finite_target[selected] = default_target
        self._last_command_finite[selected] = True


class DonorReleaseGuardedPairedJawJointPositionAction(PairedJawJointPositionAction):
    """Keep both donor jaws closed until the receiver has a measured co-hold.

    The interlock only suppresses an opening command.  It never creates a
    contact, changes the free needle, or advances the hand-off phase machine:
    those remain consequences of the measured PhysX state.  Once the previous
    post-physics sample has established ``CO_HOLD`` *and* the most recent
    receiver contact remains bilateral and opposed, the commanded donor jaw
    target passes through unchanged.  Before then, any outward movement of
    either jaw is clamped.  Losing that measured receiver grasp re-clamps the
    donor on the next control application.
    """

    cfg: DonorReleaseGuardedPairedJawJointPositionActionCfg

    def __init__(self, cfg: DonorReleaseGuardedPairedJawJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.phase_cfg is None:
            raise ValueError("donor release guard requires the shared hand-off phase configuration")
        if not math.isfinite(cfg.release_aperture_threshold_rad) or cfg.release_aperture_threshold_rad < 0.0:
            raise ValueError("donor release aperture threshold must be finite and non-negative")
        if len(cfg.hold_jaw_pos) != 2 or not all(math.isfinite(value) for value in cfg.hold_jaw_pos):
            raise ValueError("donor release guard requires two finite holding jaw positions")
        hold_target = torch.tensor(cfg.hold_jaw_pos, dtype=torch.float32, device=self.device)
        self._hold_target = hold_target.repeat(self.num_envs, 1)

    def apply_actions(self) -> None:
        # Import locally because this module is loaded before ``terminations``
        # by the public MDP namespace.
        from .terminations import HandoffPhase, get_handoff_phase_machine, jaw_needle_contact_measurements

        machine = get_handoff_phase_machine(self._env, self.cfg.phase_cfg)
        loads, normals, _ = jaw_needle_contact_measurements(self._env)
        receiver_grasp = machine._bilateral_contact(loads[:, 2:4], normals[:, 2:4], machine._receiver_engaged)
        release_requested = _donor_opening_requested(
            self.processed_actions,
            self._hold_target,
            self.cfg.release_aperture_threshold_rad,
        )
        release_allowed = donor_release_is_allowed(machine.phase, receiver_grasp, int(HandoffPhase.CO_HOLD))
        unsafe_command = ~self._last_command_finite | (release_requested & ~release_allowed)
        command = torch.where(
            unsafe_command.unsqueeze(-1),
            self._hold_target,
            self.processed_actions,
        )
        self._asset.set_joint_position_target_index(target=command, joint_ids=self._joint_ids)


__all__ = [
    "DonorReleaseGuardedPairedJawJointPositionAction",
    "PSM_JAW_JOINT_ORDER",
    "PairedJawJointPositionAction",
    "WorldFrameDifferentialInverseKinematicsAction",
    "donor_release_is_allowed",
    "donor_opening_requested",
    "world_pose_xyzw_to_root_pose_xyzw",
]
