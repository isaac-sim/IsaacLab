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
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

PSM_JAW_JOINT_ORDER = ("psm_tool_gripper1_joint", "psm_tool_gripper2_joint")


def donor_release_is_allowed(phase: torch.Tensor, receiver_grasp: torch.Tensor, co_hold_phase: int) -> torch.Tensor:
    """Return whether a donor opening may proceed in each environment.

    A completed co-hold dwell is necessary but deliberately insufficient on
    its own: the receiver must still have a bilateral, opposed measured
    contact in the latest post-physics sensor sample.  The caller passes the
    live contact result rather than a phase-derived latch so a contact loss
    cannot leave an opening request authorised.
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
    """

    if jaw_targets.shape != hold_targets.shape or jaw_targets.ndim != 2 or jaw_targets.shape[1] != 2:
        raise ValueError("jaw_targets and hold_targets must both have shape (N, 2)")
    if not torch.isfinite(jaw_targets).all() or not torch.isfinite(hold_targets).all():
        raise ValueError("jaw targets must be finite")
    if not math.isfinite(aperture_threshold_rad) or aperture_threshold_rad < 0.0:
        raise ValueError("aperture threshold must be finite and non-negative")
    return (jaw_targets[:, 0] < hold_targets[:, 0] - aperture_threshold_rad) | (
        jaw_targets[:, 1] > hold_targets[:, 1] + aperture_threshold_rad
    )


def world_pose_xyzw_to_root_pose_wxyz(
    pose_w_xyzw: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Convert absolute world poses from the public xyzw ABI to root-frame wxyz.

    Each row is converted against its matching live root transform.  The helper
    deliberately accepts batched tensors so differently placed cloned PSMs can
    never accidentally share one root transform.
    """

    if pose_w_xyzw.ndim != 2 or pose_w_xyzw.shape[1] != 7:
        raise ValueError("pose_w_xyzw must have shape (N, 7)")
    if root_pos_w.shape != pose_w_xyzw[:, :3].shape or root_quat_w_wxyz.shape != pose_w_xyzw[:, 3:].shape:
        raise ValueError("root transforms must match the batched world poses")
    if not torch.isfinite(pose_w_xyzw).all():
        raise ValueError("world-frame IK actions must be finite")

    target_quat_xyzw = pose_w_xyzw[:, 3:7]
    target_quat_norm = torch.linalg.vector_norm(target_quat_xyzw, dim=-1, keepdim=True)
    if torch.any(target_quat_norm <= 1.0e-9):
        raise ValueError("world-frame IK action quaternions must be normalisable")
    target_quat_xyzw = target_quat_xyzw / target_quat_norm
    target_quat_wxyz = torch.cat((target_quat_xyzw[:, 3:4], target_quat_xyzw[:, 0:3]), dim=-1)
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w_wxyz,
        pose_w_xyzw[:, :3],
        target_quat_wxyz,
    )
    return torch.cat((target_pos_b, target_quat_b), dim=-1)


class WorldFrameDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Absolute IK action that converts the live world target for every solve.

    Input is ``[x, y, z, qx, qy, qz, qw]`` in the shared world/XR frame.
    Immediately before each IK solve, the term reads this articulation's live
    root pose and supplies the controller with a root-frame wxyz target.  A
    command-level hold therefore keeps a target fixed; it does not disable the
    articulation's actuator drives or latch measured joint state.
    """

    cfg: WorldFrameDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: WorldFrameDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        if cfg.scale != 1.0:
            raise ValueError("world-frame absolute IK must use scale=1.0")
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Cache the world target without applying a stale root transform."""

        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"expected world-frame IK actions with shape {tuple(self._raw_actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("world-frame IK actions must be finite")
        quaternion_norm = torch.linalg.vector_norm(actions[:, 3:7], dim=-1, keepdim=True)
        if torch.any(quaternion_norm <= 1.0e-9):
            raise ValueError("world-frame IK action quaternions must be normalisable")
        self._raw_actions[:] = actions
        self._processed_actions[:, :3] = actions[:, :3]
        self._processed_actions[:, 3:7] = actions[:, 3:7] / quaternion_norm

    def apply_actions(self) -> None:
        """Convert against the live root and solve the current articulation."""

        target_pose_b = world_pose_xyzw_to_root_pose_wxyz(
            self._processed_actions,
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
        )
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        self._ik_controller.set_command(target_pose_b, ee_pos_b, ee_quat_b)
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if torch.linalg.vector_norm(ee_quat_b, dim=-1).gt(0.0).all():
            joint_pos_des = self._ik_controller.compute(
                ee_pos_b,
                ee_quat_b,
                self._compute_frame_jacobian(),
                joint_pos,
            )
        else:
            joint_pos_des = joint_pos.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear cached raw values for the selected environments."""

        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0


class PairedJawJointPositionAction(JointPositionAction):
    """Ordered two-jaw position term with a start-up ABI assertion."""

    cfg: PairedJawJointPositionActionCfg

    def __init__(self, cfg: PairedJawJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if tuple(self._joint_names) != PSM_JAW_JOINT_ORDER:
            raise ValueError(
                f"dVRK jaw action must resolve exactly {list(PSM_JAW_JOINT_ORDER)}, got {self._joint_names}"
            )


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
        hold_target = torch.tensor(cfg.hold_jaw_pos, dtype=torch.float32, device=self.device)
        if hold_target.shape != (2,) or not torch.isfinite(hold_target).all():
            raise ValueError("donor release guard requires two finite holding jaw positions")
        self._hold_target = hold_target.repeat(self.num_envs, 1)

    def apply_actions(self) -> None:
        # Import locally because this module is loaded before ``terminations``
        # by the public MDP namespace.
        from .terminations import HandoffPhase, get_handoff_phase_machine, jaw_needle_contact_measurements

        machine = get_handoff_phase_machine(self._env, self.cfg.phase_cfg)
        loads, normals, _ = jaw_needle_contact_measurements(self._env)
        receiver_grasp = machine._bilateral_contact(loads[:, 2:4], normals[:, 2:4], machine._receiver_engaged)
        release_requested = donor_opening_requested(
            self.processed_actions,
            self._hold_target,
            self.cfg.release_aperture_threshold_rad,
        )
        release_allowed = donor_release_is_allowed(machine.phase, receiver_grasp, int(HandoffPhase.CO_HOLD))
        command = torch.where(
            (release_requested & ~release_allowed).unsqueeze(-1), self._hold_target, self.processed_actions
        )
        self._asset.set_joint_position_target(command, joint_ids=self._joint_ids)


@configclass
class WorldFrameDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for live world-to-root absolute differential IK."""

    class_type: type[ActionTerm] = WorldFrameDifferentialInverseKinematicsAction


@configclass
class PairedJawJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the exact ordered paired-jaw action."""

    class_type: type[ActionTerm] = PairedJawJointPositionAction


@configclass
class DonorReleaseGuardedPairedJawJointPositionActionCfg(PairedJawJointPositionActionCfg):
    """Exact paired donor jaws with a measured receiver-grasp release interlock."""

    class_type: type[ActionTerm] = DonorReleaseGuardedPairedJawJointPositionAction
    phase_cfg: object | None = None
    release_aperture_threshold_rad: float = 0.0
    hold_jaw_pos: tuple[float, float] = (0.0, 0.0)


__all__ = [
    "DonorReleaseGuardedPairedJawJointPositionAction",
    "DonorReleaseGuardedPairedJawJointPositionActionCfg",
    "PSM_JAW_JOINT_ORDER",
    "PairedJawJointPositionAction",
    "PairedJawJointPositionActionCfg",
    "WorldFrameDifferentialInverseKinematicsAction",
    "WorldFrameDifferentialInverseKinematicsActionCfg",
    "donor_release_is_allowed",
    "donor_opening_requested",
    "world_pose_xyzw_to_root_pose_wxyz",
]
