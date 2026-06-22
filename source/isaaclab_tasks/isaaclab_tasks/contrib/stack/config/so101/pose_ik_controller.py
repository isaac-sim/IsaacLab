# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full-SE3-pose differential IK for the SO-101 over all 5 arm joints.

The SO-101 is a 5-DOF arm; a full 6-DOF pose target over its 5 arm joints (``shoulder_pan``,
``shoulder_lift``, ``elbow_flex``, ``wrist_flex``, ``wrist_roll``) is over-determined by one
DOF. This controller solves a 6-row task ``[linear_xyz, orientation_xyz]``: the 3 linear rows
keep weight 1 (position tracks exactly) while the 3 orientation rows are soft-weighted by
:attr:`SO101PoseIKControllerCfg.orientation_task_weight` (scalar or per-base-axis) so the
unreachable orientation DOF degrades gracefully rather than leaking error into position. Zeroing
the base-Z (yaw) weight drops the one DOF that over-constrains the arm (heading, served only by
``shoulder_pan``), making the task an exactly-determined 5-row [pos + tilt] solve. The position of
the IK body (``gripper_link``, on the ``wrist_roll`` axis) is roll-invariant, so position stays
served by the same 4 joints and well-conditioned; the over-determination bites only on
orientation.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.configclass import configclass


@configclass
class SO101PoseIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for the SO-101 full-pose SE3 IK controller."""

    orientation_task_weight: float | tuple[float, float, float] = 0.5
    """Soft weight on the 3 orientation task rows (the 3 linear rows stay weight 1). A scalar
    weights all rows equally; a per-axis ``(wx, wy, wz)`` weights the **base-frame** orientation
    axes, where ``wz`` is rotation about base +Z -- the gripper yaw, which only ``shoulder_pan``
    can serve and which over-constrains the 5-DOF arm. Set ``wz = 0`` (e.g. ``(0.3, 0.3, 0.0)``)
    to drop yaw, leaving 3 position + 2 tilt rows = exactly 5 matched to the arm; ``0`` recovers
    position-only IK.

    Prefer :attr:`orientation_joint_names` over ``wz = 0`` when the goal is "keep the base off
    orientation": dropping ``wz`` removes the spin-about-vertical DOF for *every* joint (so
    ``wrist_roll`` can no longer follow it either), whereas restricting orientation to the wrist
    keeps that DOF and simply routes it to ``wrist_roll``."""

    orientation_joint_names: tuple[str, ...] | None = None
    """Names of the joints permitted to serve the orientation task rows. When set, every other
    joint's orientation-Jacobian columns are zeroed, so those joints serve **position only** while
    orientation is solved purely by the listed joints (position still uses all joints). ``None``
    (default) lets all joints serve orientation.

    For the SO-101's down-pointing gripper this is set to the wrist joints
    ``("wrist_flex", "wrist_roll")``: ``wrist_roll`` then takes the gripper spin about the
    (vertical) approach axis -- the DOF otherwise redundant with ``shoulder_pan`` -- and
    ``wrist_flex`` takes the tilt, leaving ``shoulder_pan`` free to serve position (heading) so the
    base never swings to satisfy a commanded orientation. The action term resolves these names to
    Jacobian columns (asset-ordered, so the mask is order-proof) and pushes the column mask via
    :meth:`SO101PoseIKController.set_orientation_joint_mask`."""

    lambda_min: float = 0.05
    """Baseline damped-least-squares damping coefficient [matched to the weighted task-Jacobian
    singular-value scale] used away from singularities."""

    lambda_max: float = 0.2
    """Maximum DLS damping coefficient [same scale as :attr:`lambda_min`], reached as the
    smallest weighted task-Jacobian singular value -> 0."""

    sigma_thresh: float = 0.02
    """Smallest-singular-value threshold [the weighted task-Jacobian singular-value scale] below
    which damping ramps from ``lambda_min`` toward ``lambda_max`` (Maciejewski-Klein style); see
    :meth:`SO101PoseIKController._damped_least_squares` for the singular value it keys off."""

    jla_gain: float = 0.0
    """Gain for the null-space joint-limit-avoidance bias. ``0`` disables it (default); the env
    cfg enables it with a tuned value. Active only once joint limits are injected via
    :meth:`SO101PoseIKController.set_joint_pos_limits`."""

    jla_margin: float = 0.3
    """Joint-range margin [rad] within which the avoidance bias activates (1 at the limit,
    ramping to 0 at ``jla_margin`` away from it)."""

    def __post_init__(self):
        super().__post_init__()
        if self.sigma_thresh <= 0.0:
            raise ValueError(f"sigma_thresh must be > 0, got {self.sigma_thresh}.")
        if self.lambda_min > self.lambda_max:
            raise ValueError(f"lambda_min ({self.lambda_min}) must be <= lambda_max ({self.lambda_max}).")
        if self.jla_gain < 0.0:
            raise ValueError(f"jla_gain must be >= 0, got {self.jla_gain}.")
        if self.jla_margin <= 0.0:
            raise ValueError(f"jla_margin must be > 0, got {self.jla_margin}.")
        if not isinstance(self.orientation_task_weight, (int, float)) and len(self.orientation_task_weight) != 3:
            raise ValueError(
                "orientation_task_weight must be a scalar or a length-3 (wx, wy, wz) tuple, got "
                f"{self.orientation_task_weight}."
            )


class SO101PoseIKController(DifferentialIKController):
    """Differential IK over a 6-row SE3 pose task: 3 linear rows (weight 1) + 3 soft orientation rows.

    Uses a manipulability-aware DLS solver (:meth:`_damped_least_squares`). Task assembly is
    :meth:`_assemble_task`; the 7D command split is :meth:`set_command`.

    Orientation tracking is **absolute** (there is no engage clutch on orientation), so a
    Stop -> reposition -> Play snaps the gripper orientation to the controller; this is a
    known/deferred item, not a bug.

    .. note::
        The quaternion convention throughout this class is **xyzw** (scalar-last), matching the
        IsaacLab asset convention (``body_quat_w``), the expectation of
        :func:`isaaclab.utils.math.compute_pose_error` / :func:`~isaaclab.utils.math.quat_apply`,
        and the commanded quaternion in :meth:`set_command`. No conversion is applied.
    """

    cfg: SO101PoseIKControllerCfg

    def __init__(self, cfg: SO101PoseIKControllerCfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        # Per-axis orientation weights over the base-frame (x, y, z) rows; a scalar broadcasts to
        # all three. wz (about base +Z) is the gripper yaw/heading -- set it to 0 to drop yaw.
        w = cfg.orientation_task_weight
        w_tuple = (float(w),) * 3 if isinstance(w, (int, float)) else tuple(float(x) for x in w)
        self._ori_weight = torch.tensor(w_tuple, device=device)  # (3,)
        # Joint position limits for null-space limit avoidance; injected by the action term.
        self._joint_pos_lower: torch.Tensor | None = None
        self._joint_pos_upper: torch.Tensor | None = None
        # Column mask (1 = joint may serve the orientation rows) over the IK joints, pushed by the
        # action term once it has resolved ``orientation_joint_names`` to Jacobian columns. ``None``
        # leaves all joints free to serve orientation (the default).
        self._ori_joint_mask: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        """Dimension of the command: 7 = ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]``
        (position + xyzw orientation)."""
        return 7

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set the 7D pose command ``[pos_xyz, quat_xyzw]``.

        This override owns the (N, 7) command contract and the renormalization seam: it stores
        the position target and the **renormalized** commanded orientation (the upstream clutch
        may emit a non-unit quaternion). It deliberately does not call ``super().set_command()``
        so the renormalization and the readable 7-vector split stay local; do not simplify it
        back to inherit the base pose path.

        Args:
            command: Target command in shape (N, 7): ``[pos_x, pos_y, pos_z, quat_x, quat_y,
                quat_z, quat_w]``. The position slot ``command[:, 0:3]`` is in [m], base frame;
                the orientation slot ``command[:, 3:7]`` is xyzw (scalar-last). The quaternion
                need not be unit; it is renormalized here.
            ee_pos: Current end-effector position in shape (N, 3). Unused but accepted for API
                compatibility with the base class.
            ee_quat: Current end-effector orientation in xyzw shape (N, 4). Unused but accepted
                for API compatibility with the base class.
        """
        self._command[:] = command
        self.ee_pos_des[:] = command[:, 0:3]
        quat = command[:, 3:7]
        self.ee_quat_des[:] = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)

    def _assemble_task(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the 6-row SE3 pose task Jacobian and the 6-vector error.

        The orientation error is the axis-angle of ``q_des * q_cur^-1`` (base-frame,
        left-multiplicative, xyzw) from :func:`isaaclab.utils.math.compute_pose_error`. The base
        term rotates BOTH the linear and angular Jacobian rows into the base frame, so this
        base-frame axis-angle error is frame-consistent with ``jacobian[:, 3:6, :]``. The 3
        orientation rows and the orientation error are scaled per base-frame axis by
        :attr:`SO101PoseIKControllerCfg.orientation_task_weight` (a weight of 0 drops that axis).

        Args:
            ee_pos: Current end-effector position in shape (N, 3).
            ee_quat: Current end-effector orientation in xyzw shape (N, 4).
            jacobian: Full geometric Jacobian in shape (N, 6, num_joints). Rows 0-2 are linear
                velocity; rows 3-5 are angular velocity (both base-frame).

        Returns:
            A 2-tuple ``(J_task, err)``:

            - ``J_task``: Weighted task Jacobian of shape (N, 6, num_joints).
            - ``err``: Task-space error of shape (N, 6): 3 linear + 3 (weighted) orientation.
        """
        # pos_err = ee_pos_des - ee_pos; ori_err = axis-angle of q_des * q_cur^-1
        # (base frame, left-multiplicative, xyzw).
        pos_err, ori_err = math_utils.compute_pose_error(
            ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
        )  # (N,3), (N,3)

        J_pos = jacobian[:, 0:3, :]  # (N,3,M)
        J_ori = jacobian[:, 3:6, :]  # (N,3,M)
        # Restrict orientation to the allowed joints (e.g. the wrist) when a mask is set: zeroing the
        # other joints' orientation columns means only the allowed joints can reduce the orientation
        # error, so the base (shoulder_pan) serves position only and the redundant spin-about-vertical
        # DOF is routed to wrist_roll. Position rows (J_pos) keep every joint.
        if self._ori_joint_mask is not None:
            J_ori = J_ori * self._ori_joint_mask.view(1, 1, -1)
        # Soft-weight the orientation task per base-frame axis: scaling each orientation row and its
        # error by the same weight de-emphasises (or drops, at weight 0) that rotation DOF in the
        # DLS cost without changing the task dimensionality. wz=0 drops yaw (see the cfg field).
        w = self._ori_weight  # (3,)
        J_task = torch.cat([J_pos, J_ori * w.view(1, 3, 1)], dim=1)  # (N,6,M)
        err = torch.cat([pos_err, ori_err * w.view(1, 3)], dim=1)  # (N,6)
        return J_task, err

    def _adaptive_lambda_sq(self, sigma_min: torch.Tensor) -> torch.Tensor:
        """Manipulability-aware squared DLS damping per environment.

        Returns ``lambda_min**2`` where ``sigma_min >= sigma_thresh`` and ramps quadratically to
        ``lambda_max**2`` as ``sigma_min -> 0``.

        Args:
            sigma_min: Smallest singular value of the weighted task Jacobian, shape (N,).

        Returns:
            Squared damping coefficient per environment, shape (N,).
        """
        lo2 = self.cfg.lambda_min**2
        hi2 = self.cfg.lambda_max**2
        ratio = (sigma_min / self.cfg.sigma_thresh).clamp(max=1.0)
        return lo2 + (1.0 - ratio**2) * (hi2 - lo2)

    def _damped_least_squares(self, J_task: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """Solve the damped least-squares step with manipulability-aware damping.

        Computes ``dq = J^T (J J^T + lambda^2 I)^-1 e`` over the full 6-row weighted ``J_task``,
        where ``lambda`` follows :meth:`_adaptive_lambda_sq` from the smallest singular value of
        ``J_task`` itself. Keying the ramp off the full weighted Jacobian (Maciejewski-Klein
        style) damps BOTH position and orientation singularities -- a 5-DOF arm tracking a 6-DOF
        pose hits orientation-rank-loss configs the position block alone cannot see. The ramp is
        therefore coupled to :attr:`SO101PoseIKControllerCfg.orientation_task_weight`: re-tune
        ``sigma_thresh`` if the weight changes substantially, and avoid weight ~0 (the vanishing
        orientation rows would drive ``sigma_min -> 0`` and over-damp).

        Args:
            J_task: Weighted task Jacobian, shape (N, T, M) with T task rows over M joints.
            err: Task-space error, shape (N, T).

        Returns:
            Joint-space delta, shape (N, M).
        """
        sigma_min = torch.linalg.svdvals(J_task)[:, -1]  # (N,)
        lam2 = self._adaptive_lambda_sq(sigma_min)  # (N,)
        jt = J_task.transpose(1, 2)  # (N, M, T)
        n_task = J_task.shape[1]
        a = torch.bmm(J_task, jt) + lam2.view(-1, 1, 1) * torch.eye(n_task, device=self._device)  # (N, T, T)
        return torch.bmm(jt, torch.linalg.solve(a, err.unsqueeze(-1))).squeeze(-1)  # (N, M)

    def set_joint_pos_limits(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Provide the IK joints' position limits [rad] for null-space limit avoidance.

        Args:
            lower: Lower joint limits [rad], shape (M,).
            upper: Upper joint limits [rad], shape (M,).
        """
        self._joint_pos_lower = lower.to(self._device)
        self._joint_pos_upper = upper.to(self._device)

    def set_orientation_joint_mask(self, mask: torch.Tensor) -> None:
        """Restrict which joints serve the orientation task rows.

        Args:
            mask: Per-IK-joint multiplier [dimensionless], shape (M,), in the controller's joint
                (Jacobian-column) order: 1 for joints allowed to serve orientation, 0 for joints
                that should serve position only. Applied to the orientation rows of the task
                Jacobian in :meth:`_assemble_task`. See
                :attr:`SO101PoseIKControllerCfg.orientation_joint_names`.
        """
        self._ori_joint_mask = mask.to(self._device)

    def _joint_limit_avoidance(self, joint_pos: torch.Tensor, J_task: torch.Tensor) -> torch.Tensor:
        """Null-space joint-centering bias that keeps joints off their limits.

        Projects a center-seeking velocity (active only within :attr:`jla_margin` of a limit)
        into the null space of the position task rows, so it never perturbs the commanded EE
        position. Returns zeros when disabled or before limits are injected.

        Args:
            joint_pos: Current joint positions [rad], shape (N, M).
            J_task: Task Jacobian, shape (N, T, M); rows 0-2 are the position rows.

        Returns:
            Joint-space correction [rad], shape (N, M).
        """
        if self.cfg.jla_gain <= 0.0 or self._joint_pos_lower is None:
            return torch.zeros_like(joint_pos)
        lower, upper = self._joint_pos_lower, self._joint_pos_upper
        q_mid = 0.5 * (lower + upper)
        dist = torch.minimum(joint_pos - lower, upper - joint_pos)  # (N,M) margin to nearest limit
        activation = 1.0 - (dist / self.cfg.jla_margin).clamp(0.0, 1.0)  # 1 at limit, 0 mid-range
        dq_center = -self.cfg.jla_gain * activation * (joint_pos - q_mid)  # (N,M) toward center
        j_pos = J_task[:, :3, :]  # (N,3,M)
        j_pos_pinv = torch.linalg.pinv(j_pos)  # (N,M,3)
        m = J_task.shape[2]
        null_proj = torch.eye(m, device=self._device) - torch.bmm(j_pos_pinv, j_pos)  # (N,M,M)
        return torch.bmm(null_proj, dq_center.unsqueeze(-1)).squeeze(-1)

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target joint positions from the current state and stored command.

        Args:
            ee_pos: Current end-effector position in shape (N, 3).
            ee_quat: Current end-effector orientation in xyzw shape (N, 4).
            jacobian: Full geometric Jacobian in shape (N, 6, num_joints).
            joint_pos: Current joint positions in shape (N, num_joints).

        Returns:
            Target joint positions in shape (N, num_joints).
        """
        J_task, err = self._assemble_task(ee_pos, ee_quat, jacobian)
        delta = self._damped_least_squares(J_task, err)
        delta = delta + self._joint_limit_avoidance(joint_pos, J_task)
        return joint_pos + delta
