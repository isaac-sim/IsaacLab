# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full-pose return control with joint, increment and live action-filter bounds.

This controller issues ordinary actions. A target inside a joint limit does not
guarantee physical clearance or prevent inertial overshoot; task gates are unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy.optimize import lsq_linear

from isaaclab.utils import math as math_utils

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
OPTIONS = {
    "algorithm": "bounded_full_pose_return_dls_v1",
    "scope": "milk carton measured return only",
    "damping": 0.01,
    "joint_limit_interior_margin_rad": 0.020,
    "maximum_joint_increment_rad": 0.012,
    "arm_alpha": 0.2,
    "arm_scale": 0.03,
    "raw_action_limit": 1.0,
    "joint_order": list(JOINT_NAMES),
    "task_error": "unweighted world TCP position [m] and shortest world hand rotation vector [rad]",
    "jacobian": "live Newton body_link_jacobian_w, shifted from hand origin to measured TCP",
    "bounds": "intersection of inset authored joint limits, +/-0.012 rad and 0.8*live_processed +/-0.006 rad",
    "infeasible_interior": "nearest reachable endpoint toward the inset; fixed coordinate before solving",
    "infeasible_increment": "maximum braking within EMA/raw bounds; report unavoidable increment violation",
    "solver": "scipy.optimize.lsq_linear, BVLS, tolerance 1e-12, maximum 100 iterations",
    "fixed_interval_width_rad": 1e-12,
    "physical_validation_passed": False,
}

DEPENDENCY_PATHS = tuple(
    sorted(
        {
            Path(importlib.import_module(name).__file__).resolve()
            for name in (
                "scipy.optimize._lsq.lsq_linear",
                "scipy.optimize._lsq.bvls",
                "scipy.optimize._lsq.common",
                "isaaclab.utils.math",
                "isaaclab.assets.articulation.base_articulation_data",
                "isaaclab_tasks.contrib.franka_pour.mdp.actions",
            )
        }
    )
)


def return_pose_controller_identity() -> dict:
    """Return declared controls and deterministic source bindings, without success claims."""
    return {
        "schema": "bounded_return_pose_controller_v1",
        "options": copy.deepcopy(OPTIONS),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), *DEPENDENCY_PATHS)
        },
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "physics_state_modified": False,
    }


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: require finite real array with shape {shape}.")
    return result.astype(np.float64, copy=False)


def bounded_return_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    joints: np.ndarray,
    limits: np.ndarray,
    previous_delta: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Solve one full-pose action step using joint positions/increments [rad].

    Args:
        jacobian: World TCP Jacobian [m/rad for translation], shape (6, 7).
        error: Unweighted position [m] and rotation-vector [rad] error, shape (6,).
        joints: Measured named joint positions [rad], shape (7,).
        limits: Authored simulation lower/upper joint limits [rad], shape (7, 2).
        previous_delta: Live filtered arm action [rad], shape (7,).

    Returns:
        Raw bounded arm actions and JSON-safe diagnostics. If existing EMA history
        makes the increment or inset infeasible, diagnostics identify the affected
        joints. No history is reset and no robot state is written.
    """
    jacobian = _finite_array(jacobian, (6, 7), "Jacobian")
    error = _finite_array(error, (6,), "pose error")
    joints = _finite_array(joints, (7,), "joint positions")
    limits = _finite_array(limits, (7, 2), "joint limits")
    previous_delta = _finite_array(previous_delta, (7,), "previous filtered delta")
    margin = OPTIONS["joint_limit_interior_margin_rad"]
    if np.any(limits[:, 1] - limits[:, 0] <= 2 * margin):
        raise ValueError("Joint limits must have a nonempty interior after the declared margin.")
    alpha, scale = OPTIONS["arm_alpha"], OPTIONS["arm_scale"]
    step_limit = OPTIONS["maximum_joint_increment_rad"]
    center, radius = (1 - alpha) * previous_delta, alpha * scale
    reachable_lower, reachable_upper = center - radius, center + radius
    lower, upper = np.maximum(reachable_lower, -step_limit), np.minimum(reachable_upper, step_limit)
    increment_unreachable = lower > upper
    # Old EMA can make even the increment cap impossible. Brake at the reachable
    # endpoint closest to zero rather than hiding a reset or clipping filtered state.
    braking = np.clip(np.zeros(7), reachable_lower, reachable_upper)
    lower = np.where(increment_unreachable, braking, lower)
    upper = np.where(increment_unreachable, braking, upper)
    interior_lower = limits[:, 0] + margin - joints
    interior_upper = limits[:, 1] - margin - joints
    bounded_lower, bounded_upper = np.maximum(lower, interior_lower), np.minimum(upper, interior_upper)
    interior_unreachable = bounded_lower > bounded_upper
    restoring = np.where(interior_lower > upper, upper, lower)
    lower = np.where(interior_unreachable, restoring, bounded_lower)
    upper = np.where(interior_unreachable, restoring, bounded_upper)
    fixed = upper - lower <= OPTIONS["fixed_interval_width_rad"]
    delta = np.zeros(7)
    delta[fixed] = (lower[fixed] + upper[fixed]) / 2
    free = ~fixed
    iterations = 0
    if free.any():
        damping = OPTIONS["damping"]
        matrix = np.vstack((jacobian[:, free], damping * np.eye(int(free.sum()))))
        target = np.r_[error - jacobian[:, fixed] @ delta[fixed], np.zeros(int(free.sum()))]
        result = lsq_linear(matrix, target, bounds=(lower[free], upper[free]), method="bvls", tol=1e-12, max_iter=100)
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Bounded return solve failed: {result.message}")
        delta[free] = result.x
        iterations = result.nit
    raw = (delta - center) / radius
    if np.any(np.abs(raw) > 1 + 1e-10):
        raise RuntimeError("Bounded return solution escaped its EMA reachable interval.")
    raw = np.clip(raw, -1, 1)
    effective = center + radius * raw
    diagnostics = {
        "algorithm": OPTIONS["algorithm"],
        "position_error_m": float(np.linalg.norm(error[:3])),
        "rotation_error_rad": float(np.linalg.norm(error[3:])),
        "effective_delta_rad": effective.tolist(),
        "raw_arm_action": raw.tolist(),
        "effective_lower_rad": lower.tolist(),
        "effective_upper_rad": upper.tolist(),
        "joint_interior_unreachable": interior_unreachable.tolist(),
        "increment_unreachable_due_to_ema": increment_unreachable.tolist(),
        "restoration_or_braking_active": bool(interior_unreachable.any() or increment_unreachable.any()),
        "fixed_variables": fixed.tolist(),
        "linear_residual_norm": float(np.linalg.norm(jacobian @ effective - error)),
        "target_minimum_authored_joint_margin_rad": float(
            np.minimum(joints + effective - limits[:, 0], limits[:, 1] - joints - effective).min()
        ),
        "solver_iterations": int(iterations),
    }
    return raw, diagnostics


class BoundedReturnPoseController:
    """Issue ordinary filtered joint actions for a full TCP/hand return target."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.joint_ids, names = env.robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES or len(set(self.joint_ids)) != 7:
            raise ValueError("Return control requires each literal panda_joint1..7 in named order.")
        term = env.action_manager.get_term("arm_action")
        if tuple(term._joint_names) != JOINT_NAMES:
            raise ValueError("Arm action columns must match the literal named joint order.")
        self._check_action_contract()
        self.identity = return_pose_controller_identity()
        self.diagnostics: dict = {}

    def _check_action_contract(self) -> None:
        cfg = self.env.cfg.actions.arm_action
        if cfg.alpha != OPTIONS["arm_alpha"] or cfg.scale != OPTIONS["arm_scale"]:
            raise ValueError("Bounded return requires the unchanged alpha=0.2 and scale=0.03 arm action.")

    def compute(self, position: torch.Tensor, rotation: torch.Tensor, close: bool | torch.Tensor) -> torch.Tensor:
        """Map world TCP position [m] and XYZW hand orientation into raw bounded actions."""
        self._check_action_contract()
        env = self.env
        n = env.num_envs
        for value, shape, name in ((position, (n, 3), "position"), (rotation, (n, 4), "rotation")):
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.shape != shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid return {name}: require finite tensor {shape}.")
        if torch.any(torch.linalg.vector_norm(rotation, dim=-1) < 1e-12):
            raise ValueError("Return rotation quaternion must be nonzero.")
        closing = torch.as_tensor(close, device=env.device)
        if closing.dtype != torch.bool or closing.shape not in (torch.Size([]), torch.Size([n])):
            raise ValueError("Gripper command must be a bool or one bool per environment.")
        hand = env.robot.data.body_link_pose_w.torch[:, env.hand_id]
        tcp = env.tcp()
        if (
            hand.shape != (n, 7)
            or tcp.shape != (n, 3)
            or not torch.isfinite(hand).all()
            or not torch.isfinite(tcp).all()
            or torch.any(torch.linalg.vector_norm(hand[:, 3:], dim=-1) < 1e-12)
        ):
            raise ValueError("Return controller requires a finite measured hand pose and TCP.")
        jacobian = env.robot.data.body_link_jacobian_w.torch[:, env.hand_id - 1, :, self.joint_ids].clone()
        jacobian[:, :3] -= torch.bmm(math_utils.skew_symmetric_matrix(tcp - hand[:, :3]), jacobian[:, 3:])
        target_rotation = rotation / torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
        position_error, rotation_error = math_utils.compute_pose_error(
            tcp, hand[:, 3:], position, target_rotation, rot_error_type="axis_angle"
        )
        error = torch.cat((position_error, rotation_error), dim=-1)
        joints = env.robot.data.joint_pos.torch[:, self.joint_ids]
        limits = env.robot.data.joint_pos_limits.torch[:, self.joint_ids]
        previous = env.action_manager.get_term("arm_action").processed_actions
        arrays = [value.detach().cpu().numpy() for value in (jacobian, error, joints, limits, previous)]
        expected = [(n, 6, 7), (n, 6), (n, 7), (n, 7, 2), (n, 7)]
        for value, shape in zip(arrays, expected):
            if value.shape != shape:
                raise ValueError(f"Unexpected return-controller state shape: {value.shape}, require {shape}.")
        results = [bounded_return_step(*(value[index] for value in arrays)) for index in range(n)]
        actions = torch.zeros((n, 8), device=env.device, dtype=joints.dtype)
        actions[:, :7] = torch.as_tensor(np.stack([row[0] for row in results]), device=env.device, dtype=joints.dtype)
        actions[:, -1] = torch.where(closing, -1.0, 1.0)
        self.diagnostics = {"controller": OPTIONS["algorithm"], "per_environment": [row[1] for row in results]}
        return actions
