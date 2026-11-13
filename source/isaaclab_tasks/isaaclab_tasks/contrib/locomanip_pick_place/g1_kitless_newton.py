# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 action and actuator boundaries for kitless Newton teleoperation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import ManagerBasedRLEnv

    from .fixed_base_upper_body_ik_g1_env_cfg import FixedBaseUpperBodyIKG1EnvCfg

POSITION_MAX_STEP_M = 0.10
"""Maximum Cartesian wrist-target change accepted in one active frame."""

ROTATION_MAX_STEP_RAD = 0.60
"""Maximum wrist-target rotation accepted in one active frame."""

JOINT_TARGET_MAX_STEP_RAD = 0.10
"""Maximum Pink joint-target change accepted in one active frame."""

JOINT_LIMIT_MARGIN_RAD = 1.0e-3
"""Margin retained inside the common Pink/articulation joint limits."""

EXPECTED_ACTION_DIM = 28
"""Two 7-D wrist poses followed by fourteen G1 hand targets."""

WRIST_QUATERNION_NORM_BOUNDS = (0.25, 2.0)
"""Reject malformed wrist rotations before the conditioner normalizes them."""

HAND_TARGET_MAX_MAGNITUDE_RAD = 6.5
"""Reject hand joint targets outside the task's accepted angular input range [rad]."""

_WRIST_RAMP_KEYS = ("left_pos", "left_quat", "right_pos", "right_quat")

ACTUATOR_PROFILE = {
    "waist": {"stiffness": 500.0, "damping": 50.0, "armature": 0.10, "joint_effort_limit": 250.0},
    "arms": {"stiffness": 1200.0, "damping": 60.0, "armature": 0.30, "joint_effort_limit": 800.0},
    "hands": {"stiffness": 5.0, "damping": 0.5, "armature": 0.02, "joint_effort_limit": 20.0},
}
"""Newton implicit-drive settings for G1 joint groups."""


def teleop_action_target_issue(action: torch.Tensor) -> str | None:
    """Return why an incoming G1 controller action must not reach Pink/Newton.

    Args:
        action: One G1 action of :data:`EXPECTED_ACTION_DIM` elements, in any shape that flattens to
            it: left wrist position [m] and quaternion (x, y, z, w), right wrist position [m] and
            quaternion, then fourteen hand joint targets [rad].

    Returns:
        A description of the first problem found, or ``None`` when the action is acceptable.
    """
    flat = action.detach().flatten()
    if flat.numel() != EXPECTED_ACTION_DIM:
        return f"unexpected G1 controller action size {flat.numel()} (expected {EXPECTED_ACTION_DIM})"
    wrists = flat[:14].reshape(2, 7)
    quaternion_norms = torch.linalg.norm(wrists[:, 3:7], dim=1)
    min_norm, max_norm = WRIST_QUATERNION_NORM_BOUNDS
    invalid_quaternion = ((quaternion_norms < min_norm) | (quaternion_norms > max_norm)).any()
    invalid_hand = (torch.abs(flat[14:]) > HAND_TARGET_MAX_MAGNITUDE_RAD).any()
    # CPU flags are already local; accelerator flags share one host transfer.
    flags = (torch.isfinite(flat).all(), invalid_quaternion, invalid_hand)
    finite, invalid_quaternion, invalid_hand = (
        tuple(bool(flag) for flag in flags) if flat.device.type == "cpu" else torch.stack(flags).cpu().tolist()
    )
    if not finite:
        return "non-finite action target"
    if invalid_quaternion:
        return f"invalid wrist quaternion norms {quaternion_norms.detach().cpu().tolist()}"
    if invalid_hand:
        return "unreasonable hand joint target magnitude"
    return None


def bound_pink_joint_targets(
    current: torch.Tensor,
    proposed: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Keep one Pink update finite, inside limits, and close to current state.

    Targets are first limited to :data:`JOINT_TARGET_MAX_STEP_RAD` from ``current``, then
    clamped :data:`JOINT_LIMIT_MARGIN_RAD` inside ``[lower, upper]``. The joint limits take
    precedence: when ``current`` lies outside those limits, the final change can exceed
    the step bound.

    Args:
        current: Current joint positions [rad]. Shape is (num_envs, num_joints), in the Pink action's
            Isaac Lab controlled-joint order.
        proposed: Pink joint targets [rad], with the same shape and joint order.
        lower: Lower joint limits [rad], with the same shape and joint order.
        upper: Upper joint limits [rad], with the same shape and joint order.

    Returns:
        The bounded targets [rad], with the same shape and joint order.

    Raises:
        RuntimeError: If any input element is non-finite, or the margin-reduced limits of any joint do
            not overlap. The check covers the whole batch, so no environment's targets are returned.
    """
    safe_lower = lower + JOINT_LIMIT_MARGIN_RAD
    safe_upper = upper - JOINT_LIMIT_MARGIN_RAD
    flags = (
        torch.isfinite(current).all(),
        torch.isfinite(proposed).all(),
        torch.isfinite(lower).all(),
        torch.isfinite(upper).all(),
        (safe_lower > safe_upper).any(),
    )
    *finite, empty_intersection = (
        tuple(bool(flag) for flag in flags) if current.device.type == "cpu" else torch.stack(flags).cpu().tolist()
    )
    if not all(finite):
        raise RuntimeError("non-finite Pink joint target boundary")
    if empty_intersection:
        raise RuntimeError("Pink/articulation joint-limit intersection is empty")

    stepped = torch.clamp(proposed, min=current - JOINT_TARGET_MAX_STEP_RAD, max=current + JOINT_TARGET_MAX_STEP_RAD)
    return torch.maximum(torch.minimum(stepped, safe_upper), safe_lower)


class G1NewtonBoundedPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Bound Pink outputs without changing G1's public 28-D action layout."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._pink_joint_limits = None

    def _compute_ik_solutions(self) -> torch.Tensor:
        proposed = super()._compute_ik_solutions()
        current = self._asset.data.joint_pos.torch[:, self._isaaclab_controlled_joint_ids]
        articulation_limits = self._asset.data.joint_pos_limits.torch[:, self._isaaclab_controlled_joint_ids, :]

        # Pink's controlled model/order are immutable for this action's lifetime.
        # Keep articulation limits live: users may update them during a run.
        if self._pink_joint_limits is None:
            self._pink_joint_limits = tuple(
                torch.as_tensor(
                    np.stack(
                        [
                            np.asarray(getattr(controller.pink_configuration.controlled_model, limit_name))[
                                controller.pink_to_isaac_lab_controlled_ordering
                            ]
                            for controller in self._ik_controllers
                        ]
                    ),
                    device=self.device,
                    dtype=proposed.dtype,
                )
                for limit_name in ("lowerPositionLimit", "upperPositionLimit")
            )
        pink_lower, pink_upper = self._pink_joint_limits
        lower = torch.maximum(pink_lower, articulation_limits[..., 0])
        upper = torch.minimum(pink_upper, articulation_limits[..., 1])
        return bound_pink_joint_targets(current, proposed, lower, upper)


def _wrist_poses(env, state: dict[str, Any]):
    """Return current task-selected wrist poses with body indices cached in state."""
    try:
        robot = env.scene["robot"]
        if "_wrist_indices" in state:
            left_index, right_index = state["_wrist_indices"]
        else:
            target_links = env.cfg.actions.upper_body_ik.target_eef_link_names
            left_index = robot.body_names.index(target_links["left_wrist"])
            right_index = robot.body_names.index(target_links["right_wrist"])
            state["_wrist_indices"] = (left_index, right_index)
        pose_w = robot.data.body_link_pose_w.torch[0]
    except (KeyError, ValueError):
        return None
    return (
        pose_w[left_index, :3].clone(),
        pose_w[left_index, 3:7].clone(),
        pose_w[right_index, :3].clone(),
        pose_w[right_index, 3:7].clone(),
    )


def _step_vector_towards(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    delta = target - current
    distance = torch.linalg.norm(delta)
    alpha = torch.clamp(max_step / torch.clamp(distance, min=1.0e-6), max=1.0)
    return current + delta * alpha


def _step_quaternion_towards(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    current = torch.nn.functional.normalize(current, dim=0)
    target = torch.nn.functional.normalize(target, dim=0)
    target = torch.where(torch.sum(current * target) < 0.0, -target, target)
    dot = torch.clamp(torch.abs(torch.sum(current * target)), 0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    alpha = torch.clamp(max_step / torch.clamp(angle, min=1.0e-6), max=1.0)
    half_angle = angle * 0.5
    denominator = torch.clamp(torch.sin(half_angle), min=1.0e-6)
    interpolated = (
        torch.sin((1.0 - alpha) * half_angle) * current + torch.sin(alpha * half_angle) * target
    ) / denominator
    # Coincident quaternions need no interpolation and avoid a zero denominator.
    return torch.nn.functional.normalize(torch.where(angle < 1.0e-6, target, interpolated), dim=0)


def condition_kitless_newton_g1_action(
    env: ManagerBasedRLEnv, action: torch.Tensor, state: dict[str, Any]
) -> torch.Tensor:
    """Ramp both wrist targets from the current G1 poses before Pink solves IK.

    Each call moves the left and right wrist targets at most :data:`POSITION_MAX_STEP_M` and
    :data:`ROTATION_MAX_STEP_RAD` toward the requested poses; hand targets pass through unchanged.
    The ramp re-seeds from the measured wrist poses whenever environment 0 starts a new episode,
    including termination auto-resets inside ``env.step`` that the teleoperation loop does not observe.

    Args:
        env: The single-environment G1 teleoperation environment.
        action: One G1 action in the layout described by :func:`teleop_action_target_issue`. The Pink
            action reads its wrist poses relative to the environment origin, while the ramp seeds from
            the measured world-frame wrist poses, so environment 0's origin must be the world origin.
        state: Mutable dictionary owned by the caller; pass the same dictionary to every call of one
            session, starting empty. The ramp stores its targets and cached wrist body indices there.

    Returns:
        A conditioned copy of ``action`` with the same shape. An action whose size is not
        :data:`EXPECTED_ACTION_DIM` is returned unchanged and clears ``state``.

    Raises:
        RuntimeError: If the task-selected wrist bodies cannot be resolved. ``state`` is cleared.
    """
    flat = action.detach().flatten()
    if flat.numel() != EXPECTED_ACTION_DIM:
        state.clear()
        return action

    # The global step at which environment 0's episode began changes with every reset, including a
    # termination on the episode's first step, which leaves the episode counter at 0 between frames.
    episode_start = env.common_step_counter - int(env.episode_length_buf[0])
    if state.get("_episode_start") != episode_start:
        for key in _WRIST_RAMP_KEYS:
            state.pop(key, None)
    state["_episode_start"] = episode_start

    conditioned = action.clone(memory_format=torch.contiguous_format)
    conditioned_flat = conditioned.flatten()
    if "left_pos" not in state:
        current_wrist_poses = _wrist_poses(env, state)
        if current_wrist_poses is None:
            state.clear()
            raise RuntimeError("could not resolve the task-selected G1 wrist poses")
        left_pos, left_quat, right_pos, right_quat = current_wrist_poses
        state["left_pos"] = left_pos.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)
        state["left_quat"] = left_quat.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)
        state["right_pos"] = right_pos.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)
        state["right_quat"] = right_quat.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)

    state["left_pos"] = _step_vector_towards(state["left_pos"], flat[0:3], POSITION_MAX_STEP_M)
    state["left_quat"] = _step_quaternion_towards(state["left_quat"], flat[3:7], ROTATION_MAX_STEP_RAD)
    state["right_pos"] = _step_vector_towards(state["right_pos"], flat[7:10], POSITION_MAX_STEP_M)
    state["right_quat"] = _step_quaternion_towards(state["right_quat"], flat[10:14], ROTATION_MAX_STEP_RAD)
    conditioned_flat[0:3] = state["left_pos"]
    conditioned_flat[3:7] = state["left_quat"]
    conditioned_flat[7:10] = state["right_pos"]
    conditioned_flat[10:14] = state["right_quat"]
    return conditioned


def apply_g1_newton_actuator_profile(robot_cfg: ArticulationCfg) -> None:
    """Apply :data:`ACTUATOR_PROFILE` to the G1 actuator groups used by teleoperation.

    Args:
        robot_cfg: The G1 articulation configuration, modified in place.

    Raises:
        RuntimeError: If the configuration lacks the ``waist``, ``arms`` or ``hands`` actuator group.
    """
    groups = robot_cfg.actuators
    if not set(ACTUATOR_PROFILE).issubset(groups):
        raise RuntimeError(f"Expected G1 waist/arms/hands actuator groups; found {sorted(groups)}")
    for name, selected in ACTUATOR_PROFILE.items():
        actuator = groups[name]
        for key, value in selected.items():
            setattr(actuator, key, value)


def configure_g1_for_kitless_newton(env_cfg: FixedBaseUpperBodyIKG1EnvCfg) -> None:
    """Configure the controller, asset and solver for kitless Newton.

    Args:
        env_cfg: The G1 environment configuration with a Newton MJWarp physics selection, modified
            in place.

    Raises:
        RuntimeError: If the physics configuration has no MJWarp solver configuration, or the robot
            lacks an expected actuator group.
    """
    from isaaclab_tasks.contrib.packing_table import configure_packing_table_collisions
    from isaaclab_tasks.contrib.steering_wheel import condition_newton_steering_wheel_detail_collisions

    solver_cfg = getattr(getattr(env_cfg.sim, "physics", None), "solver_cfg", None)
    if solver_cfg is None:
        raise RuntimeError("G1 kitless Newton stabilization requires an MJWarp solver config")
    solver_cfg.nconmax = 4096
    solver_cfg.njmax = 1536

    apply_g1_newton_actuator_profile(env_cfg.scene.robot)
    env_cfg.actions.upper_body_ik.controller.fail_on_joint_limit_violation = False
    env_cfg.actions.upper_body_ik.class_type = G1NewtonBoundedPinkInverseKinematicsAction
    if env_cfg.isaac_teleop is not None:
        env_cfg.isaac_teleop.action_conditioner = condition_kitless_newton_g1_action
        env_cfg.isaac_teleop.action_validator = teleop_action_target_issue

    configure_packing_table_collisions(env_cfg.scene.packing_table)
    condition_newton_steering_wheel_detail_collisions(env_cfg.scene.object)
