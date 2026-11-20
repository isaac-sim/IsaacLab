# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Per-task specs for the multi-encoder actor.

A ``TaskSpec`` bundles everything needed to route one task through the actor:
the goal width and a function that converts reference motion into the
``goal_obs`` tensor consumed by the encoder. Reward is motion-tracking for every
task and lives in the WM env, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab_rl.model_based.quaternion import quat_apply_inverse
from isaaclab_rl.model_based.representations import actor_goal_dim
from isaaclab_rl.model_based.task_specs import TaskSpec

if TYPE_CHECKING:
    from isaaclab_rl.model_based.representations import StateRepresentation


def make_motion_task(representation: StateRepresentation, num_goals: int) -> TaskSpec:
    """Motion-tracking task: goal is the full motion target (actor side)."""

    def extract(motion, motion_times, aux_state):
        _, actor_goal = representation.compute_wm_goal(motion, motion_times, aux_state)
        return actor_goal

    return TaskSpec(
        name="motion",
        goal_dim=actor_goal_dim(num_goals),
        extract_training_goal=extract,
    )


def make_velocity_task(
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_z_range: tuple[float, float],
    window_frames: int,
    command_noise: tuple[float, float, float],
    device: str = "cpu",
) -> TaskSpec:
    """Velocity-command task: 3D body-frame ``[vx_b, vy_b, wz_b]``.

    Matches mjlab's velocity-tracking command convention: linear x/y in the
    root body frame, yaw rate about the body z-axis. At training time the
    command is projected from the motion's root velocity into the body frame
    and perturbed by fresh per-step uniform noise (``command_noise``,
    per-axis half-width) clamped to the configured ranges.

    ``window_frames`` controls a centered mean over N motion frames around
    ``t`` (``0`` or ``1`` = raw instantaneous). The reference motion is
    known offline, so there is no causality constraint — centering avoids
    the ~N/2 * dt lag a causal window would introduce. The window clamps to
    the concatenated motion array (not per clip), so frames within ~N/2 of a
    clip boundary average in a neighbouring clip's velocity.
    """
    lo = torch.tensor([lin_vel_x_range[0], lin_vel_y_range[0], ang_vel_z_range[0]], device=device)
    hi = torch.tensor([lin_vel_x_range[1], lin_vel_y_range[1], ang_vel_z_range[1]], device=device)
    noise_half = torch.tensor(command_noise, device=device)

    def _clamp(cmd):
        return torch.clamp(cmd, lo.to(cmd.device), hi.to(cmd.device))

    def _centered_mean(track, motion_times):
        if window_frames <= 1:
            return track[motion_times]
        half = window_frames // 2
        offsets = torch.arange(window_frames, device=motion_times.device, dtype=motion_times.dtype) - half
        idx = (motion_times.unsqueeze(-1) + offsets).clamp(min=0, max=track.shape[0] - 1)
        return track[idx].mean(dim=1)

    def _body_frame_cmd(motion, times, quat):
        v_b = quat_apply_inverse(quat, _centered_mean(motion.root_lin_vel_w, times))
        w_b = quat_apply_inverse(quat, _centered_mean(motion.root_ang_vel_w, times))
        return torch.stack([v_b[:, 0], v_b[:, 1], w_b[:, 2]], dim=-1)

    def extract(motion, motion_times, aux_state):
        cmd = _body_frame_cmd(motion, motion_times, aux_state[:, 3:7])
        noise = (torch.rand_like(cmd) * 2 - 1) * noise_half.to(cmd.device)
        return _clamp(cmd + noise)

    return TaskSpec(
        name="velocity",
        goal_dim=3,
        extract_training_goal=extract,
    )


@dataclass
class VelocityTaskCfg:
    """Body-frame velocity-command task."""

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
    window_frames: int = 25
    """Centered-mean window over motion-velocity command. 0/1 = raw instantaneous."""
    command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)
    """Uniform noise half-width per axis (lin_x, lin_y, ang_z) added to the
  velocity command each step, then clamped to the configured ranges."""


@dataclass
class TasksCfg:
    """Per-task hyperparameters. Field order defines slot block ordering."""

    velocity: VelocityTaskCfg = field(default_factory=VelocityTaskCfg)


def resolve_task_specs(
    tasks: TasksCfg,
    representation: StateRepresentation,
    num_goals: int,
    device: str = "cpu",
) -> list[TaskSpec]:
    return [
        make_motion_task(representation, num_goals),
        make_velocity_task(
            tasks.velocity.lin_vel_x_range,
            tasks.velocity.lin_vel_y_range,
            tasks.velocity.ang_vel_z_range,
            tasks.velocity.window_frames,
            tasks.velocity.command_noise,
            device,
        ),
    ]
