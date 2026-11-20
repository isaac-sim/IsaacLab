# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton_controllers import ControllerAckermann

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .ackermann_cfg import AckermannControllerCfg


_MAX_FLOAT32_STEERING_ANGLE = float(np.nextafter(np.float32(np.pi / 2.0), np.float32(0.0)))


class AckermannController:
    r"""Compute Ackermann steering positions and wheel angular velocities.

    The controller accepts a two-dimensional command ``(v_x, delta)`` for each vehicle. Here, ``v_x`` is the
    longitudinal speed [m/s] at the center of the non-steerable axle, expressed in the vehicle body frame, and
    ``delta`` is the virtual steering angle [rad]. This reference point applies to both front- and rear-steering
    vehicles. The controller produces two steering-joint position targets [rad] followed by angular-velocity targets
    [rad/s] for the left steerable wheel, right steerable wheel, and configured non-steerable wheels.

    The implementation delegates the batched Ackermann kinematics to Newton's
    ``newton_controllers.ControllerAckermann``. The vehicle yaw rate follows
    :math:`\omega_z = v_x \tan(\delta) / L`.

    .. note::

        :meth:`compute` returns persistent zero-copy views of the controller output buffers. Their addresses remain
        stable, and a later call overwrites their values. Clone an output only when it must outlive the next call.

    Args:
        cfg: Controller configuration.
        num_envs: Number of vehicles in the batch.
        device: Torch and Warp device used for computation.
    """

    def __init__(self, cfg: AckermannControllerCfg, num_envs: int, device: str) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be greater than zero, got {num_envs}.")

        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        self._num_wheels = 2 + len(cfg.non_steerable_wheel_offsets)

        # Stable Torch command buffers are exposed to Warp without copies.
        self._command = torch.zeros((num_envs, self.action_dim), dtype=torch.float32, device=device)
        self._linear_speed_input = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._steering_angle_input = torch.zeros(num_envs, dtype=torch.float32, device=device)

        wheel_velocity_indices = wp.array(
            np.arange(num_envs * self._num_wheels, dtype=np.uint32), dtype=wp.uint32, device=device
        )
        steering_angle_indices = wp.array(np.arange(num_envs * 2, dtype=np.uint32), dtype=wp.uint32, device=device)
        non_steerable_wheel_offsets = None
        if cfg.non_steerable_wheel_offsets:
            offset_values = np.broadcast_to(
                np.asarray(cfg.non_steerable_wheel_offsets, dtype=np.float32),
                (num_envs, len(cfg.non_steerable_wheel_offsets)),
            ).copy()
            non_steerable_wheel_offsets = wp.array(offset_values, dtype=wp.float32, device=device)

        self._controller = ControllerAckermann(
            num_robots=num_envs,
            wheel_radius=wp.full(num_envs, cfg.wheel_radius, dtype=wp.float32, device=device),
            wheel_base=wp.full(num_envs, cfg.wheel_base, dtype=wp.float32, device=device),
            track_width=wp.full(num_envs, cfg.track_width, dtype=wp.float32, device=device),
            default_wheel_velocity_indices=wheel_velocity_indices,
            default_steering_angle_indices=steering_angle_indices,
            max_linear_speed=wp.full(num_envs, cfg.max_linear_speed, dtype=wp.float32, device=device),
            # A valid Python float immediately below pi / 2 can round upward to float32(pi / 2). Clamp only at the
            # float32 bridge so every value accepted by the public config remains representable and non-singular.
            max_turning_angle=wp.full(
                num_envs,
                min(cfg.max_steering_angle, _MAX_FLOAT32_STEERING_ANGLE),
                dtype=wp.float32,
                device=device,
            ),
            steerable_wheels_at_rear=wp.full(num_envs, cfg.steerable_wheels_at_rear, dtype=wp.bool, device=device),
            non_steerable_wheel_offsets=non_steerable_wheel_offsets,
            device=device,
        )
        self._inputs = self._controller.input()
        self._outputs = self._controller.output()
        self._inputs.linear_speed_command = wp.from_torch(self._linear_speed_input)
        self._inputs.turning_angle_command = wp.from_torch(self._steering_angle_input)
        self._steering_joint_position_targets = wp.to_torch(self._outputs.joint_target_q).view(num_envs, 2)
        self._wheel_joint_velocity_targets = wp.to_torch(self._outputs.joint_target_qd).view(num_envs, self._num_wheels)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller command."""
        return 2

    @property
    def num_wheels(self) -> int:
        """Number of wheel angular-velocity outputs per vehicle."""
        return self._num_wheels

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset commands for selected environments.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        self._command[env_ids] = 0.0
        self._linear_speed_input[env_ids] = 0.0
        self._steering_angle_input[env_ids] = 0.0

    def compute(self, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute steering-position and wheel-velocity targets.

        Args:
            command: Longitudinal speed at the non-steerable axle center [m/s] and virtual steering angle [rad],
                shape ``(N, 2)``.

        Returns:
            A tuple containing persistent zero-copy views of steering-joint position targets [rad], shape ``(N, 2)``,
            and wheel-joint angular velocity targets [rad/s], shape ``(N, num_wheels)``. A subsequent call overwrites
            their values without changing their addresses.

        Raises:
            ValueError: If the command does not have shape ``(N, 2)``.
        """
        if command.shape != self._command.shape:
            raise ValueError(f"command must have shape {tuple(self._command.shape)}, got {tuple(command.shape)}.")

        self._command.copy_(command)
        # Preserve raw values across the stable bridge. Newton owns command validation and clamping; in particular,
        # it maps any non-finite command pair to safe zero targets instead of saturating infinity to a limit.
        self._linear_speed_input.copy_(self._command[:, 0])
        self._steering_angle_input.copy_(self._command[:, 1])
        self._controller.compute(self._inputs, self._outputs, None, None, time_step=0.0)
        return self._steering_joint_position_targets, self._wheel_joint_velocity_targets
