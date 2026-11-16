# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small runtime contract and deterministic fake joint telemetry for deployment experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class JointState:
    """Joint telemetry in session order, with unbatched CPU floating tensors of shape ``(num_joints,)``.

    Positions use [m or rad, depending on joint type]; velocities use [m/s or rad/s]. The timestamp is
    session time [s]. Tensors are borrowed until the next runtime call and must not be modified by callers.
    """

    position: torch.Tensor
    velocity: torch.Tensor
    timestamp: float


class DeploymentSession(Protocol):
    """Runtime owning joint communication and time; Lab adapters own batching and array wrappers."""

    joint_names: tuple[str, ...]
    """Names in telemetry and command order."""

    def read(self) -> JointState:
        """Read telemetry whose tensors are borrowed until the next runtime call."""
        ...

    def write(self, position_targets: torch.Tensor) -> None:
        """Consume or copy finite CPU floating targets of shape ``(num_joints,)`` before returning."""
        ...

    def step(self, dt: float) -> None:
        """Advance one runtime tick with a positive finite duration [s]."""
        ...

    def close(self) -> None:
        """Release the session; subsequent read, write, and step calls must fail."""
        ...


class FakeSession:
    """Move fake joints toward buffered targets at a bounded speed using logical time only.

    This session has no hardware connection, physical reset, or real-time timing guarantee.
    """

    def __init__(
        self,
        joint_names: tuple[str, ...] = ("shoulder", "elbow"),
        initial_position: tuple[float, ...] = (0.1, -0.2),
        max_velocity: float = 0.5,
    ) -> None:
        """Initialize measured positions and matching targets.

        Args:
            joint_names: Unique, nonempty joint names in telemetry and command order.
            initial_position: Initial joint positions [m or rad, depending on joint type].
            max_velocity: Positive speed bound [m/s or rad/s, depending on joint type].
        """
        if not joint_names or len(set(joint_names)) != len(joint_names) or any(not name for name in joint_names):
            raise ValueError("Joint names must be nonempty and unique.")
        if not math.isfinite(max_velocity) or max_velocity <= 0:
            raise ValueError("max_velocity must be positive and finite.")
        self.joint_names = tuple(joint_names)
        self._position = torch.tensor(initial_position, dtype=torch.float32, device="cpu")
        self._validate_joint_tensor(self._position)
        self._velocity = torch.zeros_like(self._position)
        self._target = self._position.clone()
        self._max_velocity = max_velocity
        self._timestamp = 0.0
        self._closed = False

    def read(self) -> JointState:
        """Read borrowed measured positions and velocities with the current session timestamp."""
        self._check_open()
        return JointState(self._position, self._velocity, self._timestamp)

    def write(self, position_targets: torch.Tensor) -> None:
        """Copy position targets into the command buffer without advancing telemetry."""
        self._check_open()
        self._validate_joint_tensor(position_targets)
        self._target.copy_(position_targets.detach())

    def step(self, dt: float) -> None:
        """Advance logical time [s] and measure velocity from the bounded position change."""
        self._check_open()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite.")
        max_delta = self._max_velocity * dt
        delta = (self._target - self._position).clamp(min=-max_delta, max=max_delta)
        self._position.add_(delta)
        self._velocity.copy_(delta / dt)
        self._timestamp += dt

    def close(self) -> None:
        """Close the session; repeated close calls are harmless."""
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Deployment session is closed.")

    def _validate_joint_tensor(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_floating_point():
            raise ValueError("Joint values must be CPU floating tensors.")
        if value.shape != (len(self.joint_names),) or not torch.isfinite(value).all():
            raise ValueError(f"Joint values must be finite with shape ({len(self.joint_names)},).")
