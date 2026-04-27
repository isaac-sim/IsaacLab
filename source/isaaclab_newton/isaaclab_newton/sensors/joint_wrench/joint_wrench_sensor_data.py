# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.joint_wrench import BaseJointWrenchSensorData


class JointWrenchSensorData(BaseJointWrenchSensorData):
    """Data container for the Newton joint-wrench sensor."""

    def __init__(self):
        self._force: wp.array | None = None
        self._torque: wp.array | None = None
        self._body_names: list[str] = []

    @property
    def force(self) -> wp.array | None:
        """Linear component of the joint reaction wrench [N].

        Shape is ``(num_envs, num_joints)``, dtype :class:`wp.vec3f`. In torch
        this resolves to ``(num_envs, num_joints, 3)``. ``None`` before the
        simulation is initialized.
        """
        return self._force

    @property
    def torque(self) -> wp.array | None:
        """Angular component of the joint reaction wrench [N·m].

        Shape is ``(num_envs, num_joints)``, dtype :class:`wp.vec3f`. In torch
        this resolves to ``(num_envs, num_joints, 3)``. ``None`` before the
        simulation is initialized.
        """
        return self._torque

    @property
    def body_names(self) -> list[str]:
        """Ordered names of the bodies whose incoming joint wrench is reported.

        Empty before the simulation is initialized.
        """
        return self._body_names

    def create_buffers(self, num_envs: int, num_joints: int, device: str) -> None:
        """Allocate internal buffers.

        Args:
            num_envs: Number of environments.
            num_joints: Number of reported joints (excludes FREE and FIXED joint types).
            device: Device for array storage.
        """
        self._force = wp.zeros((num_envs, num_joints), dtype=wp.vec3f, device=device)
        self._torque = wp.zeros((num_envs, num_joints), dtype=wp.vec3f, device=device)
