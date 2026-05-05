# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.joint_wrench import BaseJointWrenchSensorData
from isaaclab.utils.warp import ProxyArray


class JointWrenchSensorData(BaseJointWrenchSensorData):
    """Data container for the Newton joint-wrench sensor."""

    def __init__(self):
        self._force: wp.array | None = None
        self._torque: wp.array | None = None
        self._body_names: list[str] = []
        self._force_ta: ProxyArray | None = None
        self._torque_ta: ProxyArray | None = None

    @property
    def force(self) -> ProxyArray | None:
        """Linear component of the joint reaction wrench [N].

        Expressed in the frame selected by
        :attr:`~isaaclab.sensors.JointWrenchSensorCfg.convention`. Shape is
        ``(num_envs, num_joints)``, dtype ``wp.vec3f``. In torch this resolves
        to ``(num_envs, num_joints, 3)``. ``None`` before the simulation is
        initialized.
        """
        if self._force is None:
            return None
        if self._force_ta is None:
            self._force_ta = ProxyArray(self._force)
        return self._force_ta

    @property
    def torque(self) -> ProxyArray | None:
        """Angular component of the joint reaction wrench [N·m].

        Expressed in the frame selected by
        :attr:`~isaaclab.sensors.JointWrenchSensorCfg.convention`. Shape is
        ``(num_envs, num_joints)``, dtype ``wp.vec3f``. In torch this resolves
        to ``(num_envs, num_joints, 3)``. ``None`` before the simulation is
        initialized.
        """
        if self._torque is None:
            return None
        if self._torque_ta is None:
            self._torque_ta = ProxyArray(self._torque)
        return self._torque_ta

    def create_buffers(self, num_envs: int, num_joints: int, device: str) -> None:
        """Allocate internal buffers.

        Args:
            num_envs: Number of environments.
            num_joints: Number of reported joints (excludes FREE and FIXED joint types).
            device: Device for array storage.
        """
        self._force = wp.zeros((num_envs, num_joints), dtype=wp.vec3f, device=device)
        self._torque = wp.zeros((num_envs, num_joints), dtype=wp.vec3f, device=device)
        self._force_ta = None
        self._torque_ta = None
