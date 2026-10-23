# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for joint-wrench sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...utils.leapp import leapp_tensor_semantics
from ...utils.warp import ProxyArray


class BaseJointWrenchSensorData(ABC):
    """Data container for the joint reaction wrench sensor."""

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/joint/force")
    def force(self) -> ProxyArray | None:
        """Linear component of the joint reaction wrench [N].

        Expressed in the frame selected by
        :attr:`~isaaclab.sensors.JointWrenchSensorCfg.convention`. Shape is
        ``(num_envs, num_bodies)``, dtype ``wp.vec3f``. In torch this resolves
        to ``(num_envs, num_bodies, 3)``. ``None`` before the simulation is
        initialized.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/joint/torque")
    def torque(self) -> ProxyArray | None:
        """Angular component of the joint reaction wrench [N·m].

        Expressed in the frame selected by
        :attr:`~isaaclab.sensors.JointWrenchSensorCfg.convention`. Shape is
        ``(num_envs, num_bodies)``, dtype ``wp.vec3f``. In torch this resolves
        to ``(num_envs, num_bodies, 3)``. ``None`` before the simulation is
        initialized.
        """
        raise NotImplementedError
