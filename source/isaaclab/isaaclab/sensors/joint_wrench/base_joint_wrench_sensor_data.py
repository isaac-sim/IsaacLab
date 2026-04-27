# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for joint-wrench sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import warp as wp


class BaseJointWrenchSensorData(ABC):
    """Data container for the joint reaction wrench sensor."""

    @property
    @abstractmethod
    def force(self) -> wp.array | None:
        """Linear component of the joint reaction wrench [N].

        Shape is ``(num_envs, num_joints)``, dtype :class:`wp.vec3f`. In torch
        this resolves to ``(num_envs, num_joints, 3)``. ``None`` before the
        simulation is initialized.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def torque(self) -> wp.array | None:
        """Angular component of the joint reaction wrench [N·m].

        Shape is ``(num_envs, num_joints)``, dtype :class:`wp.vec3f`. In torch
        this resolves to ``(num_envs, num_joints, 3)``. ``None`` before the
        simulation is initialized.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of the bodies whose incoming joint wrench is reported."""
        raise NotImplementedError
