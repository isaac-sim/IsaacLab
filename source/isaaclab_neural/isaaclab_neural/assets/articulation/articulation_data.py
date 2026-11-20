# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint buffers of the world-model articulation."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.warp import ProxyArray

from .unsupported import raise_for_unsupported


class ArticulationData(BaseArticulationData):
    """Joint position and its command, per environment.

    Joint positions and targets wrap the manager's tensors without copying. The configured default pose stays
    separate. The world model records joint positions only: velocities and accelerations raise.
    """

    __backend_name__ = "neural"

    def __init__(
        self,
        device: str,
        joint_names: list[str],
        joint_pos: torch.Tensor,
        joint_pos_target: torch.Tensor,
        default_joint_pos: torch.Tensor,
        joint_pos_limits: torch.Tensor,
        soft_joint_pos_limit_factor: float,
    ):
        super().__init__(None, device)
        self.joint_names = joint_names
        num_instances = joint_pos.shape[0]
        shape = (num_instances, len(joint_names))
        self._joint_pos = ProxyArray(wp.from_torch(joint_pos, dtype=wp.float32))
        self._joint_pos_command = ProxyArray(wp.from_torch(joint_pos_target, dtype=wp.float32))
        self._default_joint_pos = ProxyArray(wp.from_torch(default_joint_pos, dtype=wp.float32))
        self._default_joint_vel = ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device))
        limits = joint_pos_limits.to(device).expand(num_instances, -1, -1).contiguous()
        self._joint_pos_limits = ProxyArray(wp.from_torch(limits, dtype=wp.vec2f))
        # Soft limits shrink the range around its middle, as for a simulated robot. Unbounded joints stay unbounded.
        middle, half = limits.mean(dim=-1, keepdim=True), (limits[..., 1:] - limits[..., :1]) / 2
        soft = torch.where(
            torch.isfinite(half),
            torch.cat((middle - half * soft_joint_pos_limit_factor, middle + half * soft_joint_pos_limit_factor), -1),
            limits,
        )
        self._soft_joint_pos_limits = ProxyArray(wp.from_torch(soft.contiguous(), dtype=wp.vec2f))

    def update(self, dt: float) -> None:
        """No-op: the buffers share the manager's joint state and targets."""

    @property
    def joint_pos(self) -> ProxyArray:
        """Joint positions [m or rad, depending on joint type]. Shape is (num_instances, num_joints)."""
        return self._joint_pos

    @property
    def joint_pos_target(self) -> ProxyArray:
        """Joint position targets commanded by the user. Shape is (num_instances, num_joints)."""
        return self._joint_pos_command

    @property
    def default_joint_pos(self) -> ProxyArray:
        """Default joint positions, from ``init_state.joint_pos``. Shape is (num_instances, num_joints)."""
        return self._default_joint_pos

    @property
    def default_joint_vel(self) -> ProxyArray:
        """Default joint velocities, always zero. Shape is (num_instances, num_joints)."""
        return self._default_joint_vel

    @property
    def joint_pos_limits(self) -> ProxyArray:
        """Joint position limits from the USD asset [m or rad]. Shape is (num_instances, num_joints), ``wp.vec2f``."""
        return self._joint_pos_limits

    @property
    def soft_joint_pos_limits(self) -> ProxyArray:
        """Joint position limits scaled by ``soft_joint_pos_limit_factor``. Shape as :attr:`joint_pos_limits`."""
        return self._soft_joint_pos_limits

    def _reset_pose(self, *args, **kwargs) -> None:
        """No-op: there is no root pose."""

    def _reset_velocity(self, *args, **kwargs) -> None:
        """No-op: there is no velocity."""


raise_for_unsupported(ArticulationData, BaseArticulationData)
