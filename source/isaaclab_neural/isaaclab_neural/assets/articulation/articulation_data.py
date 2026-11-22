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

UNSUPPORTED = (
    "Not available with the world-model backend: it has joint state only, no bodies, root, forces or physical"
    " properties."
)
"""Error message of the articulation members a world model does not have."""


class ArticulationData(BaseArticulationData):
    """Joint position and its command, per environment.

    The world model records joint positions only: velocities and accelerations raise.
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
        """Wrap the manager's joint state and targets, so reads and writes need no copy.

        Args:
            device: Simulation device.
            joint_names: Joint names, in the order of the buffers.
            joint_pos: The manager's joint state, shape ``(N, J)``.
            joint_pos_target: The manager's joint targets, shape ``(N, J)``.
            default_joint_pos: Default joint positions, shape ``(N, J)``.
            joint_pos_limits: Joint position limits [m or rad], shape ``(1, J, 2)``.
            soft_joint_pos_limit_factor: Fraction of the limit range that the soft limits keep.
        """
        super().__init__(None, device)
        self.joint_names = joint_names
        num_instances = joint_pos.shape[0]
        shape = (num_instances, len(joint_names))
        self._joint_pos = ProxyArray(wp.from_torch(joint_pos, dtype=wp.float32))
        self._joint_pos_command = ProxyArray(wp.from_torch(joint_pos_target, dtype=wp.float32))
        self._default_joint_pos = ProxyArray(wp.from_torch(default_joint_pos, dtype=wp.float32))
        self._default_joint_vel = ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device))
        self._soft_joint_vel_limits = ProxyArray(
            wp.from_torch(torch.full(shape, torch.inf, device=device), dtype=wp.float32)
        )
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
        """No-op: the buffers are the world model's joint state."""

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
        """Joint positions the episode starts from: the reset record's state, ``init_state.joint_pos`` before the
        first reset. Shape is (num_instances, num_joints)."""
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

    @property
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Joint velocity limits, unbounded since the world model has no joint velocity. Shape is
        (num_instances, num_joints)."""
        return self._soft_joint_vel_limits

    def _reset_pose(self, *args, **kwargs) -> None:
        """No-op: there is no root pose."""

    def _reset_velocity(self, *args, **kwargs) -> None:
        """No-op: there is no velocity."""

    # The world model has joint state only: these parts of the articulation are not available.

    @property
    def default_root_pose(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def default_root_vel(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def default_root_state(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_stiffness(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_damping(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_armature(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_friction_coeff(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_vel_limits(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_effort_limits(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def fixed_tendon_stiffness(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def fixed_tendon_damping(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def fixed_tendon_limit_stiffness(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def fixed_tendon_rest_length(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def fixed_tendon_offset(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def fixed_tendon_pos_limits(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def spatial_tendon_stiffness(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def spatial_tendon_damping(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def spatial_tendon_limit_stiffness(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def spatial_tendon_offset(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_pose_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_pose_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_state_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_state_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_state_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_mass(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_inertia(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_pose_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_pose_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_state_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_state_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_state_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_joint_wrench(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_acc_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_pose_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_vel(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def joint_acc(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def projected_gravity_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def heading_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_lin_vel_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_pos_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_quat_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_pos_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_quat_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_pos_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_quat_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_pos_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_quat_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_pos_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def body_com_quat_b(self) -> ProxyArray:
        raise NotImplementedError(UNSUPPORTED)
