# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock articulation asset for testing without Isaac Sim."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

try:
    from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
except (ImportError, ModuleNotFoundError):
    # Direct import bypassing isaaclab.assets.__init__.py (which needs omni.timeline)
    import importlib.util
    from pathlib import Path

    _file = Path(__file__).resolve().parents[3] / "assets" / "articulation" / "base_articulation_data.py"
    _spec = importlib.util.spec_from_file_location("_base_articulation_data", str(_file))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseArticulationData = _mod.BaseArticulationData


class MockArticulationData(BaseArticulationData):
    """Mock data container for articulation asset.

    This class mimics the interface of BaseArticulationData for testing purposes.
    All tensor properties return zero tensors with correct shapes if not explicitly set.
    """

    def __init__(
        self,
        num_instances: int,
        num_joints: int,
        num_bodies: int,
        joint_names: list[str] | None = None,
        body_names: list[str] | None = None,
        num_fixed_tendons: int = 0,
        num_spatial_tendons: int = 0,
        fixed_tendon_names: list[str] | None = None,
        spatial_tendon_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock articulation data.

        Args:
            num_instances: Number of articulation instances.
            num_joints: Number of joints.
            num_bodies: Number of bodies.
            joint_names: Names of joints.
            body_names: Names of bodies.
            num_fixed_tendons: Number of fixed tendons.
            num_spatial_tendons: Number of spatial tendons.
            fixed_tendon_names: Names of fixed tendons.
            spatial_tendon_names: Names of spatial tendons.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_joints = num_joints
        self._num_bodies = num_bodies
        self._num_fixed_tendons = num_fixed_tendons
        self._num_spatial_tendons = num_spatial_tendons

        # Names
        self.joint_names = joint_names or [f"joint_{i}" for i in range(num_joints)]
        self.body_names = body_names or [f"body_{i}" for i in range(num_bodies)]
        self.fixed_tendon_names = fixed_tendon_names or [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]
        self.spatial_tendon_names = spatial_tendon_names or [f"spatial_tendon_{i}" for i in range(num_spatial_tendons)]

        # Call base class init (sets self.device)
        super().__init__(root_view=None, device=device)
        self._create_buffers()

        # -- Internal storage for mock data --
        # Default states
        self._default_root_pose: wp.array | None = None
        self._default_root_vel: wp.array | None = None
        self._default_joint_pos: wp.array | None = None
        self._default_joint_vel: wp.array | None = None

        # Joint commands
        self._joint_pos_target: wp.array | None = None
        self._joint_vel_target: wp.array | None = None
        self._joint_effort_target: wp.array | None = None
        self._computed_torque: wp.array | None = None
        self._applied_torque: wp.array | None = None

        # Joint properties
        self._joint_stiffness: wp.array | None = None
        self._joint_damping: wp.array | None = None
        self._joint_armature: wp.array | None = None
        self._joint_friction_coeff: wp.array | None = None
        self._joint_dynamic_friction_coeff: wp.array | None = None
        self._joint_viscous_friction_coeff: wp.array | None = None
        self._joint_pos_limits: wp.array | None = None
        self._joint_vel_limits: wp.array | None = None
        self._joint_effort_limits: wp.array | None = None
        self._soft_joint_pos_limits: wp.array | None = None
        self._soft_joint_vel_limits: wp.array | None = None
        self._gear_ratio: wp.array | None = None

        # Joint state
        self._joint_pos: wp.array | None = None
        self._joint_vel: wp.array | None = None
        self._joint_acc: wp.array | None = None

        # Root state (link frame)
        self._root_link_pose_w: wp.array | None = None
        self._root_link_vel_w: wp.array | None = None

        # Root state (CoM frame)
        self._root_com_pose_w: wp.array | None = None
        self._root_com_vel_w: wp.array | None = None

        # Body state (link frame)
        self._body_link_pose_w: wp.array | None = None
        self._body_link_vel_w: wp.array | None = None

        # Body state (CoM frame)
        self._body_com_pose_w: wp.array | None = None
        self._body_com_vel_w: wp.array | None = None
        self._body_com_acc_w: wp.array | None = None
        self._body_com_pose_b: wp.array | None = None

        # Body properties
        self._body_mass: wp.array | None = None
        self._body_inertia: wp.array | None = None
        self._body_incoming_joint_wrench_b: wp.array | None = None

        # Tendon properties (fixed)
        self._fixed_tendon_stiffness: wp.array | None = None
        self._fixed_tendon_damping: wp.array | None = None
        self._fixed_tendon_limit_stiffness: wp.array | None = None
        self._fixed_tendon_rest_length: wp.array | None = None
        self._fixed_tendon_offset: wp.array | None = None
        self._fixed_tendon_pos_limits: wp.array | None = None

        # Tendon properties (spatial)
        self._spatial_tendon_stiffness: wp.array | None = None
        self._spatial_tendon_damping: wp.array | None = None
        self._spatial_tendon_limit_stiffness: wp.array | None = None
        self._spatial_tendon_offset: wp.array | None = None

    # -- Helper for identity quaternion --
    def _identity_quat(self, *shape: int) -> wp.array:
        """Create identity quaternion array (w, x, y, z) = (1, 0, 0, 0)."""
        np_arr = np.zeros((*shape, 4), dtype=np.float32)
        np_arr[..., 0] = 1.0
        return wp.array(np_arr, dtype=wp.float32, device=self.device)

    # -- Default state properties --

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose. dtype=wp.transformf, shape: (N,)."""
        if self._default_root_pose is None:
            pose_np = np.zeros((self._num_instances, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            self._default_root_pose = wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._default_root_pose

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity. dtype=wp.spatial_vectorf, shape: (N,)."""
        if self._default_root_vel is None:
            return wp.zeros((self._num_instances, 6), dtype=wp.float32, device=self.device).view(wp.spatial_vectorf)
        return self._default_root_vel

    @property
    def default_root_state(self) -> wp.array:
        """Default root state [pose(7), vel(6)]. Shape: (N, 13)."""
        pose_t = wp.to_torch(self.default_root_pose)
        vel_t = wp.to_torch(self.default_root_vel)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    @property
    def default_joint_pos(self) -> wp.array:
        """Default joint positions. Shape: (N, num_joints)."""
        if self._default_joint_pos is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._default_joint_pos

    @property
    def default_joint_vel(self) -> wp.array:
        """Default joint velocities. Shape: (N, num_joints)."""
        if self._default_joint_vel is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._default_joint_vel

    # -- Joint command properties --

    @property
    def joint_pos_target(self) -> wp.array:
        """Joint position targets. Shape: (N, num_joints)."""
        if self._joint_pos_target is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_pos_target

    @property
    def joint_vel_target(self) -> wp.array:
        """Joint velocity targets. Shape: (N, num_joints)."""
        if self._joint_vel_target is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_vel_target

    @property
    def joint_effort_target(self) -> wp.array:
        """Joint effort targets. Shape: (N, num_joints)."""
        if self._joint_effort_target is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_effort_target

    @property
    def computed_torque(self) -> wp.array:
        """Computed torques before clipping. Shape: (N, num_joints)."""
        if self._computed_torque is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._computed_torque

    @property
    def applied_torque(self) -> wp.array:
        """Applied torques after clipping. Shape: (N, num_joints)."""
        if self._applied_torque is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._applied_torque

    # -- Joint properties --

    @property
    def joint_stiffness(self) -> wp.array:
        """Joint stiffness. Shape: (N, num_joints)."""
        if self._joint_stiffness is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_stiffness

    @property
    def joint_damping(self) -> wp.array:
        """Joint damping. Shape: (N, num_joints)."""
        if self._joint_damping is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_damping

    @property
    def joint_armature(self) -> wp.array:
        """Joint armature. Shape: (N, num_joints)."""
        if self._joint_armature is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_armature

    @property
    def joint_friction_coeff(self) -> wp.array:
        """Joint static friction coefficient. Shape: (N, num_joints)."""
        if self._joint_friction_coeff is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_friction_coeff

    @property
    def joint_dynamic_friction_coeff(self) -> wp.array:
        """Joint dynamic friction coefficient. Shape: (N, num_joints)."""
        if self._joint_dynamic_friction_coeff is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_dynamic_friction_coeff

    @property
    def joint_viscous_friction_coeff(self) -> wp.array:
        """Joint viscous friction coefficient. Shape: (N, num_joints)."""
        if self._joint_viscous_friction_coeff is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_viscous_friction_coeff

    @property
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits [lower, upper]. dtype=wp.vec2f, shape: (N, num_joints)."""
        if self._joint_pos_limits is None:
            np_limits = np.zeros((self._num_instances, self._num_joints, 2), dtype=np.float32)
            np_limits[..., 0] = -float("inf")
            np_limits[..., 1] = float("inf")
            return wp.array(np_limits, dtype=wp.float32, device=self.device).view(wp.vec2f)
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> wp.array:
        """Joint velocity limits. Shape: (N, num_joints)."""
        if self._joint_vel_limits is None:
            return wp.full(
                (self._num_instances, self._num_joints), dtype=wp.float32, value=float("inf"), device=self.device
            )
        return self._joint_vel_limits

    @property
    def joint_effort_limits(self) -> wp.array:
        """Joint effort limits. Shape: (N, num_joints)."""
        if self._joint_effort_limits is None:
            return wp.full(
                (self._num_instances, self._num_joints), dtype=wp.float32, value=float("inf"), device=self.device
            )
        return self._joint_effort_limits

    @property
    def soft_joint_pos_limits(self) -> wp.array:
        """Soft joint position limits. Shape: (N, num_joints, 2)."""
        if self._soft_joint_pos_limits is None:
            return wp.clone(self.joint_pos_limits, self.device)
        return self._soft_joint_pos_limits

    @property
    def soft_joint_vel_limits(self) -> wp.array:
        """Soft joint velocity limits. Shape: (N, num_joints)."""
        if self._soft_joint_vel_limits is None:
            return wp.clone(self.joint_vel_limits, self.device)
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> wp.array:
        """Gear ratio. Shape: (N, num_joints)."""
        if self._gear_ratio is None:
            return wp.ones((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._gear_ratio

    # -- Joint state properties --

    @property
    def joint_pos(self) -> wp.array:
        """Joint positions. Shape: (N, num_joints)."""
        if self._joint_pos is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_pos

    @property
    def joint_vel(self) -> wp.array:
        """Joint velocities. Shape: (N, num_joints)."""
        if self._joint_vel is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_vel

    @property
    def joint_acc(self) -> wp.array:
        """Joint accelerations. Shape: (N, num_joints)."""
        if self._joint_acc is None:
            return wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        return self._joint_acc

    # -- Root state properties (link frame) --

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose in world frame. dtype=wp.transformf, shape: (N,)."""
        if self._root_link_pose_w is None:
            pose_np = np.zeros((self._num_instances, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            self._root_link_pose_w = wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._root_link_pose_w

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity in world frame. dtype=wp.spatial_vectorf, shape: (N,)."""
        if self._root_link_vel_w is None:
            return wp.zeros((self._num_instances, 6), dtype=wp.float32, device=self.device).view(wp.spatial_vectorf)
        return self._root_link_vel_w

    @property
    def root_link_state_w(self) -> wp.array:
        """Root link state in world frame. Shape: (N, 13)."""
        pose_t = wp.to_torch(self.root_link_pose_w)
        vel_t = wp.to_torch(self.root_link_vel_w)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def root_link_pos_w(self) -> wp.array:
        """Root link position in world frame. Shape: (N,), dtype=wp.vec3f."""
        t = self.root_link_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation in world frame (x, y, z, w). Shape: (N,), dtype=wp.quatf."""
        t = self.root_link_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root link linear velocity in world frame. Shape: (N,), dtype=wp.vec3f."""
        v = self.root_link_vel_w
        return wp.array(ptr=v.ptr, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in world frame. Shape: (N,), dtype=wp.vec3f."""
        v = self.root_link_vel_w
        return wp.array(ptr=v.ptr + 3 * 4, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    # -- Root state properties (CoM frame) --

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root CoM pose in world frame. dtype=wp.transformf, shape: (N,)."""
        if self._root_com_pose_w is None:
            return wp.clone(self.root_link_pose_w, self.device)
        return self._root_com_pose_w

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root CoM velocity in world frame. dtype=wp.spatial_vectorf, shape: (N,)."""
        if self._root_com_vel_w is None:
            return wp.clone(self.root_link_vel_w, self.device)
        return self._root_com_vel_w

    @property
    def root_com_state_w(self) -> wp.array:
        """Root CoM state in world frame. Shape: (N, 13)."""
        pose_t = wp.to_torch(self.root_com_pose_w)
        vel_t = wp.to_torch(self.root_com_vel_w)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    @property
    def root_state_w(self) -> wp.array:
        """Root state (link pose + CoM velocity). Shape: (N, 13)."""
        pose_t = wp.to_torch(self.root_link_pose_w)
        vel_t = wp.to_torch(self.root_com_vel_w)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def root_com_pos_w(self) -> wp.array:
        """Root CoM position in world frame. Shape: (N,), dtype=wp.vec3f."""
        t = self.root_com_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root CoM orientation in world frame. Shape: (N,), dtype=wp.quatf."""
        t = self.root_com_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root CoM linear velocity in world frame. Shape: (N,), dtype=wp.vec3f."""
        v = self.root_com_vel_w
        return wp.array(ptr=v.ptr, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root CoM angular velocity in world frame. Shape: (N,), dtype=wp.vec3f."""
        v = self.root_com_vel_w
        return wp.array(ptr=v.ptr + 3 * 4, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    # -- Body state properties (link frame) --

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link poses in world frame. dtype=wp.transformf, shape: (N, num_bodies)."""
        if self._body_link_pose_w is None:
            pose_np = np.zeros((self._num_instances, self._num_bodies, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            self._body_link_pose_w = wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._body_link_pose_w

    @property
    def body_link_vel_w(self) -> wp.array:
        """Body link velocities in world frame. dtype=wp.spatial_vectorf, shape: (N, num_bodies)."""
        if self._body_link_vel_w is None:
            return wp.zeros((self._num_instances, self._num_bodies, 6), dtype=wp.float32, device=self.device).view(
                wp.spatial_vectorf
            )
        return self._body_link_vel_w

    @property
    def body_link_state_w(self) -> wp.array:
        """Body link states in world frame. Shape: (N, num_bodies, 13)."""
        pose_t = wp.to_torch(self.body_link_pose_w)
        vel_t = wp.to_torch(self.body_link_vel_w)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def body_link_pos_w(self) -> wp.array:
        """Body link positions. Shape: (N, num_bodies), dtype=wp.vec3f."""
        t = self.body_link_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def body_link_quat_w(self) -> wp.array:
        """Body link orientations. Shape: (N, num_bodies), dtype=wp.quatf."""
        t = self.body_link_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Body link linear velocities in world frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        v = self.body_link_vel_w
        return wp.array(ptr=v.ptr, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Body link angular velocities in world frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        v = self.body_link_vel_w
        return wp.array(ptr=v.ptr + 3 * 4, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    # -- Body state properties (CoM frame) --

    @property
    def body_com_pose_w(self) -> wp.array:
        """Body CoM poses in world frame. dtype=wp.transformf, shape: (N, num_bodies)."""
        if self._body_com_pose_w is None:
            return wp.clone(self.body_link_pose_w, self.device)
        return self._body_com_pose_w

    @property
    def body_com_vel_w(self) -> wp.array:
        """Body CoM velocities in world frame. dtype=wp.spatial_vectorf, shape: (N, num_bodies)."""
        if self._body_com_vel_w is None:
            return wp.clone(self.body_link_vel_w, self.device)
        return self._body_com_vel_w

    @property
    def body_com_state_w(self) -> wp.array:
        """Body CoM states in world frame. Shape: (N, num_bodies, 13)."""
        pose_t = wp.to_torch(self.body_com_pose_w)
        vel_t = wp.to_torch(self.body_com_vel_w)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    @property
    def body_state_w(self) -> wp.array:
        """Body states (link pose + CoM velocity). Shape: (N, num_bodies, 13)."""
        pose_t = wp.to_torch(self.body_link_pose_w)
        vel_t = wp.to_torch(self.body_com_vel_w)
        return wp.from_torch(torch.cat([pose_t, vel_t], dim=-1))

    @property
    def body_com_acc_w(self) -> wp.array:
        """Body CoM accelerations in world frame. dtype=wp.spatial_vectorf, shape: (N, num_bodies)."""
        if self._body_com_acc_w is None:
            return wp.zeros((self._num_instances, self._num_bodies, 6), dtype=wp.float32, device=self.device).view(
                wp.spatial_vectorf
            )
        return self._body_com_acc_w

    @property
    def body_com_pose_b(self) -> wp.array:
        """Body CoM poses in body frame. dtype=wp.transformf, shape: (N, num_bodies)."""
        if self._body_com_pose_b is None:
            pose_np = np.zeros((self._num_instances, self._num_bodies, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            self._body_com_pose_b = wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._body_com_pose_b

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def body_com_pos_w(self) -> wp.array:
        """Body CoM positions. Shape: (N, num_bodies), dtype=wp.vec3f."""
        t = self.body_com_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def body_com_quat_w(self) -> wp.array:
        """Body CoM orientations. Shape: (N, num_bodies), dtype=wp.quatf."""
        t = self.body_com_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Body CoM linear velocities in world frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        v = self.body_com_vel_w
        return wp.array(ptr=v.ptr, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Body CoM angular velocities in world frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        v = self.body_com_vel_w
        return wp.array(ptr=v.ptr + 3 * 4, shape=v.shape, dtype=wp.vec3f, strides=v.strides, device=self.device)

    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Body CoM linear accelerations in world frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        a = self.body_com_acc_w
        return wp.array(ptr=a.ptr, shape=a.shape, dtype=wp.vec3f, strides=a.strides, device=self.device)

    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Body CoM angular accelerations in world frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        a = self.body_com_acc_w
        return wp.array(ptr=a.ptr + 3 * 4, shape=a.shape, dtype=wp.vec3f, strides=a.strides, device=self.device)

    @property
    def body_com_pos_b(self) -> wp.array:
        """Body CoM positions in body frame. Shape: (N, num_bodies), dtype=wp.vec3f."""
        t = self.body_com_pose_b
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def body_com_quat_b(self) -> wp.array:
        """Body CoM orientations in body frame. Shape: (N, num_bodies), dtype=wp.quatf."""
        t = self.body_com_pose_b
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    # -- Body properties --

    @property
    def body_mass(self) -> wp.array:
        """Body masses. Shape: (N, num_bodies)."""
        if self._body_mass is None:
            return wp.ones((self._num_instances, self._num_bodies), dtype=wp.float32, device=self.device)
        return self._body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Body inertias (flattened 3x3). Shape: (N, num_bodies, 9)."""
        if self._body_inertia is None:
            np_inertia = np.zeros((self._num_instances, self._num_bodies, 9), dtype=np.float32)
            np_inertia[..., 0] = 1.0  # Ixx
            np_inertia[..., 4] = 1.0  # Iyy
            np_inertia[..., 8] = 1.0  # Izz
            return wp.array(np_inertia, dtype=wp.float32, device=self.device)
        return self._body_inertia

    @property
    def body_incoming_joint_wrench_b(self) -> wp.array:
        """Body incoming joint wrenches. dtype=wp.spatial_vectorf, shape: (N, num_bodies)."""
        if self._body_incoming_joint_wrench_b is None:
            return wp.zeros((self._num_instances, self._num_bodies, 6), dtype=wp.float32, device=self.device).view(
                wp.spatial_vectorf
            )
        return self._body_incoming_joint_wrench_b

    # -- Derived properties --

    @property
    def projected_gravity_b(self) -> wp.array:
        """Gravity projection on base. Shape: (N,), dtype=wp.vec3f."""
        np_gravity = np.zeros((self._num_instances, 3), dtype=np.float32)
        np_gravity[:, 2] = -1.0
        return wp.array(np_gravity, dtype=wp.float32, device=self.device).view(wp.vec3f)

    @property
    def heading_w(self) -> wp.array:
        """Yaw heading in world frame. Shape: (N,)."""
        return wp.zeros((self._num_instances,), dtype=wp.float32, device=self.device)

    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in body frame. Shape: (N,), dtype=wp.vec3f."""
        return wp.clone(self.root_link_lin_vel_w, self.device)

    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in body frame. Shape: (N,), dtype=wp.vec3f."""
        return wp.clone(self.root_link_ang_vel_w, self.device)

    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root CoM linear velocity in body frame. Shape: (N,), dtype=wp.vec3f."""
        return wp.clone(self.root_com_lin_vel_w, self.device)

    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root CoM angular velocity in body frame. Shape: (N,), dtype=wp.vec3f."""
        return wp.clone(self.root_com_ang_vel_w, self.device)

    # com_pos_b and com_quat_b are inherited from BaseArticulationData
    # (they delegate to body_com_pos_b and body_com_quat_b respectively)

    # -- Fixed tendon properties --

    @property
    def fixed_tendon_stiffness(self) -> wp.array:
        """Fixed tendon stiffness. Shape: (N, num_fixed_tendons)."""
        if self._num_fixed_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._fixed_tendon_stiffness is None:
            return wp.zeros((self._num_instances, self._num_fixed_tendons), dtype=wp.float32, device=self.device)
        return self._fixed_tendon_stiffness

    @property
    def fixed_tendon_damping(self) -> wp.array:
        """Fixed tendon damping. Shape: (N, num_fixed_tendons)."""
        if self._num_fixed_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._fixed_tendon_damping is None:
            return wp.zeros((self._num_instances, self._num_fixed_tendons), dtype=wp.float32, device=self.device)
        return self._fixed_tendon_damping

    @property
    def fixed_tendon_limit_stiffness(self) -> wp.array:
        """Fixed tendon limit stiffness. Shape: (N, num_fixed_tendons)."""
        if self._num_fixed_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._fixed_tendon_limit_stiffness is None:
            return wp.zeros((self._num_instances, self._num_fixed_tendons), dtype=wp.float32, device=self.device)
        return self._fixed_tendon_limit_stiffness

    @property
    def fixed_tendon_rest_length(self) -> wp.array:
        """Fixed tendon rest length. Shape: (N, num_fixed_tendons)."""
        if self._num_fixed_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._fixed_tendon_rest_length is None:
            return wp.zeros((self._num_instances, self._num_fixed_tendons), dtype=wp.float32, device=self.device)
        return self._fixed_tendon_rest_length

    @property
    def fixed_tendon_offset(self) -> wp.array:
        """Fixed tendon offset. Shape: (N, num_fixed_tendons)."""
        if self._num_fixed_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._fixed_tendon_offset is None:
            return wp.zeros((self._num_instances, self._num_fixed_tendons), dtype=wp.float32, device=self.device)
        return self._fixed_tendon_offset

    @property
    def fixed_tendon_pos_limits(self) -> wp.array:
        """Fixed tendon position limits. dtype=wp.vec2f, shape: (N, num_fixed_tendons)."""
        if self._num_fixed_tendons == 0:
            return wp.zeros((self._num_instances, 0, 2), dtype=wp.float32, device=self.device)
        if self._fixed_tendon_pos_limits is None:
            np_limits = np.zeros((self._num_instances, self._num_fixed_tendons, 2), dtype=np.float32)
            np_limits[..., 0] = -float("inf")
            np_limits[..., 1] = float("inf")
            return wp.array(np_limits, dtype=wp.float32, device=self.device).view(wp.vec2f)
        return self._fixed_tendon_pos_limits

    # -- Spatial tendon properties --

    @property
    def spatial_tendon_stiffness(self) -> wp.array:
        """Spatial tendon stiffness. Shape: (N, num_spatial_tendons)."""
        if self._num_spatial_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._spatial_tendon_stiffness is None:
            return wp.zeros((self._num_instances, self._num_spatial_tendons), dtype=wp.float32, device=self.device)
        return self._spatial_tendon_stiffness

    @property
    def spatial_tendon_damping(self) -> wp.array:
        """Spatial tendon damping. Shape: (N, num_spatial_tendons)."""
        if self._num_spatial_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._spatial_tendon_damping is None:
            return wp.zeros((self._num_instances, self._num_spatial_tendons), dtype=wp.float32, device=self.device)
        return self._spatial_tendon_damping

    @property
    def spatial_tendon_limit_stiffness(self) -> wp.array:
        """Spatial tendon limit stiffness. Shape: (N, num_spatial_tendons)."""
        if self._num_spatial_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._spatial_tendon_limit_stiffness is None:
            return wp.zeros((self._num_instances, self._num_spatial_tendons), dtype=wp.float32, device=self.device)
        return self._spatial_tendon_limit_stiffness

    @property
    def spatial_tendon_offset(self) -> wp.array:
        """Spatial tendon offset. Shape: (N, num_spatial_tendons)."""
        if self._num_spatial_tendons == 0:
            return wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        if self._spatial_tendon_offset is None:
            return wp.zeros((self._num_instances, self._num_spatial_tendons), dtype=wp.float32, device=self.device)
        return self._spatial_tendon_offset

    # -- Update method --

    def update(self, dt: float) -> None:
        """Update data (no-op for mock)."""
        pass

    # -- Internal helper --

    def _identity_quat_np(self, *shape: int) -> np.ndarray:
        """Create identity quaternion numpy array (w, x, y, z) = (1, 0, 0, 0)."""
        quat = np.zeros((*shape, 4), dtype=np.float32)
        quat[..., 0] = 1.0
        return quat

    # -- Setters --

    def set_default_root_pose(self, value: torch.Tensor) -> None:
        self._default_root_pose = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_default_root_vel(self, value: torch.Tensor) -> None:
        self._default_root_vel = wp.from_torch(value.to(self.device).contiguous())

    def set_default_joint_pos(self, value: torch.Tensor) -> None:
        self._default_joint_pos = wp.from_torch(value.to(self.device).contiguous())

    def set_default_joint_vel(self, value: torch.Tensor) -> None:
        self._default_joint_vel = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_pos_target(self, value: torch.Tensor) -> None:
        self._joint_pos_target = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_vel_target(self, value: torch.Tensor) -> None:
        self._joint_vel_target = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_effort_target(self, value: torch.Tensor) -> None:
        self._joint_effort_target = wp.from_torch(value.to(self.device).contiguous())

    def set_computed_torque(self, value: torch.Tensor) -> None:
        self._computed_torque = wp.from_torch(value.to(self.device).contiguous())

    def set_applied_torque(self, value: torch.Tensor) -> None:
        self._applied_torque = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_stiffness(self, value: torch.Tensor) -> None:
        self._joint_stiffness = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_damping(self, value: torch.Tensor) -> None:
        self._joint_damping = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_armature(self, value: torch.Tensor) -> None:
        self._joint_armature = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_friction_coeff(self, value: torch.Tensor) -> None:
        self._joint_friction_coeff = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_pos_limits(self, value: torch.Tensor) -> None:
        self._joint_pos_limits = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_vel_limits(self, value: torch.Tensor) -> None:
        self._joint_vel_limits = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_effort_limits(self, value: torch.Tensor) -> None:
        self._joint_effort_limits = wp.from_torch(value.to(self.device).contiguous())

    def set_soft_joint_pos_limits(self, value: torch.Tensor) -> None:
        self._soft_joint_pos_limits = wp.from_torch(value.to(self.device).contiguous())

    def set_soft_joint_vel_limits(self, value: torch.Tensor) -> None:
        self._soft_joint_vel_limits = wp.from_torch(value.to(self.device).contiguous())

    def set_gear_ratio(self, value: torch.Tensor) -> None:
        self._gear_ratio = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_pos(self, value: torch.Tensor) -> None:
        self._joint_pos = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_vel(self, value: torch.Tensor) -> None:
        self._joint_vel = wp.from_torch(value.to(self.device).contiguous())

    def set_joint_acc(self, value: torch.Tensor) -> None:
        self._joint_acc = wp.from_torch(value.to(self.device).contiguous())

    def set_root_link_pose_w(self, value: torch.Tensor) -> None:
        self._root_link_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_root_link_vel_w(self, value: torch.Tensor) -> None:
        self._root_link_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_root_com_pose_w(self, value: torch.Tensor) -> None:
        self._root_com_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_root_com_vel_w(self, value: torch.Tensor) -> None:
        self._root_com_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_link_pose_w(self, value: torch.Tensor) -> None:
        self._body_link_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_body_link_vel_w(self, value: torch.Tensor) -> None:
        self._body_link_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_com_pose_w(self, value: torch.Tensor) -> None:
        self._body_com_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_body_com_vel_w(self, value: torch.Tensor) -> None:
        self._body_com_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_com_acc_w(self, value: torch.Tensor) -> None:
        self._body_com_acc_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_com_pose_b(self, value: torch.Tensor) -> None:
        self._body_com_pose_b = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_body_mass(self, value: torch.Tensor) -> None:
        self._body_mass = wp.from_torch(value.to(self.device).contiguous())

    def set_body_inertia(self, value: torch.Tensor) -> None:
        self._body_inertia = wp.from_torch(value.to(self.device).contiguous())

    def set_body_incoming_joint_wrench_b(self, value: torch.Tensor) -> None:
        self._body_incoming_joint_wrench_b = wp.from_torch(value.to(self.device).contiguous())

    def set_fixed_tendon_stiffness(self, value: torch.Tensor) -> None:
        self._fixed_tendon_stiffness = wp.from_torch(value.to(self.device).contiguous())

    def set_fixed_tendon_damping(self, value: torch.Tensor) -> None:
        self._fixed_tendon_damping = wp.from_torch(value.to(self.device).contiguous())

    def set_fixed_tendon_limit_stiffness(self, value: torch.Tensor) -> None:
        self._fixed_tendon_limit_stiffness = wp.from_torch(value.to(self.device).contiguous())

    def set_fixed_tendon_rest_length(self, value: torch.Tensor) -> None:
        self._fixed_tendon_rest_length = wp.from_torch(value.to(self.device).contiguous())

    def set_fixed_tendon_offset(self, value: torch.Tensor) -> None:
        self._fixed_tendon_offset = wp.from_torch(value.to(self.device).contiguous())

    def set_fixed_tendon_pos_limits(self, value: torch.Tensor) -> None:
        self._fixed_tendon_pos_limits = wp.from_torch(value.to(self.device).contiguous())

    def set_spatial_tendon_stiffness(self, value: torch.Tensor) -> None:
        self._spatial_tendon_stiffness = wp.from_torch(value.to(self.device).contiguous())

    def set_spatial_tendon_damping(self, value: torch.Tensor) -> None:
        self._spatial_tendon_damping = wp.from_torch(value.to(self.device).contiguous())

    def set_spatial_tendon_limit_stiffness(self, value: torch.Tensor) -> None:
        self._spatial_tendon_limit_stiffness = wp.from_torch(value.to(self.device).contiguous())

    def set_spatial_tendon_offset(self, value: torch.Tensor) -> None:
        self._spatial_tendon_offset = wp.from_torch(value.to(self.device).contiguous())

    def set_mock_data(self, **kwargs) -> None:
        """Bulk setter for mock data.

        Accepts any property name as a keyword argument with a tensor value.
        """
        for key, value in kwargs.items():
            setter = getattr(self, f"set_{key}", None)
            if setter is not None:
                setter(value)
            else:
                raise ValueError(f"Unknown property: {key}")


class MockArticulation:
    """Mock articulation asset for testing without Isaac Sim.

    This class mimics the interface of BaseArticulation for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        num_joints: int,
        num_bodies: int,
        joint_names: list[str] | None = None,
        body_names: list[str] | None = None,
        is_fixed_base: bool = False,
        num_fixed_tendons: int = 0,
        num_spatial_tendons: int = 0,
        fixed_tendon_names: list[str] | None = None,
        spatial_tendon_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock articulation.

        Args:
            num_instances: Number of articulation instances.
            num_joints: Number of joints.
            num_bodies: Number of bodies.
            joint_names: Names of joints.
            body_names: Names of bodies.
            is_fixed_base: Whether the articulation has a fixed base.
            num_fixed_tendons: Number of fixed tendons.
            num_spatial_tendons: Number of spatial tendons.
            fixed_tendon_names: Names of fixed tendons.
            spatial_tendon_names: Names of spatial tendons.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_joints = num_joints
        self._num_bodies = num_bodies
        self._is_fixed_base = is_fixed_base
        self._num_fixed_tendons = num_fixed_tendons
        self._num_spatial_tendons = num_spatial_tendons
        self._device = device

        self._joint_names = joint_names or [f"joint_{i}" for i in range(num_joints)]
        self._body_names = body_names or [f"body_{i}" for i in range(num_bodies)]
        self._fixed_tendon_names = fixed_tendon_names or [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]
        self._spatial_tendon_names = spatial_tendon_names or [f"spatial_tendon_{i}" for i in range(num_spatial_tendons)]

        self._data = MockArticulationData(
            num_instances=num_instances,
            num_joints=num_joints,
            num_bodies=num_bodies,
            joint_names=self._joint_names,
            body_names=self._body_names,
            num_fixed_tendons=num_fixed_tendons,
            num_spatial_tendons=num_spatial_tendons,
            fixed_tendon_names=self._fixed_tendon_names,
            spatial_tendon_names=self._spatial_tendon_names,
            device=device,
        )

        # Actuators (empty dict for mock)
        self.actuators: dict = {}

    # -- Properties --

    @property
    def data(self) -> MockArticulationData:
        """Data container for the articulation."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of articulation instances."""
        return self._num_instances

    @property
    def device(self) -> str:
        """Device for tensor allocation."""
        return self._device

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation has a fixed base."""
        return self._is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        return self._num_joints

    @property
    def num_bodies(self) -> int:
        """Number of bodies."""
        return self._num_bodies

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons."""
        return self._num_fixed_tendons

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons."""
        return self._num_spatial_tendons

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names."""
        return self._joint_names

    @property
    def body_names(self) -> list[str]:
        """Ordered body names."""
        return self._body_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered fixed tendon names."""
        return self._fixed_tendon_names

    @property
    def spatial_tendon_names(self) -> list[str]:
        """Ordered spatial tendon names."""
        return self._spatial_tendon_names

    @property
    def root_view(self) -> None:
        """Returns None (no physics view in mock)."""
        return None

    @property
    def instantaneous_wrench_composer(self) -> None:
        """Returns None (no wrench composer in mock)."""
        return None

    @property
    def permanent_wrench_composer(self) -> None:
        """Returns None (no wrench composer in mock)."""
        return None

    # -- Core methods --

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Reset articulation state for specified environments."""
        pass

    def write_data_to_sim(self) -> None:
        """Write data to simulation (no-op for mock)."""
        pass

    def update(self, dt: float) -> None:
        """Update articulation data."""
        self._data.update(dt)

    # -- Finder methods --

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies by name regex patterns."""
        return self._find_by_regex(self._body_names, name_keys, preserve_order)

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find joints by name regex patterns."""
        names = self._joint_names
        if joint_subset is not None:
            subset_indices = [self._joint_names.index(n) for n in joint_subset]
            names = joint_subset
            indices, matched_names = self._find_by_regex(names, name_keys, preserve_order)
            indices = [subset_indices[i] for i in indices]
            return indices, matched_names
        return self._find_by_regex(names, name_keys, preserve_order)

    def find_fixed_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find fixed tendons by name regex patterns."""
        names = self._fixed_tendon_names
        if tendon_subsets is not None:
            subset_indices = [self._fixed_tendon_names.index(n) for n in tendon_subsets]
            names = tendon_subsets
            indices, matched_names = self._find_by_regex(names, name_keys, preserve_order)
            indices = [subset_indices[i] for i in indices]
            return indices, matched_names
        return self._find_by_regex(names, name_keys, preserve_order)

    def find_spatial_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find spatial tendons by name regex patterns."""
        names = self._spatial_tendon_names
        if tendon_subsets is not None:
            subset_indices = [self._spatial_tendon_names.index(n) for n in tendon_subsets]
            names = tendon_subsets
            indices, matched_names = self._find_by_regex(names, name_keys, preserve_order)
            indices = [subset_indices[i] for i in indices]
            return indices, matched_names
        return self._find_by_regex(names, name_keys, preserve_order)

    def _find_by_regex(
        self, names: list[str], name_keys: str | Sequence[str], preserve_order: bool
    ) -> tuple[list[int], list[str]]:
        """Find items by regex patterns."""
        if isinstance(name_keys, str):
            name_keys = [name_keys]

        matched_indices = []
        matched_names = []

        if preserve_order:
            for key in name_keys:
                pattern = re.compile(key)
                for i, name in enumerate(names):
                    if pattern.fullmatch(name) and i not in matched_indices:
                        matched_indices.append(i)
                        matched_names.append(name)
        else:
            for i, name in enumerate(names):
                for key in name_keys:
                    pattern = re.compile(key)
                    if pattern.fullmatch(name):
                        matched_indices.append(i)
                        matched_names.append(name)
                        break

        return matched_indices, matched_names

    # -- State writer methods (no-op for mock) --

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root state to simulation."""
        pass

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root CoM state to simulation."""
        pass

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root link state to simulation."""
        pass

    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root pose to simulation."""
        pass

    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root link pose to simulation."""
        pass

    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root CoM pose to simulation."""
        pass

    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root velocity to simulation."""
        pass

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root CoM velocity to simulation."""
        pass

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write root link velocity to simulation."""
        pass

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor | wp.array,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint state to simulation."""
        pass

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint positions to simulation."""
        pass

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint velocities to simulation."""
        pass

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint stiffness to simulation."""
        pass

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint damping to simulation."""
        pass

    def write_joint_position_limit_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        """Write joint position limits to simulation."""
        pass

    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint velocity limits to simulation."""
        pass

    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint effort limits to simulation."""
        pass

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint armature to simulation."""
        pass

    def write_joint_friction_coefficient_to_sim(
        self,
        coeff: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint friction coefficient to simulation."""
        pass

    def write_joint_friction_to_sim(
        self,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint friction to simulation."""
        pass

    def write_joint_limits_to_sim(
        self,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Write joint limits to simulation."""
        pass

    # -- Setter methods --

    def set_masses(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set body masses."""
        pass

    def set_coms(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set body centers of mass."""
        pass

    def set_inertias(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set body inertias."""
        pass

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        positions: torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        is_global: bool = False,
    ) -> None:
        """Set external forces and torques."""
        pass

    def set_joint_position_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set joint position targets."""
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        if self._data._joint_pos_target is None:
            self._data._joint_pos_target = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.float32, device=self._device
            )
        pos_target = wp.to_torch(self._data._joint_pos_target)
        pos_target[env_ids, joint_ids] = target.to(self._device)
        self._data._joint_pos_target = wp.from_torch(pos_target)

    def set_joint_velocity_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set joint velocity targets."""
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        if self._data._joint_vel_target is None:
            self._data._joint_vel_target = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.float32, device=self._device
            )
        vel_target = wp.to_torch(self._data._joint_vel_target)
        vel_target[env_ids, joint_ids] = target.to(self._device)
        self._data._joint_vel_target = wp.from_torch(vel_target)

    def set_joint_effort_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set joint effort targets."""
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        if self._data._joint_effort_target is None:
            self._data._joint_effort_target = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.float32, device=self._device
            )
        effort_target = wp.to_torch(self._data._joint_effort_target)
        effort_target[env_ids, joint_ids] = target.to(self._device)
        self._data._joint_effort_target = wp.from_torch(effort_target)

    # -- Tendon methods (fixed) --

    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon stiffness."""
        pass

    def set_fixed_tendon_damping(
        self,
        damping: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon damping."""
        pass

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon limit stiffness."""
        pass

    def set_fixed_tendon_position_limit(
        self,
        limit: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon position limit."""
        pass

    def set_fixed_tendon_limit(
        self,
        limit: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon limit (alias for set_fixed_tendon_position_limit)."""
        pass

    def set_fixed_tendon_rest_length(
        self,
        rest_length: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon rest length."""
        pass

    def set_fixed_tendon_offset(
        self,
        offset: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set fixed tendon offset."""
        pass

    def write_fixed_tendon_properties_to_sim(
        self,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write fixed tendon properties to simulation."""
        pass

    # -- Tendon methods (spatial) --

    def set_spatial_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set spatial tendon stiffness."""
        pass

    def set_spatial_tendon_damping(
        self,
        damping: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set spatial tendon damping."""
        pass

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set spatial tendon limit stiffness."""
        pass

    def set_spatial_tendon_offset(
        self,
        offset: torch.Tensor,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set spatial tendon offset."""
        pass

    def write_spatial_tendon_properties_to_sim(
        self,
        tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write spatial tendon properties to simulation."""
        pass

    # -- Shape validation helpers --

    _DTYPE_TO_TORCH_TRAILING_DIMS: dict[type, tuple[int, ...]] = {
        wp.float32: (),
        wp.int32: (),
        wp.vec2f: (2,),
        wp.vec3f: (3,),
        wp.vec4f: (4,),
        wp.transformf: (7,),
        wp.spatial_vectorf: (6,),
    }

    def assert_shape_and_dtype(
        self, tensor: float | torch.Tensor | wp.array, shape: tuple[int, ...], dtype: type, name: str = ""
    ) -> None:
        if __debug__:
            cls = type(self).__name__
            prefix = f"{cls}: '{name}' " if name else f"{cls}: "
            if isinstance(tensor, (int, float)):
                return
            elif isinstance(tensor, wp.array):
                assert tensor.dtype == dtype, f"{prefix}Dtype mismatch: {tensor.dtype} != {dtype}"
                assert tensor.shape == shape, f"{prefix}Shape mismatch: {tensor.shape} != {shape}"
            elif isinstance(tensor, torch.Tensor):
                offset = self._DTYPE_TO_TORCH_TRAILING_DIMS.get(dtype)
                if offset is None:
                    raise ValueError(f"Unsupported dtype: {dtype}")
                assert tensor.shape == (*shape, *offset), (
                    f"{prefix}Shape mismatch: {tensor.shape} != {(*shape, *offset)}"
                )

    def _resolve_n_envs(self, env_ids):
        if env_ids is None:
            return self._num_instances
        if isinstance(env_ids, (torch.Tensor, wp.array)):
            return env_ids.shape[0]
        return len(env_ids)

    def _resolve_n_items(self, item_ids, total):
        if item_ids is None or item_ids == slice(None):
            return total
        if isinstance(item_ids, (torch.Tensor, wp.array)):
            return item_ids.shape[0]
        return len(item_ids)

    # -- Index/Mask methods --

    # Root pose/velocity

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n = self._resolve_n_envs(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n = self._resolve_n_envs(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n = self._resolve_n_envs(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n = self._resolve_n_envs(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n = self._resolve_n_envs(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n = self._resolve_n_envs(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")

    # Joint write methods

    def write_joint_position_to_sim_index(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(position, (n_e, n_j), wp.float32, "position")

    def write_joint_position_to_sim_mask(
        self,
        *,
        position: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(position, (self._num_instances, self._num_joints), wp.float32, "position")

    def write_joint_velocity_to_sim_index(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(velocity, (n_e, n_j), wp.float32, "velocity")

    def write_joint_velocity_to_sim_mask(
        self,
        *,
        velocity: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(velocity, (self._num_instances, self._num_joints), wp.float32, "velocity")

    def write_joint_stiffness_to_sim_index(
        self,
        *,
        stiffness: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(stiffness, (n_e, n_j), wp.float32, "stiffness")

    def write_joint_stiffness_to_sim_mask(
        self,
        *,
        stiffness: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._num_joints), wp.float32, "stiffness")

    def write_joint_damping_to_sim_index(
        self,
        *,
        damping: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(damping, (n_e, n_j), wp.float32, "damping")

    def write_joint_damping_to_sim_mask(
        self,
        *,
        damping: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(damping, (self._num_instances, self._num_joints), wp.float32, "damping")

    def write_joint_position_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        if isinstance(limits, (int, float)):
            raise ValueError("Float scalars are not supported for position limits (vec2f dtype)")
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(limits, (n_e, n_j), wp.vec2f, "limits")

    def write_joint_position_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        warn_limit_violation: bool = True,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        if isinstance(limits, (int, float)):
            raise ValueError("Float scalars are not supported for position limits (vec2f dtype)")
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.vec2f, "limits")

    def write_joint_velocity_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(limits, (n_e, n_j), wp.float32, "limits")

    def write_joint_velocity_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.float32, "limits")

    def write_joint_effort_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(limits, (n_e, n_j), wp.float32, "limits")

    def write_joint_effort_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.float32, "limits")

    def write_joint_armature_to_sim_index(
        self,
        *,
        armature: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(armature, (n_e, n_j), wp.float32, "armature")

    def write_joint_armature_to_sim_mask(
        self,
        *,
        armature: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(armature, (self._num_instances, self._num_joints), wp.float32, "armature")

    def write_joint_friction_coefficient_to_sim_index(
        self,
        *,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(joint_friction_coeff, (n_e, n_j), wp.float32, "joint_friction_coeff")

    def write_joint_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(
            joint_friction_coeff, (self._num_instances, self._num_joints), wp.float32, "joint_friction_coeff"
        )

    # Body setter methods

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_b = self._resolve_n_items(body_ids, self._num_bodies)
        self.assert_shape_and_dtype(masses, (n_e, n_b), wp.float32, "masses")

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_b = self._resolve_n_items(body_ids, self._num_bodies)
        self.assert_shape_and_dtype(coms, (n_e, n_b), wp.transformf, "coms")

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_b = self._resolve_n_items(body_ids, self._num_bodies)
        self.assert_shape_and_dtype(inertias, (n_e, n_b, 9), wp.float32, "inertias")

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")

    # Joint target methods

    def set_joint_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(target, (n_e, n_j), wp.float32, "target")

    def set_joint_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(target, (self._num_instances, self._num_joints), wp.float32, "target")

    def set_joint_velocity_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(target, (n_e, n_j), wp.float32, "target")

    def set_joint_velocity_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(target, (self._num_instances, self._num_joints), wp.float32, "target")

    def set_joint_effort_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_j = self._resolve_n_items(joint_ids, self._num_joints)
        self.assert_shape_and_dtype(target, (n_e, n_j), wp.float32, "target")

    def set_joint_effort_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(target, (self._num_instances, self._num_joints), wp.float32, "target")

    # Fixed tendon methods

    def set_fixed_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(fixed_tendon_ids, self._num_fixed_tendons)
        self.assert_shape_and_dtype(stiffness, (n_e, n_t), wp.float32, "stiffness")

    def set_fixed_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._num_fixed_tendons), wp.float32, "stiffness")

    def set_fixed_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(fixed_tendon_ids, self._num_fixed_tendons)
        self.assert_shape_and_dtype(damping, (n_e, n_t), wp.float32, "damping")

    def set_fixed_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(damping, (self._num_instances, self._num_fixed_tendons), wp.float32, "damping")

    def set_fixed_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(fixed_tendon_ids, self._num_fixed_tendons)
        self.assert_shape_and_dtype(limit_stiffness, (n_e, n_t), wp.float32, "limit_stiffness")

    def set_fixed_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(
            limit_stiffness, (self._num_instances, self._num_fixed_tendons), wp.float32, "limit_stiffness"
        )

    def set_fixed_tendon_position_limit_index(
        self,
        *,
        limit: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(fixed_tendon_ids, self._num_fixed_tendons)
        self.assert_shape_and_dtype(limit, (n_e, n_t), wp.float32, "limit")

    def set_fixed_tendon_position_limit_mask(
        self,
        *,
        limit: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(limit, (self._num_instances, self._num_fixed_tendons), wp.float32, "limit")

    def set_fixed_tendon_rest_length_index(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(fixed_tendon_ids, self._num_fixed_tendons)
        self.assert_shape_and_dtype(rest_length, (n_e, n_t), wp.float32, "rest_length")

    def set_fixed_tendon_rest_length_mask(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(
            rest_length, (self._num_instances, self._num_fixed_tendons), wp.float32, "rest_length"
        )

    def set_fixed_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(fixed_tendon_ids, self._num_fixed_tendons)
        self.assert_shape_and_dtype(offset, (n_e, n_t), wp.float32, "offset")

    def set_fixed_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(offset, (self._num_instances, self._num_fixed_tendons), wp.float32, "offset")

    def write_fixed_tendon_properties_to_sim_index(
        self,
        *,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_fixed_tendon_properties_to_sim_mask(
        self,
        *,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    # Spatial tendon methods

    def set_spatial_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(spatial_tendon_ids, self._num_spatial_tendons)
        self.assert_shape_and_dtype(stiffness, (n_e, n_t), wp.float32, "stiffness")

    def set_spatial_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(
            stiffness, (self._num_instances, self._num_spatial_tendons), wp.float32, "stiffness"
        )

    def set_spatial_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(spatial_tendon_ids, self._num_spatial_tendons)
        self.assert_shape_and_dtype(damping, (n_e, n_t), wp.float32, "damping")

    def set_spatial_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(damping, (self._num_instances, self._num_spatial_tendons), wp.float32, "damping")

    def set_spatial_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(spatial_tendon_ids, self._num_spatial_tendons)
        self.assert_shape_and_dtype(limit_stiffness, (n_e, n_t), wp.float32, "limit_stiffness")

    def set_spatial_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(
            limit_stiffness, (self._num_instances, self._num_spatial_tendons), wp.float32, "limit_stiffness"
        )

    def set_spatial_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        n_e = self._resolve_n_envs(env_ids)
        n_t = self._resolve_n_items(spatial_tendon_ids, self._num_spatial_tendons)
        self.assert_shape_and_dtype(offset, (n_e, n_t), wp.float32, "offset")

    def set_spatial_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        self.assert_shape_and_dtype(offset, (self._num_instances, self._num_spatial_tendons), wp.float32, "offset")

    def write_spatial_tendon_properties_to_sim_index(
        self,
        *,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_spatial_tendon_properties_to_sim_mask(
        self,
        *,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        pass
