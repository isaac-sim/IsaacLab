# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent PVA output storage and feature contracts."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ...utils.backend_utils import FactoryBase
from ...utils.leapp import (
    POSE7_ELEMENT_NAMES,
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
    leapp_tensor_semantics,
)
from ...utils.warp import ProxyArray
from ...utils.warp.memory import split_views
from .base_pva_data import BasePvaData


class _PvaDataStorage(BasePvaData):
    """Shared PVA measurements with explicit optional output groups.

    Enabled properties return ``None`` before allocation. Disabled properties raise
    ``NotImplementedError``. Pose/component views share memory, including mutations,
    and may be noncontiguous. Reading a property never launches a kernel.

    Backends subclass this container for compatibility or additional measurements while
    retaining these contracts; native input layouts do not need to match this output layout.
    """

    def __init__(self, data_types: Sequence[str] = ("pose", "velocity", "acceleration", "projected_gravity")):
        self._data_types = frozenset(data_types)
        if "pose" not in self._data_types:
            raise ValueError("PVA data_types must include 'pose'")
        unknown = self._data_types - {"pose", "velocity", "acceleration", "projected_gravity"}
        if unknown:
            raise ValueError(f"Unknown core PVA data types: {sorted(unknown)}")
        self._pose_w = self._pos_w = self._quat_w = None
        self._velocity_b = self._acceleration_b = self._gravity_b = None
        self._lin_vel_b = self._ang_vel_b = None
        self._lin_acc_b = self._ang_acc_b = None
        self._projected_gravity_b = None

    @property
    def data_types(self) -> frozenset[str]:
        """Enabled standard output groups, resolved once before allocating buffers."""
        return self._data_types

    def create_buffers(self, num_envs: int, device: str) -> None:
        """Allocate enabled outputs and rebuild views after simulation initialization.

        Reset defaults are identity pose, zero velocity/acceleration, and projected
        gravity along -Z until the first sample supplies the actual scene direction.
        """
        pose_w = wp.full(num_envs, wp.transform_identity(), dtype=wp.transformf, device=device)
        pos, quat = split_views(pose_w, (wp.vec3f, wp.quatf))
        self._pose_w, self._pos_w, self._quat_w = (ProxyArray(a) for a in (pose_w, pos, quat))
        if "velocity" in self.data_types:
            self._velocity_b = wp.zeros(num_envs, dtype=wp.spatial_vectorf, device=device)
            self._lin_vel_b, self._ang_vel_b = (
                ProxyArray(a) for a in split_views(self._velocity_b, (wp.vec3f, wp.vec3f))
            )
        if "acceleration" in self.data_types:
            self._acceleration_b = wp.zeros(num_envs, dtype=wp.spatial_vectorf, device=device)
            self._lin_acc_b, self._ang_acc_b = (
                ProxyArray(a) for a in split_views(self._acceleration_b, (wp.vec3f, wp.vec3f))
            )
        if "projected_gravity" in self.data_types:
            self._gravity_b = wp.full(num_envs, wp.vec3f(0.0, 0.0, -1.0), dtype=wp.vec3f, device=device)
            self._projected_gravity_b = ProxyArray(self._gravity_b)

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def pose_w(self) -> ProxyArray | None:
        """Pose of the sensor origin in world frame [m, unitless].

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        return self._pose_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def pos_w(self) -> ProxyArray | None:
        """Position of the sensor origin in world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._pos_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def quat_w(self) -> ProxyArray | None:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        return self._quat_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def projected_gravity_b(self) -> ProxyArray | None:
        """Gravity direction unit vector projected on the PVA frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if "projected_gravity" not in self.data_types:
            raise NotImplementedError("PVA projected_gravity output was not enabled in PvaCfg.data_types")
        return self._projected_gravity_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def lin_vel_b(self) -> ProxyArray | None:
        """PVA frame linear velocity relative to the world expressed in PVA frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if "velocity" not in self.data_types:
            raise NotImplementedError("PVA velocity output was not enabled in PvaCfg.data_types")
        return self._lin_vel_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def ang_vel_b(self) -> ProxyArray | None:
        """PVA frame angular velocity relative to the world expressed in PVA frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if "velocity" not in self.data_types:
            raise NotImplementedError("PVA velocity output was not enabled in PvaCfg.data_types")
        return self._ang_vel_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_ACCELERATION, element_names=XYZ_ELEMENT_NAMES)
    def lin_acc_b(self) -> ProxyArray | None:
        """Linear acceleration (coordinate) in the PVA frame [m/s^2].

        Equal to gravity in freefall, zero at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if "acceleration" not in self.data_types:
            raise NotImplementedError("PVA acceleration output was not enabled in PvaCfg.data_types")
        return self._lin_acc_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_ACCELERATION, element_names=XYZ_ELEMENT_NAMES)
    def ang_acc_b(self) -> ProxyArray | None:
        """PVA frame angular acceleration relative to the world expressed in PVA frame [rad/s^2].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if "acceleration" not in self.data_types:
            raise NotImplementedError("PVA acceleration output was not enabled in PvaCfg.data_types")
        return self._ang_acc_b


class PvaData(FactoryBase, BasePvaData):
    """Create the backend PVA data container using the active simulation backend."""
