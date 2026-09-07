# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for contact sensor data containers."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

from isaaclab.utils.leapp import (
    POSE7_ELEMENT_NAMES,
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
    leapp_tensor_semantics,
)
from isaaclab.utils.warp import ProxyArray


class BaseContactSensorData(ABC):
    """Data container for the contact reporting sensor.

    This base class defines the interface for contact sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.

    :attr:`net_forces_w` is the total contact force (normal + friction). Newton reports this
    quantity directly. PhysX and OVPhysX cannot compute a total force, so they return
    :attr:`net_normal_forces_w` and warn. The same applies to :attr:`net_forces_w_history`,
    :attr:`force_matrix_w`, and :attr:`force_matrix_w_history`.

    :attr:`friction_forces_w` is the aggregate friction force. Newton reports this as
    :attr:`net_friction_forces_w`. PhysX and OVPhysX only provide filtered friction, so they
    return :attr:`friction_force_matrix_w` and warn.
    """

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def pose_w(self) -> ProxyArray | None:
        """Pose of the sensor origin in world frame.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def pos_w(self) -> ProxyArray | None:
        """Position of the sensor origin in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def quat_w(self) -> ProxyArray | None:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_sensors, 4). The orientation is provided in (x, y, z, w) format.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def net_normal_forces_w(self) -> ProxyArray | None:
        """The net normal contact forces [N] in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        The net total contact force is the sum of this quantity and
        :attr:`net_friction_forces_w`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def net_normal_forces_w_history(self) -> ProxyArray | None:
        """History of net normal contact forces [N].

        Shape is (num_instances, history_length, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, history_length, num_sensors, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def normal_force_matrix_w(self) -> ProxyArray | None:
        """Normal contact forces [N] filtered between sensor and filtered bodies.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.

        The total contact force matrix is the sum of this quantity and
        :attr:`friction_force_matrix_w`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def normal_force_matrix_w_history(self) -> ProxyArray | None:
        """History of filtered normal contact forces [N].

        Shape is (num_instances, history_length, num_sensors, num_filter_shapes), dtype = wp.vec3f.
        In torch this resolves to (num_instances, history_length, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def contact_pos_w(self) -> ProxyArray | None:
        """Average position of contact points.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_contact_points` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def net_friction_forces_w(self) -> ProxyArray | None:
        """The net friction contact forces [N] in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def net_friction_forces_w_history(self) -> ProxyArray | None:
        """History of net friction contact forces [N] in world frame.

        Shape is (num_instances, history_length, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, history_length, num_sensors, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def friction_force_matrix_w(self) -> ProxyArray | None:
        """Friction contact forces [N] filtered between sensor and filtered bodies.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False or no filter objects are configured.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def friction_force_matrix_w_history(self) -> ProxyArray | None:
        """History of filtered friction contact forces [N].

        Shape is (num_instances, history_length, num_sensors, num_filter_shapes), dtype = wp.vec3f.
        In torch this resolves to (num_instances, history_length, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False or no filter objects are configured.
        """
        raise NotImplementedError

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def net_forces_w(self) -> ProxyArray | None:
        """The net total contact forces [N] in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        This is the sum of :attr:`net_normal_forces_w` and :attr:`net_friction_forces_w`.
        PhysX and OVPhysX cannot compute a total force, so they return
        :attr:`net_normal_forces_w` and warn.
        """
        warnings.warn(
            "PhysX does not return a total contact force. This is a known limitation in PhysX"
            " and we are fixing it for the next release. Returning 'net_normal_forces_w'.",
            UserWarning,
            stacklevel=2,
        )
        return self.net_normal_forces_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def net_forces_w_history(self) -> ProxyArray | None:
        """History of net total contact forces [N] in world frame.

        Shape is (num_instances, history_length, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, history_length, num_sensors, 3).

        PhysX and OVPhysX cannot compute a total force, so they return
        :attr:`net_normal_forces_w_history` and warn.
        """
        warnings.warn(
            "PhysX does not return a total contact force. This is a known limitation in PhysX"
            " and we are fixing it for the next release. Returning 'net_normal_forces_w_history'.",
            UserWarning,
            stacklevel=2,
        )
        return self.net_normal_forces_w_history

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def force_matrix_w(self) -> ProxyArray | None:
        """Total filtered contact forces [N] between sensor and filtered bodies.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        This is the sum of :attr:`normal_force_matrix_w` and :attr:`friction_force_matrix_w`.
        PhysX and OVPhysX cannot compute a total force, so they return
        :attr:`normal_force_matrix_w` and warn.

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        warnings.warn(
            "PhysX does not return a total contact force. This is a known limitation in PhysX"
            " and we are fixing it for the next release. Returning 'normal_force_matrix_w'.",
            UserWarning,
            stacklevel=2,
        )
        return self.normal_force_matrix_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def force_matrix_w_history(self) -> ProxyArray | None:
        """History of total filtered contact forces [N].

        Shape is (num_instances, history_length, num_sensors, num_filter_shapes), dtype = wp.vec3f.
        In torch this resolves to (num_instances, history_length, num_sensors, num_filter_shapes, 3).

        PhysX and OVPhysX cannot compute a total force, so they return
        :attr:`normal_force_matrix_w_history` and warn.

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        warnings.warn(
            "PhysX does not return a total contact force. This is a known limitation in PhysX"
            " and we are fixing it for the next release. Returning 'normal_force_matrix_w_history'.",
            UserWarning,
            stacklevel=2,
        )
        return self.normal_force_matrix_w_history

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def friction_forces_w(self) -> ProxyArray | None:
        """The net total friction contact forces [N] in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        This is the same quantity as :attr:`net_friction_forces_w`. PhysX and OVPhysX only
        provide filtered friction, so they return :attr:`friction_force_matrix_w` and warn.

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        warnings.warn(
            "PhysX does not return an aggregate friction force; it only provides filtered friction."
            " This is a known limitation in PhysX and we are fixing it for the next release."
            " Returning 'friction_force_matrix_w'.",
            UserWarning,
            stacklevel=2,
        )
        return self.friction_force_matrix_w

    @property
    @abstractmethod
    @leapp_tensor_semantics()
    def last_air_time(self) -> ProxyArray | None:
        """Time spent in air before last contact.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics()
    def current_air_time(self) -> ProxyArray | None:
        """Time spent in air since last detach.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics()
    def last_contact_time(self) -> ProxyArray | None:
        """Time spent in contact before last detach.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics()
    def current_contact_time(self) -> ProxyArray | None:
        """Time spent in contact since last contact.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError
