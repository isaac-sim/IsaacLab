# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab.utils.warp import ProxyArray

import isaaclab_ovphysx.tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager

from .contact_sensor_data import ContactSensorData
from .kernels import (
    compute_first_transition_kernel,
    reset_contact_sensor_kernel,
    split_flat_pose_to_pos_quat,
    unpack_contact_buffer_data,  # noqa: F401  -- reserved for v2 contact-points support
    update_net_forces_kernel,
)

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg


class ContactSensor(BaseContactSensor):
    """An ovphysx contact reporting sensor.

    Reports normal contact forces in world frame using the ovphysx
    :class:`ContactBinding` API. The `PhysxContactReportAPI` USD schema must
    be applied to each sensor body (set
    :attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sensors`
    on the asset spawner).

    Optional features tracked by :attr:`ContactSensorCfg`:

    * ``track_pose`` — sensor body pose via a ``RIGID_BODY_POSE`` tensor binding.
    * ``filter_prim_paths_expr`` — per-partner filtered forces via
      :meth:`ContactBinding.read_force_matrix`.
    * ``track_air_time`` — air/contact time tracking and
      :meth:`compute_first_contact` / :meth:`compute_first_air`.

    The following config flags are not supported on the ovphysx backend yet
    (the underlying ovphysx APIs do not expose tensor-friendly per-sensor
    reads — see ``docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md``):

    * ``track_contact_points``
    * ``track_friction_forces``

    Setting either flag raises :class:`NotImplementedError` at initialization.
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the contact sensor."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        # Reject the v1 unsupported optional features early, before USD discovery.
        if cfg.track_contact_points or cfg.track_friction_forces:
            raise NotImplementedError(
                "ovphysx ContactSensor does not yet support 'track_contact_points' or 'track_friction_forces'."
                " ovphysx 0.3.7 lacks tensor-friendly per-sensor read APIs for these features."
                " See docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md for the maintainer asks."
            )

        self._data: ContactSensorData = ContactSensorData()
        # Backend handles, populated in _initialize_impl.
        self._physx_instance: Any = None
        self._contact_binding: Any = None
        self._pose_binding: Any = None
        # Pre-allocated read buffers, populated in _create_buffers.
        self._net_forces_flat_buf: wp.array | None = None
        self._force_matrix_flat_buf: wp.array | None = None
        self._poses_flat_buf: wp.array | None = None
        # Body names (resolved during init).
        self._body_names: list[str] = []
        # Default backend tunables matching the PhysX backend.
        if self.cfg.max_contact_data_count_per_prim is None:
            self.cfg.max_contact_data_count_per_prim = 4
        if self.cfg.force_threshold is None:
            self.cfg.force_threshold = 1.0

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tbackend           : ovphysx\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of bodies  : {self.num_sensors}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int | None:
        if self._contact_binding is None:
            return None
        return self._contact_binding.sensor_count

    @property
    def data(self) -> ContactSensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_sensors(self) -> int:
        return self._num_sensors

    @property
    def body_names(self) -> list[str] | None:
        if not self._body_names:
            return None
        return list(self._body_names)

    @property
    def contact_view(self) -> Any:
        """The underlying ovphysx :class:`ContactBinding` (or ``None`` before init).

        .. note::
            Use this view with caution. It owns native handles released at
            simulation stop.
        """
        return self._contact_binding

    @property
    def pose_binding(self) -> Any:
        """The underlying ovphysx ``RIGID_BODY_POSE`` :class:`TensorBinding`.

        ``None`` if ``cfg.track_pose`` is False or before initialization.
        """
        return self._pose_binding

    """
    Implementation.
    """

    def _initialize_impl(self) -> None:
        super()._initialize_impl()

        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._physx_instance = physx_instance

        # Discover sensor bodies. Mirror the PhysX discovery path.
        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names: list[str] = []
        for prim in sim_utils.find_matching_prims(template_prim_path + "/" + leaf_pattern):
            if "PhysxContactReportAPI" in prim.GetAppliedSchemas():
                body_names.append(prim.GetPath().pathString.rsplit("/", 1)[-1])
        if not body_names:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )
        self._body_names = body_names
        self._num_sensors = len(body_names)

        # Build glob patterns: one per (env, sensor body).
        # IsaacLab path forms map to ovphysx fnmatch globs the same way Articulation does.
        base_glob = self.cfg.prim_path.rsplit("/", 1)[0]
        base_glob = re.sub(r"\{ENV_REGEX_NS\}", "*", base_glob)
        base_glob = re.sub(r"\.\*", "*", base_glob)
        sensor_patterns = [f"{base_glob}/{name}" for name in body_names]

        # Build filter patterns (flat: len = n_sensors * filters_per_sensor).
        filter_globs = [
            re.sub(r"\.\*", "*", re.sub(r"\{ENV_REGEX_NS\}", "*", expr))
            for expr in self.cfg.filter_prim_paths_expr
        ]
        filters_per_sensor = len(filter_globs)
        if filters_per_sensor > 0:
            filter_patterns: list[str] | None = filter_globs * self._num_sensors
        else:
            filter_patterns = None

        # Create the contact binding (must happen BEFORE the next step()).
        max_count = self.cfg.max_contact_data_count_per_prim * self._num_sensors * self._num_envs
        self._contact_binding = physx_instance.create_contact_binding(
            sensor_patterns=sensor_patterns,
            filter_patterns=filter_patterns,
            filters_per_sensor=filters_per_sensor,
            max_contact_data_count=max_count,
        )

        # Validate that ovphysx matched what we expected. sensor_count is the
        # global total (envs * bodies); the binding does not split per env.
        expected_sensors = self._num_sensors * self._num_envs
        if self._contact_binding.sensor_count != expected_sensors:
            raise RuntimeError(
                "Failed to initialize contact binding for specified bodies."
                f"\n\tInput prim path     : {self.cfg.prim_path}"
                f"\n\tExpected sensors    : {expected_sensors} ({self._num_envs} envs * {self._num_sensors} bodies)"
                f"\n\tBound sensors       : {self._contact_binding.sensor_count}"
            )

        # Optional: pose tracking via a RIGID_BODY_POSE tensor binding.
        # ovphysx uses fnmatch and does not brace-expand, so we widen to a single
        # "*" leaf pattern under the base glob. This relies on the prim_path
        # already isolating the sensor bodies (e.g. ".*_FOOT" matches all four
        # feet and no siblings). The post-bind count check below catches a
        # mismatch.
        if self.cfg.track_pose:
            single_pose_pattern = f"{base_glob}/*"
            self._pose_binding = physx_instance.create_tensor_binding(
                pattern=single_pose_pattern, tensor_type=TT.RIGID_BODY_POSE,
            )
            if self._pose_binding.count != expected_sensors:
                raise RuntimeError(
                    "RIGID_BODY_POSE binding count mismatch."
                    f"\n\tPattern: {single_pose_pattern}"
                    f"\n\tBound  : {self._pose_binding.count}"
                    f"\n\tExpect : {expected_sensors}"
                )

        self._create_buffers()

    def _create_buffers(self) -> None:
        """Allocate Warp buffers, including the pre-allocated ovphysx read tensors."""
        self._num_filter_shapes = self._contact_binding.filter_count if self.cfg.filter_prim_paths_expr else 0
        self._history_length = max(self.cfg.history_length, 1)

        # Sensor data buffers (delegated to the data container).
        self._data.create_buffers(
            num_envs=self._num_envs,
            num_sensors=self._num_sensors,
            num_filter_shapes=self._num_filter_shapes,
            history_length=self.cfg.history_length,
            track_pose=self.cfg.track_pose,
            track_air_time=self.cfg.track_air_time,
            track_contact_points=self.cfg.track_contact_points,
            track_friction_forces=self.cfg.track_friction_forces,
            device=self._device,
        )

        # ovphysx ContactBinding writes into pre-allocated tensors. We allocate
        # them once here and reuse every step. Shape: [S, 3] for net forces,
        # [S, F, 3] for the force matrix (S = num_envs * num_sensors).
        flat_count = self._num_envs * self._num_sensors
        self._net_forces_flat_buf = wp.zeros((flat_count, 3), dtype=wp.float32, device=self._device)
        if self._num_filter_shapes > 0:
            self._force_matrix_flat_buf = wp.zeros(
                (flat_count, self._num_filter_shapes, 3), dtype=wp.float32, device=self._device,
            )
        else:
            self._force_matrix_flat_buf = None

        # Pose buffer: [S, 7] for RIGID_BODY_POSE (px,py,pz,qx,qy,qz,qw).
        if self.cfg.track_pose:
            self._poses_flat_buf = wp.zeros((flat_count, 7), dtype=wp.float32, device=self._device)
        else:
            self._poses_flat_buf = None
