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
