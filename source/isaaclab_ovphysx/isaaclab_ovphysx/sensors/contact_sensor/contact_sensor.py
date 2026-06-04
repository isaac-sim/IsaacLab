# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source
from isaaclab.utils.warp import ProxyArray

import isaaclab_ovphysx.tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager

from .contact_sensor_data import ContactSensorData
from .kernels import (
    compute_first_transition_kernel,
    reset_contact_sensor_kernel,
    split_flat_pose_to_pos_quat,
    unpack_contact_buffer_data,  # noqa: F401  -- reserved for v2 contact-points support
    update_net_forces_ovphysx_kernel,
)

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg

logger = logging.getLogger(__name__)


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
    def body_names(self) -> list[str]:
        """The leaf-prim names of the sensor bodies.

        Raises:
            RuntimeError: If accessed before the sensor has been initialized
                (matches the eager non-``None`` contract PhysX provides).
        """
        if not self._body_names:
            raise RuntimeError(
                "OvPhysxContactSensor.body_names accessed before initialization. "
                "Step the simulation once (or wait for PhysicsEvent.PHYSICS_READY) so the "
                "sensor can discover its bodies."
            )
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

        # Discover sensor bodies.  We use ``GetPrimTypeInfo().GetAppliedAPISchemas()``
        # (raw apiSchemas listOp) instead of ``GetAppliedSchemas()`` so that codeless
        # USDs without ``omni.physx``'s plugin loaded still report
        # ``PhysxContactReportAPI``.  Under the kitless ovphysx flow the
        # ``PhysxSchema`` USD plugin is registered by
        # :meth:`OvPhysxManager.initialize` so the wheel-side schema check passes,
        # but the Python-side filtered API still hides ``PhysxContactReportAPI``
        # because the schema TYPE registration only happens when the C++ plugin
        # library is loaded by ``omni.physx``.  The unfiltered API matches what
        # the underlying USD apiSchemas listOp actually carries (verified against
        # :class:`pxr.Sdf.PrimSpec.GetInfo("apiSchemas")`).
        parent_expr, leaf_pattern = self.cfg.prim_path.rsplit("/", 1)
        name_pattern = re.compile(leaf_pattern)

        def has_contact_report(prim) -> bool:
            return bool(name_pattern.fullmatch(prim.GetName())) and (
                "PhysxContactReportAPI" in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            )

        asset_prim, body_parent = resolve_matching_prims_from_source(parent_expr)[0]
        walk_root = asset_prim.GetPath().pathString
        prims = get_all_matching_child_prims(walk_root, predicate=has_contact_report, traverse_instance_prims=False)
        body_names = [prim.GetPath().pathString.rsplit("/", 1)[-1] for prim in prims]
        if not body_names:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )
        self._body_names = body_names
        self._num_sensors = len(body_names)

        # Build glob patterns: one per (env, sensor body).
        # IsaacLab path forms map to ovphysx fnmatch globs the same way Articulation does.
        base_glob = re.sub(r"\{ENV_REGEX_NS\}", "*", body_parent)
        base_glob = re.sub(r"\.\*", "*", base_glob)
        sensor_patterns = [f"{base_glob}/{name}" for name in body_names]

        # Build filter patterns (flat: len = n_sensors * filters_per_sensor).
        filter_globs = [
            re.sub(r"\.\*", "*", re.sub(r"\{ENV_REGEX_NS\}", "*", expr)) for expr in self.cfg.filter_prim_paths_expr
        ]
        filters_per_sensor = len(filter_globs)
        if filters_per_sensor > 0:
            filter_patterns: list[str] | None = filter_globs * self._num_sensors
        else:
            filter_patterns = None

        # Create the contact binding (must happen BEFORE the next step()).
        # OVPhysX's ``InteractiveScene`` runs in ``clone_usd=False`` mode:
        # env_1..N have no USD prim — they're physics-layer clones via
        # ``physx.clone()``.  The parent class's ``find_matching_prims`` walk
        # therefore sees only env_0 and sets ``self._num_envs = 1`` even when
        # the scene is configured for many envs.  We size the
        # ``max_contact_data_count`` for env_0 only here; the binding's
        # ``sensor_count`` after creation gives us the real env count.
        max_count = self.cfg.max_contact_data_count_per_prim * self._num_sensors * self._num_envs
        self._contact_binding = physx_instance.create_contact_binding(
            sensor_patterns=sensor_patterns,
            filter_patterns=filter_patterns,
            filters_per_sensor=filters_per_sensor,
            max_contact_data_count=max_count,
        )

        # Validate: sensor_count must be a non-zero multiple of num_sensors.
        if self._contact_binding.sensor_count == 0 or self._contact_binding.sensor_count % self._num_sensors != 0:
            raise RuntimeError(
                "Failed to initialize contact binding for specified bodies."
                f"\n\tInput prim path     : {self.cfg.prim_path}"
                f"\n\tNum sensor bodies   : {self._num_sensors}"
                f"\n\tBound sensors       : {self._contact_binding.sensor_count}"
            )

        # Override ``_num_envs`` with the binding's view if it differs (it does
        # for any OVPhysX scene with ``num_envs > 1`` due to ``clone_usd=False``).
        # Re-allocate the env-sized buffers from the parent class so they match
        # the real env count.
        binding_num_envs = self._contact_binding.sensor_count // self._num_sensors
        if binding_num_envs != self._num_envs:
            self._num_envs = binding_num_envs
            self._ALL_ENV_MASK = wp.ones((self._num_envs,), dtype=wp.bool, device=self._device)
            self._reset_mask = wp.zeros((self._num_envs,), dtype=wp.bool, device=self._device)
            self._reset_mask_torch = wp.to_torch(self._reset_mask)
            self._is_outdated = wp.ones(self._num_envs, dtype=wp.bool, device=self._device)
            self._timestamp = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
            self._timestamp_last_update = wp.zeros_like(self._timestamp)

        # Optional: pose tracking via a RIGID_BODY_POSE tensor binding.
        # ovphysx fnmatch does not brace-expand, so we cannot match multiple
        # body names with a single glob.  Single-body sensors (the common case
        # — one prim path per sensor) use a tight per-body pattern.  Multi-body
        # sensors are rejected here; they need per-body bindings + an
        # interleaved-read kernel that does not exist yet.
        if self.cfg.track_pose:
            if self._num_sensors != 1:
                raise NotImplementedError(
                    "ovphysx ContactSensor.track_pose is not yet supported for sensors that "
                    f"resolve to more than one body per env (got {self._num_sensors} bodies "
                    f"under '{self.cfg.prim_path}').  Workaround: create one ContactSensor "
                    "per body."
                )
            single_pose_pattern = f"{base_glob}/{body_names[0]}"
            self._pose_binding = physx_instance.create_tensor_binding(
                pattern=single_pose_pattern,
                tensor_type=TT.RIGID_BODY_POSE,
            )
            if self._pose_binding.count != self._contact_binding.sensor_count:
                raise RuntimeError(
                    "RIGID_BODY_POSE binding count mismatch."
                    f"\n\tPattern: {single_pose_pattern}"
                    f"\n\tBound  : {self._pose_binding.count}"
                    f"\n\tExpect : {self._contact_binding.sensor_count}"
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
                (flat_count, self._num_filter_shapes, 3),
                dtype=wp.float32,
                device=self._device,
            )
        else:
            self._force_matrix_flat_buf = None

        # Pose buffer: [S, 7] for RIGID_BODY_POSE (px,py,pz,qx,qy,qz,qw).
        if self.cfg.track_pose:
            self._poses_flat_buf = wp.zeros((flat_count, 7), dtype=wp.float32, device=self._device)
        else:
            self._poses_flat_buf = None

    def _update_buffers_impl(self, env_mask: wp.array | None = None) -> None:
        """Read contact data from ovphysx and update sensor buffers."""
        env_mask = self._resolve_indices_and_mask(None, env_mask)

        # Pull aggregate forces into the pre-allocated flat buffer:
        # shape [num_envs * num_sensors, 3] float32 -> [num_envs * num_sensors] vec3f.
        self._contact_binding.read_net_forces(self._net_forces_flat_buf)
        net_forces_flat = self._net_forces_flat_buf.view(wp.vec3f)

        if self._force_matrix_flat_buf is not None:
            self._contact_binding.read_force_matrix(self._force_matrix_flat_buf)
            force_matrix_flat = self._force_matrix_flat_buf.view(wp.vec3f)
        else:
            force_matrix_flat = None

        wp.launch(
            update_net_forces_ovphysx_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[
                net_forces_flat,
                force_matrix_flat,
                env_mask,
                self._num_envs,
                self._num_sensors,
                self._num_filter_shapes,
                self._history_length,
                self.cfg.force_threshold,
                self._timestamp,
                self._timestamp_last_update,
            ],
            outputs=[
                self._data._net_forces_w,
                self._data._net_forces_w_history,
                self._data._force_matrix_w,
                self._data._force_matrix_w_history,
                self._data._current_air_time,
                self._data._current_contact_time,
                self._data._last_air_time,
                self._data._last_contact_time,
            ],
            device=self._device,
        )

        if self.cfg.track_pose:
            # Read pose into [num_envs * num_sensors, 7] float32 -> view as transformf.
            self._pose_binding.read(self._poses_flat_buf)
            poses_flat = self._poses_flat_buf.view(wp.transformf)
            wp.launch(
                split_flat_pose_to_pos_quat,
                dim=(self._num_envs, self._num_sensors),
                inputs=[poses_flat, env_mask, self._num_sensors],
                outputs=[self._data._pos_w, self._data._quat_w],
                device=self._device,
            )

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)

        wp.launch(
            reset_contact_sensor_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[
                self._history_length,
                self._num_filter_shapes,
                env_mask,
                self._data._net_forces_w,
                self._data._net_forces_w_history,
                self._data._force_matrix_w,
            ],
            outputs=[
                self._data._current_air_time,
                self._data._last_air_time,
                self._data._current_contact_time,
                self._data._last_contact_time,
                self._data._friction_forces_w,
                self._data._contact_pos_w,
            ],
            device=self._device,
        )

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> ProxyArray:
        """Boolean mask (as float) of bodies that established contact within ``dt`` [s].

        Args:
            dt: Time window since contact establishment [s].
            abs_tol: Absolute tolerance for the comparison [s].

        Returns:
            Boolean tensor (1.0/0.0) of shape ``(num_envs, num_sensors)``.

        Raises:
            RuntimeError: If :attr:`ContactSensorCfg.track_air_time` is False.
        """
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                " Please enable 'track_air_time' in the sensor configuration."
            )
        wp.launch(
            compute_first_transition_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[float(dt + abs_tol), self._data._current_contact_time],
            outputs=[self._data._first_transition],
            device=self._device,
        )
        return self._data._first_transition_ta

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> ProxyArray:
        """Boolean mask (as float) of bodies that broke contact within ``dt`` [s].

        Args:
            dt: Time window since contact break [s].
            abs_tol: Absolute tolerance for the comparison [s].

        Returns:
            Boolean tensor (1.0/0.0) of shape ``(num_envs, num_sensors)``.

        Raises:
            RuntimeError: If :attr:`ContactSensorCfg.track_air_time` is False.
        """
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track air time."
                " Please enable 'track_air_time' in the sensor configuration."
            )
        wp.launch(
            compute_first_transition_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[float(dt + abs_tol), self._data._current_air_time],
            outputs=[self._data._first_transition],
            device=self._device,
        )
        return self._data._first_transition_ta

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle contact-marker visibility.

        The kitless OVPhysX flow has no Kit-based renderer, so visualization
        markers are effectively invisible. The hook is still wired so that
        callers setting ``cfg.debug_vis=True`` get an explicit warning rather
        than silent no-op behaviour.
        """
        if debug_vis and not getattr(self, "_warned_debug_vis_unavailable", False):
            logger.warning(
                "OVPhysX ContactSensor: debug visualization markers are not rendered under the "
                "kitless OVPhysX flow (no Kit renderer present). The hook runs but marker "
                "geometry will not appear."
            )
            self._warned_debug_vis_unavailable = True

    def _debug_vis_callback(self, event) -> None:
        """Per-frame visualization update.

        Under kitless OVPhysX this is a no-op -- there is no renderer driving
        the per-frame marker positions. The method exists so the base
        sensor's debug-vis lifecycle hooks have a callable target.
        """
        return

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Release native handles when the simulation stops."""
        super()._invalidate_initialize_callback(event)
        # Drop strong references; ovphysx native handles are torn down on the
        # next reset() of OvPhysxManager.
        if self._contact_binding is not None:
            with contextlib.suppress(Exception):
                self._contact_binding.destroy()
        self._contact_binding = None
        if self._pose_binding is not None:
            with contextlib.suppress(Exception):
                self._pose_binding.destroy()
        self._pose_binding = None
        self._physx_instance = None
