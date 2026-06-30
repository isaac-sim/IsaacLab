# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`isaaclab.assets.AssetBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

import inspect
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim.utils.queries import get_first_matching_ancestor_prim
from isaaclab.sim.utils.transforms import resolve_prim_pose

from .kernels import reset_envs_kernel, update_outdated_envs_kernel, update_timestamp_kernel

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan

    from .sensor_base_cfg import SensorBaseCfg

logger = logging.getLogger(__name__)


class SensorBase(ABC):
    """The base class for implementing a sensor.

    The implementation is based on lazy evaluation. The sensor data is only updated when the user
    tries accessing the data through the :attr:`data` property or sets ``force_compute=True`` in
    the :meth:`update` method. This is done to avoid unnecessary computation when the sensor data
    is not used.

    The sensor is updated at the specified update period. If the update period is zero, then the
    sensor is updated at every simulation step.
    """

    def __init__(self, cfg: SensorBaseCfg):
        """Initialize the sensor class.

        Args:
            cfg: The configuration parameters for the sensor.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self._source_cfg = cfg
        self.cfg = cfg.copy()
        # flag for whether the sensor is initialized
        self._is_initialized = False
        # flag for whether the sensor is in visualization mode
        self._is_visualizing = False
        # clone plan used for this sensor's latest initialization
        self._clone_plan: ClonePlan | None = None
        self.stage = sim_utils.get_current_stage()

        # register various callback functions
        self._register_callbacks()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # clear physics events handles
        self._clear_callbacks()

    """
    Properties
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the sensor is initialized.

        Returns True if the sensor is initialized, False otherwise.
        """
        return self._is_initialized

    @property
    def num_instances(self) -> int:
        """Number of instances of the sensor.

        This is equal to the number of sensors per environment multiplied by the number of environments.
        """
        return self._num_envs

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self._device

    @property
    @abstractmethod
    def data(self) -> Any:
        """Data from the sensor.

        This property is only updated when the user tries to access the data. This is done to avoid
        unnecessary computation when the sensor data is not used.

        For updating the sensor when this property is accessed, you can use the following
        code snippet in your sensor implementation:

        .. code-block:: python

            # update sensors if needed
            self._update_outdated_buffers()
            # return the data (where `_data` is the data for the sensor)
            return self._data
        """
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the sensor has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the sensor data.

        Args:
            debug_vis: Whether to visualize the sensor data.

        Returns:
            Whether the debug visualization was successfully set. False if the sensor
            does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization flag
        self._is_visualizing = debug_vis
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                sim_ctx = sim_utils.SimulationContext.instance()
                if sim_ctx is not None:
                    self._debug_vis_handle = sim_ctx.vis_marker_registry.add_debug_vis_callback(self)
        else:
            # remove the subscriber if it exists
            sim_ctx = sim_utils.SimulationContext.instance()
            if sim_ctx is not None:
                sim_ctx.vis_marker_registry.clear_debug_vis_callback(self)
            else:
                self._debug_vis_handle = None
        # return success
        return True

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Resets the sensor internals.

        Args:
            env_ids: The environment indices to reset. Defaults to None, in which case all
                environments are reset.
            env_mask: A boolean warp array indicating which environments to reset. If provided,
                takes priority over ``env_ids``. Defaults to None.
        """
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        wp.launch(
            reset_envs_kernel,
            dim=self._num_envs,
            inputs=[env_mask, self._is_outdated, self._timestamp, self._timestamp_last_update],
            device=self._device,
        )

    def update(self, dt: float, force_recompute: bool = False):
        # Skip update if sensor is not initialized
        if not self._is_initialized:
            return
        # Update the timestamp for the sensors
        wp.launch(
            update_timestamp_kernel,
            dim=self._num_envs,
            inputs=[
                self._is_outdated,
                self._timestamp,
                self._timestamp_last_update,
                dt,
                self.cfg.update_period,
            ],
            device=self._device,
        )
        # Update the buffers
        if force_recompute or self._is_visualizing:
            self._update_outdated_buffers()

    """
    Implementation specific.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        # Obtain Simulation Context
        sim = sim_utils.SimulationContext.instance()
        if sim is None:
            raise RuntimeError("Simulation Context is not initialized!")
        # Obtain device and backend
        self._device = sim.device
        self._backend = sim.backend
        self._sim_physics_dt = sim.get_physics_dt()
        # Count number of environments. Prefer the active simulation's clone plan when USD
        # only carries the env_0 prototype (e.g. Newton clones solver-side).
        self._clone_plan = sim.get_clone_plan()
        clone_plan = self._clone_plan
        clone_plan_matches = ()
        if clone_plan is not None:
            clone_plan_matches = tuple(iter_clone_plan_matches(clone_plan, self.cfg.prim_path))
        if clone_plan_matches:
            self._parent_prims = []
            self._num_envs = int(clone_plan.clone_mask.shape[1])
        elif clone_plan is not None:
            env_prim_path_expr = self.cfg.prim_path.rsplit("/", 1)[0]
            self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
            self._num_envs = int(clone_plan.env_ids.numel())
        else:
            env_prim_path_expr = self.cfg.prim_path.rsplit("/", 1)[0]
            self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
            self._num_envs = len(self._parent_prims)
        # Create warp env mask arrays for "all envs" cases and resets.
        # Note: We use wp.to_torch() to create zero-copy torch tensor views of warp arrays.
        # This allows warp arrays to be passed to warp kernels while the corresponding torch
        # views support fancy indexing (e.g. tensor[env_ids] = True) without any memory copies.
        # Both the warp array and torch view share the same underlying device memory.
        self._ALL_ENV_MASK = wp.ones((self._num_envs), dtype=wp.bool, device=self._device)
        self._reset_mask = wp.zeros((self._num_envs), dtype=wp.bool, device=self._device)
        self._reset_mask_torch = wp.to_torch(self._reset_mask)
        # timestamp and outdated flags
        self._is_outdated = wp.ones(self._num_envs, dtype=wp.bool, device=self._device)
        self._timestamp = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._timestamp_last_update = wp.zeros_like(self._timestamp)

        # Initialize debug visualization handle
        if self._debug_vis_handle is None:
            # set initial state of debug visualization
            self.set_debug_vis(self.cfg.debug_vis)

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the sensor data for provided environment ids.

        This function does not perform any time-based checks and directly fills the data into the
        data container.

        Args:
            env_mask: The mask of the environments that are ready to capture.
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    """
    Internal simulation callbacks.
    """

    def _register_callbacks(self):
        """Registers physics lifecycle callbacks via the current backend's physics manager."""
        physics_mgr_cls = sim_utils.SimulationContext.instance().physics_manager

        obj_ref = weakref.proxy(self)

        def _invoke(callback_name, event):
            getattr(obj_ref, callback_name)(event)

        # Backend-agnostic: PHYSICS_READY (init) and STOP (invalidate)
        self._initialize_handle = physics_mgr_cls.register_callback(
            lambda payload: PhysicsManager.safe_callback_invoke(
                _invoke, "_initialize_callback", payload, physics_manager=physics_mgr_cls
            ),
            PhysicsEvent.PHYSICS_READY,
            order=10,
        )
        self._invalidate_initialize_handle = physics_mgr_cls.register_callback(
            lambda payload: PhysicsManager.safe_callback_invoke(
                _invoke, "_invalidate_initialize_callback", payload, physics_manager=physics_mgr_cls
            ),
            PhysicsEvent.STOP,
            order=10,
        )
        # Optional: prim deletion (only supported by PhysX backend; the substring
        # check would also match ``OvPhysxManager``, which does not expose
        # ``IsaacEvents``, so use an exact class-name match).
        self._prim_deletion_handle = None
        if physics_mgr_cls.__name__ == "PhysxManager":
            from isaaclab_physx.physics import IsaacEvents  # noqa: PLC0415

            self._prim_deletion_handle = physics_mgr_cls.register_callback(
                lambda event: PhysicsManager.safe_callback_invoke(
                    _invoke, "_on_prim_deletion", event, physics_manager=physics_mgr_cls
                ),
                IsaacEvents.PRIM_DELETION,
            )

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        .. note::
            Physics handles are only valid once the simulation is ready. This callback runs when
            :attr:`PhysicsEvent.PHYSICS_READY` is dispatched by the current backend.
        """
        if not self._is_initialized:
            self._initialize_impl()
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False
        self._clone_plan = None
        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is not None:
            sim_ctx.vis_marker_registry.clear_debug_vis_callback(self)
        else:
            self._debug_vis_handle = None

    def _on_prim_deletion(self, event) -> None:
        """Invalidates and deletes the callbacks when the prim is deleted.

        Args:
            event: The prim deletion event containing the prim path in payload.

        Note:
            This function is called when the prim is deleted.
        """
        prim_path = event.payload["prim_path"]
        if prim_path == "/":
            self._clear_callbacks()
            return
        if sim_utils.matches_path_expr_prefix(self.cfg.prim_path, prim_path):
            self._clear_callbacks()

    def _clear_callbacks(self) -> None:
        """Clears the callbacks."""
        if self._initialize_handle is not None:
            self._initialize_handle.deregister()
            self._initialize_handle = None
        if self._invalidate_initialize_handle is not None:
            self._invalidate_initialize_handle.deregister()
            self._invalidate_initialize_handle = None
        if self._prim_deletion_handle is not None:
            self._prim_deletion_handle.deregister()
            self._prim_deletion_handle = None
        # Clear debug visualization
        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is not None:
            sim_ctx.vis_marker_registry.clear_debug_vis_callback(self)
        else:
            self._debug_vis_handle = None

    """
    Helper functions.
    """

    def _update_outdated_buffers(self):
        """Fills the sensor data for the outdated sensors."""
        self._update_buffers_impl(self._is_outdated)
        # update timestamps and clear outdated flags
        wp.launch(
            update_outdated_envs_kernel,
            dim=self._num_envs,
            inputs=[self._is_outdated, self._timestamp, self._timestamp_last_update],
            device=self._device,
        )

    def _resolve_indices_and_mask(
        self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None
    ) -> wp.array:
        """Resolve environment indices to a warp array and mask."""
        if env_ids is None and env_mask is None:
            return self._ALL_ENV_MASK
        elif env_mask is not None:
            return env_mask
        else:
            self._reset_mask.zero_()
            self._reset_mask_torch[env_ids] = True
            return self._reset_mask

    def _resolve_rigid_body_ancestor_expr(
        self,
    ) -> tuple[str, tuple[float, float, float] | None, tuple[float, float, float, float] | None]:
        """Resolve the rigid-body ancestor view expression and the sensor-to-body offset.

        The sensor's :attr:`SensorBaseCfg.prim_path` may point to any frame
        inside the asset. To create a physics view, this helper walks ancestors
        from that prim until it finds one with ``UsdPhysics.RigidBodyAPI``,
        builds the corresponding destination-side expression, and computes the
        fixed transform from that body to the configured sensor frame.

        Combines two resolution paths:

        1. When an active :class:`~isaaclab.cloner.ClonePlan` exists, the
           source-side env path is taken from the plan via
           :func:`~isaaclab.cloner.resolve_clone_plan_source`, the rigid-body ancestor
           is located on that source env, and the destination expression is
           reconstructed by trimming the sensor-relative suffix from the plan's
           destination glob.
        2. Otherwise (stage scan fallback for non-cloned setups), the first
           matching env is located via
           :func:`~isaaclab.sim.utils.queries.find_first_matching_prim`, the
           rigid-body ancestor is located on that env, and the destination
           expression is the configured :attr:`SensorBaseCfg.prim_path` minus
           the sensor-relative suffix.

        The returned expression may still contain regex-style wildcards (e.g.
        ``.*``); callers are responsible for converting to glob form for their
        physics view (e.g. ``.replace(".*", "*")``).

        Returns:
            A tuple of:

            * ``rigid_parent_expr``: destination-side view expression that
              matches the rigid-body ancestor across envs.
            * ``fixed_pos_b``: sensor-relative-to-body translation [m] (xyz),
              or ``None`` when the sensor is mounted directly at the body
              origin.
            * ``fixed_quat_b``: sensor-relative-to-body rotation as a
              quaternion ``(x, y, z, w)``, or ``None`` when the sensor is
              mounted directly at the body origin.
        """
        prim, target_expr = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)[0]

        ancestor_prim = get_first_matching_ancestor_prim(
            prim.GetPath(), predicate=lambda _prim: _prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if ancestor_prim is None:
            raise RuntimeError(f"Failed to find a rigid body ancestor prim at path expression: {self.cfg.prim_path}")

        if ancestor_prim == prim:
            return target_expr, None, None

        relative_path = prim.GetPath().MakeRelativePath(ancestor_prim.GetPath()).pathString
        suffix = "/" + relative_path
        if not target_expr.endswith(suffix):
            raise RuntimeError(
                f"Failed to build rigid body ancestor expression: target expression {target_expr!r} does not end "
                f"with relative path {relative_path!r}."
            )
        rigid_parent_expr = target_expr[: -len(suffix)]
        fixed_pos_b, fixed_quat_b = resolve_prim_pose(prim, ancestor_prim)
        return rigid_parent_expr, fixed_pos_b, fixed_quat_b
