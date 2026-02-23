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
import re
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

from isaaclab.physics import PhysicsEvent
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from .sensor_base_cfg import SensorBaseCfg


class SensorBase(ABC):
    """The base class for implementing a sensor.

    The implementation is based on lazy evaluation. The sensor data is only updated when the user
    tries accessing the data through the :attr:`data` property or sets ``force_compute=True`` in
    the :meth:`update` method. This is done to avoid unnecessary computation when the sensor data
    is not used.

    The sensor is updated at the specified update period. If the update period is zero, then the
    sensor is updated at every simulation step.

    Backend-specific implementations should inherit from this class and implement:
    - :meth:`_initialize_impl` - Initialize sensor handles and buffers
    - :meth:`_update_outdated_buffers` - Update sensor data for outdated environments
    - Either :meth:`_update_buffers_impl_index` (PhysX) or :meth:`_update_buffers_impl_mask` (Newton)
    """

    def __init__(self, cfg: SensorBaseCfg):
        """Initialize the sensor class.

        Args:
            cfg: The configuration parameters for the sensor.
        """
        # check that config is valid
        if cfg.history_length < 0:
            raise ValueError(f"History length must be greater than 0! Received: {cfg.history_length}")
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg.copy()
        # flag for whether the sensor is initialized
        self._is_initialized = False
        # flag for whether the sensor is in visualization mode
        self._is_visualizing = False
        # get stage handle
        self.stage = get_current_stage()

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
        # toggle debug visualization handles (Kit/omni only for PhysX backend)
        if debug_vis:
            if self._debug_vis_handle is None:
                sim_ctx = SimulationContext.instance()
                if sim_ctx and "Physx" in sim_ctx.physics_manager.__name__:
                    import omni.kit.app

                    app_interface = omni.kit.app.get_app_interface()
                    self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                        lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                    )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Resets the sensor internals.

        Args:
            env_ids: The sensor ids to reset. Defaults to None (all instances).
            env_mask: The sensor mask to reset. Defaults to None (all instances).

        Note:
            Backend implementations should handle either env_ids or env_mask based on their preference.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float, force_recompute: bool = False):
        """Updates the sensor data.

        Args:
            dt: The time step for the update.
            force_recompute: Whether to force recomputation of sensor data.
        """
        raise NotImplementedError

    """
    Implementation specific.
    """

    _device: str
    """Memory device for computation."""
    _backend: str
    """Simulation backend (e.g., "newton", "physx")."""
    _num_envs: int
    """Number of environments."""

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers.

        Backend implementations should:
        1. Get device and backend from SimulationContext
        2. Determine number of environments
        3. Initialize internal state (timestamps, outdated flags, etc.)
        4. Initialize sensor-specific buffers
        """
        raise NotImplementedError

    @abstractmethod
    def _update_outdated_buffers(self):
        """Fills the sensor data for the outdated sensors.

        Backend implementations should:
        1. Determine which environments need updating (based on timestamps/outdated flags)
        2. Call the appropriate _update_buffers_impl method
        3. Update timestamps and clear outdated flags
        """
        raise NotImplementedError

    def _update_buffers_impl_index(self, env_ids: Sequence[int]):
        """Fills the sensor data for provided environment indices.

        This is the index-based API primarily used by PhysX backend.

        Args:
            env_ids: The indices of the environments to update.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _update_buffers_impl_index. "
            "This method is typically used by PhysX backend sensors."
        )

    def _update_buffers_impl_mask(self, env_mask: wp.array):
        """Fills the sensor data for environments specified by mask.

        This is the mask-based API primarily used by Newton backend.

        Args:
            env_mask: Boolean mask array indicating which environments to update.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _update_buffers_impl_mask. "
            "This method is typically used by Newton backend sensors."
        )

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
        physics_mgr_cls = SimulationContext.instance().physics_manager

        def safe_callback(callback_name, event, obj_ref):
            """Safely invoke a callback on a weakly-referenced object, ignoring ReferenceError if deleted."""
            try:
                obj = obj_ref
                getattr(obj, callback_name)(event)
            except ReferenceError:
                # Object has been deleted; ignore.
                pass

        # note: use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        obj_ref = weakref.proxy(self)

        # Backend-agnostic: PHYSICS_READY (init) and STOP (invalidate)
        self._initialize_handle = physics_mgr_cls.register_callback(
            lambda payload, obj_ref=obj_ref: safe_callback("_initialize_callback", payload, obj_ref),
            PhysicsEvent.PHYSICS_READY,
            order=10,
        )
        self._invalidate_initialize_handle = physics_mgr_cls.register_callback(
            lambda payload, obj_ref=obj_ref: safe_callback("_invalidate_initialize_callback", payload, obj_ref),
            PhysicsEvent.STOP,
            order=10,
        )
        # Optional: prim deletion (only supported by PhysX backend)
        self._prim_deletion_handle = None
        physics_backend = physics_mgr_cls.__name__
        if "Physx" in physics_backend:
            from isaaclab_physx.physics import IsaacEvents

            self._prim_deletion_handle = physics_mgr_cls.register_callback(
                lambda event, obj_ref=obj_ref: safe_callback("_on_prim_deletion", event, obj_ref),
                IsaacEvents.PRIM_DELETION,
            )

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        .. note::
            Physics handles are only valid once the simulation is ready. This callback runs when
            :attr:`PhysicsEvent.PHYSICS_READY` is dispatched by the current backend.
        """
        if not self._is_initialized:
            try:
                self._initialize_impl()
            except Exception as e:
                store_fn = getattr(
                    SimulationContext.instance().physics_manager,
                    "store_callback_exception",
                    None,
                )
                if callable(store_fn):
                    store_fn(e)
                else:
                    raise
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _on_prim_deletion(self, event) -> None:
        """Invalidates and clears callbacks when the prim is deleted.

        Only used when the backend supports prim deletion events (e.g. PhysX).
        """
        payload = getattr(event, "payload", event) if not isinstance(event, dict) else event
        prim_path = payload.get("prim_path", "") if isinstance(payload, dict) else ""
        if prim_path == "/":
            self._clear_callbacks()
            return
        result = re.match(
            pattern="^" + "/".join(self.cfg.prim_path.split("/")[: prim_path.count("/") + 1]) + "$", string=prim_path
        )
        if result:
            self._clear_callbacks()

    def _clear_callbacks(self) -> None:
        """Clears all registered callbacks."""
        if getattr(self, "_initialize_handle", None) is not None:
            self._initialize_handle.deregister()
            self._initialize_handle = None
        if getattr(self, "_invalidate_initialize_handle", None) is not None:
            self._invalidate_initialize_handle.deregister()
            self._invalidate_initialize_handle = None
        if getattr(self, "_prim_deletion_handle", None) is not None:
            self._prim_deletion_handle.deregister()
            self._prim_deletion_handle = None
        if getattr(self, "_debug_vis_handle", None) is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None
