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
from isaaclab_physx.physics import IsaacEvents, PhysxManager

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.utils.stage import get_current_stage

from .kernels import reset_envs_kernel, update_outdated_envs_kernel, update_timestamp_kernel

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
    """

    def __init__(self, cfg: SensorBaseCfg):
        """Initialize the sensor class.

        Args:
            cfg: The configuration parameters for the sensor.
        """
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
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                import omni.kit.app  # noqa: PLC0415

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
        # Count number of environments
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
        """Registers physics lifecycle and prim deletion callbacks."""

        # register simulator callbacks (with weakref safety to avoid crashes on deletion)
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

        # Register PHYSICS_READY callback for initialization (order=10 for lower priority)
        self._initialize_handle = PhysxManager.register_callback(
            lambda payload, obj_ref=obj_ref: safe_callback("_initialize_callback", payload, obj_ref),
            PhysicsEvent.PHYSICS_READY,
            order=10,
        )
        # Register TIMELINE_STOP callback for invalidation (PhysX-specific)
        self._invalidate_initialize_handle = PhysxManager.register_callback(
            lambda event, obj_ref=obj_ref: safe_callback("_invalidate_initialize_callback", event, obj_ref),
            IsaacEvents.TIMELINE_STOP,
            order=10,
        )
        # Register PRIM_DELETION callback (PhysX-specific)
        self._prim_deletion_handle = PhysxManager.register_callback(
            lambda event, obj_ref=obj_ref: safe_callback("_on_prim_deletion", event, obj_ref),
            IsaacEvents.PRIM_DELETION,
        )

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        .. note::
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:
            try:
                self._initialize_impl()
                self._is_initialized = True
            except Exception as e:
                # Store exception to be raised after callback completes
                PhysxManager.store_callback_exception(e)

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
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
        result = re.match(
            pattern="^" + "/".join(self.cfg.prim_path.split("/")[: prim_path.count("/") + 1]) + "$", string=prim_path
        )
        if result:
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
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
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
