# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`isaaclab.assets.AssetBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

import builtins
import contextlib
import inspect
import re
import torch
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim.utils.stage import get_current_stage

# import omni.kit.app
# import omni.timeline
# from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager


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
        # add callbacks for stage play/stop
        obj_ref = weakref.proxy(self)
        # timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()

        # # the order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        NewtonManager.add_on_start_callback(lambda: safe_callback("_initialize_callback", None, obj_ref))
        # # register timeline STOP event callback (lower priority with order=10)
        # self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        #     int(omni.timeline.TimelineEventType.STOP),
        #     lambda event, obj_ref=obj_ref: safe_callback("_invalidate_initialize_callback", event, obj_ref),
        #     order=10,
        # )
        # # register prim deletion callback
        # self._prim_deletion_callback_id = SimulationManager.register_callback(
        #     lambda event, obj_ref=obj_ref: safe_callback("_on_prim_deletion", event, obj_ref),
        #     event=IsaacEvents.PRIM_DELETION,
        # )

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # Suppress errors during Python shutdown
        # Note: contextlib may be None during interpreter shutdown
        if contextlib is not None:
            with contextlib.suppress(ImportError, AttributeError, TypeError):
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

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Resets the sensor internals.

        Args:
            env_ids: The sensor ids to reset. Defaults to None (all instances).
            env_mask: The sensor mask to reset. Defaults to None (all instances).
        """
        # Resolve sensor ids
        if env_ids is None:
            env_ids = slice(None)
        # Reset the timestamp for the sensors
        self._timestamp[env_ids] = 0.0

        self._timestamp_last_update[env_ids] = 0.0
        # Set all reset sensors to outdated so that they are updated when data is called the next time.
        self._is_outdated[env_ids] = True

    def update(self, dt: float, force_recompute: bool = False):
        # Skip update if sensor is not initialized
        if not self._is_initialized:
            return
        # Update the timestamp for the sensors
        self._timestamp += dt
        self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
        # Update the buffers
        # TODO (from @mayank): Why is there a history length here when it doesn't mean anything in the sensor base?!?
        #   It is only for the contact sensor but there we should redefine the update function IMO.
        if force_recompute or self._is_visualizing or (self.cfg.history_length > 0):
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
        self._num_envs = NewtonManager._num_envs
        # Boolean tensor indicating whether the sensor data has to be refreshed
        self._is_outdated = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        # Current timestamp (in seconds)
        self._timestamp = torch.zeros(self._num_envs, device=self._device)
        # Timestamp from last update
        self._timestamp_last_update = torch.zeros_like(self._timestamp)

    @abstractmethod
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the sensor data for provided environment ids.

        This function does not perform any time-based checks and directly fills the data into the
        data container.

        Args:
            env_ids: The indices of the sensors that are ready to capture.
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

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:
            try:
                self._initialize_impl()
                self._is_initialized = True
            except Exception as e:
                if builtins.ISAACLAB_CALLBACK_EXCEPTION is None:
                    builtins.ISAACLAB_CALLBACK_EXCEPTION = e

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _on_prim_deletion(self, prim_path: str) -> None:
        """Invalidates and deletes the callbacks when the prim is deleted.

        Args:
            prim_path: The path to the prim that is being deleted.

        Note:
            This function is called when the prim is deleted.
        """
        # skip callback if required
        if getattr(SimulationContext.instance(), "_skip_next_prim_deletion_callback_fn", False):
            return
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
        import contextlib

        if getattr(self, "_prim_deletion_callback_id", None):
            with contextlib.suppress(ImportError, NameError):
                from isaacsim.core.simulation_manager import SimulationManager

                SimulationManager.deregister_callback(self._prim_deletion_callback_id)
            self._prim_deletion_callback_id = None
        if getattr(self, "_invalidate_initialize_handle", None):
            self._invalidate_initialize_handle.unsubscribe()
            self._invalidate_initialize_handle = None
        # clear debug visualization
        if getattr(self, "_debug_vis_handle", None):
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Helper functions.
    """

    def _update_outdated_buffers(self):
        """Fills the sensor data for the outdated sensors."""
        outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
        if len(outdated_env_ids) > 0:
            # obtain new data
            self._update_buffers_impl(outdated_env_ids)
            # update the timestamp from last update
            self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]
            # set outdated flag to false for the updated sensors
            self._is_outdated[outdated_env_ids] = False
