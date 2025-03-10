# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
import re
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import omni.kit.app
import omni.timeline

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from .asset_base_cfg import AssetBaseCfg


class AssetBase(ABC):
    """The base interface class for assets.

    An asset corresponds to any physics-enabled object that can be spawned in the simulation. These include
    rigid objects, articulated objects, deformable objects etc. The core functionality of an asset is to
    provide a set of buffers that can be used to interact with the simulator. The buffers are updated
    by the asset class and can be written into the simulator using the their respective ``write`` methods.
    This allows a convenient way to perform post-processing operations on the buffers before writing them
    into the simulator and obtaining the corresponding simulation results.

    The class handles both the spawning of the asset into the USD stage as well as initialization of necessary
    physics handles to interact with the asset. Upon construction of the asset instance, the prim corresponding
    to the asset is spawned into the USD stage if the spawn configuration is not None. The spawn configuration
    is defined in the :attr:`AssetBaseCfg.spawn` attribute. In case the configured :attr:`AssetBaseCfg.prim_path`
    is an expression, then the prim is spawned at all the matching paths. Otherwise, a single prim is spawned
    at the configured path. For more information on the spawn configuration, see the
    :mod:`isaaclab.sim.spawners` module.

    Unlike Isaac Sim interface, where one usually needs to call the
    :meth:`isaacsim.core.prims.XFormPrim.initialize` method to initialize the PhysX handles, the asset
    class automatically initializes and invalidates the PhysX handles when the stage is played/stopped. This
    is done by registering callbacks for the stage play/stop events.

    Additionally, the class registers a callback for debug visualization of the asset if a debug visualization
    is implemented in the asset class. This can be enabled by setting the :attr:`AssetBaseCfg.debug_vis` attribute
    to True. The debug visualization is implemented through the :meth:`_set_debug_vis_impl` and
    :meth:`_debug_vis_callback` methods.
    """

    def __init__(self, cfg: AssetBaseCfg):
        """Initialize the asset base.

        Args:
            cfg: The configuration class for the asset.

        Raises:
            RuntimeError: If no prims found at input prim path or prim path expression.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg.copy()
        # flag for whether the asset is initialized
        self._is_initialized = False

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        asset_path = self.cfg.prim_path.split("/")[-1]
        asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None
        # spawn the asset
        if self.cfg.spawn is not None and not asset_path_is_regex:
            self.cfg.spawn.func(
                self.cfg.prim_path,
                self.cfg.spawn,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
            )
        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")

        # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor is called.
        # add callbacks for stage play/stop
        # The order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # clear physics events handles
        if self._initialize_handle:
            self._initialize_handle.unsubscribe()
            self._initialize_handle = None
        if self._invalidate_initialize_handle:
            self._invalidate_initialize_handle.unsubscribe()
            self._invalidate_initialize_handle = None
        # clear debug visualization
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the asset is initialized.

        Returns True if the asset is initialized, False otherwise.
        """
        return self._is_initialized

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of instances of the asset.

        This is equal to the number of asset instances per environment multiplied by the number of environments.
        """
        return NotImplementedError

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self._device

    @property
    @abstractmethod
    def data(self) -> Any:
        """Data related to the asset."""
        return NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the asset has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the asset data.

        Args:
            debug_vis: Whether to visualize the asset data.

        Returns:
            Whether the debug visualization was successfully set. False if the asset
            does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
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
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets all internal buffers of selected environments.

        Args:
            env_ids: The indices of the object to reset. Defaults to None (all instances).
        """
        raise NotImplementedError

    @abstractmethod
    def write_data_to_sim(self):
        """Writes data to the simulator."""
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of quantities such as joint
        accelerations which are not provided by the simulator.

        Args:
            dt: The amount of time passed from last ``update`` call.
        """
        raise NotImplementedError

    """
    Implementation specific.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the PhysX handles and internal buffers."""
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
            # obtain simulation related information
            sim = sim_utils.SimulationContext.instance()
            if sim is None:
                raise RuntimeError("SimulationContext is not initialized! Please initialize SimulationContext first.")
            self._backend = sim.backend
            self._device = sim.device
            # initialize the asset
            self._initialize_impl()
            # set flag
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False
