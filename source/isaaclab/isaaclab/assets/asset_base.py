# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import torch
from isaaclab_physx.physics import IsaacEvents, PhysxManager

import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage

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
        # get stage handle
        self.stage = get_current_stage()

        # spawn the asset
        # determine path where prims should exist after spawn
        if self.cfg.spawn is not None:
            # Use spawn_path if set (by InteractiveScene), otherwise fall back to prim_path
            check_path = self.cfg.spawn.spawn_path if self.cfg.spawn.spawn_path is not None else self.cfg.prim_path
            self.cfg.spawn.func(
                check_path,
                self.cfg.spawn,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
            )
        else:
            # asset should already exist at prim_path
            check_path = self.cfg.prim_path

        # check that prims exist
        matching_prims = sim_utils.find_matching_prims(check_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {check_path}.")

        # register various callback functions
        self._register_callbacks()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # clear events handles
        self._clear_callbacks()

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

    def set_visibility(self, visible: bool, env_ids: Sequence[int] | None = None):
        """Set the visibility of the prims corresponding to the asset.

        This operation affects the visibility of the prims corresponding to the asset in the USD stage.
        It is useful for toggling the visibility of the asset in the simulator. For instance, one can
        hide the asset when it is not being used to reduce the rendering overhead.

        .. note::
            This operation uses the PXR API to set the visibility of the prims. Thus, the operation
            may have an overhead if the number of prims is large.

        Args:
            visible: Whether to make the prims visible or not.
            env_ids: The indices of the object to set visibility. Defaults to None (all instances).
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = range(len(self._prims))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()

        # obtain the prims corresponding to the asset
        # note: we only want to find the prims once since this is a costly operation
        if not hasattr(self, "_prims"):
            self._prims = sim_utils.find_matching_prims(self.cfg.prim_path)

        # iterate over the environment ids
        for env_id in env_ids:
            sim_utils.set_prim_visibility(self._prims[env_id], visible)

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
            # obtain simulation related information
            self._backend = PhysicsManager.get_backend()
            self._device = PhysicsManager.get_device()
            # initialize the asset
            try:
                self._initialize_impl()
            except Exception as e:
                # Store exception to be raised after callback completes
                PhysxManager.store_callback_exception(e)
            # set flag
            self._is_initialized = True

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

    def _clear_callbacks(self, event: Any = None) -> None:
        """Clears the callbacks.

        Args:
            event: Optional event that triggered the callback (unused but required for event handlers).
        """
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
