# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.kit.app
import omni.physx

from .asset_base_cfg import AssetBaseCfg


class AssetBase(ABC):
    """The interface class for assets."""

    def __init__(self, cfg: AssetBaseCfg):
        """Initialize the asset base.

        Args:
            cfg (AssetBaseCfg): The configuration class for the asset.

        Raises:
            RuntimeError: If no prims found at input prim path or prim path expression.
        """
        # store inputs
        self.cfg = cfg
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
        matching_prim_paths = prim_utils.find_matching_prim_paths(self.cfg.prim_path)
        if len(matching_prim_paths) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")

        # add callbacks for stage play/stop
        physx_interface = omni.physx.acquire_physx_interface()
        self._initialize_handle = physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop_by_type(
            int(omni.physx.bindings._physx.SimulationEvent.RESUMED), self._initialize_callback
        )
        self._invalidate_initialize_handle = (
            physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop_by_type(
                int(omni.physx.bindings._physx.SimulationEvent.STOPPED), self._invalidate_initialize_callback
            )
        )
        # add callback for debug visualization
        if self.cfg.debug_vis:
            app_interface = omni.kit.app.get_app_interface()
            self._debug_visualization_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                self._debug_vis_callback
            )
        else:
            self._debug_visualization_handle = None

    def __del__(self):
        """Unsubscribe from the callbacks."""
        self._initialize_handle.unsubscribe()
        self._invalidate_initialize_handle.unsubscribe()
        if self._debug_visualization_handle is not None:
            self._debug_visualization_handle.unsubscribe()

    """
    Properties
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Memory device for computation."""
        return NotImplementedError

    @property
    @abstractmethod
    def data(self) -> Any:
        """Data related to the asset."""
        return NotImplementedError

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the asset data.

        Args:
            debug_vis (bool): Whether to visualize the asset data.

        Raises:
            RuntimeError: If the asset debug visualization is not enabled.
        """
        if not self.cfg.debug_vis:
            raise RuntimeError("Debug visualization is not enabled for this sensor.")

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets all internal buffers of selected environments.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the object to reset.
                Defaults to None (all instances).
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
            dt (float): The amount of time passed from last ``update`` call.
        """
        raise NotImplementedError

    """
    Implementation specific.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the PhysX handles and internal buffers."""
        raise NotImplementedError

    def _debug_vis_impl(self):
        """Perform debug visualization of the asset."""
        pass

    """
    Simulation callbacks.
    """

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:
            self._initialize_impl()
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False

    def _debug_vis_callback(self, event):
        """Visualizes the asset data."""
        self._debug_vis_impl()
