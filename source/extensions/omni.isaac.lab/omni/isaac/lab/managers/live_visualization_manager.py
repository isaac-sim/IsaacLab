# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
import weakref
from collections.abc import Sequence
import carb

import omni.kit.app
from omni.ui import CollapsableFrame, Frame, VStack, Window
from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.lab.utils import configclass
import omni.isaac.lab.managers as managers
from omni.isaac.lab.ui.widgets.line_plot import LiveLinePlot

@configclass
class ManagerLiveVisualizerCfg():
    debug_vis = False

class ManagerLiveVisualizer():
    def __init__(self, manager,cfg: ManagerLiveVisualizerCfg = ManagerLiveVisualizerCfg()):
        """"""
        self._manager = manager
        self.debug_vis = cfg.debug_vis    
        self._env_idx: int = 0
        self.cfg = cfg

    #
    # Implementation checks
    # 

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    @property
    def has_vis_frame_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self.set_vis_frame)
        return "NotImplementedError" not in source_code

    @property
    def has_window_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self.set_window)
        return "NotImplementedError" not in source_code

    @property
    def has_env_selection_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self.set_env_selection)
        return "NotImplementedError" not in source_code

    @property
    def get_vis_frame(self) -> Frame:
        return self._vis_frame
    
    #
    # Setters
    #

    def set_env_selection(self,idx: int):
        if idx > 0 and idx < self._manager.num_envs:
            self._env_idx = idx
        else:
            carb.log_warn(f"Environment index is out of range (0,{self._manager.num_envs})")

    def set_vis_frame(self, frame: Frame):
        """Updates the assigned frame that can be used for visualizations.
        Args:
            frame: The debug visualization frame.
        """
        self._vis_frame = frame

    def set_window(self, window: Window):
        pass 

    def set_debug_vis(self, debug_vis: bool):
        """Set the debug visualization external facing function.
        
        Args: 
            debug_vis: Whether to enable or disable the debug visualization.
        """
        self._set_debug_vis_impl(debug_vis)

    #
    # Implementations
    #

    def _debug_vis_callback(self, event):
        """Callback for the debug visualization event."""

        if not SimulationContext.instance().is_playing():
            # Visualizers have not been created yet.
            return
        # update the visualization
        for (_, terms), vis in zip(self._manager.get_active_iterable_terms(env_idx=self._env_idx), self._term_visualizers):
            vis.add_datapoint(terms)



    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization implementation.
        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """

        if not hasattr(self, "_viewer_env_idx"):
            self._viewer_env_idx = 0

        if not hasattr(self, "_vis_frame"):
            raise RuntimeError("No frame set for debug visualization.")

        # Clear internal visualizers
        self._term_visualizers = []
        self._vis_frame.clear()

        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if not hasattr(self, "_debug_vis_handle") or self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

            self._vis_frame.visible = False
            return

        self._vis_frame.visible = True

        with self._vis_frame:
            with VStack():
                # Add a plot in a collapsible frame for each action term
                for name, terms in self._manager.get_active_iterable_terms(env_idx=self._env_idx):
                    frame = CollapsableFrame(
                        name,
                        collapsed=False,
                        style={"border_color": 0xFF8A8777, "padding": 4},
                    )
                    with frame:
                        plot = LiveLinePlot(
                            y_data=[[term] for term in terms],
                            plot_height=150,
                            show_legend=True,
                        )
                        self._term_visualizers.append(plot)
                    frame.collapsed = True

        self._debug_vis = debug_vis
