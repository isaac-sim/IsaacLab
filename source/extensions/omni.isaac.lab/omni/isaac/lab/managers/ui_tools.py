# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixin utility to visualize action terms as live-plots."""

from __future__ import annotations

import weakref
from collections.abc import Sequence

import omni.kit.app
from omni.isaac.core.simulation_context import SimulationContext
from omni.ui import CollapsableFrame, Frame, VStack

from omni.isaac.lab.ui.widgets.line_plot import LiveLinePlot
from omni.isaac.lab.ui.widgets.ui_visualizer_mixin import UiVisualizerMixin


class ManagerLivePlotMixin(UiVisualizerMixin):
    """Mixin utility to visualize manager terms as live-plots.
    This mixin class can be used to visualize manager terms as live-plots. The class implements the functionality
    to create a collapsible frame for each term and add a live-plot for each term. The live-plots are updated
    automatically when the debug visualization is enabled.
    To use this mixin class, the following functions need to be implemented:
    * :func:`get_active_iterable_terms`: Returns the active terms as iterable sequence of tuples. The first element
        of the tuple is the name of the term and the second element is the raw value(s) of the term.
    Other than that, no additional code is required to use this mixin class.
    """

    @property
    def debug_vis(self) -> bool:
        """Whether to enable debug visualization."""
        if not hasattr(self, "_debug_vis"):
            return False

        return self._debug_vis

    """ Debug visualization Internal Functions """

    def get_active_iterable_terms(self) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.
        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.
        Returns:
            The active terms.
        """
        raise NotImplementedError(f"Active terms are not implemented for {self.__class__.__name__}.")

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets the debug visualization state.
        Args:
            debug_vis: Whether to enable or disable debug visualization.
        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # set debug visualization
        self._set_debug_vis_impl(debug_vis)
        return True

    """
    Helper functions.
    """

    def _set_env_selection_impl(self, env_selection: int):
        """Updates the currently selected environment.
        Args:
            env_selection: The environment selection.
        """
        self._viewer_env_idx = env_selection

        if hasattr(self, "_action_term_visualizers"):
            for vis in self._action_term_visualizers:
                vis.clear()

    def _set_vis_frame_impl(self, frame: Frame):
        """Updates the assigned frame that can be used for visualizations.
        Args:
            frame: The debug visualization frame.
        """
        self._vis_frame = frame

    def _debug_vis_callback(self, event):
        """Callback for the debug visualization event."""

        if not hasattr(self, "_action_term_visualizers") or not SimulationContext.instance().is_playing():
            # Visualizers have not been created yet.
            return
        # update the visualization
        for (_, terms), vis in zip(self.get_active_iterable_terms(), self._action_term_visualizers):
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
        self._action_term_visualizers = []
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
                for name, terms in self.get_active_iterable_terms():
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
                        self._action_term_visualizers.append(plot)
                    frame.collapsed = True

        self._debug_vis = debug_vis
