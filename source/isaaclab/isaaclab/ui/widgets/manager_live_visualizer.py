# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy
import weakref
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.kit.app
import omni.log
from isaacsim.core.api.simulation_context import SimulationContext

from isaaclab.managers import ManagerBase
from isaaclab.utils import configclass

from .image_plot import ImagePlot
from .line_plot import LiveLinePlot
from .ui_visualizer_base import UiVisualizerBase

if TYPE_CHECKING:
    import omni.ui


@configclass
class ManagerLiveVisualizerCfg:
    """Configuration for the :class:`ManagerLiveVisualizer` class."""

    debug_vis: bool = False
    """Flag used to set status of the live visualizers on startup. Defaults to False, which means closed."""

    manager_name: str = MISSING
    """Manager name that corresponds to the manager of interest in the ManagerBasedEnv and ManagerBasedRLEnv"""

    term_names: list[str] | dict[str, list[str]] | None = None
    """Specific term names specified in a Manager config that are chosen to be plotted. Defaults to None.

    If None all terms will be plotted. For managers that utilize Groups (i.e. ObservationGroup) use a dictionary of
    {group_names: [term_names]}.
    """


class ManagerLiveVisualizer(UiVisualizerBase):
    """A interface object used to transfer data from a manager to a UI widget.

    This class handles the creation of UI Widgets for selected terms given a :class:`ManagerLiveVisualizerCfg`.
    It iterates through the terms of the manager and creates a visualizer for each term. If the term is a single
    variable or a multi-variable signal, it creates a :class:`LiveLinePlot`. If the term is an image (2D or RGB),
    it creates an :class:`ImagePlot`. The visualizer can be toggled on and off using the
    :attr:`ManagerLiveVisualizerCfg.debug_vis` flag in the configuration.
    """

    def __init__(self, manager: ManagerBase, cfg: ManagerLiveVisualizerCfg = ManagerLiveVisualizerCfg()):
        """Initialize ManagerLiveVisualizer.

        Args:
            manager: The manager with terms to be plotted. The manager must have a :meth:`get_active_iterable_terms` method.
            cfg: The configuration file used to select desired manager terms to be plotted.
        """

        self._manager = manager
        self.debug_vis = cfg.debug_vis
        self._env_idx: int = 0
        self.cfg = cfg
        self._viewer_env_idx = 0
        self._vis_frame: omni.ui.Frame
        self._vis_window: omni.ui.Window

        # evaluate chosen terms if no terms provided use all available.
        self.term_names = []

        if self.cfg.term_names is not None:
            # extract chosen terms
            if isinstance(self.cfg.term_names, list):
                for term_name in self.cfg.term_names:
                    if term_name in self._manager.active_terms:
                        self.term_names.append(term_name)
                    else:
                        omni.log.error(
                            f"ManagerVisualizer Failure: ManagerTerm ({term_name}) does not exist in"
                            f" Manager({self.cfg.manager_name})"
                        )

            # extract chosen group-terms
            elif isinstance(self.cfg.term_names, dict):
                # if manager is using groups and terms are saved as a dictionary
                if isinstance(self._manager.active_terms, dict):
                    for group, terms in self.cfg.term_names:
                        if group in self._manager.active_terms.keys():
                            for term_name in terms:
                                if term_name in self._manager.active_terms[group]:
                                    self.term_names.append(f"{group}-{term_name}")
                                else:
                                    omni.log.error(
                                        f"ManagerVisualizer Failure: ManagerTerm ({term_name}) does not exist in"
                                        f" Group({group})"
                                    )
                        else:
                            omni.log.error(
                                f"ManagerVisualizer Failure: Group ({group}) does not exist in"
                                f" Manager({self.cfg.manager_name})"
                            )
                else:
                    omni.log.error(
                        f"ManagerVisualizer Failure: Manager({self.cfg.manager_name}) does not utilize grouping of"
                        " terms."
                    )

    #
    # Implementation checks
    #

    @property
    def get_vis_frame(self) -> omni.ui.Frame:
        """Returns the UI Frame object tied to this visualizer."""
        return self._vis_frame

    @property
    def get_vis_window(self) -> omni.ui.Window:
        """Returns the UI Window object tied to this visualizer."""
        return self._vis_window

    #
    # Setters
    #

    def set_debug_vis(self, debug_vis: bool):
        """Set the debug visualization external facing function.

        Args:
            debug_vis: Whether to enable or disable the debug visualization.
        """
        self._set_debug_vis_impl(debug_vis)

    #
    # Implementations
    #

    def _set_env_selection_impl(self, env_idx: int):
        """Update the index of the selected environment to display.

        Args:
            env_idx: The index of the selected environment.
        """
        if env_idx > 0 and env_idx < self._manager.num_envs:
            self._env_idx = env_idx
        else:
            omni.log.warn(f"Environment index is out of range (0, {self._manager.num_envs - 1})")

    def _set_vis_frame_impl(self, frame: omni.ui.Frame):
        """Updates the assigned frame that can be used for visualizations.

        Args:
            frame: The debug visualization frame.
        """
        self._vis_frame = frame

    def _debug_vis_callback(self, event):
        """Callback for the debug visualization event."""

        if not SimulationContext.instance().is_playing():
            # Visualizers have not been created yet.
            return

        # get updated data and update visualization
        for (_, term), vis in zip(
            self._manager.get_active_iterable_terms(env_idx=self._env_idx), self._term_visualizers
        ):
            if isinstance(vis, LiveLinePlot):
                vis.add_datapoint(term)
            elif isinstance(vis, ImagePlot):
                vis.update_image(numpy.array(term))

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization implementation.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """

        if not hasattr(self, "_vis_frame"):
            raise RuntimeError("No frame set for debug visualization.")

        # Clear internal visualizers
        self._term_visualizers = []
        self._vis_frame.clear()

        if debug_vis:
            # if enabled create a subscriber for the post update event if it doesn't exist
            if not hasattr(self, "_debug_vis_handle") or self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # if disabled remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

            self._vis_frame.visible = False
            return

        self._vis_frame.visible = True

        with self._vis_frame:
            with omni.ui.VStack():
                # Add a plot in a collapsible frame for each term available
                for name, term in self._manager.get_active_iterable_terms(env_idx=self._env_idx):
                    if name in self.term_names or len(self.term_names) == 0:
                        frame = omni.ui.CollapsableFrame(
                            name,
                            collapsed=False,
                            style={"border_color": 0xFF8A8777, "padding": 4},
                        )
                        with frame:
                            # create line plot for single or multi-variable signals
                            len_term_shape = len(numpy.array(term).shape)
                            if len_term_shape <= 2:
                                plot = LiveLinePlot(y_data=[[elem] for elem in term], plot_height=150, show_legend=True)
                                self._term_visualizers.append(plot)
                            # create an image plot for 2d and greater data (i.e. mono and rgb images)
                            elif len_term_shape == 3:
                                image = ImagePlot(image=numpy.array(term), label=name)
                                self._term_visualizers.append(image)
                            else:
                                omni.log.warn(
                                    f"ManagerLiveVisualizer: Term ({name}) is not a supported data type for"
                                    " visualization."
                                )
                        frame.collapsed = True

        self._debug_vis = debug_vis


@configclass
class DefaultManagerBasedEnvLiveVisCfg:
    """Default configuration to use for the ManagerBasedEnv. Each chosen manager assumes all terms will be plotted."""

    action_live_vis = ManagerLiveVisualizerCfg(manager_name="action_manager")
    observation_live_vis = ManagerLiveVisualizerCfg(manager_name="observation_manager")


@configclass
class DefaultManagerBasedRLEnvLiveVisCfg(DefaultManagerBasedEnvLiveVisCfg):
    """Default configuration to use for the ManagerBasedRLEnv. Each chosen manager assumes all terms will be plotted."""

    curriculum_live_vis = ManagerLiveVisualizerCfg(manager_name="curriculum_manager")
    command_live_vis = ManagerLiveVisualizerCfg(manager_name="command_manager")
    reward_live_vis = ManagerLiveVisualizerCfg(manager_name="reward_manager")
    termination_live_vis = ManagerLiveVisualizerCfg(manager_name="termination_manager")


class EnvLiveVisualizer:
    """A class to handle all ManagerLiveVisualizers used in an Environment."""

    def __init__(self, cfg: object, managers: dict[str, ManagerBase]):
        """Initialize the EnvLiveVisualizer.

        Args:
            cfg: The configuration file containing terms of ManagerLiveVisualizers.
            managers: A dictionary of labeled managers. i.e. {"manager_name",manager}.
        """
        self.cfg = cfg
        self.managers = managers
        self._prepare_terms()

    def _prepare_terms(self):
        self._manager_visualizers: dict[str, ManagerLiveVisualizer] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check if term config is viable
            if isinstance(term_cfg, ManagerLiveVisualizerCfg):
                # find appropriate manager name
                manager = self.managers[term_cfg.manager_name]
                self._manager_visualizers[term_cfg.manager_name] = ManagerLiveVisualizer(manager=manager, cfg=term_cfg)
            else:
                raise TypeError(
                    f"Provided EnvLiveVisualizer term: '{term_name}' is not of type ManagerLiveVisualizerCfg"
                )

    @property
    def manager_visualizers(self) -> dict[str, ManagerLiveVisualizer]:
        """A dictionary of labeled ManagerLiveVisualizers associated manager name as key."""
        return self._manager_visualizers
