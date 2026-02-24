# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.devices.device_base import DevicesCfg

if TYPE_CHECKING:
    from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .common import ViewerCfg
from .ui import BaseEnvWindow


@configclass
class DefaultEventManagerCfg:
    """Configuration of the default event manager.

    This manager is used to reset the scene to a default state. The default state is specified
    by the scene configuration.
    """

    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ManagerBasedEnvCfg:
    """Base configuration of the environment."""

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""

    # ui settings
    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # general settings
    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt.

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """

    recorders: object = DefaultEmptyRecorderManagerCfg()
    """Recorder settings. Defaults to recording nothing.

    Please refer to the :class:`isaaclab.managers.RecorderManager` class for more details.
    """

    observations: object = MISSING
    """Observation space settings.

    Please refer to the :class:`isaaclab.managers.ObservationManager` class for more details.
    """

    actions: object = MISSING
    """Action space settings.

    Please refer to the :class:`isaaclab.managers.ActionManager` class for more details.
    """

    events: object = DefaultEventManagerCfg()
    """Event settings. Defaults to the basic configuration that resets the scene to its default state.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    .. deprecated:: 2.3.1
        This attribute is deprecated and will be removed in the future. Please use
        :attr:`num_rerenders_on_reset` instead.

        To get the same behaviour as setting this parameter to ``True`` or ``False``, set
        :attr:`num_rerenders_on_reset` to 1 or 0, respectively.
    """

    num_rerenders_on_reset: int = 0
    """Number of render steps to perform after reset. Defaults to 0, which means no render step will be
    performed after reset.

    * When this is 0, no render step will be performed after reset. Data collected from sensors after performing
      reset will be stale and will not reflect the latest states in simulation caused by the reset.
    * When this is greater than 0, the specified number of extra render steps will be performed to update the
      sensor data to reflect the latest states from the reset. This comes at a cost of performance as additional
      render steps will be performed after each time an environment is reset.
    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""

    xr: XrCfg | None = None
    """Configuration for viewing and interacting with the environment through an XR device."""

    teleop_devices: DevicesCfg = field(default_factory=DevicesCfg)
    """Configuration for teleoperation devices."""

    isaac_teleop: object | None = None
    """Configuration for IsaacTeleop-based teleoperation.

    When set, the environment uses the IsaacTeleop stack for XR teleoperation instead
    of the native Isaac Lab teleop devices. This should be a IsaacTeleopCfg instance
    from the isaaclab_teleop package.

    The teleop scripts will automatically detect this configuration and use the
    IsaacTeleop stack when present.
    """

    export_io_descriptors: bool = False
    """Whether to export the IO descriptors for the environment. Defaults to False."""

    log_dir: str | None = None
    """Directory for logging experiment artifacts. Defaults to None, in which case no specific log directory is set."""
