# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.orbit.envs.mdp as mdp
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sim import SimulationCfg
from omni.isaac.orbit.utils import configclass

from .ui import BaseEnvWindow


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in m). Default is (0.0, 0.0, 0.0)."""
    cam_prim_path: str = "/OmniverseKit_Persp"
    """The camera prim path to record images from. Default is "/OmniverseKit_Persp", which is the
    default camera in the default viewport.
    """
    resolution: tuple[int, int] = (1280, 720)
    """The resolution (width, height) of the camera specified using :attr:`cam_prim_path`.
    Default is (1280, 720).
    """


@configclass
class DefaultRandomizationManagerCfg:
    """Configuration of the default randomization manager.

    This manager is used to reset the scene to a default state. The default state is specified
    by the scene configuration.
    """

    reset_scene_to_default = RandTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class BaseEnvCfg:
    """Base configuration of the environment."""

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""
    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""
    # ui settings
    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Defaults to None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`omni.isaac.orbit.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # general settings
    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt."""

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """Scene settings"""
    observations: object = MISSING
    """Observation space settings."""
    actions: object = MISSING
    """Action space settings."""
    randomization: object = DefaultRandomizationManagerCfg()
    """Randomization settings. Defaults to the default randomization manager, which resets
    the scene to its default state."""
