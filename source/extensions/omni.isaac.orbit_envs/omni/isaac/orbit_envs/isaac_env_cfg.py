# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from dataclasses import MISSING
from typing import Tuple

from omni.isaac.orbit.sim import SimulationCfg
from omni.isaac.orbit.utils import configclass

__all__ = ["IsaacEnvCfg", "EnvCfg", "ViewerCfg"]


##
# General environment configuration
##


@configclass
class EnvCfg:
    """Configuration of the common environment information."""

    num_envs: int = MISSING
    """Number of environment instances to create."""
    env_spacing: float = MISSING
    """Spacing between cloned environments."""
    episode_length_s: float = None
    """Duration of an episode (in seconds). Default is None (no limit)."""
    send_time_outs: bool = True
    """Whether to send time-outs to the algorithm. Default is True."""
    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is False.

    Note:
        In Isaac Sim 2022.2.0, domain randomization of material properties is not supported when
        ``replicate_physics`` is set to True.
    """


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    debug_vis: bool = False
    """Whether to enable/disable debug visualization in the scene."""
    eye: Tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in m). Default is (0.0, 0.0, 0.0)."""
    resolution: Tuple[int, int] = (1280, 720)
    """The resolution (width, height) of the default viewport (in pixels). Default is (1280, 720).

    This is the resolution of the camera "/OmniverseKit_Persp", that is used in default viewport.
    The camera is also used for rendering RGB images of the simulation.
    """


##
# Environment configuration
##


@configclass
class IsaacEnvCfg:
    """Base configuration of the environment."""

    env: EnvCfg = MISSING
    """General environment configuration."""
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""
    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""
