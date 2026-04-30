# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for visualization markers."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class NewtonMarkerCfg:
    """Newton-specific fallback used when a marker cannot be rendered from its Kit spawner config directly."""

    renderer: Literal["mesh", "frame", "none"] = "none"
    """Renderer path used by Newton-family visualizers."""

    mesh_type: Literal["arrow", "box", "sphere", "cylinder", "capsule", "cone"] | None = None
    """Procedural Newton mesh type for ``renderer='mesh'``."""

    mesh_params: dict[str, float | tuple[float, float, float]] = {}
    """Parameters used to construct the procedural Newton mesh."""

    scale: tuple[float, float, float] | None = None
    """Optional scale applied before per-instance marker scales."""

    color: tuple[float, float, float] | None = None
    """Optional Newton fallback color. If None, the marker visual material is used when possible."""


@configclass
class VisualizationMarkersCfg:
    """A class to configure a :class:`VisualizationMarkers`."""

    prim_path: str = MISSING
    """The prim path where the :class:`UsdGeom.PointInstancer` will be created."""

    markers: dict[str, SpawnerCfg] = MISSING
    """The dictionary of marker configurations.

    The key is the name of the marker, and the value is the configuration of the marker.
    The key is used to identify the marker in the class.
    """

    newton_markers: dict[str, NewtonMarkerCfg] = {}
    """Optional Newton-specific fallback markers keyed by entries in :attr:`markers`.

    These are used by Newton-family visualizers only. If a key is omitted, the Newton renderer attempts to infer a
    lightweight fallback from the corresponding Kit marker config. Generic USD-to-Newton mesh conversion is
    intentionally deferred; common USD marker assets such as arrows and frames are handled by built-in fallbacks.
    """
