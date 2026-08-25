# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the lift event terms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.managers import ManagerTermBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg as SuccessMonitorCfg


@configclass
class MeshClearanceCfg(ManagerTermBaseCfg):
    """Validity criterion for :class:`conditional_reset`: object clear of the robot.

    Configuration for :class:`mesh_clearance`, which checks the object's surface point cloud
    against the robot collision meshes and the robot collision vertices against the object
    mesh with Warp signed-distance queries. The implementation is referenced by name and
    resolved at play, so importing this config pulls no implementation dependencies.
    """

    func: Callable | str = "{DIR}.events:mesh_clearance"

    asset_name: str = MISSING
    """Robot whose collision meshes are queried."""

    body_names: str | list[str] = MISSING
    """Robot bodies whose collision meshes participate, as a regex or list of regexes."""

    object_name: str = MISSING
    """Rigid object whose surface point cloud is checked."""

    num_object_points: int = 64
    """Surface points sampled on the object.

    Defaults to the observation point cloud's count so the sampler's geometry-keyed cache is
    reused instead of sampling a second cloud.
    """

    min_clearance: float = 0.02
    """Required clearance [m]; states with any signed distance below this are invalid."""


@configclass
class GraspTravelDistanceCfg(ManagerTermBaseCfg):
    """Spread descriptor for :class:`conditional_reset`: how far the hand and the object each have to go.

    Configuration for :class:`grasp_travel_distance`. The implementation is referenced by name
    and resolved at play, so importing this config pulls no implementation dependencies.
    """

    func: Callable | str = "{DIR}.events:grasp_travel_distance"

    asset_name: str = MISSING
    """Robot whose bodies are measured."""

    body_names: str | list[str] = MISSING
    """Robot bodies to measure from, as a regex or list of regexes.

    Use the grasping surfaces (fingertips, palm) rather than the whole arm: a link that barely
    moves relative to the object contributes a near-constant descriptor and washes out the spread.
    """

    object_name: str = MISSING
    """Rigid object the distance is measured to."""

    command_name: str = MISSING
    """Name of the pose command term whose sampling range locates the goal region."""

    log_scale: bool = False
    """Whether to report the logarithm of the distances rather than the distances themselves.

    A centimetre at 3 cm changes the problem far more than a centimetre at 60 cm, and the spread
    this feeds covers its input evenly. On a linear scale that hands most of the bank to the far
    half of the reach and inflates the tail of near-hopeless starts; on a log scale the near band
    takes about half the span and the far tail compresses into a slice.
    """


@configclass
class SlabClearanceCfg(ManagerTermBaseCfg):
    """Validity criterion for :class:`conditional_reset`: robot and object clear of slabs.

    Configuration for :class:`slab_clearance`, which checks the object's surface point cloud
    and the robot's collision-mesh vertices against horizontal obstacle slabs. The
    implementation is referenced by name and resolved at play, so importing this config pulls
    no implementation dependencies.
    """

    func: Callable | str = "{DIR}.events:slab_clearance"

    asset_name: str = MISSING
    """Robot whose collision-mesh vertices are checked."""

    body_names: str | list[str] = MISSING
    """Robot bodies whose collision-mesh vertices participate, as a regex or list of regexes."""

    object_name: str = MISSING
    """Rigid object whose surface point cloud is checked."""

    obstacle_slabs: list[tuple[tuple[float, float] | None, tuple[float, float] | None, float]] = MISSING
    """Horizontal obstacles as ``(x_range, y_range, top_z)`` in the environment frame [m].

    A range of ``None`` means unbounded (e.g. the ground plane is ``(None, None, 0.0)``).
    """

    num_object_points: int = 64
    """Surface points sampled on the object.

    Defaults to the observation point cloud's count so the sampler's geometry-keyed cache is
    reused instead of sampling a second cloud.
    """

    min_clearance: float = 0.02
    """Required clearance [m]; states with any point below this above a slab are invalid."""
