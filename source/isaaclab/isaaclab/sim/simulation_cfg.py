# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from __future__ import annotations

from typing import Literal

from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.rendering_mode.rendering_mode_cfg import RenderingModeCfg
from isaaclab.visualizers import VisualizerCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class SimulationCfg:
    """Configuration for simulation physics.

    This class contains the main simulation parameters including physics time-step, gravity,
    device settings, and physics backend configuration.
    """

    device: str = "cuda:0"
    """The device to run the simulation on. Default is ``"cuda:0"``.

    Valid options are:

    - ``"cpu"``: Use CPU.
    - ``"cuda"``: Use GPU, where the device ID is inferred from :class:`~isaaclab.app.AppLauncher`'s config.
    - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
    """

    dt: float = 1.0 / 60.0
    """The physics simulation time-step (in seconds). Default is 0.0167 seconds."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81)."""

    physics_prim_path: str = "/physicsScene"
    """The prim path where the USD PhysicsScene is created. Default is "/physicsScene"."""

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    """Default physics material settings for rigid bodies. Default is RigidBodyMaterialCfg.

    The physics engine defaults to this physics material for all the rigid body prims that do not have any
    physics material specified on them.

    The material is created at the path: ``{physics_prim_path}/defaultMaterial``.
    """

    use_fabric: bool = True
    """Enable/disable reading of physics buffers directly. Default is True.

    When running the simulation, updates in the states in the scene is normally synchronized with USD.
    This leads to an overhead in reading the data and does not scale well with massive parallelization.
    This flag allows disabling the synchronization and reading the data directly from the physics buffers.

    It is recommended to set this flag to :obj:`True` when running the simulation with a large number
    of primitives in the scene.
    """

    render_interval: int = 1
    """The number of physics simulation steps per rendering step. Default is 1."""

    enable_scene_query_support: bool = False
    """Enable/disable scene query support for collision shapes. Default is False.

    This flag allows performing collision queries (raycasts, sweeps, and overlaps) on actors and
    attached shapes in the scene. This is useful for implementing custom collision detection logic
    outside of the physics engine.

    If set to False, the physics engine does not create the scene query manager and the scene query
    functionality will not be available. However, this provides some performance speed-up.

    Note:
        This flag is overridden to True inside the :class:`SimulationContext` class when running the simulation
        with the GUI enabled. This is to allow certain GUI features to work properly.
    """

    physics: PhysicsCfg | None = None
    """Physics manager configuration. Default is None (uses PhysxCfg()).

    This configuration determines which physics manager to use. Override with
    a different config (e.g., NewtonManagerCfg) to use a different physics backend.
    """

    rendering_quality_cfgs: dict[str, RenderingModeCfg] = {
        "performance": RenderingModeCfg(kit_rendering_preset="performance"),
        "balanced": RenderingModeCfg(kit_rendering_preset="balanced"),
        "quality": RenderingModeCfg(kit_rendering_preset="quality"),
    }
    """Named rendering quality profiles available to visualizers/renderers."""

    create_stage_in_memory: bool = False
    """If stage is first created in memory. Default is False.

    Creating the stage in memory can reduce start-up time.
    """

    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    """The logging level. Default is "WARNING"."""

    save_logs_to_file: bool = True
    """Save logs to a file. Default is True."""

    log_dir: str | None = None
    """The directory to save the logs to. Default is None.

    If :attr:`save_logs_to_file` is True, the logs will be saved to the directory specified by :attr:`log_dir`.
    If None, the logs will be saved to the temp directory.
    """

    visualizer_cfgs: list[VisualizerCfg] | VisualizerCfg | None = None
    """The list of visualizer configurations. Default is None."""

    renderer_cfgs: list[RendererCfg] | RendererCfg | None = None
    """Placeholder list of renderer configurations. Default is None.

    TODO: Hook renderer lifecycle and execution flow into SimulationContext.
    """
