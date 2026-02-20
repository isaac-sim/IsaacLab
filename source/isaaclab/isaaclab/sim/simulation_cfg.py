# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from __future__ import annotations

from typing import Any, Literal  # Literal used by RenderCfg

from isaaclab.physics import PhysicsCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg


@configclass
class RenderCfg:
    """Configuration for Omniverse RTX Renderer.

    These parameters are used to configure the Omniverse RTX Renderer.

    The defaults for IsaacLab are set in the experience files:

    * ``apps/isaaclab.python.rendering.kit``: Setting used when running the simulation with the GUI enabled.
    * ``apps/isaaclab.python.headless.rendering.kit``: Setting used when running the simulation in headless mode.

    Setting any value here will override the defaults of the experience files.

    For more information, see the `Omniverse RTX Renderer documentation`_.

    .. _Omniverse RTX Renderer documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html
    """

    enable_translucency: bool | None = None
    """Enables translucency for specular transmissive surfaces such as glass.

    This comes at the cost of some performance. Default is False.
    This is set by the variable: ``/rtx/translucency/enabled``.
    """

    enable_reflections: bool | None = None
    """Enables reflections at the cost of some performance. Default is False.

    This is set by the variable: ``/rtx/reflections/enabled``.
    """

    enable_global_illumination: bool | None = None
    """Enables Diffused Global Illumination at the cost of some performance. Default is False.

    This is set by the variable: ``/rtx/indirectDiffuse/enabled``.
    """

    antialiasing_mode: Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"] | None = None
    """Selects the anti-aliasing mode to use. Defaults to DLSS.

    - **DLSS**: Boosts performance by using AI to output higher resolution frames from a lower resolution input.
      DLSS samples multiple lower resolution images and uses motion data and feedback from prior frames to reconstruct
      native quality images.
    - **DLAA**: Provides higher image quality with an AI-based anti-aliasing technique. DLAA uses the same
      Super Resolution technology developed for DLSS, reconstructing a native resolution image to maximize
      image quality.

    This is set by the variable: ``/rtx/post/dlss/execMode``.
    """

    enable_dlssg: bool | None = None
    """"Enables the use of DLSS-G. Default is False.

    DLSS Frame Generation boosts performance by using AI to generate more frames. DLSS analyzes sequential frames
    and motion data to create additional high quality frames.

    .. note::

        This feature requires an Ada Lovelace architecture GPU. Enabling this feature also enables additional
        thread-related activities, which can hurt performance.

    This is set by the variable: ``/rtx-transient/dlssg/enabled``.
    """

    enable_dl_denoiser: bool | None = None
    """Enables the use of a DL denoiser.

    The DL denoiser can help improve the quality of renders, but comes at a cost of performance.

    This is set by the variable: ``/rtx-transient/dldenoiser/enabled``.
    """

    dlss_mode: Literal[0, 1, 2, 3] | None = None
    """For DLSS anti-aliasing, selects the performance/quality tradeoff mode. Default is 0.

    Valid values are:

    * 0 (Performance)
    * 1 (Balanced)
    * 2 (Quality)
    * 3 (Auto)

    This is set by the variable: ``/rtx/post/dlss/execMode``.
    """

    enable_direct_lighting: bool | None = None
    """Enable direct light contributions from lights. Default is False.

    This is set by the variable: ``/rtx/directLighting/enabled``.
    """

    samples_per_pixel: int | None = None
    """Defines the Direct Lighting samples per pixel. Default is 1.

    A higher value increases the direct lighting quality at the cost of performance.

    This is set by the variable: ``/rtx/directLighting/sampledLighting/samplesPerPixel``.
    """

    enable_shadows: bool | None = None
    """Enables shadows at the cost of performance. Defaults to True.

    When disabled, lights will not cast shadows.

    This is set by the variable: ``/rtx/shadows/enabled``.
    """

    enable_ambient_occlusion: bool | None = None
    """Enables ambient occlusion at the cost of some performance. Default is False.

    This is set by the variable: ``/rtx/ambientOcclusion/enabled``.
    """

    dome_light_upper_lower_strategy: Literal[0, 3, 4] | None = None
    """Selects how to sample the Dome Light. Default is 0.
    For more information, refer to the `documentation`_.

    .. _documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#dome-light

    Valid values are:

    * 0: **Image-Based Lighting (IBL)** - Most accurate even for high-frequency Dome Light textures.
      Can introduce sampling artifacts in real-time mode.
    * 3: **Limited Image-Based Lighting** - Only sampled for reflection and refraction. Fastest, but least
      accurate. Good for cases where the Dome Light contributes less than other light sources.
    * 4: **Approximated Image-Based Lighting** - Fast and artifacts-free sampling in real-time mode but only
      works well with a low-frequency texture (e.g., a sky with no sun disc where the sun is instead a separate
      Distant Light). Requires enabling Direct Lighting denoiser.

    This is set by the variable: ``/rtx/domeLight/upperLowerStrategy``.
    """

    carb_settings: dict[str, Any] | None = None
    """A general dictionary for users to supply all carb rendering settings with native names.

    The keys of the dictionary can be formatted like a carb setting, .kit file setting, or python variable.
    For instance, a key value pair can be ``/rtx/translucency/enabled: False`` (carb),
    ``rtx.translucency.enabled: False`` (.kit), or ``rtx_translucency_enabled: False`` (python).
    """

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    """The rendering mode.

    This behaves the same as the passing the CLI arg ``--rendering_mode`` to an executable script.
    """


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
    a different config (e.g., NewtonCfg) to use a different physics backend.
    """

    render: RenderCfg = RenderCfg()
    """Render settings. Default is RenderCfg()."""

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
