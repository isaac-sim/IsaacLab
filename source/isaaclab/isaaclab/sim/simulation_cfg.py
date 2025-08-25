# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from typing import Literal

from isaaclab.utils import configclass

from ._impl.newton_manager_cfg import NewtonCfg
from .spawners.materials import RigidBodyMaterialCfg


@configclass
class RenderCfg:
    """Configuration for Omniverse RTX Renderer.

    These parameters are used to configure the Omniverse RTX Renderer. The defaults for IsaacLab are set in the
    experience files: `apps/isaaclab.python.rendering.kit` and `apps/isaaclab.python.headless.rendering.kit`. Setting any
    value here will override the defaults of the experience files.

    For more information, see the `Omniverse RTX Renderer documentation`_.

    .. _Omniverse RTX Renderer documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html
    """

    enable_translucency: bool | None = None
    """Enables translucency for specular transmissive surfaces such as glass at the cost of some performance. Default is False.

    Set variable: /rtx/translucency/enabled
    """

    enable_reflections: bool | None = None
    """Enables reflections at the cost of some performance. Default is False.

    Set variable: /rtx/reflections/enabled
    """

    enable_global_illumination: bool | None = None
    """Enables Diffused Global Illumination at the cost of some performance. Default is False.

    Set variable: /rtx/indirectDiffuse/enabled
    """

    antialiasing_mode: Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"] | None = None
    """Selects the anti-aliasing mode to use. Defaults to DLSS.
       - DLSS: Boosts performance by using AI to output higher resolution frames from a lower resolution input. DLSS samples multiple lower resolution images and uses motion data and feedback from prior frames to reconstruct native quality images.
       - DLAA: Provides higher image quality with an AI-based anti-aliasing technique. DLAA uses the same Super Resolution technology developed for DLSS, reconstructing a native resolution image to maximize image quality.

    Set variable: /rtx/post/dlss/execMode
    """

    enable_dlssg: bool | None = None
    """"Enables the use of DLSS-G.
        DLSS Frame Generation boosts performance by using AI to generate more frames.
        DLSS analyzes sequential frames and motion data to create additional high quality frames.
        This feature requires an Ada Lovelace architecture GPU.
        Enabling this feature also enables additional thread-related activities, which can hurt performance.
        Default is False.

    Set variable: /rtx-transient/dlssg/enabled
    """

    enable_dl_denoiser: bool | None = None
    """Enables the use of a DL denoiser.
       The DL denoiser can help improve the quality of renders, but comes at a cost of performance.

    Set variable: /rtx-transient/dldenoiser/enabled
    """

    dlss_mode: Literal[0, 1, 2, 3] | None = None
    """For DLSS anti-aliasing, selects the performance/quality tradeoff mode.
       Valid values are 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto). Default is 0.

    Set variable: /rtx/post/dlss/execMode
    """

    enable_direct_lighting: bool | None = None
    """Enable direct light contributions from lights.

    Set variable: /rtx/directLighting/enabled
    """

    samples_per_pixel: int | None = None
    """Defines the Direct Lighting samples per pixel.
       Higher values increase the direct lighting quality at the cost of performance. Default is 1.

    Set variable: /rtx/directLighting/sampledLighting/samplesPerPixel"""

    enable_shadows: bool | None = None
    """Enables shadows at the cost of performance. When disabled, lights will not cast shadows. Defaults to True.

    Set variable: /rtx/shadows/enabled
    """

    enable_ambient_occlusion: bool | None = None
    """Enables ambient occlusion at the cost of some performance. Default is False.

    Set variable: /rtx/ambientOcclusion/enabled
    """

    carb_settings: dict | None = None
    """Provides a general dictionary for users to supply all carb rendering settings with native names.
        - Name strings can be formatted like a carb setting, .kit file setting, or python variable.
        - For instance, a key value pair can be
            /rtx/translucency/enabled: False # carb
             rtx.translucency.enabled: False # .kit
             rtx_translucency_enabled: False # python"""

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    """Sets the rendering mode. Behaves the same as the CLI arg '--rendering_mode'"""


@configclass
class SimulationCfg:
    """Configuration for simulation physics."""

    physics_prim_path: str = "/physicsScene"
    """The prim path where the USD PhysicsScene is created. Default is "/physicsScene"."""

    device: str = "cuda:0"
    """The device to run the simulation on. Default is ``"cuda:0"``.

    Valid options are:

    - ``"cpu"``: Use CPU.
    - ``"cuda"``: Use GPU, where the device ID is inferred from :class:`~isaaclab.app.AppLauncher`'s config.
    - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
    """

    dt: float = 1.0 / 60.0
    """The physics simulation time-step (in seconds). Default is 0.0167 seconds."""

    render_interval: int = 1
    """The number of physics simulation steps per rendering step. Default is 1."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81).

    If set to (0.0, 0.0, 0.0), gravity is disabled.
    """

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

    use_fabric: bool = True
    """Enable/disable reading of physics buffers directly. Default is True.

    When running the simulation, updates in the states in the scene is normally synchronized with USD.
    This leads to an overhead in reading the data and does not scale well with massive parallelization.
    This flag allows disabling the synchronization and reading the data directly from the physics buffers.

    It is recommended to set this flag to :obj:`True` when running the simulation with a large number
    of primitives in the scene.

    Note:
        When enabled, the GUI will not update the physics parameters in real-time. To enable real-time
        updates, please set this flag to :obj:`False`.

        When using GPU simulation, it is required to enable Fabric to visualize updates in the renderer.
        Transform updates are propagated to the renderer through Fabric. If Fabric is disabled with GPU simulation,
        the renderer will not be able to render any updates in the simulation, although simulation will still be
        running under the hood.
    """

    newton_cfg: NewtonCfg = NewtonCfg()
    """Newton manager settings. Default is NewtonCfg()."""

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    """Default physics material settings for rigid bodies. Default is RigidBodyMaterialCfg().

    The physics engine defaults to this physics material for all the rigid body prims that do not have any
    physics material specified on them.

    The material is created at the path: ``{physics_prim_path}/defaultMaterial``.
    """

    render: RenderCfg = RenderCfg()
    """Render settings. Default is RenderCfg()."""

    enable_newton_rendering: bool = False
    """Enable/disable rendering using Newton. Default is False.

    When enabled, the Newton to renderer will be called every time the simulation is rendered. If Isaac Sim's
    renderer is also enabled, both will be called.
    """

    create_stage_in_memory: bool = False
    """If stage is first created in memory. Default is False.

    Creating the stage in memory can reduce start-up time.
    """
