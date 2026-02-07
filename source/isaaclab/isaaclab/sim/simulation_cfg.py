# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

import warnings
from typing import Any, Literal  # Literal used by RenderCfg

from isaaclab.utils import configclass

from isaaclab.physics.physics_manager_cfg import PhysicsManagerCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxManagerCfg
from isaaclab.visualizers import VisualizerCfg


# Deprecated alias for PhysxCfg -> PhysxManagerCfg
# This supports old code that uses `from isaaclab.sim.simulation_cfg import PhysxCfg`
class PhysxCfg(PhysxManagerCfg):
    """DEPRECATED: Use PhysxManagerCfg from isaaclab_physx.physics instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PhysxCfg is deprecated. Use PhysxManagerCfg from isaaclab_physx.physics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Mapping of deprecated SimulationCfg fields to their new location in physics_manager_cfg
_DEPRECATED_FIELDS = {
    "dt": "physics_manager_cfg.dt",
    "gravity": "physics_manager_cfg.gravity",
    "physics_prim_path": "physics_manager_cfg.physics_prim_path",
    "physics_material": "physics_manager_cfg.physics_material",
    "use_fabric": "physics_manager_cfg.use_fabric",
    "physx": "physics_manager_cfg (PhysxManagerCfg attributes directly)",
}


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
    """Enables translucency for specular transmissive surfaces such as glass at the cost of some performance. Default is False.

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
    - **DLAA**: Provides higher image quality with an AI-based anti-aliasing technique. DLAA uses the same Super Resolution
      technology developed for DLSS, reconstructing a native resolution image to maximize image quality.

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
    For instance, a key value pair can be ``/rtx/translucency/enabled: False`` (carb), ``rtx.translucency.enabled: False`` (.kit),
    or ``rtx_translucency_enabled: False`` (python).
    """

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    """The rendering mode.

    This behaves the same as the passing the CLI arg ``--rendering_mode`` to an executable script.
    """


@configclass
class SimulationCfg:
    """Configuration for simulation physics.

    .. note::
        The following fields have been moved to ``physics_manager_cfg`` and are deprecated:

        - ``dt`` → ``physics_manager_cfg.dt``
        - ``gravity`` → ``physics_manager_cfg.gravity``
        - ``physics_prim_path`` → ``physics_manager_cfg.physics_prim_path``
        - ``physics_material`` → ``physics_manager_cfg.physics_material``
        - ``use_fabric`` → ``physics_manager_cfg.use_fabric``
        - ``physx`` → Use ``PhysxManagerCfg`` attributes directly

        Using the old field names will issue a deprecation warning and forward
        the values to the new location.
    """

    device: str = "cuda:0"
    """The device to run the simulation on. Default is ``"cuda:0"``.

    Valid options are:

    - ``"cpu"``: Use CPU.
    - ``"cuda"``: Use GPU, where the device ID is inferred from :class:`~isaaclab.app.AppLauncher`'s config.
    - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
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

    physics_manager_cfg: PhysicsManagerCfg = PhysxManagerCfg()
    """Physics manager configuration. Default is PhysxManagerCfg().

    This configuration determines which physics manager to use. Override with
    a different config (e.g., NewtonManagerCfg) to use a different physics backend.
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

    # Deprecated fields - accepted in constructor for backward compatibility
    dt: float | None = None
    """DEPRECATED: Use physics_manager_cfg.dt instead."""

    gravity: tuple[float, float, float] | None = None
    """DEPRECATED: Use physics_manager_cfg.gravity instead."""

    physics_prim_path: str | None = None
    """DEPRECATED: Use physics_manager_cfg.physics_prim_path instead."""

    physics_material: Any | None = None
    """DEPRECATED: Use physics_manager_cfg.physics_material instead."""

    use_fabric: bool | None = None
    """DEPRECATED: Use physics_manager_cfg.use_fabric instead."""

    physx: Any | None = None
    """DEPRECATED: Use physics_manager_cfg (PhysxManagerCfg) directly instead.
    
    After initialization, this field is set to physics_manager_cfg for backward compatibility.
    """

    def __post_init__(self):
        """Forward deprecated constructor arguments to physics_manager_cfg."""
        deprecated_fields = ["dt", "gravity", "physics_prim_path", "physics_material", "use_fabric"]

        for field_name in deprecated_fields:
            # Use getattr with None default - field might not exist during class definition
            value = getattr(self, field_name, None)
            if value is not None:
                warnings.warn(
                    f"SimulationCfg({field_name}=...) is deprecated. "
                    f"Use SimulationCfg(physics_manager_cfg=PhysxManagerCfg({field_name}=...)) instead.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                # Forward to physics_manager_cfg
                if hasattr(self.physics_manager_cfg, field_name):
                    setattr(self.physics_manager_cfg, field_name, value)

        # Delete deprecated fields so __getattr__ is called when accessing them
        # This allows runtime access like self.sim.dt to work via __getattr__
        for field_name in deprecated_fields:
            if field_name != "physics_material":  # physics_material needs object access
                try:
                    delattr(self, field_name)
                except AttributeError:
                    pass

        # Set physics_material to point to physics_manager_cfg.physics_material for backward-compatible access
        if hasattr(self.physics_manager_cfg, "physics_material"):
            object.__setattr__(self, "physics_material", self.physics_manager_cfg.physics_material)

        # Handle physx=PhysxCfg(...) - copy PhysX-specific attributes to physics_manager_cfg
        # The old PhysxCfg only had PhysX-specific settings, not dt/gravity/etc.
        physx_cfg = getattr(self, "physx", None)
        if physx_cfg is not None:
            warnings.warn(
                "SimulationCfg(physx=...) is deprecated. "
                "Use SimulationCfg(physics_manager_cfg=PhysxManagerCfg(...)) instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            # PhysX-specific fields that should be copied (not general physics settings)
            physx_specific_fields = {
                "bounce_threshold_velocity",
                "friction_offset_threshold",
                "friction_correlation_distance",
                "solver_type",
                "enable_stabilization",
                "max_depenetration_velocity",
                "enable_enhanced_determinism",
                "min_position_iteration_count",
                "max_position_iteration_count",
                "min_velocity_iteration_count",
                "max_velocity_iteration_count",
                "enable_ccd",
                "gpu_max_rigid_contact_count",
                "gpu_max_rigid_patch_count",
                "gpu_found_lost_pairs_capacity",
                "gpu_found_lost_aggregate_pairs_capacity",
                "gpu_total_aggregate_pairs_capacity",
                "gpu_heap_capacity",
                "gpu_temp_buffer_capacity",
                "gpu_max_num_partitions",
                "gpu_max_soft_body_contacts",
                "gpu_max_particle_contacts",
                "gpu_collision_stack_size",
            }

            import dataclasses

            if dataclasses.is_dataclass(physx_cfg):
                for field in dataclasses.fields(physx_cfg):
                    if field.name in physx_specific_fields:
                        value = getattr(physx_cfg, field.name)
                        # Get field default
                        if field.default is not dataclasses.MISSING:
                            default = field.default
                        elif field.default_factory is not dataclasses.MISSING:
                            default = field.default_factory()
                        else:
                            default = None
                        # Only copy if different from default
                        if value != default and hasattr(self.physics_manager_cfg, field.name):
                            setattr(self.physics_manager_cfg, field.name, value)

        # Set physx to physics_manager_cfg for backward-compatible access (sim.physx.some_setting)
        object.__setattr__(self, "physx", self.physics_manager_cfg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept deprecated attribute assignment and forward to physics_manager_cfg."""
        # Mapping of deprecated fields to their new location
        deprecated_map = {
            "dt": "physics_manager_cfg.dt",
            "gravity": "physics_manager_cfg.gravity",
            "physics_prim_path": "physics_manager_cfg.physics_prim_path",
            "physics_material": "physics_manager_cfg.physics_material",
            "use_fabric": "physics_manager_cfg.use_fabric",
        }

        if name in deprecated_map and value is not None:
            # Only forward non-None values (None means "not set" for deprecated fields)
            try:
                physics_cfg = object.__getattribute__(self, "physics_manager_cfg")
                if hasattr(physics_cfg, name):
                    setattr(physics_cfg, name, value)
                    warnings.warn(
                        f"SimulationCfg.{name} is deprecated. "
                        f"Use {deprecated_map[name]} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return
            except AttributeError:
                # physics_manager_cfg not yet set, fall through to normal setattr
                pass
        # Default behavior
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """Intercept deprecated attribute access and forward to physics_manager_cfg."""
        # Mapping of deprecated fields to their new location
        deprecated_map = {
            "dt": "physics_manager_cfg.dt",
            "gravity": "physics_manager_cfg.gravity",
            "physics_prim_path": "physics_manager_cfg.physics_prim_path",
            "physics_material": "physics_manager_cfg.physics_material",
            "use_fabric": "physics_manager_cfg.use_fabric",
        }

        if name in deprecated_map:
            try:
                physics_cfg = object.__getattribute__(self, "physics_manager_cfg")
                if hasattr(physics_cfg, name):
                    warnings.warn(
                        f"SimulationCfg.{name} is deprecated. "
                        f"Use {deprecated_map[name]} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return getattr(physics_cfg, name)
            except AttributeError:
                pass

        # Note: 'physx' is now a field set to physics_manager_cfg in __post_init__
        # for backward compatibility with sim.physx.some_setting access

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
