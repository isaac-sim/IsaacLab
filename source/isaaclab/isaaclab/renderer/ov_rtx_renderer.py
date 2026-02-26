# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse RTX renderer for applying RTX settings."""

from __future__ import annotations

import logging
import os
import toml
from typing import TYPE_CHECKING

import flatdict

from .ov_rtx_renderer_cfg import OVRTXRendererCfg
from .renderer import RendererBase

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class OVRTXRenderer(RendererBase):
    """Omniverse RTX renderer for applying RTX settings.

    Unlike camera renderers (NewtonWarpRenderer), this renderer doesn't produce
    image output. It configures the RTX rendering pipeline settings from
    SimulationCfg.render_cfg.

    Lifecycle is managed by RendererInterface, but actual rendering is handled
    by Omniverse's RTX pipeline.
    """

    def __init__(self, cfg: OVRTXRendererCfg, sim_context: SimulationContext | None = None):
        """Initialize RTX renderer.

        Args:
            cfg: Renderer configuration.
            sim_context: Optional simulation context for settings access.
                        If None, settings must be applied later via apply_settings().
        """
        super().__init__(cfg)
        self._sim = sim_context
        self._fabric_iface = None
        self._update_fabric = None

        if sim_context is not None:
            self._sim.settings.set("/isaaclab/fabric_enabled", False)
            self._apply_render_settings()

    def apply_settings(self, sim_context: SimulationContext) -> None:
        """Apply RTX settings from simulation context.

        Args:
            sim_context: Simulation context with render_cfg.
        """
        self._sim = sim_context
        self._sim.settings.set("/isaaclab/fabric_enabled", False)
        self._apply_render_settings()

    def _apply_render_settings(self) -> None:
        """Apply RTX settings from RenderCfg."""
        if self._sim is None:
            return

        rendering_setting_name_mapping = {
            "enable_translucency": "/rtx/translucency/enabled",
            "enable_reflections": "/rtx/reflections/enabled",
            "enable_global_illumination": "/rtx/indirectDiffuse/enabled",
            "enable_dlssg": "/rtx-transient/dlssg/enabled",
            "enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
            "dlss_mode": "/rtx/post/dlss/execMode",
            "enable_direct_lighting": "/rtx/directLighting/enabled",
            "samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
            "enable_shadows": "/rtx/shadows/enabled",
            "enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
        }

        not_carb_settings = ["rendering_mode", "carb_settings", "antialiasing_mode"]

        rendering_mode = self._sim.cfg.render_cfg.rendering_mode
        if rendering_mode is not None:
            supported_rendering_modes = ["performance", "balanced", "quality"]
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes {supported_rendering_modes}."
                )

            import carb

            repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
            preset_filename = os.path.join(repo_path, f"apps/rendering_modes/{rendering_mode}.kit")
            with open(preset_filename) as file:
                preset_dict = toml.load(file)
            preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

            for key, value in preset_dict.items():
                key = "/" + key.replace(".", "/")
                self._sim.settings.set(key, value)

        for key, value in vars(self._sim.cfg.render_cfg).items():
            if value is None or key in not_carb_settings:
                continue
            if key not in rendering_setting_name_mapping:
                raise ValueError(
                    f"'{key}' in RenderCfg not found. Note: internal 'rendering_setting_name_mapping' dictionary might"
                    " need to be updated."
                )
            key = rendering_setting_name_mapping[key]
            self._sim.settings.set(key, value)

        carb_settings = self._sim.cfg.render_cfg.carb_settings
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")
                elif "." in key:
                    key = "/" + key.replace(".", "/")
                if self._sim.settings.get(key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.general_parameters does not map to a carb setting.")
                self._sim.settings.set(key, value)

        if self._sim.cfg.render_cfg.antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=self._sim.cfg.render_cfg.antialiasing_mode)
            except Exception:
                pass

        render_mode = self._sim.settings.get("/rtx/rendermode")
        if render_mode is not None and render_mode.lower() == "raytracedlighting":
            self._sim.settings.set("/rtx/rendermode", "RaytracedLighting")

    def initialize(self) -> None:
        """Initialize renderer (no-op for RTX settings renderer)."""
        pass

    def step(self) -> None:
        """Step renderer (no-op for RTX settings renderer)."""
        pass

    def reset(self) -> None:
        """Reset renderer (no-op for RTX settings renderer)."""
        pass

    def close(self) -> None:
        """Clean up renderer resources."""
        self._fabric_iface = None
        self._update_fabric = None

    def _initialize_output(self) -> None:
        """Initialize output (no-op - RTX renderer doesn't produce direct output)."""
        pass

    def load_fabric_interface(self) -> None:
        """Load fabric interface if enabled."""
        if self._sim is not None and self._sim.cfg.use_fabric:
            from omni.physxfabric import get_physx_fabric_interface

            self._fabric_iface = get_physx_fabric_interface()
            if hasattr(self._fabric_iface, "force_update"):
                self._update_fabric = self._fabric_iface.force_update
            else:
                self._update_fabric = self._fabric_iface.update

            self._sim.settings.set("/isaaclab/fabric_enabled", True)
