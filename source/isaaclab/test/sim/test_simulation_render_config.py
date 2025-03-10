# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
if not AppLauncher.instance() or AppLauncher.instance()._enable_cameras is False:
    AppLauncher.clear_instance()
    simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import carb

from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext


def test_render_cfg():
    """Test that the simulation context is created with the correct render cfg."""
    enable_translucency = True
    enable_reflections = True
    enable_global_illumination = True
    antialiasing_mode = "DLAA"
    enable_dlssg = True
    enable_dl_denoiser = True
    dlss_mode = 0
    enable_direct_lighting = True
    samples_per_pixel = 4
    enable_shadows = True
    enable_ambient_occlusion = True

    render_cfg = RenderCfg(
        enable_translucency=enable_translucency,
        enable_reflections=enable_reflections,
        enable_global_illumination=enable_global_illumination,
        antialiasing_mode=antialiasing_mode,
        enable_dlssg=enable_dlssg,
        dlss_mode=dlss_mode,
        enable_dl_denoiser=enable_dl_denoiser,
        enable_direct_lighting=enable_direct_lighting,
        samples_per_pixel=samples_per_pixel,
        enable_shadows=enable_shadows,
        enable_ambient_occlusion=enable_ambient_occlusion,
    )

    cfg = SimulationCfg(render=render_cfg)

    sim = SimulationContext(cfg)

    assert sim.cfg.render.enable_translucency == enable_translucency
    assert sim.cfg.render.enable_reflections == enable_reflections
    assert sim.cfg.render.enable_global_illumination == enable_global_illumination
    assert sim.cfg.render.antialiasing_mode == antialiasing_mode
    assert sim.cfg.render.enable_dlssg == enable_dlssg
    assert sim.cfg.render.dlss_mode == dlss_mode
    assert sim.cfg.render.enable_dl_denoiser == enable_dl_denoiser
    assert sim.cfg.render.enable_direct_lighting == enable_direct_lighting
    assert sim.cfg.render.samples_per_pixel == samples_per_pixel
    assert sim.cfg.render.enable_shadows == enable_shadows
    assert sim.cfg.render.enable_ambient_occlusion == enable_ambient_occlusion

    carb_settings_iface = carb.settings.get_settings()
    assert carb_settings_iface.get("/rtx/translucency/enabled") == sim.cfg.render.enable_translucency
    assert carb_settings_iface.get("/rtx/reflections/enabled") == sim.cfg.render.enable_reflections
    assert carb_settings_iface.get("/rtx/indirectDiffuse/enabled") == sim.cfg.render.enable_global_illumination
    assert carb_settings_iface.get("/rtx-transient/dlssg/enabled") == sim.cfg.render.enable_dlssg
    assert carb_settings_iface.get("/rtx-transient/dldenoiser/enabled") == sim.cfg.render.enable_dl_denoiser
    assert carb_settings_iface.get("/rtx/post/dlss/execMode") == sim.cfg.render.dlss_mode
    assert carb_settings_iface.get("/rtx/directLighting/enabled") == sim.cfg.render.enable_direct_lighting
    assert carb_settings_iface.get("/rtx/directLighting/sampledLighting/samplesPerPixel") == sim.cfg.render.samples_per_pixel
    assert carb_settings_iface.get("/rtx/shadows/enabled") == sim.cfg.render.enable_shadows
    assert carb_settings_iface.get("/rtx/ambientOcclusion/enabled") == sim.cfg.render.enable_ambient_occlusion
    assert carb_settings_iface.get("/rtx/post/aa/op") == 4  # dlss = 3, dlaa=4


def test_render_cfg_defaults():
    """Test that the simulation context is created with the correct render cfg."""
    enable_translucency = False
    enable_reflections = False
    enable_global_illumination = False
    antialiasing_mode = "DLSS"
    enable_dlssg = False
    enable_dl_denoiser = False
    dlss_mode = 2
    enable_direct_lighting = False
    samples_per_pixel = 1
    enable_shadows = False
    enable_ambient_occlusion = False

    render_cfg = RenderCfg(
        enable_translucency=enable_translucency,
        enable_reflections=enable_reflections,
        enable_global_illumination=enable_global_illumination,
        antialiasing_mode=antialiasing_mode,
        enable_dlssg=enable_dlssg,
        enable_dl_denoiser=enable_dl_denoiser,
        dlss_mode=dlss_mode,
        enable_direct_lighting=enable_direct_lighting,
        samples_per_pixel=samples_per_pixel,
        enable_shadows=enable_shadows,
        enable_ambient_occlusion=enable_ambient_occlusion,
    )

    cfg = SimulationCfg(render=render_cfg)

    sim = SimulationContext(cfg)

    assert sim.cfg.render.enable_translucency == enable_translucency
    assert sim.cfg.render.enable_reflections == enable_reflections
    assert sim.cfg.render.enable_global_illumination == enable_global_illumination
    assert sim.cfg.render.antialiasing_mode == antialiasing_mode
    assert sim.cfg.render.enable_dlssg == enable_dlssg
    assert sim.cfg.render.enable_dl_denoiser == enable_dl_denoiser
    assert sim.cfg.render.dlss_mode == dlss_mode
    assert sim.cfg.render.enable_direct_lighting == enable_direct_lighting
    assert sim.cfg.render.samples_per_pixel == samples_per_pixel
    assert sim.cfg.render.enable_shadows == enable_shadows
    assert sim.cfg.render.enable_ambient_occlusion == enable_ambient_occlusion

    carb_settings_iface = carb.settings.get_settings()
    assert carb_settings_iface.get("/rtx/translucency/enabled") == sim.cfg.render.enable_translucency
    assert carb_settings_iface.get("/rtx/reflections/enabled") == sim.cfg.render.enable_reflections
    assert carb_settings_iface.get("/rtx/indirectDiffuse/enabled") == sim.cfg.render.enable_global_illumination
    assert carb_settings_iface.get("/rtx-transient/dlssg/enabled") == sim.cfg.render.enable_dlssg
    assert carb_settings_iface.get("/rtx-transient/dldenoiser/enabled") == sim.cfg.render.enable_dl_denoiser
    assert carb_settings_iface.get("/rtx/post/dlss/execMode") == sim.cfg.render.dlss_mode
    assert carb_settings_iface.get("/rtx/directLighting/enabled") == sim.cfg.render.enable_direct_lighting
    assert carb_settings_iface.get("/rtx/directLighting/sampledLighting/samplesPerPixel") == sim.cfg.render.samples_per_pixel
    assert carb_settings_iface.get("/rtx/shadows/enabled") == sim.cfg.render.enable_shadows
    assert carb_settings_iface.get("/rtx/ambientOcclusion/enabled") == sim.cfg.render.enable_ambient_occlusion
    assert carb_settings_iface.get("/rtx/post/aa/op") == 3  # dlss = 3, dlaa=4
