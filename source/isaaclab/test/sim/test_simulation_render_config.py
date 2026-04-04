# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import os

import flatdict
import pytest
import toml

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.version import get_isaac_sim_version


@pytest.mark.skip(reason="Timeline not stopped")
@pytest.mark.isaacsim_ci
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

    # FIXME: when running all tests, the timeline is not stopped, force stop it here but also that does not the timeline
    # omni.timeline.get_timeline_interface().stop()

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

    assert sim.get_setting("/rtx/translucency/enabled") == sim.cfg.render.enable_translucency
    assert sim.get_setting("/rtx/reflections/enabled") == sim.cfg.render.enable_reflections
    assert sim.get_setting("/rtx/indirectDiffuse/enabled") == sim.cfg.render.enable_global_illumination
    assert sim.get_setting("/rtx-transient/dlssg/enabled") == sim.cfg.render.enable_dlssg
    assert sim.get_setting("/rtx-transient/dldenoiser/enabled") == sim.cfg.render.enable_dl_denoiser
    assert sim.get_setting("/rtx/post/dlss/execMode") == sim.cfg.render.dlss_mode
    assert sim.get_setting("/rtx/directLighting/enabled") == sim.cfg.render.enable_direct_lighting
    assert sim.get_setting("/rtx/directLighting/sampledLighting/samplesPerPixel") == sim.cfg.render.samples_per_pixel
    assert sim.get_setting("/rtx/shadows/enabled") == sim.cfg.render.enable_shadows
    assert sim.get_setting("/rtx/ambientOcclusion/enabled") == sim.cfg.render.enable_ambient_occlusion
    assert sim.get_setting("/rtx/post/aa/op") == 4  # dlss = 3, dlaa=4


@pytest.mark.isaacsim_ci
def test_render_cfg_presets():
    """Test that the simulation context is created with the correct render cfg preset with overrides."""

    # carb setting dictionary overrides
    carb_settings = {"/rtx/raytracing/subpixel/mode": 3, "/rtx/pathtracing/maxSamplesPerLaunch": 999999}
    # user-friendly setting overrides
    dlss_mode = ("/rtx/post/dlss/execMode", 5)

    rendering_modes = ["performance", "balanced", "quality"]

    for rendering_mode in rendering_modes:
        # Clear any existing simulation context before creating a new one
        SimulationContext.clear_instance()

        # grab isaac lab apps path
        isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
        # for Isaac Sim 5 compatibility, we use the 5 rendering mode app files in a different folder
        if get_isaac_sim_version().major < 6:
            isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_5")

        # grab preset settings
        preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
        with open(preset_filename) as file:
            preset_dict = toml.load(file)
        preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

        render_cfg = RenderCfg(
            rendering_mode=rendering_mode,
            dlss_mode=dlss_mode[1],
            carb_settings=carb_settings,
        )

        cfg = SimulationCfg(render=render_cfg)

        SimulationContext(cfg)

        settings = get_settings_manager()
        for key, val in preset_dict.items():
            setting_name = "/" + key.replace(".", "/")  # convert to setting path format

            if setting_name in carb_settings:
                setting_gt = carb_settings[setting_name]
            elif setting_name == dlss_mode[0]:
                setting_gt = dlss_mode[1]
            else:
                setting_gt = val

            setting_val = settings.get(setting_name)

            assert setting_gt == setting_val, (
                f"Mismatch for '{setting_name}' in mode '{rendering_mode}': "
                f"expected {setting_gt!r}, got {setting_val!r}"
            )

    # Clean up after the test
    SimulationContext.clear_instance()


@pytest.mark.skip(reason="Timeline not stopped")
@pytest.mark.isaacsim_ci
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

    assert sim.get_setting("/rtx/translucency/enabled") == sim.cfg.render.enable_translucency
    assert sim.get_setting("/rtx/reflections/enabled") == sim.cfg.render.enable_reflections
    assert sim.get_setting("/rtx/indirectDiffuse/enabled") == sim.cfg.render.enable_global_illumination
    assert sim.get_setting("/rtx-transient/dlssg/enabled") == sim.cfg.render.enable_dlssg
    assert sim.get_setting("/rtx-transient/dldenoiser/enabled") == sim.cfg.render.enable_dl_denoiser
    assert sim.get_setting("/rtx/post/dlss/execMode") == sim.cfg.render.dlss_mode
    assert sim.get_setting("/rtx/directLighting/enabled") == sim.cfg.render.enable_direct_lighting
    assert sim.get_setting("/rtx/directLighting/sampledLighting/samplesPerPixel") == sim.cfg.render.samples_per_pixel
    assert sim.get_setting("/rtx/shadows/enabled") == sim.cfg.render.enable_shadows
    assert sim.get_setting("/rtx/ambientOcclusion/enabled") == sim.cfg.render.enable_ambient_occlusion
    assert sim.get_setting("/rtx/post/aa/op") == 3  # dlss = 3, dlaa=4
