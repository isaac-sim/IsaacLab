# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import pytest
from isaaclab_physx.visualizers import KitVisualizerCfg

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.rendering_mode import RenderingModeCfg, get_kit_rendering_preset
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext


@pytest.mark.isaacsim_ci
def test_rendering_mode_presets():
    """Preset profiles should apply and allow explicit field overrides."""
    dlss_override = 3

    for mode_name in ["performance", "balanced", "quality"]:
        SimulationContext.clear_instance()
        preset_dict = get_kit_rendering_preset(mode_name)
        profile_name = f"profile_{mode_name}"

        cfg = SimulationCfg(
            rendering_mode_cfgs={
                profile_name: RenderingModeCfg(
                    rendering_mode_preset=mode_name,
                    kit_dlss_mode=dlss_override,
                )
            },
            visualizer_cfgs=KitVisualizerCfg(rendering_mode=profile_name),
        )
        sim = SimulationContext(cfg)
        sim.reset()

        settings = get_settings_manager()
        for key, val in preset_dict.items():
            setting_name = "/" + key.replace(".", "/")
            expected = dlss_override if setting_name == "/rtx/post/dlss/execMode" else val
            assert settings.get(setting_name) == expected, (
                f"Mismatch for '{setting_name}' in mode '{mode_name}': "
                f"expected {expected!r}, got {settings.get(setting_name)!r}"
            )

    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_rendering_mode_field_overrides():
    """Explicit RenderingModeCfg kit_* fields should map to carb settings."""
    mode_cfg = RenderingModeCfg(
        kit_enable_translucency=True,
        kit_enable_reflections=True,
        kit_enable_global_illumination=True,
        kit_antialiasing_mode="DLAA",
        kit_enable_dlssg=True,
        kit_enable_dl_denoiser=True,
        kit_dlss_mode=0,
        kit_enable_direct_lighting=True,
        kit_samples_per_pixel=4,
        kit_enable_shadows=True,
        kit_enable_ambient_occlusion=True,
    )
    cfg = SimulationCfg(
        rendering_mode_cfgs={"custom": mode_cfg},
        visualizer_cfgs=KitVisualizerCfg(rendering_mode="custom"),
    )
    sim = SimulationContext(cfg)
    sim.reset()

    assert sim.get_setting("/rtx/translucency/enabled") is True
    assert sim.get_setting("/rtx/reflections/enabled") is True
    assert sim.get_setting("/rtx/indirectDiffuse/enabled") is True
    assert sim.get_setting("/rtx-transient/dlssg/enabled") is True
    assert sim.get_setting("/rtx-transient/dldenoiser/enabled") is True
    assert sim.get_setting("/rtx/post/dlss/execMode") == 0
    assert sim.get_setting("/rtx/directLighting/enabled") is True
    assert sim.get_setting("/rtx/directLighting/sampledLighting/samplesPerPixel") == 4
    assert sim.get_setting("/rtx/shadows/enabled") is True
    assert sim.get_setting("/rtx/ambientOcclusion/enabled") is True
