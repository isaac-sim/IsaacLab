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

from isaaclab.rendering.rendering_mode.rendering_mode_presets import get_kit_rendering_preset
from isaaclab.rendering.visualizers import KitVisualizerCfg
from isaaclab.sim.simulation_cfg import RenderingModeCfg, SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext


# @pytest.mark.isaacsim_ci
def test_render_cfg_presets():
    """Test that quality presets are applied and can be overridden via RenderingModeCfg."""

    # user-friendly field override
    dlss_mode = ("/rtx/post/dlss/execMode", 5)

    rendering_modes = ["performance", "balanced", "quality"]

    for rendering_mode in rendering_modes:
        # Clear any existing simulation context before creating a new one
        SimulationContext.clear_instance()

        preset_dict = get_kit_rendering_preset(rendering_mode)

        profile_name = f"profile_{rendering_mode}"
        quality_cfg = RenderingModeCfg(
            kit_rendering_preset=rendering_mode,
            kit_dlss_mode=dlss_mode[1],
        )
        cfg = SimulationCfg(
            rendering_quality_cfgs={profile_name: quality_cfg},
            visualizer_cfgs=KitVisualizerCfg(rendering_quality=profile_name),
        )

        sim = SimulationContext(cfg)
        sim.reset()

        for setting_name, val in preset_dict.items():
            if setting_name == dlss_mode[0]:
                setting_gt = dlss_mode[1]
            else:
                setting_gt = val

            setting_val = sim.get_setting(setting_name)

            assert setting_gt == setting_val, (
                f"Mismatch for '{setting_name}' in mode '{rendering_mode}': "
                f"expected {setting_gt!r}, got {setting_val!r}"
            )

    # Clean up after the test
    SimulationContext.clear_instance()


@pytest.mark.skip(reason="Timeline not stopped")
# @pytest.mark.isaacsim_ci
def test_rendering_quality_cfg_field_overrides():
    """Test that explicit RenderingModeCfg fields map to carb settings."""
    quality_cfg = RenderingModeCfg(
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
        rendering_quality_cfgs={"custom": quality_cfg},
        visualizer_cfgs=KitVisualizerCfg(rendering_quality="custom"),
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
    assert sim.get_setting("/rtx/post/aa/op") == 4  # dlss = 3, dlaa=4
