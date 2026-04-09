# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering mode: offline resolution checks and Isaac Sim integration tests.

**Import order:** Do not import Isaac Sim / USD / ``pxr`` at module scope. Pytest loads this file
before the ``simulation_app`` fixture runs :class:`~isaaclab.app.AppLauncher`; loading ``pxr`` from
conda before Kit starts causes USD extension failures. Integration tests import sim APIs *inside*
the test body (after the fixture has launched SimulationApp).

Offline tests only use lightweight imports below.
"""

import logging

import pytest

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.rendering_mode import RenderingModeCfg
from isaaclab.rendering_mode.rendering_mode_utils import (
    apply_mode_profile_to_renderer_cfg,
    resolve_rendering_mode_name_for_renderer_cfg,
)


@pytest.fixture(scope="module")
def simulation_app():
    """Launch Kit once for Isaac Sim tests in this module (must run before any ``pxr`` import)."""
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=True, enable_cameras=True).app


# ---------------------------------------------------------------------------
# Offline: resolution / gating (no live Kit session)
# ---------------------------------------------------------------------------


class _FakeSettings:
    """Minimal carb settings view for CLI explicit + rendering_mode paths."""

    def __init__(self, explicit: bool, mode):
        self._data: dict[str, object] = {
            "/isaaclab/rendering/rendering_mode/explicit": explicit,
            "/isaaclab/rendering/rendering_mode": mode,
        }

    def get(self, key: str):
        return self._data.get(key)


def test_cli_explicit_rendering_mode_overrides_renderer_cfg():
    """CLI explicit flag should win over ``RendererCfg.rendering_mode`` (Kit RTX path)."""
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="quality")
    settings = _FakeSettings(explicit=True, mode="performance")
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "performance"


def test_cli_explicit_coerces_carb_subtree_to_profile_name():
    """Some Kit builds return a dict subtree from ``get()``; profile name must still resolve."""
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="performance")
    settings = _FakeSettings(explicit=True, mode={"outer": {"inner": "balanced"}})
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "balanced"


def test_apply_rtx_profile_skips_non_rtx_renderer_backend():
    """Only default / isaac_rtx / rtx renderers receive ``SimulationCfg.rendering_mode_cfgs`` RTX applies."""
    recorded: list[tuple[str, object]] = []

    def get_setting(key: str):
        return _FakeSettings(explicit=False, mode="").get(key)

    def set_setting(name: str, value: object) -> None:
        recorded.append((name, value))

    r_cfg = RendererCfg(renderer_type="newton_warp", rendering_mode="performance")
    apply_mode_profile_to_renderer_cfg(
        get_setting,
        set_setting,
        r_cfg,
        {"performance": RenderingModeCfg(rendering_mode_preset="performance")},
        logger=logging.getLogger(__name__),
    )
    assert recorded == []


# ---------------------------------------------------------------------------
# Isaac Sim: presets and overrides via visualizer and via RTX camera renderer
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
@pytest.mark.usefixtures("simulation_app")
def test_rendering_mode_presets_apply_via_visualizer():
    """Built-in presets + ``kit_*`` overrides should propagate to carb via Kit visualizer init."""
    from isaaclab_physx.visualizers import KitVisualizerCfg

    from isaaclab.app.settings_manager import get_settings_manager
    from isaaclab.rendering_mode import get_kit_rendering_preset
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext

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
            setting_name = key if key.startswith("/") else "/" + key.replace(".", "/")
            expected = dlss_override if setting_name == "/rtx/post/dlss/execMode" else val
            assert settings.get(setting_name) == expected, (
                f"Mismatch for '{setting_name}' in mode '{mode_name}': "
                f"expected {expected!r}, got {settings.get(setting_name)!r}"
            )

    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.usefixtures("simulation_app")
def test_rendering_mode_kit_field_overrides_via_visualizer():
    """Explicit ``RenderingModeCfg`` fields should map to carb when the Kit visualizer applies the profile."""
    from isaaclab_physx.visualizers import KitVisualizerCfg

    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext

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

    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.usefixtures("simulation_app")
def test_rendering_mode_kit_field_overrides_via_camera_renderer():
    """Same ``RenderingModeCfg`` profile should apply when a Kit RTX camera initializes (renderer path)."""
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext

    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    mode_cfg = RenderingModeCfg(
        kit_enable_translucency=True,
        kit_enable_reflections=True,
        kit_enable_global_illumination=True,
        kit_dlss_mode=0,
        kit_enable_direct_lighting=True,
        kit_samples_per_pixel=4,
        kit_enable_shadows=True,
        kit_enable_ambient_occlusion=True,
    )
    sim_cfg = SimulationCfg(
        dt=0.01,
        rendering_mode_cfgs={"cam_profile": mode_cfg},
    )
    sim = SimulationContext(sim_cfg)
    sim_utils.update_stage()

    camera_cfg = CameraCfg(
        height=64,
        width=64,
        prim_path="/World/Camera",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        renderer_cfg=IsaacRtxRendererCfg(rendering_mode="cam_profile"),
    )
    camera = Camera(camera_cfg)
    sim.reset()
    assert camera.is_initialized

    assert sim.get_setting("/rtx/translucency/enabled") is True
    assert sim.get_setting("/rtx/reflections/enabled") is True
    assert sim.get_setting("/rtx/indirectDiffuse/enabled") is True
    assert sim.get_setting("/rtx/post/dlss/execMode") == 0
    assert sim.get_setting("/rtx/directLighting/enabled") is True
    assert sim.get_setting("/rtx/directLighting/sampledLighting/samplesPerPixel") == 4
    assert sim.get_setting("/rtx/shadows/enabled") is True
    assert sim.get_setting("/rtx/ambientOcclusion/enabled") is True

    SimulationContext.clear_instance()
