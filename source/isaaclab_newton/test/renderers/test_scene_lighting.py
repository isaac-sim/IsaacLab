# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene-authored lighting reaches Newton camera pixels through cloning."""

import newton
import numpy as np
import pytest
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg
from isaaclab_newton.physics.newton_builder import import_builder_lights
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from pxr import Sdf, Usd, UsdGeom, UsdLux

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

pytestmark = [pytest.mark.kitless, pytest.mark.rendering]


@configclass
class LightingSceneCfg(InteractiveSceneCfg):
    box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/box",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        ),
    )
    dome = AssetBaseCfg(
        prim_path="/World/dome",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 0.0, 0.0), intensity=0.125, exposure=1.0),
    )
    sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(color=(0.0, 0.0, 1.0), intensity=0.25))
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        height=16,
        width=16,
        data_types=["rgb"],
        renderer_cfg=NewtonWarpRendererCfg(use_cuda_graph=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="opengl"),
    )


def _render_center_pixels(device, num_envs, rotation=(0.0, 0.0, 0.0, 1.0), visible=True, lighting="authored"):
    cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), num_substeps=1), device=device)
    with build_simulation_context(sim_cfg=cfg) as sim:
        scene_cfg = LightingSceneCfg(num_envs=num_envs, env_spacing=3.0)
        scene_cfg.sun.init_state.rot = rotation
        scene_cfg.sun.spawn.visible = scene_cfg.dome.spawn.visible = visible
        if lighting != "authored":
            scene_cfg.sun = None
            scene_cfg.dome = (
                scene_cfg.dome.replace(spawn=sim_utils.DomeLightCfg(texture_file="unsupported.exr"))
                if lighting == "unsupported"
                else None
            )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        camera = scene["camera"]
        camera.update(cfg.dt, force_recompute=True)
        return camera.data.output["rgb"].torch.cpu().numpy()[:, 8, 8, :].copy()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_authored_lighting_changes_pixels_without_scaling_with_world_count(device):
    """Exposure scales ambient radiance; rotating the sun removes blue; cloning preserves brightness."""
    single = _render_center_pixels(device, 1)
    cloned = _render_center_pixels(device, 4)
    away = _render_center_pixels(device, 4, (1.0, 0.0, 0.0, 0.0))
    dark = _render_center_pixels(device, 1, visible=False)
    native = _render_center_pixels(device, 1, lighting="native")
    unsupported = _render_center_pixels(device, 1, lighting="unsupported")
    np.testing.assert_array_equal(unsupported, native)
    assert native.any(), "Unsupported lights must retain native illumination."
    # White albedo under 0.25 red ambient + 0.25 blue direct light, converted to sRGB bytes.
    np.testing.assert_allclose(single, [[136, 0, 136]], atol=1)
    np.testing.assert_array_equal(cloned, np.repeat(single, 4, axis=0))
    np.testing.assert_allclose(away, np.tile([136, 0, 0], (4, 1)), atol=1)
    np.testing.assert_array_equal(dark, [[0, 0, 0]])


def test_light_colors_are_owned_by_scene_assets():
    """Renderer configuration must not introduce a second source for scene light colors."""
    for name in ("ambient_light_color", "default_light_color"):
        with pytest.raises(TypeError, match=name):
            NewtonWarpRendererCfg(**{name: (1.0, 1.0, 1.0)})


def test_unsupported_lights_are_reported_and_multiple_suns_are_rejected(caplog):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/lights")
    UsdLux.SphereLight.Define(stage, "/World/lights/area")
    dome = UsdLux.DomeLight.Define(stage, "/World/lights/textured")
    dome.CreateTextureFileAttr(Sdf.AssetPath("unused.exr"))
    UsdLux.DistantLight.Define(stage, "/World/env_0/local_light")
    builder = newton.ModelBuilder()
    import_builder_lights(builder, stage, global_paths=["/World/lights"], source_paths=["/World/env_0"])
    assert "SphereLight" in caplog.text
    assert "textured dome" in caplog.text
    assert "per-world light" in caplog.text

    UsdLux.DistantLight.Define(stage, "/World/lights/sun_a")
    UsdLux.DistantLight.Define(stage, "/World/lights/sun_b")
    with pytest.raises(ValueError, match="only one global distant light"):
        import_builder_lights(newton.ModelBuilder(), stage, global_paths=["/World/lights"], source_paths=[])
