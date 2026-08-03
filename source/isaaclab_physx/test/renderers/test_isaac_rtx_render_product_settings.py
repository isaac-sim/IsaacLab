# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RTX RenderProduct settings integration tests."""

from isaaclab.app import AppLauncher

# RenderProduct materialization and HydraTexture attachment require a camera-enabled Kit app.
simulation_app = AppLauncher(headless=True, enable_cameras=True, device="cpu").app

"""Rest everything follows."""

import pytest
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

import carb
import omni.replicator.core as rep
import usdrt.Usd as UsdRtUsd
from pxr import UsdUtils

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sensors.camera import Camera, CameraCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

_GLOBAL_RTX_SETTINGS = (
    "/rtx/post/dlss/execMode",
    "/rtx-transient/dldenoiser/enabled",
    "/rtx/dldenoiser/responsiveDenoising",
)


@pytest.fixture(autouse=True)
def restore_global_rtx_settings():
    """Restore every process-global RTX setting changed by these tests."""
    settings = carb.settings.get_settings()
    previous = {path: settings.get(path) for path in _GLOBAL_RTX_SETTINGS}
    try:
        yield
    finally:
        for path, value in previous.items():
            if value is None:
                settings.destroy_item(path)
            else:
                settings.set(path, value)


def _read_dlss_settings(prim) -> tuple[str, bool]:
    """Read the schema-backed settings from a USD or Fabric RenderProduct prim."""
    exec_mode = prim.GetAttribute("omni:rtx:post:dlss:execMode").Get()
    ray_reconstruction = prim.GetAttribute("omni:rtx:newDenoiser:enabled").Get()
    return str(exec_mode), bool(ray_reconstruction)


def test_camera_local_dlss_settings_survive_annotator_attachment():
    """Per-camera settings override conflicting globals in both USD and Fabric after attachment."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=1.0 / 60.0))
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            height=96,
            width=96,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(),
            renderer_cfg=IsaacRtxRendererCfg(
                global_settings=IsaacRtxRendererGlobalSettingsCfg(
                    dlss_mode=0,
                    enable_dl_denoiser=True,
                    carb_settings={"/rtx/dldenoiser/responsiveDenoising": True},
                ),
                enable_dlss_ray_reconstruction=False,
                dlss_exec_mode="quality",
            ),
        )
    )

    try:
        sim.reset()
        for _ in range(2):
            sim.step()
            camera.update(sim.cfg.dt)

        stage = sim_utils.get_current_stage()
        render_product_path = camera._render_data.render_product.path
        usd_prim = stage.GetPrimAtPath(render_product_path)

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        fabric_stage = UsdRtUsd.Stage.Attach(stage_id)
        fabric_prim = fabric_stage.GetPrimAtPath(render_product_path)

        expected = ("quality", False)
        assert _read_dlss_settings(usd_prim) == expected
        assert _read_dlss_settings(fabric_prim) == expected
        assert get_settings_manager().get("/rtx/dldenoiser/responsiveDenoising") is True
    finally:
        camera._invalidate_initialize_callback(None)
        rep.vp_manager.destroy_hydra_textures("Replicator")
        sim.stop()
        sim.clear_instance()


def test_cpu_simulation_camera_can_publish_cuda_pixels():
    """Keep camera pose state on CPU while Isaac RTX publishes its image on CUDA."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=1.0 / 60.0))
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            height=64,
            width=64,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(),
            renderer_cfg=IsaacRtxRendererCfg(camera_output_device="cuda:0"),
        )
    )

    try:
        sim.reset()
        for _ in range(2):
            sim.step()
            camera.update(sim.cfg.dt)

        assert camera.data.pos_w.torch.device.type == "cpu"
        assert camera.data.output["rgb"].torch.device.type == "cuda"
        assert camera.data.output["rgb"].torch.device.index == 0
    finally:
        camera._invalidate_initialize_callback(None)
        rep.vp_manager.destroy_hydra_textures("Replicator")
        sim.stop()
        sim.clear_instance()
