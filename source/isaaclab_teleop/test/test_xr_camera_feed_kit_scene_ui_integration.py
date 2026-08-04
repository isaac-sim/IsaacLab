# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch real Scene UI extensions without starting an XR runtime."""

from isaaclab.app import AppLauncher

_SCENE_UI_KIT_ARGS = " ".join(
    (
        "--enable omni.kit.xr.core",
        "--enable omni.kit.scene_view.xr",
        "--enable omni.kit.scene_view.xr_utils",
    )
)
simulation_app = AppLauncher(
    headless=True,
    enable_cameras=True,
    device="cpu",
    kit_args=_SCENE_UI_KIT_ARGS,
).app

import pytest
from isaaclab_teleop.camera_feed import _PanelDescriptor
from isaaclab_teleop.camera_feed_kit_scene_ui import _KitSceneUiCameraFeedPresenter

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


def test_real_scene_ui_imports_and_constructs_world_panel():
    """The real Scene UI extensions provide the signatures used by PiP."""
    sim_utils.create_new_stage()
    presenter = _KitSceneUiCameraFeedPresenter()
    descriptor = _PanelDescriptor(
        label="Camera",
        width_m=0.48,
        offset_m=(0.0, 0.0),
        distance_m=0.8,
        placement="world",
        world_position_m=(0.0, 0.8, 1.6),
        world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    panel = presenter.create_panel(descriptor, width=720, height=450)
    try:
        assert panel._container is not None
        assert panel._component is not None
    finally:
        panel.close()

    assert panel._closed


def test_real_feed_source_reads_cuda_from_cpu_camera_render_product():
    """PiP owns a CUDA annotator while the Camera-owned RGBA buffer stays on CPU."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=1.0 / 60.0))
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            height=64,
            width=64,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(),
        )
    )
    source = None

    try:
        sim.reset()
        for _ in range(2):
            sim.step()
            camera.update(sim.cfg.dt)

        fallback = camera.data.output["rgba"].torch[0]
        assert fallback.device.type == "cpu"

        presenter = _KitSceneUiCameraFeedPresenter()
        source = presenter.create_image_source("camera", camera)
        assert source is not None

        for _ in range(2):
            sim.step()
            camera.update(sim.cfg.dt)
            simulation_app.update()
        image = source.get_image(tuple(fallback.shape))

        assert image.device.type == "cuda"
        assert tuple(image.shape) == tuple(fallback.shape)
        assert image.data_ptr() == source._annotator.get_data().ptr
        assert image.dtype == fallback.dtype
    finally:
        if source is not None:
            source.close()
        camera._invalidate_initialize_callback(None)
        rep.vp_manager.destroy_hydra_textures("Replicator")
        sim.stop()
        sim.clear_instance()
