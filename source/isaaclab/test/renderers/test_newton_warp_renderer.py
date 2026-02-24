# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TestNewtonWarpRenderer."""

import pytest

# Skip entire module if ovrtx not available (ovrtx_renderer imports it)
# pytest.importorskip("ovrtx")
# Warp/GPU tests may need app context
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import isaaclab.sim as isaaclab_sim

sim_cfg = isaaclab_sim.SimulationCfg(device="cuda:0")
sim = isaaclab_sim.SimulationContext(sim_cfg)

"""Rest everything follows."""

import torch
import warp as wp

from isaaclab.renderers import NewtonWarpRenderer


@pytest.fixture
def device():
    """Use CUDA if available, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


class TestNewtonWarpRenderer:
    """Tests for TestNewtonWarpRenderer buffer creation."""

    def test_render_init(self, device: str):
        renderer = NewtonWarpRenderer()
        assert renderer.newton_sensor is not None

    def test_dummy_tiled_camera(self, device: str):
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

        scene_cfg = InteractiveSceneCfg(num_envs=9, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
            data_types=["rgb"],
            spawn=isaaclab_sim.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=400,
            height=300,
        )

        renderer = NewtonWarpRenderer()
        scene.sensors["tiled_camera"] = TiledCamera(tiled_camera_cfg, renderer)

        render_data = renderer.create_render_data(scene.sensors["tiled_camera"])
        assert render_data is not None

        render_data.render_context.world_count = 9

        assert render_data.render_context is not None
        assert render_data.width == 400
        assert render_data.height == 300
        assert render_data.num_cameras == 1
        assert render_data.camera_rays is None
        assert render_data.camera_transforms is None

        outputs = {
            "rgba": torch.zeros((9, 300, 400, 4), dtype=torch.uint8, device=device).contiguous(),
            "normals": torch.zeros((9, 300, 400, 3), dtype=torch.float32, device=device).transpose(1, 2),
        }

        renderer.set_outputs(render_data, outputs)

        assert render_data.outputs.color_image is not None
        assert render_data.outputs.normals_image is not None

        assert render_data.outputs.albedo_image is None
        assert render_data.outputs.depth_image is None
        assert render_data.outputs.instance_segmentation_image is None

        assert render_data.outputs.color_image.ptr == outputs["rgba"].data_ptr()
        assert render_data.outputs.normals_image.ptr != outputs["normals"].data_ptr()

        assert render_data.get_output("rgba").ptr == outputs["rgba"].data_ptr()
        assert render_data.get_output("rgba").shape == (9, 1, 300, 400)
        assert render_data.get_output("rgba").dtype == wp.uint32
        assert render_data.get_output("normals").shape == (9, 1, 300, 400)
        assert render_data.get_output("normals").dtype == wp.vec3f

        camera_positions = torch.tensor([(1.0, 2.0, 3.0)], device=device)
        camera_orientations = torch.tensor([(0.0, 0.0, 0.0, 0.5)], device=device)
        camera_intrinsics = torch.tensor([[[45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [45.0, 0.0, 0.0]]], device=device)
        renderer.update_camera(render_data, camera_positions, camera_orientations, camera_intrinsics)

        camera_transforms = render_data.camera_transforms.numpy()
        assert camera_transforms[0, 0, 0] == pytest.approx(1.0)
        assert camera_transforms[0, 0, 1] == pytest.approx(2.0)
        assert camera_transforms[0, 0, 2] == pytest.approx(3.0)
        assert camera_transforms[0, 0, 3] == pytest.approx(0.5)
        assert camera_transforms[0, 0, 4] == pytest.approx(-0.5)
        assert camera_transforms[0, 0, 5] == pytest.approx(-0.5)
        assert camera_transforms[0, 0, 6] == pytest.approx(0.5)
