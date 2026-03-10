# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton Warp renderer."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import warp as wp

newton = pytest.importorskip("newton")

from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer, RenderData
from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

MODULE = "isaaclab_newton.renderers.newton_warp_renderer"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_ENVS = 2
WIDTH = 4
HEIGHT = 4


class _MockSensor:
    """Minimal sensor mock that supports weak references."""

    def __init__(self, width=WIDTH, height=HEIGHT):
        self.cfg = Mock(width=width, height=height)


def _make_render_context(world_count=NUM_ENVS):
    ctx = MagicMock()
    ctx.world_count = world_count
    return ctx


def _make_render_data(world_count=NUM_ENVS, width=WIDTH, height=HEIGHT):
    return RenderData(_make_render_context(world_count), _MockSensor(width, height))


# ---------------------------------------------------------------------------
# RenderData.__init__
# ---------------------------------------------------------------------------


class TestRenderDataInit:
    def test_weakref_clears_when_sensor_deleted(self):
        sensor = _MockSensor()
        rd = RenderData(_make_render_context(), sensor)
        del sensor
        assert rd.sensor() is None

    def test_default_dimensions_without_config(self):
        ctx = _make_render_context()
        sensor = MagicMock(spec=[])
        sensor.cfg = MagicMock(spec=[])
        rd = RenderData(ctx, sensor)
        assert rd.width == 100
        assert rd.height == 100


# ---------------------------------------------------------------------------
# RenderData.get_output
# ---------------------------------------------------------------------------


class TestGetOutput:
    @pytest.mark.parametrize(
        "output_name,attr_name",
        [
            (RenderData.OutputNames.RGBA, "color_image"),
            (RenderData.OutputNames.ALBEDO, "albedo_image"),
            (RenderData.OutputNames.DEPTH, "depth_image"),
            (RenderData.OutputNames.NORMALS, "normals_image"),
            (RenderData.OutputNames.INSTANCE_SEGMENTATION, "instance_segmentation_image"),
        ],
    )
    def test_returns_correct_attribute(self, output_name, attr_name):
        rd = _make_render_data()
        sentinel = object()
        setattr(rd.outputs, attr_name, sentinel)
        assert rd.get_output(output_name) is sentinel

    def test_unknown_name_returns_none(self):
        assert _make_render_data().get_output("nonexistent") is None

    def test_rgb_returns_none(self):
        rd = _make_render_data()
        rd.outputs.color_image = object()
        assert rd.get_output(RenderData.OutputNames.RGB) is None


# ---------------------------------------------------------------------------
# RenderData.set_outputs  (mock _from_torch to isolate routing logic)
# ---------------------------------------------------------------------------


class TestSetOutputs:
    @pytest.fixture()
    def rd(self):
        rd = _make_render_data()
        rd._from_torch = Mock(side_effect=lambda t, dtype: (t, dtype))
        return rd

    @pytest.mark.parametrize(
        "output_name,attr_name,expected_dtype",
        [
            (RenderData.OutputNames.RGBA, "color_image", wp.uint32),
            (RenderData.OutputNames.ALBEDO, "albedo_image", wp.uint32),
            (RenderData.OutputNames.DEPTH, "depth_image", wp.float32),
            (RenderData.OutputNames.NORMALS, "normals_image", wp.vec3f),
            (RenderData.OutputNames.INSTANCE_SEGMENTATION, "instance_segmentation_image", wp.uint32),
        ],
    )
    def test_routes_to_correct_attribute_and_dtype(self, rd, output_name, attr_name, expected_dtype):
        tensor = torch.zeros(4)
        rd.set_outputs({output_name: tensor})
        assert getattr(rd.outputs, attr_name) == (tensor, expected_dtype)

    def test_rgb_is_ignored(self, rd):
        rd.set_outputs({RenderData.OutputNames.RGB: torch.zeros(4)})
        rd._from_torch.assert_not_called()
        assert rd.outputs.color_image is None

    def test_unknown_type_logs_warning(self, caplog):
        rd = _make_render_data()
        rd._from_torch = Mock(return_value=object())
        with caplog.at_level(logging.WARNING):
            rd.set_outputs({"mystery_type": torch.zeros(4)})
        assert "not yet supported" in caplog.text

    def test_multiple_outputs_in_single_call(self, rd):
        rd.set_outputs(
            {
                RenderData.OutputNames.RGBA: torch.zeros(4),
                RenderData.OutputNames.DEPTH: torch.ones(4),
            }
        )
        assert rd.outputs.color_image is not None
        assert rd.outputs.depth_image is not None


# ---------------------------------------------------------------------------
# RenderData._from_torch  (real torch + warp)
# ---------------------------------------------------------------------------


class TestFromTorch:
    @pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "cuda"])
    def device(self, request):
        if request.param == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return request.param

    def test_contiguous_returns_correct_shape(self, device):
        rd = _make_render_data(world_count=1, width=4, height=4)
        tensor = torch.zeros(1 * 1 * 4 * 4, dtype=torch.float32, device=device)
        result = rd._from_torch(tensor, dtype=wp.float32)
        assert result.shape == (1, 1, 4, 4)
        assert result.dtype == wp.float32

    def test_contiguous_aliases_memory(self, device):
        """Test if zero-copy view is created when tensor is contiguous."""
        rd = _make_render_data(world_count=1, width=2, height=2)
        tensor = torch.arange(4, dtype=torch.float32, device=device)
        result = rd._from_torch(tensor, dtype=wp.float32)
        result_torch = wp.to_torch(result).flatten()
        assert torch.equal(result_torch, tensor)

    def test_non_contiguous_returns_zeros_and_warns(self, device, caplog):
        rd = _make_render_data(world_count=1, width=2, height=2)
        base = torch.zeros(8, dtype=torch.float32, device=device)
        non_contiguous = base[::2]
        assert not non_contiguous.is_contiguous()

        with caplog.at_level(logging.WARNING):
            result = rd._from_torch(non_contiguous, dtype=wp.float32)
        assert result.shape == (1, 1, 2, 2)
        assert "non-contiguous" in caplog.text

    def test_uint32_dtype(self, device):
        rd = _make_render_data(world_count=1, width=2, height=2)
        tensor = torch.zeros(4, dtype=torch.int32, device=device)
        result = rd._from_torch(tensor, dtype=wp.uint32)
        assert result.dtype == wp.uint32
        assert result.shape == (1, 1, 2, 2)

    def test_multi_env_shape(self, device):
        rd = _make_render_data(world_count=3, width=2, height=2)
        tensor = torch.zeros(3 * 1 * 2 * 2, dtype=torch.float32, device=device)
        result = rd._from_torch(tensor, dtype=wp.float32)
        assert result.shape == (3, 1, 2, 2)


# ---------------------------------------------------------------------------
# RenderData.update  (real warp kernel launch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp kernel launch requires CUDA")
class TestUpdate:
    def test_allocates_camera_transforms(self):
        rd = _make_render_data(world_count=2)
        positions = torch.zeros(2, 3, dtype=torch.float32, device=DEVICE)
        orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32, device=DEVICE)
        intrinsics = torch.eye(3, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(2, -1, -1).clone()
        intrinsics[:, 1, 1] = 100.0

        rd.update(positions, orientations, intrinsics)
        assert rd.camera_transforms is not None
        assert rd.camera_transforms.shape == (1, 2)

    def test_calls_compute_pinhole_camera_rays(self):
        rd = _make_render_data(world_count=2, width=8, height=6)
        positions = torch.zeros(2, 3, dtype=torch.float32, device=DEVICE)
        orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32, device=DEVICE)
        intrinsics = torch.eye(3, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(2, -1, -1).clone()
        intrinsics[:, 1, 1] = 100.0

        rd.update(positions, orientations, intrinsics)
        rd.render_context.utils.compute_pinhole_camera_rays.assert_called_once()
        args = rd.render_context.utils.compute_pinhole_camera_rays.call_args
        assert args[0][0] == 8
        assert args[0][1] == 6

    def test_stores_camera_rays_from_render_context(self):
        rd = _make_render_data(world_count=2)
        expected_rays = wp.zeros((1, 2, HEIGHT, WIDTH), dtype=wp.vec3f, device=DEVICE)
        rd.render_context.utils.compute_pinhole_camera_rays.return_value = expected_rays

        positions = torch.zeros(2, 3, dtype=torch.float32, device=DEVICE)
        orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32, device=DEVICE)
        intrinsics = torch.eye(3, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(2, -1, -1).clone()
        intrinsics[:, 1, 1] = 100.0

        rd.update(positions, orientations, intrinsics)
        assert rd.camera_rays is expected_rays


# ---------------------------------------------------------------------------
# NewtonWarpRenderer (mocked SimulationContext + newton sensor)
# ---------------------------------------------------------------------------


@pytest.fixture()
def renderer():
    """Construct a NewtonWarpRenderer with mocked external dependencies."""
    mock_sim_ctx = MagicMock()
    with (
        patch(f"{MODULE}.SimulationContext") as mock_sim_cls,
        patch("newton.sensors.SensorTiledCamera"),
    ):
        mock_sim_cls.instance.return_value = mock_sim_ctx
        r = NewtonWarpRenderer(NewtonWarpRendererCfg())
        r._mock_sim_cls = mock_sim_cls
        r._mock_sim_ctx = mock_sim_ctx
        yield r


class TestNewtonWarpRendererInit:
    def test_creates_sensor_from_newton_model(self, renderer):
        assert renderer.newton_sensor is not None

    def test_queries_scene_data_provider(self, renderer):
        renderer._mock_sim_ctx.initialize_scene_data_provider.assert_called_once()


class TestNewtonWarpRendererCreateRenderData:
    def test_render_data_does_not_prevent_sensor_gc(self, renderer):
        sensor = _MockSensor()
        renderer.newton_sensor.render_context = _make_render_context()
        rd = renderer.create_render_data(sensor)
        del sensor
        assert rd.sensor() is None  # avoid cyclic reference
        renderer.cleanup(rd)


class TestNewtonWarpRendererRender:
    def test_passes_all_output_fields_to_newton_sensor(self, renderer):
        rd = _make_render_data()
        rd.outputs.color_image = "color"
        rd.outputs.albedo_image = "albedo"
        rd.outputs.depth_image = "depth"
        rd.outputs.normals_image = "normals"
        rd.outputs.instance_segmentation_image = "seg"
        rd.camera_transforms = "transforms"
        rd.camera_rays = "rays"

        with patch(f"{MODULE}.SimulationContext") as mock_sim_cls:
            mock_sim_cls.instance.return_value = renderer._mock_sim_ctx
            renderer.render(rd)

        renderer.newton_sensor.update.assert_called_once_with(
            renderer._mock_sim_ctx.initialize_scene_data_provider().get_newton_state(),
            "transforms",
            "rays",
            color_image="color",
            albedo_image="albedo",
            depth_image="depth",
            normal_image="normals",
            shape_index_image="seg",
        )


class TestNewtonWarpRendererWriteOutput:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp arrays require CUDA")
    def test_copies_when_pointers_differ(self, renderer):
        src = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        rd = _make_render_data()
        rd.outputs.depth_image = src

        dst = torch.zeros(4, dtype=torch.float32, device=DEVICE)
        renderer.write_output(rd, RenderData.OutputNames.DEPTH, dst)
        assert torch.equal(wp.to_torch(src), dst)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp arrays require CUDA")
    def test_skips_copy_when_pointers_match(self, renderer):
        dst = torch.zeros(4, dtype=torch.float32, device=DEVICE)
        src = wp.from_torch(dst)
        rd = _make_render_data()
        rd.outputs.depth_image = src

        with patch(f"{MODULE}.wp.copy") as mock_copy:
            renderer.write_output(rd, RenderData.OutputNames.DEPTH, dst)
            mock_copy.assert_not_called()

    def test_noop_when_output_not_set(self, renderer):
        rd = _make_render_data()
        dst = torch.zeros(4)
        renderer.write_output(rd, RenderData.OutputNames.DEPTH, dst)


class TestNewtonWarpRendererCleanup:
    def test_clears_sensor_reference(self, renderer):
        rd = _make_render_data()
        rd.sensor = "not_none"
        renderer.cleanup(rd)
        assert rd.sensor is None

    def test_noop_when_render_data_is_none(self, renderer):
        renderer.cleanup(None)


class TestNewtonWarpRendererUpdateTransforms:
    def test_calls_update_scene_data_provider(self, renderer):
        with patch(f"{MODULE}.SimulationContext") as mock_sim_cls:
            mock_sim_cls.instance.return_value = renderer._mock_sim_ctx
            renderer.update_transforms()
        renderer._mock_sim_ctx.update_scene_data_provider.assert_called_with(True)
