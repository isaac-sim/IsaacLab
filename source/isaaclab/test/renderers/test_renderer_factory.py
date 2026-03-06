# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Renderer factory."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from isaaclab.renderers import Renderer
from isaaclab.renderers.base_renderer import BaseRenderer

pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")
pytest.importorskip("isaaclab_ov")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg


def _make_mock_renderer_class(name: str):
    """Create a minimal concrete BaseRenderer subclass for testing."""

    class MockRenderer(BaseRenderer):
        def __init__(self, cfg=None):
            pass

        def prepare_stage(self, stage, num_envs):
            pass

        def create_render_data(self, sensor):
            return None

        def set_outputs(self, render_data, output_data):
            pass

        def update_transforms(self):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            pass

        def write_output(self, render_data, output_name, output_data):
            pass

        def cleanup(self, render_data):
            pass

    MockRenderer.__name__ = name
    return MockRenderer


def test_renderer_factory_backend_mapping():
    """Renderer._get_backend maps config renderer_type to correct backend."""
    assert Renderer._get_backend(IsaacRtxRendererCfg()) == "physx"
    assert Renderer._get_backend(NewtonWarpRendererCfg()) == "newton"
    assert Renderer._get_backend(OVRTXRendererCfg()) == "ov"

    # If someone decide to hack and specify renderer_type it should default to physx
    assert Renderer._get_backend(Mock(renderer_type="unknown")) == "physx"


@pytest.mark.parametrize(
    "cfg_cls,expected_class_name",
    [
        (IsaacRtxRendererCfg, "IsaacRtxRenderer"),
        (NewtonWarpRendererCfg, "NewtonWarpRenderer"),
        (OVRTXRendererCfg, "OVRTXRenderer"),
    ],
    ids=["IsaacRtxRendererCfg", "NewtonWarpRendererCfg", "OVRTXRendererCfg"],
)
def test_renderer_factory_returns_correct_backend(cfg_cls, expected_class_name):
    """Renderer(cfg) returns instance of correct class when registry is populated."""
    cfg = cfg_cls()
    backend = Renderer._get_backend(cfg)

    mock_cls = _make_mock_renderer_class(expected_class_name)
    original = Renderer._registry.get(backend)
    Renderer._registry[backend] = mock_cls

    try:
        renderer = Renderer(cfg)
        assert type(renderer).__name__ == expected_class_name
        assert isinstance(renderer, BaseRenderer)
    finally:
        if original is not None:
            Renderer._registry[backend] = original
        else:
            Renderer._registry.pop(backend, None)


@pytest.mark.parametrize(
    "cfg_cls,expected_class_name",
    [
        (IsaacRtxRendererCfg, "IsaacRtxRenderer"),
        (NewtonWarpRendererCfg, "NewtonWarpRenderer"),
        (OVRTXRendererCfg, "OVRTXRenderer"),
    ],
    ids=["IsaacRtxRendererCfg", "NewtonWarpRendererCfg", "OVRTXRendererCfg"],
)
def test_renderer_factory_instantiation_backends(cfg_cls, expected_class_name):
    """Renderer(cfg) with registry returns correct backend class.
    NewtonWarpRenderer needs SimulationContext and SensorTiledCamera mocked."""
    cfg = cfg_cls()
    if cfg_cls is NewtonWarpRendererCfg:
        mock_provider = MagicMock()
        mock_provider.get_newton_model.return_value = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.initialize_scene_data_provider.return_value = mock_provider
        with (
            patch("isaaclab_newton.renderers.newton_warp_renderer.SimulationContext") as mock_sc_cls,
            patch("isaaclab_newton.renderers.newton_warp_renderer.newton.sensors.SensorTiledCamera") as mock_stc,
        ):
            mock_sc_cls.instance.return_value = mock_ctx
            mock_stc.return_value = MagicMock()
            renderer = Renderer(cfg)
    else:
        renderer = Renderer(cfg)
    assert type(renderer).__name__ == expected_class_name
    assert isinstance(renderer, BaseRenderer)
