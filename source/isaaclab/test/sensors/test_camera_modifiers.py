# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera modifier chains: renderer input resolution and once-per-capture processing."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass, modifiers, replace

pytestmark = pytest.mark.unit


def _to_uint8(data: torch.Tensor) -> torch.Tensor:
    return (data * 255.0).to(torch.uint8)


@configclass
class _ToRgbCfg(modifiers.ModifierCfg):
    """Stand-in for an image modifier that declares the camera output it produces."""

    func: Callable[..., torch.Tensor] = _to_uint8
    output: str = "rgb"


def _camera_cfg(data_types: list[str], camera_modifiers: dict) -> CameraCfg:
    return CameraCfg(
        prim_path="/World/Camera",
        height=2,
        width=3,
        spawn=PinholeCameraCfg(),
        data_types=data_types,
        modifiers=camera_modifiers,
    )


def test_modifier_outputs_replace_renderer_outputs_and_inputs_are_requested():
    """Chain outputs are not rendered, chain inputs are, and an undeclared output keeps the input name."""
    scale = modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0})
    cfg = _camera_cfg(["rgb", "depth"], {"rgb_radiance": [_ToRgbCfg(), scale], "depth": [scale]})

    assert cfg.modifier_outputs() == {"rgb_radiance": "rgb", "depth": "depth"}
    assert cfg.render_data_types() == ["depth", "rgb_radiance"]


def test_modifier_chains_must_produce_distinct_outputs():
    cfg = _camera_cfg(["rgb"], {"rgb_radiance": [_ToRgbCfg()], "rgb_hdr": [_ToRgbCfg()]})

    with pytest.raises(ValueError, match="distinct outputs"):
        cfg.modifier_outputs()


class _UninitializedCamera(Camera):
    """Camera built without a simulation; it owns no callbacks or renderer resources to release."""

    def __del__(self):
        pass


def _camera_with_chain(calls: list) -> Camera:
    """A camera whose renderer published ``rgb_radiance`` and that converts it to ``rgb``."""

    def record(data: torch.Tensor) -> torch.Tensor:
        calls.append(data.clone())
        return data

    camera = _UninitializedCamera.__new__(_UninitializedCamera)
    camera._data = CameraData.allocate(
        data_types=["rgb_radiance"],
        height=2,
        width=3,
        num_views=2,
        device="cpu",
        supported_specs={RenderBufferKind.RGB_RADIANCE: RenderBufferSpec(3, wp.float32)},
    )
    camera._render_outputs = dict(camera._data.output)
    camera._render_outputs["rgb_radiance"].torch.fill_(0.5)
    chain = modifiers.ModifierChain([modifiers.ModifierCfg(func=record), _ToRgbCfg()], "cpu")
    camera._modifier_chains = {"rgb_radiance": ("rgb", chain)}
    camera._modifier_buffers = {}
    camera._modifier_capture = None
    return camera


def test_camera_publishes_the_chain_output_without_changing_the_rendered_input():
    calls = []
    camera = _camera_with_chain(calls)
    radiance = camera._render_outputs["rgb_radiance"]

    camera._apply_modifiers()
    rgb = camera._data.output["rgb"]
    radiance.torch.fill_(1.0)
    camera._apply_modifiers()

    assert camera._data.output["rgb"] is rgb
    assert rgb.torch.dtype == torch.uint8
    assert torch.equal(rgb.torch, torch.full((2, 2, 3, 3), 255, dtype=torch.uint8))
    assert torch.equal(radiance.torch, torch.ones(2, 2, 3, 3))


def test_camera_runs_chains_once_per_published_capture():
    """A delayed renderer that republishes its last capture does not rerun the chains."""
    calls = []
    camera = _camera_with_chain(calls)
    capture = {"frame": object()}
    camera._data.info["rgb_radiance"] = {"capture": capture}

    camera._apply_modifiers()
    camera._apply_modifiers()
    assert len(calls) == 1

    camera._data.info["rgb_radiance"] = {"capture": {"frame": object()}}
    camera._apply_modifiers()
    assert len(calls) == 2


def test_modifier_inputs_stay_private_and_outputs_receive_renderer_metadata(monkeypatch):
    """Only requested outputs and modifier results are published; renderers attach capture metadata to them."""
    cfg = _camera_cfg(["rgb"], {"rgb_radiance": [_ToRgbCfg()]})
    specs = {
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB_RADIANCE: RenderBufferSpec(3, wp.float32),
    }
    bound = {}
    monkeypatch.setattr(Camera, "_initialize_intrinsics", lambda self: None)
    monkeypatch.setattr(Camera, "_update_poses", lambda self: None)
    camera = _UninitializedCamera.__new__(_UninitializedCamera)
    camera.cfg = cfg
    camera._render_cfg = replace(cfg, data_types=cfg.render_data_types())
    camera._device = "cpu"
    camera._view = SimpleNamespace(count=2)
    camera._render_data = None
    camera._renderer = SimpleNamespace(
        supported_output_types=lambda: specs, set_outputs=lambda data, outputs: bound.update(outputs)
    )

    camera._create_buffers()

    assert set(bound) == {"rgb_radiance"}
    assert set(camera._data.output) == set()
    assert set(camera._data.info) == {"rgb"}

    capture = {"frame": object()}
    camera._data.info["rgb"] = {"capture": capture}
    bound["rgb_radiance"].torch.fill_(0.5)
    camera._apply_modifiers()

    assert set(camera._data.output) == {"rgb"}
    assert camera._data.info["rgb"]["capture"] is capture
