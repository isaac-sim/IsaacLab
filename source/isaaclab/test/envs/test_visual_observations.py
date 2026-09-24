# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-observation ownership, caching, and lifecycle without a renderer runtime."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.envs.mdp import processed_image
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg, SceneEntityCfg
from isaaclab.renderers import RenderBufferSpec
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind
from isaaclab.utils.visual_processing import VisualProcessor, VisualProcessorCfg
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


class CameraSource:
    """Persistent raw frames with the camera's lazy-read contract."""

    def __init__(self):
        self.cfg = SimpleNamespace(isp_cfg=None, height=2, width=3)
        self.camera_prim_paths = ("/World/envs/env_0/Camera",)
        self.render_buffer_specs = {
            "rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb"),
            "rgba": RenderBufferSpec(4, wp.uint8, color_space="srgb"),
            "rgb_hdr": RenderBufferSpec(3, wp.float32, color_space="scene_linear"),
        }
        self.render_outputs = CameraData.allocate(
            data_types=["rgb", "rgb_hdr"],
            height=2,
            width=3,
            num_views=2,
            device="cpu",
            supported_specs={RenderBufferKind(name): spec for name, spec in self.render_buffer_specs.items()},
        ).output
        self.render_outputs["rgb"].torch.copy_(torch.arange(36, dtype=torch.uint8).reshape(2, 2, 3, 3))
        self.render_generation = 1
        self.frame = ProxyArray(wp.ones(2, dtype=wp.int64, device="cpu"))
        self.requests = []

    def request_render_inputs(self, data_types, *, neutral_exposure=False):
        self.requests.append((data_types, neutral_exposure))


def make_term_cfg(events, *, source="rgb", increment=1, **params):
    """Build processors with independently observable state and bindings."""
    source_spec = RenderBufferSpec(
        3,
        wp.float32 if source == "rgb_hdr" else wp.uint8,
        color_space="scene_linear" if source == "rgb_hdr" else "srgb",
    )
    bindings = []

    def factory(cfg, context):
        buffers = {}

        def initialize(inputs, outputs):
            buffers.update(input=inputs[source].torch, output=outputs["rgb"].torch)
            bindings.append(buffers)

        def process(mask):
            events.append(("process", mask.numpy().copy()))
            torch.add(buffers["input"], increment, out=buffers["output"])

        return VisualProcessor(
            inputs={source: source_spec},
            outputs={"rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb")},
            initialize=initialize,
            process=process,
            reset=lambda mask: events.append(("reset", mask.numpy().copy())),
            close=lambda: events.append(("close", None)),
            neutral_exposure=source == "rgb_hdr",
            in_place=True,
        )

    cfg = ObservationTermCfg(
        func=processed_image,
        params={"sensor_cfg": SceneEntityCfg("camera"), "processors": [VisualProcessorCfg(func=factory)], **params},
    )
    return cfg, bindings


def make_env(camera):
    return SimpleNamespace(
        num_envs=2, device="cpu", scene={"camera": camera}, sim=SimpleNamespace(stage=None, is_playing=lambda: True)
    )


def test_terms_share_raw_camera_without_sharing_processed_state():
    """In-place processing never writes shared renderer inputs, and cached reads do no work."""
    camera = CameraSource()
    env = make_env(camera)
    first_events, second_events = [], []
    first_cfg, first_bindings = make_term_cfg(first_events)
    second_cfg, second_bindings = make_term_cfg(second_events, increment=5)
    first = processed_image.prepare_scene(first_cfg, env)
    second = processed_image.prepare_scene(second_cfg, env)
    raw = camera.render_outputs["rgb"].torch.clone()
    first_output = first(env, **first_cfg.params)
    second_output = second(env, **second_cfg.params)
    assert first_output.data_ptr() != second_output.data_ptr()
    assert first_bindings[0]["input"].data_ptr() == second_bindings[0]["input"].data_ptr()
    assert first_output.data_ptr() != camera.render_outputs["rgba"].warp.ptr
    torch.testing.assert_close(first_output, raw + 1)
    torch.testing.assert_close(second_output, raw + 5)
    torch.testing.assert_close(camera.render_outputs["rgb"].torch, raw)
    assert first(env, **first_cfg.params) is first_output
    assert [kind for kind, _ in first_events] == ["process"]
    first.reset([1])
    np.testing.assert_array_equal(first_events[-1][1], [False, True])
    assert first(env, **first_cfg.params) is first_output
    assert [kind for kind, _ in first_events] == ["process", "reset"]
    camera.render_generation += 1
    camera.frame.torch[1] += 1
    assert first(env, **first_cfg.params) is first_output
    np.testing.assert_array_equal(first_events[-1][1], [False, True])
    assert [kind for kind, _ in second_events] == ["process"]
    first.close()
    first.close()
    second.close()
    assert [kind for kind, _ in first_events].count("close") == 1
    with pytest.raises(RuntimeError, match="closed"):
        first(env, **first_cfg.params)


def test_manager_adopts_prepared_image_and_snapshots_its_output():
    """Requirements resolve before playback; manager reset/close reach the same term."""
    camera = CameraSource()
    env = make_env(camera)
    events = []
    term_cfg, bindings = make_term_cfg(events)
    group = ObservationGroupCfg(concatenate_terms=False, enable_corruption=False)
    group.image = term_cfg
    cfg = {"policy": group}
    prepared = ObservationManager.prepare_scene(cfg, env)
    assert camera.requests == [(("rgb", "rgba"), False)]
    assert not bindings
    manager = ObservationManager(cfg, env, prepared_terms=prepared)
    first = manager.compute()["policy"]["image"]
    second = manager.compute()["policy"]["image"]
    assert first.data_ptr() != second.data_ptr()
    assert sum(kind == "process" for kind, _ in events) == 1
    first.zero_()
    assert torch.count_nonzero(second).item() == second.numel()
    manager.reset([1])
    np.testing.assert_array_equal(events[-1][1], [False, True])
    manager.close()
    assert sum(kind == "close" for kind, _ in events) == 1


def test_normalized_permuted_output_reuses_storage_and_matches_image_math():
    """Strided RGB input normalizes into a persistent float32 output and BCHW view."""
    camera = CameraSource()
    env = make_env(camera)
    cfg, _ = make_term_cfg([], normalize=True, permute=True)
    term = processed_image.prepare_scene(cfg, env)
    expected = (camera.render_outputs["rgb"].torch.float() + 1) / 255
    expected -= expected.mean(dim=(1, 2), keepdim=True)
    result = term(env, **cfg.params)
    torch.testing.assert_close(result, expected.permute(0, 3, 1, 2))
    camera.render_generation += 1
    assert term(env, **cfg.params) is result
    term.close()


def test_term_requests_private_hdr_and_neutral_exposure_before_binding():
    camera = CameraSource()
    env = make_env(camera)
    events = []
    cfg, bindings = make_term_cfg(events, source="rgb_hdr")
    term = processed_image.prepare_scene(cfg, env)
    assert camera.requests == [(("rgb_hdr",), True)]
    assert not bindings
    term.close()


def test_processed_image_rejects_legacy_isp_before_requesting_inputs():
    camera = CameraSource()
    camera.cfg.isp_cfg = object()
    cfg, _ = make_term_cfg([])
    with pytest.raises(ValueError, match="Move its value into PpispProcessorCfg"):
        processed_image.prepare_scene(cfg, make_env(camera))
    assert camera.requests == []


def test_term_rejects_recreated_camera_buffers_instead_of_reading_stale_storage():
    """Stopping/restarting the simulation cannot silently preserve invalid bindings."""
    camera = CameraSource()
    env = make_env(camera)
    cfg, _ = make_term_cfg([])
    term = processed_image.prepare_scene(cfg, env)
    term(env, **cfg.params)
    camera.render_outputs = CameraSource().render_outputs
    with pytest.raises(RuntimeError, match="Camera render buffers were recreated"):
        term(env, **cfg.params)
    term.close()


def test_term_mask_covers_all_views_changed_since_its_previous_read():
    """Selective group computation cannot lose updates from earlier camera batches."""
    camera = CameraSource()
    env = make_env(camera)
    events = []
    cfg, _ = make_term_cfg(events)
    term = processed_image.prepare_scene(cfg, env)
    term(env, **cfg.params)
    camera.frame.torch[0] += 1
    camera.render_generation += 1
    camera.frame.torch[1] += 1
    camera.render_generation += 1
    term(env, **cfg.params)
    np.testing.assert_array_equal(events[-1][1], [True, True])
    term.close()
