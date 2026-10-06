# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-observation ownership, caching, and lifecycle without a renderer runtime."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.envs.mdp import processed_image
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg, SceneEntityCfg
from isaaclab.renderers import RenderBufferSpec
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind
from isaaclab.sensors.post_processing import CameraPostProcessingChain, SensorPostProcessor, SensorPostProcessorCfg
from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


class CameraSource:
    """Persistent raw frames with the camera's lazy-read contract."""

    def __init__(self, device="cpu"):
        self.cfg = SimpleNamespace(height=2, width=3)
        self.camera_prim_paths = ("/World/envs/env_0/Camera",)
        self.render_buffer_specs = {
            "rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb"),
            "rgba": RenderBufferSpec(4, wp.uint8, color_space="srgb"),
            "rgb_radiance": RenderBufferSpec(3, wp.float32, color_space="scene_linear"),
        }
        self.render_outputs = CameraData.allocate(
            data_types=["rgb", "rgb_radiance"],
            height=2,
            width=3,
            num_views=2,
            device=device,
            supported_specs={RenderBufferKind(name): spec for name, spec in self.render_buffer_specs.items()},
        ).output
        self.render_outputs["rgb"].torch.copy_(torch.arange(36, dtype=torch.uint8).reshape(2, 2, 3, 3))
        self.render_generation = 1
        self.frame = ProxyArray(wp.ones(2, dtype=wp.int64, device=device))
        self.render_frame = self.frame
        self.requests = []

    def request_render_inputs(self, data_types):
        self.requests.append(data_types)


def make_term_cfg(events, *, source="rgb", increment=1, **params):
    """Build processors with independently observable state and bindings."""
    source_spec = RenderBufferSpec(
        3,
        wp.float32 if source == "rgb_radiance" else wp.uint8,
        color_space="scene_linear" if source == "rgb_radiance" else "srgb",
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

        return SensorPostProcessor(
            inputs={source: source_spec},
            outputs={"rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb")},
            initialize=initialize,
            process=process,
            reset=lambda mask: events.append(("reset", mask.numpy().copy())),
            close=lambda: events.append(("close", None)),
            in_place=True,
        )

    cfg = ObservationTermCfg(
        func=processed_image,
        params={"sensor_cfg": SceneEntityCfg("camera"), "processors": [SensorPostProcessorCfg(func=factory)], **params},
    )
    return cfg, bindings


def make_env(camera):
    return SimpleNamespace(
        num_envs=2,
        device=str(camera.frame.warp.device),
        scene={"camera": camera},
        sim=SimpleNamespace(stage=None, is_playing=lambda: True),
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
    term_cfg.func = "isaaclab.envs.mdp:processed_image"
    group = ObservationGroupCfg(concatenate_terms=False, enable_corruption=False)
    group.image = term_cfg
    cfg = {"policy": group}
    prepared = ObservationManager.prepare_scene(cfg, env)
    instance = prepared["policy/image"]
    assert camera.requests == [("rgb", "rgba")]
    assert not bindings
    manager = ObservationManager(cfg, env, prepared_terms=prepared)
    assert not prepared
    assert manager.cfg["policy"].image.func is instance
    first = manager.compute()["policy"]["image"]
    second = manager.compute()["policy"]["image"]
    assert first.data_ptr() != second.data_ptr()
    assert sum(kind == "process" for kind, _ in events) == 1
    first.zero_()
    assert torch.count_nonzero(second).item() == second.numel()
    manager.reset([1])
    np.testing.assert_array_equal(events[-1][1], [False, True])
    manager.close()
    manager.close()
    assert sum(kind == "close" for kind, _ in events) == 1


def _encode_image(env, image: ObservationTermCfg):
    """Wrapper observation that reads a nested image term."""
    return image.func(env, **image.params)


def test_manager_prepares_nested_image_terms_before_startup():
    """An image term nested in another term's params requests renderer inputs before startup."""
    camera = CameraSource()
    env = make_env(camera)
    events = []
    image_cfg, bindings = make_term_cfg(events)
    image_cfg.func = "isaaclab.envs.mdp:processed_image"
    group = ObservationGroupCfg(concatenate_terms=False, enable_corruption=False)
    group.encoded = ObservationTermCfg(func=_encode_image, params={"image": image_cfg})
    cfg = {"policy": group}
    prepared = ObservationManager.prepare_scene(cfg, env)
    instance = prepared["policy/encoded.params.image"]
    assert camera.requests == [("rgb", "rgba")]
    assert not bindings
    manager = ObservationManager(cfg, env, prepared_terms=prepared)
    assert manager.cfg["policy"].encoded.params["image"].func is instance
    manager.compute()
    assert sum(kind == "process" for kind, _ in events) == 1
    manager.reset([1])
    resets = [mask for kind, mask in events if kind == "reset"]
    assert len(resets) == 1
    np.testing.assert_array_equal(resets[0], [False, True])
    manager.close()
    assert sum(kind == "close" for kind, _ in events) == 1


@pytest.mark.parametrize("source", ["default", "shared"])
def test_manager_prepares_defaulted_and_shared_nested_image_terms(source):
    """Nested image terms supplied as defaults, or referenced twice, are prepared once before startup."""
    camera = CameraSource()
    env = make_env(camera)
    events = []
    image_cfg, _ = make_term_cfg(events)
    image_cfg.func = "isaaclab.envs.mdp:processed_image"
    if source == "default":

        def encode(env, image=image_cfg):
            return image.func(env, **image.params)

        term_cfg, expected = ObservationTermCfg(func=encode), "policy/encoded.params.image"
    else:

        def encode(env, images):
            return torch.cat([image.func(env, **image.params) for image in images], dim=-1)

        term_cfg = ObservationTermCfg(func=encode, params={"images": (image_cfg, image_cfg)})
        expected = "policy/encoded.params.images.0"
    group = ObservationGroupCfg(concatenate_terms=False, enable_corruption=False)
    group.encoded = term_cfg
    cfg = {"policy": group}
    prepared = ObservationManager.prepare_scene(cfg, env)
    assert list(prepared) == [expected]
    assert camera.requests == [("rgb", "rgba")]
    manager = ObservationManager(cfg, env, prepared_terms=prepared)
    manager.compute()
    assert sum(kind == "process" for kind, _ in events) == 1
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
    # Output selection is fixed during preparation; a different call-time value must not be ignored.
    with pytest.raises(ValueError, match="prepared"):
        term(env, **{**cfg.params, "permute": False})
    term.close()


def test_term_requests_private_radiance_before_binding():
    camera = CameraSource()
    env = make_env(camera)
    events = []
    cfg, bindings = make_term_cfg(events, source="rgb_radiance")
    term = processed_image.prepare_scene(cfg, env)
    assert camera.requests == [("rgb_radiance",)]
    assert not bindings
    term.close()


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
    # Delayed images carry their own frame numbers while the live camera keeps advancing.
    camera.render_frame = ProxyArray(wp.clone(camera.frame.warp))
    camera.frame.torch.add_(2)
    camera.render_frame.torch[0] += 1
    camera.render_generation += 1
    term(env, **cfg.params)
    np.testing.assert_array_equal(events[-1][1], [True, False])
    term.close()


@pytest.mark.parametrize("env_ids", [None, [1]])
@pytest.mark.parametrize("device", test_devices())
def test_reset_excludes_unconsumed_pre_reset_capture(env_ids, device):
    """A skipped observation preserves peer updates but cannot advance reset state from old pixels."""
    with contextlib.ExitStack() as stack:
        if device.startswith("cuda"):
            # Camera writes and reset bookkeeping originate on a non-default Torch stream;
            # the observation must order its Warp mask kernels with that same stream.
            stack.enter_context(torch.cuda.stream(torch.cuda.Stream(device=device)))
        _check_reset_excludes_unconsumed_pre_reset_capture(env_ids, device)


def _check_reset_excludes_unconsumed_pre_reset_capture(env_ids, device):
    camera = CameraSource(device)
    env = make_env(camera)
    events = []
    cfg, _ = make_term_cfg(events)
    term = processed_image.prepare_scene(cfg, env)
    term(env, **cfg.params)
    camera.frame.torch.fill_(2)
    camera.render_generation += 1
    term.reset(env_ids)
    term(env, **cfg.params)
    if env_ids is None:
        assert [kind for kind, _ in events] == ["process", "reset"]
    else:
        np.testing.assert_array_equal(events[-1][1], [True, False])

    # Episode-local frame numbers can repeat: this is a genuinely new post-reset capture.
    camera.frame.torch[slice(None) if env_ids is None else env_ids] = 1
    camera.render_generation += 1
    term(env, **cfg.params)
    np.testing.assert_array_equal(events[-1][1], [env_ids is None, True])
    term.close()


def test_chain_processes_camera_outputs_without_an_observation_manager():
    """Direct post-processing tracks captures and resets like the observation term."""
    camera = CameraSource()
    events = []
    cfg, _ = make_term_cfg(events, increment=2)
    chain = CameraPostProcessingChain(camera, cfg.params["processors"], ["rgb"], num_views=2, device="cpu", stage=None)
    assert len(camera.requests) == 1 and chain.outputs is None

    assert chain.update()
    torch.testing.assert_close(chain.outputs["rgb"].torch, camera.render_outputs["rgb"].torch + 2)
    assert not chain.update()

    chain.reset([0])
    camera.render_generation += 1
    assert chain.update()
    np.testing.assert_array_equal(events[-1][1], [True, False])

    chain.close()
    assert events[-1] == ("close", None)
    with pytest.raises(RuntimeError, match="closed"):
        chain.update()
