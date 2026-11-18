# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton camera output and calibration contracts without Isaac Sim."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer

from pxr import Usd

from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.utils.semantics import add_labels
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_variant_switch_renders_into_camera_outputs_without_moving_bodies(device):
    """Native selection changes color, preserves depth/pose, and works in a captured render graph."""
    variants = []
    for color in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        variant = newton.ModelBuilder()
        body = variant.add_body()
        variant.add_shape_sphere(body, radius=0.5, color=color, label="/World/sphere/geom")
        variants.append(variant)
    prototype = newton.ModelBuilder()
    body = prototype.add_body(label="/World/sphere", xform=wp.transform((0.0, 0.0, -2.0), wp.quat_identity()))
    prototype.add_body_variants(body, variants)
    builder = newton.ModelBuilder()
    builder.replicate(prototype, 2)
    model = builder.finalize(device=device)
    state = model.state()
    original_pose = state.body_q.numpy().copy()
    stage = Usd.Stage.CreateInMemory()
    for label in model.shape_label:
        prim = stage.DefinePrim(label, "Sphere")
        add_labels(prim, labels=["sphere"], instance_name="class")
    backend = SimpleNamespace(model=model, state_0=state)
    provider = SimpleNamespace(create_mapping=lambda labels: None)
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=None),
        device=device,
        requires_usd_stage=False,
        requires_newton_model=False,
        stage=stage,
        get_or_create_backend=lambda cfg: backend,
        get_scene_data_provider=lambda: provider,
        get_clone_plan=lambda: None,
    )
    renderer = NewtonWarpRenderer(
        NewtonWarpRendererCfg(colorize_semantic_segmentation=False, depth_clipping_behavior="max"), sim=sim
    )
    renderer.initialize()
    kinds = (
        "rgba", "rgb_hdr", "albedo", "normals", "distance_to_camera", "distance_to_image_plane", "depth",
        "semantic_segmentation", "instance_segmentation",
    )  # fmt: skip
    camera_cfg = CameraCfg(
        prim_path="/World/camera", width=9, height=9, data_types=list(kinds),
        spawn=PinholeCameraCfg(clipping_range=(0.01, 5.0)), background_color=(0.2, 0.3, 0.4),
    )  # fmt: skip
    data = renderer.create_render_data(CameraRenderSpec(camera_cfg, device, 2, (), 2))
    outputs = {}
    for kind in kinds:
        spec = renderer.supported_output_types()[kind]
        outputs[kind] = ProxyArray(wp.zeros((2, 9, 9, spec.channels), dtype=spec.dtype, device=device))
    renderer.set_outputs(data, outputs)
    positions = ProxyArray(wp.zeros(2, dtype=wp.vec3f, device=device))
    # World-camera +X forward/+Z up, oriented to look down global -Z with +Y up.
    orientations = ProxyArray(wp.array([(-0.5, 0.5, 0.5, 0.5)] * 2, dtype=wp.quatf, device=device))
    intrinsics = ProxyArray(
        wp.array([((9.0, 0.0, 4.5), (0.0, 9.0, 4.5), (0.0, 0.0, 1.0))] * 2, dtype=wp.mat33f, device=device)
    )
    renderer.update_camera(data, positions, orientations, intrinsics)
    pointers = {kind: output.warp.ptr for kind, output in outputs.items()}
    graph = None
    for choices in ([0, 1], [1, 1]):
        model.select_body_variants(wp.array(choices, dtype=wp.int32, device=device))
        model.bvh_refit_shapes(state)
        if graph is None:
            # Warm kernels and native scene synchronization before capture.
            renderer._launch_render(data)
            with wp.ScopedCapture(device=device) as capture:
                renderer._launch_render(data)
            graph = capture.graph
        wp.capture_launch(graph)
        for env, variant in enumerate(choices):
            rgba = outputs["rgba"].warp.numpy()[env]
            channel, other = (0, 2) if variant == 0 else (2, 0)
            assert rgba[4, 4, channel] > rgba[4, 4, other]
            np.testing.assert_array_equal(rgba[0, 0], [51, 76, 102, 255])
            for kind in ("rgb_hdr", "albedo"):
                center = outputs[kind].warp.numpy()[env, 4, 4]
                assert center[channel] > center[other]
            np.testing.assert_allclose(outputs["normals"].warp.numpy()[env, 4, 4], [0, 0, 1], atol=1e-5)
            for kind in ("distance_to_camera", "distance_to_image_plane", "depth"):
                np.testing.assert_allclose(outputs[kind].warp.numpy()[env, 4, 4, 0], 1.5, atol=1e-5)
                assert outputs[kind].warp.numpy()[env, 0, 0, 0] == 5.0
            segmentation = outputs["semantic_segmentation"].warp.numpy()[env, :, :, 0]
            assert segmentation[4, 4] >= 2
            assert segmentation[0, 0] == 0
            colorized = outputs["instance_segmentation"].warp.numpy()[env]
            assert colorized[4, 4, 3] == 255
            np.testing.assert_array_equal(colorized[0, 0], [0, 0, 0, 0])
        assert {kind: output.warp.ptr for kind, output in outputs.items()} == pointers
        np.testing.assert_array_equal(state.body_q.numpy(), original_pose)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_intrinsic_updates_preserve_shared_ray_storage(device):
    """Uniform updates reuse the ray field; nonuniform calibration fails without changing rays."""
    data = SimpleNamespace(
        _distortion=None,
        width=16,
        height=8,
        _intrinsic_status=wp.zeros(1, dtype=wp.int32, device=device),
        camera_rays=wp.empty((1, 8, 16, 2), dtype=wp.vec3f, device=device),
    )
    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    matrix = np.array([[20.0, 0, 8], [0, 20.0, 4], [0, 0, 1]], dtype=np.float32)
    matrices = wp.array([matrix, matrix], dtype=wp.mat33f, device=device)
    parameters = wp.empty((5, 2), dtype=wp.float32, device=device)
    pointer = data.camera_rays.ptr
    renderer.update_camera_intrinsics(data, matrices, parameters)
    before = data.camera_rays.numpy().copy()
    matrix[0, 0] *= 0.5
    matrices.assign(np.stack([matrix, matrix]))
    renderer.update_camera_intrinsics(data, matrices, parameters)
    after = data.camera_rays.numpy().copy()
    assert data.camera_rays.ptr == pointer
    assert not np.array_equal(before, after)
    different = np.stack([matrix, matrix])
    different[1, 0, 0] *= 0.5
    matrices.assign(different)
    with pytest.raises(ValueError, match="identical camera intrinsics"):
        renderer.update_camera_intrinsics(data, matrices, parameters)
    np.testing.assert_array_equal(data.camera_rays.numpy(), after)
