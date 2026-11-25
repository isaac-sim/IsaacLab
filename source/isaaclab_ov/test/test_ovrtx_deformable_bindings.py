# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX bindings consume backend-independent SDP point and transform publications."""

from __future__ import annotations

import contextlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sim import SimulationContext

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
pytestmark = [
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {_MISSING_MODULES}"),
    pytest.mark.ov,
]

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.stage import OvstageBackend

    from pxr import Gf, Usd, UsdGeom


@pytest.fixture(autouse=True)
def _simulation(monkeypatch):
    monkeypatch.setattr(
        SimulationContext, "_instance", SimpleNamespace(device="cpu", get_scene_data_provider=lambda: None)
    )


def _make_renderer_without_backend() -> tuple[OVRTXRenderer, MagicMock]:
    renderer = OVRTXRenderer(OVRTXRendererCfg(), sim=SimulationContext.instance())
    renderer.backend = SimpleNamespace(renderer=MagicMock())
    renderer.scene = renderer.backend
    renderer.scene.query = OvstageBackend.query.__get__(renderer.scene)
    renderer._device = "cpu"
    renderer._warp_device = SimpleNamespace(ordinal=0, stream=SimpleNamespace(cuda_stream=99))
    return renderer, renderer.backend.renderer


def test_geometry_bindings_follow_mixed_sdp_points_and_pointer_swaps():
    """Mesh, particle and curve points share one binding and retain publication ordering."""
    renderer, native = _make_renderer_without_backend()
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=42))
    points = {
        path: wp.array(np.arange(count * 3, dtype=np.float32).reshape(count, 3), dtype=wp.vec3f, device="cpu")
        for path, count in (
            ("/Cells/cell_3/Cloth/mesh", 3),
            ("/Cells/cell_7/Volume/visual", 5),
            ("/Shared/Particles", 4),
            ("/Cells/cell_3/Cable/curve", 2),
        )
    }
    publication = SimpleNamespace(points=points, geometry_timestamp=0)
    last_points = points

    def read_points(_format):
        nonlocal last_points
        if publication.points is not last_points:
            publication.geometry_timestamp += 1
            last_points = publication.points
        batches = []
        for path in points if publication.points else ():
            array = publication.points[path]
            source = SceneDataFormat.Points()
            source.points = array
            batches.append((source, {path: (0, len(array))}))
        return batches

    publication.get_geometry_batches = read_points
    renderer._sdp = SceneDataProvider(publication)
    renderer.scene.ordinal = 7
    renderer.scene.paths = SimpleNamespace(
        create_path_list_from_strings=lambda paths: paths, destroy_path_list=lambda paths: None
    )
    renderer.scene.stage = MagicMock()
    renderer.scene.stage.query_from_path_list.side_effect = contextlib.nullcontext
    publication.points = {}
    renderer._setup_geometry_bindings()
    renderer.update_geometries()
    renderer.scene.stage.write_attribute.assert_not_called()
    publication.points = points
    renderer._sdp = SceneDataProvider(publication)
    renderer._setup_geometry_bindings()
    assert renderer._geometry_points_binding == list(points)
    write = renderer.scene.stage.write_attribute
    assert [call.args[1] for call in write.call_args_list] == ["omni:resetXformStack", "omni:xform"]
    np.testing.assert_array_equal(write.call_args_list[0].kwargs["tensors"], np.ones(len(points), dtype=np.bool_))
    write.reset_mock()

    write.side_effect = RuntimeError("native write failed")
    with pytest.raises(RuntimeError, match="native write failed"):
        renderer.update_geometries()
    write.side_effect = None
    write.reset_mock()
    renderer.update_geometries()
    renderer.update_geometries()
    assert write.call_count == 1
    data = write.call_args.kwargs["tensors"]
    for item, array in zip(data, points.values(), strict=True):
        assert (item.data) == array.ptr
    kwargs = write.call_args.kwargs
    assert kwargs["cuda_stream"] == 42
    assert kwargs["ordinal"] == 7 and kwargs["is_array"]

    # The producer advances its timestamp during the request, not before the consumer calls it.
    publication.points = {path: wp.clone(array) for path, array in reversed(tuple(points.items()))}
    renderer.update_geometries()
    renderer.update_geometries()
    assert write.call_count == 2
    data = write.call_args.kwargs["tensors"]
    for item, path in zip(data, points, strict=True):
        array = publication.points[path]
        assert (item.data) == array.ptr


@pytest.mark.parametrize("scaled", [False, True])
def test_update_transforms_consumes_sdp_matrices_once_per_publication(monkeypatch, scaled):
    """Reuse SDP publications or retained scaled transform buffers."""
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    renderer, _ = _make_renderer_without_backend()
    paths = ["/World/Shared", "/World/envs/env_1/Object"]
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]], dtype=np.float32)
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
    backend = SimpleNamespace(transforms=transforms, transforms_timestamp=0, transform_count=2, transform_paths=paths)
    backend.get_transforms = lambda _format: transforms
    renderer._sdp = SceneDataProvider(backend)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, paths[0]).AddScaleOp().Set(Gf.Vec3d(2, 3, 4) if scaled else Gf.Vec3d(1))
    renderer.scene.clone_copies = [(paths[0], [])]
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=0))
    renderer.scene.ordinal = 5
    writes = []

    renderer.scene.paths = SimpleNamespace(
        create_path_list_from_strings=lambda paths: paths, destroy_path_list=lambda paths: None
    )
    renderer.scene.stage = SimpleNamespace(
        query_from_path_list=contextlib.nullcontext,
        write_attribute=lambda query, attribute, **kwargs: (
            writes.append((query, attribute, kwargs)) or SimpleNamespace(wait=lambda: None)
        ),
    )
    monkeypatch.setattr(ovrtx_renderer_module, "xform_tensor_from_warp", lambda matrices: matrices)
    renderer._setup_xform_bindings(stage)
    assert renderer._object_xform_binding == paths
    writes.clear()

    renderer.update_transforms()
    renderer.update_transforms()
    assert len(writes) == 1
    matrices = writes[0][2]["tensors"]
    expected = np.tile(np.eye(4), (2, 1, 1))
    if scaled:
        expected[0, :3, :3] = np.diag([2, 3, 4])
    expected[:, 3, :3] = poses[:, :3]
    np.testing.assert_array_equal(matrices.numpy(), expected)
    assert writes[0][2]["cuda_stream"] == 1  # Torch's default stream, not "no sync"
    assert writes[0][2]["ordinal"] == 5

    if not scaled:
        other, native = _make_renderer_without_backend()
        other._sdp = renderer._sdp
        other.scene = renderer.scene
        other.scene.stage.write_attribute = lambda query, attribute, **kwargs: SimpleNamespace(wait=lambda: None)
        other._setup_xform_bindings(stage)
        other.update_transforms()
        assert other._transform_buffer.matrices is matrices

    renderer.scene.stage.write_attribute = lambda query, attribute, **kwargs: (
        writes.append((query, attribute, kwargs)) or SimpleNamespace(wait=lambda: None)
    )
    poses[:, 0] += 10
    transforms.transforms.assign(poses)
    backend.transforms_timestamp += 1
    renderer.update_transforms()
    assert len(writes) == 2
    updated = writes[1][2]["tensors"]
    assert updated is matrices
    expected[:, 3, :3] = poses[:, :3]
    np.testing.assert_array_equal(updated.numpy(), expected)
