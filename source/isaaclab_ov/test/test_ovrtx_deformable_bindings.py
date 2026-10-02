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
pytestmark = pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {_MISSING_MODULES}")

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.stage import OvstageBackend
    from ovrtx import DataAccess


@pytest.fixture(autouse=True)
def _simulation(monkeypatch):
    monkeypatch.setattr(SimulationContext, "_instance", SimpleNamespace(get_scene_data_provider=lambda: None))


def _make_renderer_without_backend() -> tuple[OVRTXRenderer, MagicMock]:
    renderer = OVRTXRenderer(OVRTXRendererCfg())
    renderer.backend = SimpleNamespace(renderer=MagicMock())
    renderer.scene = renderer.backend
    renderer.scene.query = OvstageBackend.query.__get__(renderer.scene)
    renderer._device = "cpu"
    renderer._warp_device = SimpleNamespace(ordinal=0, stream=SimpleNamespace(cuda_stream=99))
    return renderer, renderer.backend.renderer


@pytest.mark.parametrize("mode", ["sync", "async", "ovstage"])
def test_geometry_bindings_follow_mixed_sdp_points_and_pointer_swaps(mode):
    """Mesh, particle and curve points share one binding and retain publication ordering."""
    renderer, native = _make_renderer_without_backend()
    renderer.cfg.async_rendering = mode == "async"
    use_ovstage = mode == "ovstage"
    renderer._use_ovstage = use_ovstage
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
    if use_ovstage:
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
    else:
        publication.points = {}
        renderer._setup_geometry_bindings()
        renderer.update_geometries()
        native.bind_array_attribute.assert_not_called()
        publication.points = points
        renderer._sdp = SceneDataProvider(publication)
        renderer._setup_geometry_bindings()
        assert native.bind_array_attribute.call_args.kwargs["prim_paths"] == list(points)
        np.testing.assert_array_equal(
            native.write_attribute.call_args_list[0].kwargs["tensor"], np.ones(len(points), dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            native.write_attribute.call_args_list[1].kwargs["tensor"], np.tile(np.eye(4), (len(points), 1, 1))
        )
        binding = renderer._geometry_points_binding
        write = binding.write_async if mode == "async" else binding.write

    write.side_effect = RuntimeError("native write failed")
    with pytest.raises(RuntimeError, match="native write failed"):
        renderer.update_geometries()
    write.side_effect = None
    write.reset_mock()
    renderer.update_geometries()
    renderer.update_geometries()
    assert write.call_count == 1
    data = write.call_args.kwargs["tensors"] if use_ovstage else write.call_args.args[0]
    for item, array in zip(data, points.values(), strict=True):
        if mode == "async":
            assert item.ptr != array.ptr
            np.testing.assert_array_equal(item.numpy(), array.numpy())
        else:
            assert (item.data if use_ovstage else item.ptr) == array.ptr
    kwargs = write.call_args.kwargs
    assert kwargs["cuda_stream"] == 42
    if use_ovstage:
        assert kwargs["ordinal"] == 7 and kwargs["is_array"]
    else:
        assert kwargs["data_access"] is DataAccess.ASYNC

    # The producer advances its timestamp during the request, not before the consumer calls it.
    publication.points = {path: wp.clone(array) for path, array in reversed(tuple(points.items()))}
    renderer.update_geometries()
    renderer.update_geometries()
    assert write.call_count == 2
    data = write.call_args.kwargs["tensors"] if use_ovstage else write.call_args.args[0]
    for item, path in zip(data, points, strict=True):
        array = publication.points[path]
        if mode == "async":
            np.testing.assert_array_equal(item.numpy(), array.numpy())
        else:
            assert (item.data if use_ovstage else item.ptr) == array.ptr
    if mode == "async":
        # Mutating live physics storage cannot change either queued snapshot.
        snapshots = [call.args[0] for call in write.call_args_list]
        for array in publication.points.values():
            array.fill_(wp.vec3f(-1))
        for snapshot in snapshots:
            for item, array in zip(snapshot, points.values(), strict=True):
                np.testing.assert_array_equal(item.numpy(), array.numpy())
        publication.geometry_timestamp += 1
        renderer.update_geometries()
        write.return_value.wait.assert_called()
        assert write.call_args.args[0][0].ptr == snapshots[0][0].ptr
        binding.write.assert_not_called()


@pytest.mark.parametrize("mode, scaled", [("sync", False), ("async", True), ("ovstage", True)])
def test_update_transforms_consumes_sdp_matrices_once_per_publication(monkeypatch, mode, scaled):
    """Borrow synchronous publications or convert directly into retained asynchronous write buffers."""
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    renderer, _ = _make_renderer_without_backend()
    renderer.cfg.async_rendering = mode == "async"
    use_ovstage = mode == "ovstage"
    paths = ["/World/Shared", "/World/envs/env_1/Object"]
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]], dtype=np.float32)
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
    backend = SimpleNamespace(transforms=transforms, transforms_timestamp=0, transform_count=2, transform_paths=paths)
    backend.get_transforms = lambda _format: transforms
    renderer._sdp = SceneDataProvider(backend)
    renderer._object_scales_by_path = {paths[0]: (2, 3, 4)} if scaled else {}
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=99))
    renderer._use_ovstage = use_ovstage
    renderer.scene.ordinal = 5
    writes = []

    if use_ovstage:
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
        renderer._setup_xform_bindings()
        assert renderer._object_xform_binding == paths
        writes.clear()
    else:
        renderer._setup_xform_bindings()
        assert renderer.backend.renderer.bind_attribute.call_args.kwargs["prim_paths"] == paths
        renderer._object_xform_binding.write = lambda matrices, **kwargs: writes.append((None, matrices, kwargs))
        operation = MagicMock()
        renderer._object_xform_binding.write_async = lambda matrices, **kwargs: (
            writes.append((None, matrices, kwargs)) or operation
        )

    renderer.update_transforms()
    renderer.update_transforms()
    assert len(writes) == 1
    matrices = writes[0][2]["tensors"] if use_ovstage else writes[0][1]
    expected = np.tile(np.eye(4), (2, 1, 1))
    if scaled:
        expected[0, :3, :3] = np.diag([2, 3, 4])
    expected[:, 3, :3] = poses[:, :3]
    np.testing.assert_array_equal(matrices.numpy(), expected)
    assert writes[0][2]["cuda_stream"] == 99
    if use_ovstage:
        assert writes[0][2]["ordinal"] == 5
    else:
        assert writes[0][2]["data_access"] is DataAccess.ASYNC

    if not scaled:
        other, native = _make_renderer_without_backend()
        other._sdp = renderer._sdp
        other._setup_xform_bindings()
        other.update_transforms()
        assert native.bind_attribute.return_value.write.call_args.args[0] is matrices

    poses[:, 0] += 10
    transforms.transforms.assign(poses)
    backend.transforms_timestamp += 1
    renderer.update_transforms()
    assert len(writes) == 2
    updated = writes[1][2]["tensors"] if use_ovstage else writes[1][1]
    assert (updated is matrices) == (mode != "async")
    expected[:, 3, :3] = poses[:, :3]
    np.testing.assert_array_equal(updated.numpy(), expected)
    if mode == "async":
        # The new capture has not overwritten the previous borrowed input.
        np.testing.assert_array_equal(matrices.numpy()[:, 3, 0], poses[:, 0] - 10)
        operation.wait.assert_not_called()
        backend.transforms_timestamp += 1
        renderer.update_transforms()
        operation.wait.assert_called_once_with()
        assert writes[2][1] is matrices
