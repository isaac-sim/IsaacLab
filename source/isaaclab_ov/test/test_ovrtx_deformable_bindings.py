# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX bindings consume backend-independent SDP point and transform publications."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
pytestmark = pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {_MISSING_MODULES}")

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from ovrtx import DataAccess


def _make_renderer_without_backend() -> tuple[OVRTXRenderer, MagicMock]:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer.backend = SimpleNamespace(renderer=MagicMock())
    renderer._device = "cpu"
    renderer._geometry_paths = []
    renderer._geometry_version = -1
    renderer._use_ovstage = False
    renderer._init_fields_legacy()
    return renderer, renderer.backend.renderer


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_geometry_bindings_borrow_mixed_sdp_points_and_follow_pointer_swaps(use_ovstage):
    """Mesh, particle and curve points share one binding and retain publication ordering."""
    renderer, native = _make_renderer_without_backend()
    assert not renderer.cfg.cloning_contexts
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
    publication = SimpleNamespace(points=points, geometry_version=0)
    last_points = points

    def read_points():
        nonlocal last_points
        if publication.points is not last_points:
            publication.geometry_version += 1
            last_points = publication.points
        return publication.points

    renderer._sdp = SimpleNamespace(backend=publication, get_geometry_points=read_points)
    if use_ovstage:
        renderer._init_fields_ovstage()
        renderer._current_ordinal = 7
        renderer.backend.paths = SimpleNamespace(create_path_list_from_strings=lambda paths: paths)
        renderer.backend.stage = MagicMock()
        renderer.backend.stage.query_from_path_list.side_effect = lambda paths: paths
        publication.points = {}
        renderer._setup_geometry_bindings_ovstage()
        renderer.update_geometries()
        renderer.backend.stage.write_attribute.assert_not_called()
        publication.points = points
        renderer._setup_geometry_bindings_ovstage()
        assert renderer._geometry_points_query == list(points)
        write = renderer.backend.stage.write_attribute
        assert [call.args[1] for call in write.call_args_list] == ["omni:resetXformStack", "omni:xform"]
        np.testing.assert_array_equal(write.call_args_list[0].kwargs["tensors"], np.ones(len(points), dtype=np.bool_))
        write.reset_mock()
    else:
        publication.points = {}
        renderer._setup_geometry_bindings_legacy()
        renderer.update_geometries()
        native.bind_array_attribute.assert_not_called()
        publication.points = points
        renderer._setup_geometry_bindings_legacy()
        assert native.bind_array_attribute.call_args.kwargs["prim_paths"] == list(points)
        np.testing.assert_array_equal(
            native.write_attribute.call_args_list[0].kwargs["tensor"], np.ones(len(points), dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            native.write_attribute.call_args_list[1].kwargs["tensor"], np.tile(np.eye(4), (len(points), 1, 1))
        )
        write = renderer._geometry_points_binding.write

    renderer.update_geometries()
    renderer.update_geometries()
    assert write.call_count == 1
    data = write.call_args.kwargs["tensors"] if use_ovstage else write.call_args.args[0]
    assert [item.data if use_ovstage else item.ptr for item in data] == [array.ptr for array in points.values()]
    kwargs = write.call_args.kwargs
    assert kwargs["cuda_stream"] == 42
    if use_ovstage:
        assert kwargs["ordinal"] == 7 and kwargs["is_array"]
    else:
        assert kwargs["data_access"] is DataAccess.ASYNC

    # The producer advances its version during the request, not before the consumer calls it.
    publication.points = {path: wp.clone(array) for path, array in reversed(tuple(points.items()))}
    renderer.update_geometries()
    renderer.update_geometries()
    assert write.call_count == 2
    data = write.call_args.kwargs["tensors"] if use_ovstage else write.call_args.args[0]
    assert [item.data if use_ovstage else item.ptr for item in data] == [
        publication.points[path].ptr for path in points
    ]


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_update_transforms_consumes_sdp_matrices_once_per_publication(monkeypatch, use_ovstage):
    """Both OVRTX paths bind published bodies and consume SDP's scaled, transposed matrices."""
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    renderer, _ = _make_renderer_without_backend()
    paths = ["/World/Shared", "/World/envs/env_1/Object"]
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]], dtype=np.float32)
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
    backend = SimpleNamespace(transforms=transforms, transforms_version=0, transform_count=2, transform_paths=paths)
    backend.get_transforms = lambda _format: transforms
    renderer._sdp = SceneDataProvider(backend)
    renderer._transform_version = -1
    renderer._object_scales_by_path = {paths[0]: (2, 3, 4)}
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=99))
    renderer._use_ovstage = use_ovstage
    renderer._current_ordinal = 5
    writes = []

    if use_ovstage:
        renderer.backend.paths = SimpleNamespace(create_path_list_from_strings=lambda actual: actual)
        renderer.backend.stage = SimpleNamespace(
            query_from_path_list=lambda actual: actual,
            write_attribute=lambda query, attribute, **kwargs: (
                writes.append((query, attribute, kwargs)) or SimpleNamespace(wait=lambda: None)
            ),
        )
        monkeypatch.setattr(ovrtx_renderer_module, "xform_tensor_from_warp", lambda matrices: matrices)
        renderer._setup_xform_bindings_ovstage()
        assert renderer._object_xform_query == paths
        writes.clear()
    else:
        renderer._setup_xform_bindings_legacy()
        assert renderer.backend.renderer.bind_attribute.call_args.kwargs["prim_paths"] == paths
        renderer._object_xform_binding.write = lambda matrices, **kwargs: writes.append((None, matrices, kwargs))

    renderer.update_transforms()
    renderer.update_transforms()
    assert len(writes) == 1
    matrices = writes[0][2]["tensors"] if use_ovstage else writes[0][1]
    expected = np.tile(np.eye(4), (2, 1, 1))
    expected[0, :3, :3] = np.diag([2, 3, 4])
    expected[:, 3, :3] = poses[:, :3]
    np.testing.assert_array_equal(matrices.numpy(), expected)
    assert writes[0][2]["cuda_stream"] == 99
    if use_ovstage:
        assert writes[0][2]["ordinal"] == 5
    else:
        assert writes[0][2]["data_access"] is DataAccess.ASYNC

    poses[:, 0] += 10
    transforms.transforms.assign(poses)
    backend.transforms_version += 1
    renderer.update_transforms()
    assert len(writes) == 2
    updated = writes[1][2]["tensors"] if use_ovstage else writes[1][1]
    assert updated is matrices
    expected[:, 3, :3] = poses[:, :3]
    np.testing.assert_array_equal(updated.numpy(), expected)
