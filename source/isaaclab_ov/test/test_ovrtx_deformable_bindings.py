# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX deformable mesh point bindings."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module  # noqa: E402

    # ovstage is an unconditional dependency of isaaclab_ov, so it is importable here.
    import ovstage  # noqa: E402
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_newton.physics.visualization_deformables import (  # noqa: E402
        ShadowDeformableEntity,
        ShadowDeformableRegistryGroup,
    )
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402
    from ovrtx import BindingFlag, DataAccess  # noqa: E402

    from isaaclab.cloner import ClonePlan
else:
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    BindingFlag = None
    DataAccess = None


class _FakePointsBinding:
    """Capture array writes made through an OVRTX array attribute binding."""

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name
        self.written = None
        self.write_kwargs: dict | None = None
        self.unbound = False

    def write(self, data, **kwargs):
        self.written = data
        self.write_kwargs = kwargs

    def map(self, device=None, device_id=0):  # noqa: ARG002
        raise RuntimeError("bind_array_attribute bindings do not expose mapped point buffers")

    def unbind(self):
        self.unbound = True


class _FakeOVRTXBackend:
    """Minimal OVRTX backend stub for deformable binding setup."""

    def __init__(self):
        self.bindings = {}
        self.calls = []
        self.writes = []

    def bind_array_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakePointsBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def bind_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakePointsBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def query_prims(self, **kwargs):  # noqa: ARG002
        return {
            "/World/envs/env_0/Deformable/mesh": {},
            "/World/envs/env_0/Deformable/geometry/mesh": {},
        }

    def write_attribute(self, **kwargs):
        self.writes.append(kwargs)


def _make_renderer_without_backend(device: str = "cpu") -> tuple[OVRTXRenderer, _FakeOVRTXBackend]:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer.backend = SimpleNamespace()
    renderer._device = device
    renderer._camera_prim_path = "/World/envs/env_0/Camera"
    renderer._clone_plan = ClonePlan(
        sources=("/Prototype",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=bool),
        env_ids=np.array([0, 1]),
    )
    renderer.backend.renderer = _FakeOVRTXBackend()
    renderer._deformable_points_binding = None
    renderer._deformable_particle_offsets = []
    renderer._deformable_particle_counts = []
    renderer._particle_points_binding = None
    renderer._particle_visual_offsets = []
    renderer._particle_visual_counts = []
    renderer._particle_workaround_applied = False
    # Cable bindings are set in __init__, which this fixture bypasses via __new__. Without them
    # _update_geometries_legacy raises AttributeError on its cable check before reaching anything
    # this module is testing.
    renderer._cable_points_binding = None
    renderer._cable_segment_counts = []
    renderer._use_ovstage = False
    return renderer, renderer.backend.renderer


@pytest.mark.parametrize(
    "template, env_ids, mask, selected",
    [
        ("/World/envs/env_{}", [0, 1], [True, True], [0, 1]),
        ("/Scenes/{}", [3, 17, 42], [True, False, True], [3, 42]),
    ],
)
def test_setup_deformable_bindings_binds_mixed_surface_and_volume_entries(
    monkeypatch: pytest.MonkeyPatch, template, env_ids, mask, selected
):
    """Registry metadata binds every surface and volume instance without a USD stage."""
    renderer, backend = _make_renderer_without_backend()
    renderer._clone_plan = replace(
        renderer._clone_plan, destinations=(template,), env_ids=np.array(env_ids), clone_mask=np.array([mask])
    )
    pattern = template.format("[^/]+")
    surface_entry = SimpleNamespace(
        prim_path=f"{pattern}/DeformableSurface",
        vis_mesh_prim_path=f"{pattern}/DeformableSurface/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3],
        particles_per_body=3,
    )
    volume_entry = SimpleNamespace(
        prim_path=f"{pattern}/DeformableVolume",
        vis_mesh_prim_path=f"{pattern}/DeformableVolume/mesh",
        deformable_type="volume",
        particle_offsets=[6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: None)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [surface_entry, volume_entry])

    renderer._setup_deformable_bindings_legacy()

    paths = [
        f"{template.format(env_id)}/{asset}/mesh"
        for asset in ("DeformableSurface", "DeformableVolume")
        for env_id in selected
    ]
    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == paths
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["dtype"] is np.float32
    assert backend.calls[0]["shape"] == (3,)
    assert [write["attribute_name"] for write in backend.writes] == ["omni:resetXformStack", "omni:xform"]
    assert all(write["prim_paths"] == paths for write in backend.writes)
    particle_q = wp.array(np.arange(36, dtype=np.float32).reshape(12, 3), dtype=wp.vec3f, device="cpu")
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=42))
    renderer.update_geometries()
    written = backend.bindings["points"].written
    assert len(written) == len(paths)
    for points, offset in zip(written, surface_entry.particle_offsets + volume_entry.particle_offsets, strict=True):
        np.testing.assert_array_equal(points.numpy(), particle_q.numpy()[offset : offset + 3])


def test_shadow_bindings_keep_heterogeneous_prototype_offsets(monkeypatch: pytest.MonkeyPatch):
    """Equal destination patterns do not merge different prototype particle layouts."""
    renderer, backend = _make_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/Source/A", "/Source/B"),
        destinations=("/Copies/{}/Body",) * 2,
        clone_mask=np.array([[True, False, True], [False, True, False]]),
        env_ids=np.array([2, 10, 30]),
    )
    groups = [
        ShadowDeformableRegistryGroup(
            prim_path="/Copies/[^/]+/Body",
            sim_mesh_prim_path="/Copies/[^/]+/Body/Sim",
            vis_mesh_prim_path="/Copies/[^/]+/Body/Visual",
            deformable_type="volume",
            particles_per_body=count,
            register_usd_vis_point_bindings=True,
            particle_offsets=offsets,
            entities=[
                ShadowDeformableEntity(f"/Copies/{env_id}/Body", 0, 4, offset, count)
                for env_id, offset in zip(env_ids, offsets, strict=True)
            ],
        )
        for count, env_ids, offsets in ((3, (2, 30), [4, 11]), (5, (10,), [20]))
    ]
    monkeypatch.setattr(
        NewtonManager, "_deformable_registry", [*groups, replace(groups[0], register_usd_vis_point_bindings=False)]
    )
    renderer._setup_deformable_bindings_legacy()
    assert backend.calls[0]["prim_paths"] == [f"/Copies/{env_id}/Body/Visual" for env_id in (2, 30, 10)]
    particle_q = wp.array(np.arange(75, dtype=np.float32).reshape(25, 3), dtype=wp.vec3f, device="cpu")
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=42))
    renderer.update_geometries()
    for points, offset, count in zip(backend.bindings["points"].written, (4, 11, 20), (3, 3, 5), strict=True):
        np.testing.assert_array_equal(points.numpy(), particle_q.numpy()[offset : offset + count])


def test_setup_deformable_bindings_rejects_offset_count_mismatch(monkeypatch: pytest.MonkeyPatch):
    """Native offsets must match the instances actually declared in the plan."""
    renderer, _backend = _make_renderer_without_backend()
    bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Deformable",
        vis_mesh_prim_path="/World/envs/env_[^/]+/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [bad_entry])
    with pytest.raises(ValueError, match="zip"):
        renderer._setup_deformable_bindings_legacy()


def test_update_geometries_rejects_inconsistent_deformable_mapping(monkeypatch: pytest.MonkeyPatch):
    """Geometry sync fails fast when offset and count metadata drift out of alignment."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("points")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [3, 3]
    particle_q = wp.array(
        [wp.vec3f(float(i), 0.0, 0.0) for i in range(4)],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    with pytest.raises(ValueError, match="zip"):
        renderer.update_geometries()


def test_setup_particle_points_bindings_binds_multiple_mpm_assets(monkeypatch: pytest.MonkeyPatch):
    """Multiple MPM assets bind as ``num_assets * num_envs`` points prims, like deformables."""
    renderer, backend = _make_renderer_without_backend()
    particle_visual_prims = {
        "/World/envs/env_0/Media/Particles": SimpleNamespace(offset=0, count=5),
        "/World/envs/env_1/Media/Particles": SimpleNamespace(offset=5, count=5),
        "/World/envs/env_0/Foam/Particles": SimpleNamespace(offset=10, count=3),
        "/World/envs/env_1/Foam/Particles": SimpleNamespace(offset=13, count=3),
    }

    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", particle_visual_prims)

    renderer._setup_particle_bindings_legacy()

    assert len(backend.calls) == 1
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["flags"] is BindingFlag.OPTIMIZE
    assert [write["attribute_name"] for write in backend.writes] == ["omni:resetXformStack", "omni:xform"]
    # Binding order follows dict insertion order (no path sort).
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
        "/World/envs/env_0/Foam/Particles",
        "/World/envs/env_1/Foam/Particles",
    ]
    particle_q = wp.array(np.arange(48, dtype=np.float32).reshape(16, 3), dtype=wp.vec3f, device="cpu")
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=42))
    renderer.update_geometries()
    for points, visual in zip(backend.bindings["points"].written, particle_visual_prims.values(), strict=True):
        np.testing.assert_array_equal(points.numpy(), particle_q.numpy()[visual.offset : visual.offset + visual.count])


def test_update_geometries_writes_deformable_and_mpm_bindings(monkeypatch: pytest.MonkeyPatch):
    """Deformable and MPM points use GPU ASYNC writes from the first update."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("deformable_points")
    renderer._deformable_particle_offsets = [1]
    renderer._deformable_particle_counts = [1]
    renderer._particle_points_binding = _FakePointsBinding("points")
    renderer._particle_visual_offsets = [2]
    renderer._particle_visual_counts = [2]
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(2.0, 3.0, 4.0),
            wp.vec3f(5.0, 6.0, 7.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    class _FakeStream:
        cuda_stream = 42

    renderer._warp_device = SimpleNamespace(stream=_FakeStream())

    renderer.update_geometries()

    deformable_written = renderer._deformable_points_binding.written
    assert deformable_written is not None
    assert len(deformable_written) == 1
    assert deformable_written[0].ptr == particle_q[1:2].ptr
    np.testing.assert_array_equal(deformable_written[0].numpy(), particle_q.numpy()[1:2])
    assert renderer._deformable_points_binding.write_kwargs is not None
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._deformable_points_binding.write_kwargs["cuda_stream"] == 42

    assert len(backend.writes) == 0
    mpm_written = renderer._particle_points_binding.written
    assert mpm_written is not None
    assert len(mpm_written) == 1
    assert mpm_written[0].ptr == particle_q[2:4].ptr
    np.testing.assert_array_equal(mpm_written[0].numpy(), particle_q.numpy()[2:4])
    assert renderer._particle_points_binding.write_kwargs is not None
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._particle_points_binding.write_kwargs["cuda_stream"] == 42


def _install_cable_shapes(shapes: dict[str, list[int]], monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake :meth:`NewtonManager.collect_cable_segment_shape_ids` result."""
    monkeypatch.setattr(NewtonManager, "collect_cable_segment_shape_ids", classmethod(lambda cls: dict(shapes)))


def test_setup_cable_bindings_binds_curve_points(monkeypatch: pytest.MonkeyPatch):
    """Renderable cables create a ``points`` array binding over their curve prims."""
    renderer, backend = _make_renderer_without_backend()
    _install_cable_shapes({"/World/envs/env_0/Cable/geometry/mesh": [4, 5, 6]}, monkeypatch)

    renderer._setup_cable_bindings_legacy()

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Cable/geometry/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["dtype"] is np.float32
    assert backend.calls[0]["shape"] == (3,)
    assert backend.calls[0]["flags"] is BindingFlag.OPTIMIZE
    assert renderer._cable_points_binding is backend.bindings["points"]

    # World-space points are written directly, so the inherited env transform must be neutralised
    # or it is applied twice -- the same contract the deformable path relies on.
    assert [write["attribute_name"] for write in backend.writes] == ["omni:resetXformStack", "omni:xform"]


def test_setup_cable_bindings_noop_without_cables(monkeypatch: pytest.MonkeyPatch):
    """A scene with no renderable cables binds nothing rather than failing."""
    renderer, backend = _make_renderer_without_backend()
    _install_cable_shapes({}, monkeypatch)

    renderer._setup_cable_bindings_legacy()

    assert renderer._cable_points_binding is None
    assert backend.calls == []


def test_update_geometries_writes_one_slice_per_cable(monkeypatch: pytest.MonkeyPatch):
    """Cable updates use disjoint point slices and GPU interop for unequal-length curves."""
    renderer, _ = _make_renderer_without_backend()
    _install_cable_shapes(
        {
            "/World/envs/env_0/Cable/geometry/mesh": [0, 1, 2],
            "/World/envs/env_1/Cable/geometry/mesh": [3, 4, 5, 6, 7],
            "/World/envs/env_2/Cable/geometry/mesh": [8, 9],
        },
        monkeypatch,
    )
    renderer._setup_cable_bindings_legacy()

    model = SimpleNamespace(shape_body=None, shape_transform=None, shape_scale=None)
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(body_q=None)))
    # The kernel needs a live Newton model; this test covers the slicing around it, not the maths in it.
    launch_kwargs: dict = {}

    def _capture_launch(*args, **kwargs):
        launch_kwargs.update(kwargs)
        if args:
            launch_kwargs["kernel"] = args[0]

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _capture_launch)
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=1234))

    renderer.update_geometries()

    assert launch_kwargs["dim"] == (3, 6)
    written = renderer._cable_points_binding.written
    assert written is not None
    assert [len(slice_) for slice_ in written] == [4, 6, 3]
    assert written[0].ptr == renderer._cable_points[0:4].ptr
    assert written[1].ptr == renderer._cable_points[4:10].ptr
    assert written[2].ptr == renderer._cable_points[10:13].ptr
    # Zero-copy: OVRTX is handed the Warp stream so it waits on the kernel instead of forcing a host
    # round-trip. Switching to SYNC would silently reintroduce a per-frame device copy, and is the
    # only guard against that -- the downgrade does not raise, it just renders from a stale copy.
    assert renderer._cable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._cable_points_binding.write_kwargs["cuda_stream"] == 1234


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="requires a CUDA device")
def test_write_particle_q_slices_ovstage_passes_device_slices_zero_copy():
    """The ovstage points write hands ``particle_q`` slices to ovstage as CUDA DLTensors, without a host copy."""
    renderer, _backend = _make_renderer_without_backend(device="cuda:0")
    particle_q = wp.array(
        [
            wp.vec3f(-1.0, -1.0, -1.0),
            wp.vec3f(1.0, 2.0, 3.0),
            wp.vec3f(4.0, 5.0, 6.0),
            wp.vec3f(7.0, 8.0, 9.0),
        ],
        dtype=wp.vec3f,
        device="cuda:0",
    )
    writes: list[dict] = []

    def _write(query, attribute, **kwargs):
        writes.append({"query": query, "attribute": attribute, **kwargs})
        return SimpleNamespace(wait=lambda: None)

    renderer.backend.stage = SimpleNamespace(write_attribute=_write)
    renderer._current_ordinal = 7
    renderer._warp_device = wp.get_device("cuda:0")

    renderer._write_particle_q_slices_ovstage("points_query", particle_q, [1], [3])

    assert len(writes) == 1
    assert writes[0]["attribute"] == "points"
    assert writes[0]["is_array"] is True
    # The slices alias ``particle_q``, so ovstage is handed the producing Warp stream to order its
    # read against, rather than the caller blocking the host on a device synchronize.
    assert writes[0]["cuda_stream"] == wp.get_stream("cuda:0").cuda_stream
    tensors = writes[0]["tensors"]
    assert len(tensors) == 1
    # A zero-copy device view: the descriptor points straight at the slice's own CUDA buffer with
    # the trailing component axis folded into point3f's three lanes.
    assert tensors[0].device.device_type.value == ovstage.DLDeviceType.kDLCUDA
    assert tensors[0].data == particle_q[1:4].ptr
    assert tensors[0].shape_tuple == (3,)
    assert tensors[0].dtype.lanes == 3


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_update_transforms_consumes_sdp_matrices_once_per_publication(monkeypatch, use_ovstage):
    """Both OVRTX paths bind published bodies and consume SDP's scaled, transposed matrices."""
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    def reject_newton_access(*args, **kwargs):
        raise AssertionError("Rigid transform transport must not read Newton state")

    monkeypatch.setattr(NewtonManager, "get_model", reject_newton_access)
    monkeypatch.setattr(NewtonManager, "get_state", reject_newton_access)
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
        assert renderer.backend.renderer.calls[0]["prim_paths"] == paths
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


def test_update_camera_writes_without_mapping(monkeypatch: pytest.MonkeyPatch):
    """Camera xforms are handed to ``write()`` instead of copied into a mapped OVRTX buffer."""
    renderer, _ = _make_renderer_without_backend()
    render_data = SimpleNamespace(camera_xform_binding=_FakePointsBinding("omni:xform"))
    camera_transforms = []

    monkeypatch.setattr(ovrtx_renderer_module, "convert_camera_frame_orientation_convention_wp", lambda **kwargs: None)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "empty", lambda *args, **kwargs: object())

    def _fake_zeros(*args, **kwargs):
        arr = object()
        camera_transforms.append(arr)
        return arr

    monkeypatch.setattr(ovrtx_renderer_module.wp, "zeros", _fake_zeros)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", lambda *args, **kwargs: None)
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=7))

    positions = SimpleNamespace(shape=(2,), warp=object())
    renderer.update_camera(render_data, positions, SimpleNamespace(warp=object()), object())

    assert render_data.camera_xform_binding.written is camera_transforms[0]
    assert render_data.camera_xform_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert render_data.camera_xform_binding.write_kwargs["cuda_stream"] == 7
