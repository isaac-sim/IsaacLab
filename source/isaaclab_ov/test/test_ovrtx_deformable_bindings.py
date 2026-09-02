# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX deformable mesh point bindings."""

from __future__ import annotations

import importlib.util
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
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402
    from ovrtx import BindingFlag, DataAccess  # noqa: E402
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
    renderer._device = device
    renderer._env_paths = ("/World/envs/env_0",)
    renderer._camera_paths = ("/World/envs/env_0/Camera",)
    renderer._clone_plan = None
    renderer._renderer = _FakeOVRTXBackend()
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
    return renderer, renderer._renderer


def test_points_array_binding_uses_write_not_map():
    """OVRTX array bindings accept ``List[DLTensor]`` via ``write()``, not mapped tensors."""
    binding = _FakePointsBinding("points")
    with pytest.raises(RuntimeError, match="do not expose mapped point buffers"):
        binding.map()


def test_setup_deformable_bindings_binds_surface_mesh_points(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries create OVRTX ``points`` array bindings."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Deformable",
        vis_mesh_prim_path="/World/envs/env_[^/]+/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["dtype"] is np.float32
    assert backend.calls[0]["shape"] == (3,)
    assert renderer._deformable_points_binding is backend.bindings["points"]
    assert len(backend.writes) == 2
    assert backend.writes[0]["attribute_name"] == "omni:resetXformStack"
    assert backend.writes[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.writes[1]["attribute_name"] == "omni:xform"
    assert backend.writes[1]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert len(renderer._deformable_particle_counts) == 1
    assert renderer._deformable_particle_counts[0] == 3
    assert renderer._deformable_particle_offsets == [7]


def test_setup_deformable_bindings_binds_volume_mesh_points(monkeypatch: pytest.MonkeyPatch):
    """Volume deformable registry entries create OVRTX ``points`` bindings."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Deformable",
        vis_mesh_prim_path="/World/envs/env_[^/]+/Deformable/mesh",
        deformable_type="volume",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert renderer._deformable_points_binding is backend.bindings["points"]
    assert renderer._deformable_particle_offsets == [7]


def test_setup_deformable_bindings_binds_mixed_surface_and_volume_entries(monkeypatch: pytest.MonkeyPatch):
    """Surface and volume deformable registry entries bind together with distinct offsets."""
    renderer, backend = _make_renderer_without_backend()
    surface_entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/DeformableSurface",
        vis_mesh_prim_path="/World/envs/env_[^/]+/DeformableSurface/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3],
        particles_per_body=3,
    )
    volume_entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/DeformableVolume",
        vis_mesh_prim_path="/World/envs/env_[^/]+/DeformableVolume/mesh",
        deformable_type="volume",
        particle_offsets=[6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [surface_entry, volume_entry])

    renderer._setup_deformable_bindings(num_envs=2)

    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/DeformableSurface/mesh",
        "/World/envs/env_1/DeformableSurface/mesh",
        "/World/envs/env_0/DeformableVolume/mesh",
        "/World/envs/env_1/DeformableVolume/mesh",
    ]
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


def test_setup_deformable_bindings_works_without_stage(monkeypatch: pytest.MonkeyPatch):
    """Deformable bindings are created from registry metadata without a USD stage."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Deformable",
        vis_mesh_prim_path="/World/envs/env_[^/]+/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: None)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_points_binding is backend.bindings["points"]


def test_setup_deformable_bindings_binds_all_surface_mesh_instances(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries bind every cloned visual mesh instance."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Deformable",
        vis_mesh_prim_path="/World/envs/env_[^/]+/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=4)

    expected_paths = [f"/World/envs/env_{i}/Deformable/mesh" for i in range(4)]
    assert backend.calls[0]["prim_paths"] == expected_paths
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


def test_update_deformable_points_writes_world_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` slices are handed to OVRTX through :meth:`OVRTXRenderer.update_geometries`."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("points")
    renderer._deformable_particle_offsets = [1]
    renderer._deformable_particle_counts = [3]
    particle_q = wp.array(
        [
            wp.vec3f(-1.0, -1.0, -1.0),
            wp.vec3f(1.0, 2.0, 3.0),
            wp.vec3f(4.0, 5.0, 6.0),
            wp.vec3f(7.0, 8.0, 9.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    class _FakeStream:
        cuda_stream = 42

    renderer._warp_device = SimpleNamespace(stream=_FakeStream())

    renderer.update_geometries()

    written = renderer._deformable_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert written[0].ptr == particle_q[1:4].ptr
    assert renderer._deformable_points_binding.write_kwargs is not None
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._deformable_points_binding.write_kwargs["cuda_stream"] == 42
    assert written[0].numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


def test_setup_deformable_bindings_rejects_offset_count_mismatch(monkeypatch: pytest.MonkeyPatch):
    """Registry entries must provide one particle offset per environment, listing every bad entry."""
    renderer, _backend = _make_renderer_without_backend()
    bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Deformable",
        vis_mesh_prim_path="/World/envs/env_[^/]+/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )
    other_bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/DeformableOther",
        vis_mesh_prim_path="/World/envs/env_[^/]+/DeformableOther/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [bad_entry, other_bad_entry])

    with pytest.raises(RuntimeError, match="one particle offset per environment") as excinfo:
        renderer._setup_deformable_bindings(num_envs=2)

    message = str(excinfo.value)
    assert bad_entry.prim_path in message
    assert other_bad_entry.prim_path in message


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


def test_setup_particle_points_bindings_binds_mpm_visual_prims(monkeypatch: pytest.MonkeyPatch):
    """MPM particle visual prims create an OPTIMIZE ``points`` array binding."""
    renderer, backend = _make_renderer_without_backend()
    particle_visual_prims = {
        "/World/envs/env_0/Media/Particles": SimpleNamespace(offset=10, count=5),
        "/World/envs/env_1/Media/Particles": SimpleNamespace(offset=15, count=5),
    }

    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", particle_visual_prims)

    renderer._setup_particle_bindings()

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
    ]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["flags"] is BindingFlag.OPTIMIZE
    assert renderer._particle_points_binding is backend.bindings["points"]
    assert renderer._particle_visual_offsets == [10, 15]
    assert renderer._particle_visual_counts == [5, 5]
    assert len(backend.writes) == 2
    assert backend.writes[0]["attribute_name"] == "omni:resetXformStack"
    assert backend.writes[1]["attribute_name"] == "omni:xform"


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

    renderer._setup_particle_bindings()

    # Binding order follows dict insertion order (no path sort).
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
        "/World/envs/env_0/Foam/Particles",
        "/World/envs/env_1/Foam/Particles",
    ]
    assert renderer._particle_visual_offsets == [0, 5, 10, 13]
    assert renderer._particle_visual_counts == [5, 5, 3, 3]


def test_update_particle_points_writes_world_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """The first MPM ``points`` update writes world-space positions through GPU ASYNC."""
    renderer, backend = _make_renderer_without_backend()
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

    assert len(backend.writes) == 0
    written = renderer._particle_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert written[0].ptr == particle_q[2:4].ptr
    assert written[0].numpy().tolist() == [
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
    ]
    assert renderer._particle_points_binding.write_kwargs is not None
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._particle_points_binding.write_kwargs["cuda_stream"] == 42


def test_update_geometries_writes_deformable_and_mpm_bindings(monkeypatch: pytest.MonkeyPatch):
    """Deformable and MPM points use GPU ASYNC writes from the first update."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("deformable_points")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [2]
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
    assert deformable_written[0].ptr == particle_q[0:2].ptr
    assert renderer._deformable_points_binding.write_kwargs is not None
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._deformable_points_binding.write_kwargs["cuda_stream"] == 42

    assert len(backend.writes) == 0
    mpm_written = renderer._particle_points_binding.written
    assert mpm_written is not None
    assert len(mpm_written) == 1
    assert mpm_written[0].ptr == particle_q[2:4].ptr
    assert renderer._particle_points_binding.write_kwargs is not None
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._particle_points_binding.write_kwargs["cuda_stream"] == 42
    assert len(backend.writes) == 0


def _install_cable_shapes(shapes: dict[str, list[int]], monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake :meth:`NewtonManager.collect_cable_segment_shape_ids` result."""
    monkeypatch.setattr(NewtonManager, "collect_cable_segment_shape_ids", classmethod(lambda cls: dict(shapes)))


def test_setup_cable_bindings_binds_curve_points(monkeypatch: pytest.MonkeyPatch):
    """Renderable cables create a ``points`` array binding over their curve prims."""
    renderer, backend = _make_renderer_without_backend()
    _install_cable_shapes({"/World/envs/env_0/Cable/geometry/mesh": [4, 5, 6]}, monkeypatch)

    renderer._setup_cable_bindings()

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

    renderer._setup_cable_bindings()

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
    renderer._setup_cable_bindings()

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

    renderer._stage = SimpleNamespace(write_attribute=_write)
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


def test_update_transforms_writes_caller_owned_buffer(monkeypatch: pytest.MonkeyPatch):
    """Object xforms fill a persistent GPU buffer and blocking ASYNC write, not map/unmap."""
    renderer, _ = _make_renderer_without_backend()
    buffer = object()
    renderer._object_xform_binding = _FakePointsBinding("omni:xform")
    renderer._object_newton_indices = [0, 1]
    renderer._object_scales = object()
    renderer._object_transform_buffer = buffer

    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(body_q=object())))
    launch_kwargs: dict = {}

    def _capture_launch(*args, **kwargs):
        launch_kwargs.update(kwargs)

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _capture_launch)
    renderer._warp_device = SimpleNamespace(stream=SimpleNamespace(cuda_stream=99))

    renderer.update_transforms()

    assert launch_kwargs["inputs"][0] is buffer
    assert launch_kwargs["dim"] == 2
    assert renderer._object_xform_binding.written is buffer
    assert renderer._object_xform_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._object_xform_binding.write_kwargs["cuda_stream"] == 99


def test_update_camera_writes_without_mapping(monkeypatch: pytest.MonkeyPatch):
    """Camera xforms are handed to ``write()`` instead of copied into a mapped OVRTX buffer."""
    renderer, _ = _make_renderer_without_backend()
    renderer._camera_xform_binding = _FakePointsBinding("omni:xform")
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
    renderer.update_camera(object(), positions, SimpleNamespace(warp=object()), object())

    assert renderer._camera_xform_binding.written is camera_transforms[0]
    assert renderer._camera_xform_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._camera_xform_binding.write_kwargs["cuda_stream"] == 7
