# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the :class:`OvPhysxView` string-keyed binding manager.

These exercise the pure-Python name<->enum logic and the view's get/set/read-into
dispatch (including the float32 reinterpret of structured buffers and the
no-implicit-conversion device policy) against a fake ``PhysX`` + fake ``TensorBinding``.

Scope is intentionally the **API mechanics** against mock bindings. Live coverage against
real ovphysx bindings -- CPU-only properties on a GPU sim, read-only/write-only failures,
and structured ``read_into`` round-trips -- is provided by the asset-integration tests
(``test_articulation.py`` / ``test_rigid_object.py`` / ``test_rigid_object_collection.py``)
that adopt this view as ``root_view``; it is not (and is not meant to be) re-covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

# The OVPhysX runtime wheel is optional. ``ovphysx.types`` is pure Python (no native
# dependency), so the import-skip guards only the wheel's presence.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import warp as wp  # noqa: E402
from isaaclab_ovphysx.sim.views.ovphysx_view import (  # noqa: E402
    OvPhysxView,
    OvPhysxViewError,
    attribute_vocabulary,
    is_cpu_only,
    is_read_only,
    resolve_tensor_type,
    tensor_type_name,
)
from ovphysx.types import TensorType  # noqa: E402

wp.init()
_HAS_CUDA = wp.get_cuda_device_count() > 0

# Per-type shapes used by the fakes (only the types touched by the tests).
_SHAPES = {
    TensorType.RIGID_BODY_POSE: lambda n: (n, 7),
    TensorType.RIGID_BODY_VELOCITY: lambda n: (n, 6),
    TensorType.RIGID_BODY_MASS: lambda n: (n,),  # CPU-only
    TensorType.RIGID_BODY_ACCELERATION: lambda n: (n, 6),  # read-only
}


class _FakeBinding:
    """Minimal stand-in for an ovphysx ``TensorBinding``."""

    def __init__(self, tensor_type, n: int):
        self.tensor_type = tensor_type
        self.shape = _SHAPES.get(tensor_type, lambda k: (k, 1))(n)
        self.count = n
        self.prim_paths = [f"/World/env_{i}/body" for i in range(n)]
        self.dof_names: list[str] = []
        self.body_names = ["body"]
        self.joint_names: list[str] = []
        self.dof_count = 0
        self.body_count = 1
        self.joint_count = 0
        self.is_fixed_base = True
        self.fixed_tendon_count = 0
        self.spatial_tendon_count = 0
        self.read_calls = 0
        self.last_read: tuple | None = None
        self.last_read_obj = None  # the actual array object handed to read (for cache-warmth checks)
        self.write_calls: list[tuple] = []

    def read(self, dst) -> None:
        self.read_calls += 1
        self.last_read = (dst.dtype, tuple(dst.shape), str(dst.device))
        self.last_read_obj = dst

    def write(self, tensor, indices=None, mask=None) -> None:
        self.write_calls.append((tensor.dtype, tuple(tensor.shape), indices, mask))


class _FakePhysX:
    """Fake ``PhysX`` whose ``create_tensor_binding`` hands back ``_FakeBinding`` instances."""

    def __init__(self, n: int = 3, unavailable: set | None = None, all_unavailable: bool = False):
        self.n = n
        self._unavailable = unavailable or set()
        self._all_unavailable = all_unavailable
        self.created: list[tuple] = []

    def create_tensor_binding(self, *, tensor_type, pattern=None, prim_paths=None):
        self.created.append((tensor_type, pattern, prim_paths))
        if self._all_unavailable or tensor_type in self._unavailable:
            return _FakeBinding(tensor_type, 0)  # wheel returns a 0-count binding on no match
        return _FakeBinding(tensor_type, self.n)


def _make_view(n: int = 3, unavailable: set | None = None, device: str = "cpu") -> OvPhysxView:
    return OvPhysxView(_FakePhysX(n=n, unavailable=unavailable), pattern="/World/env_*/body", device=device)


# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------


def test_vocabulary_is_lowercased_enum_without_invalid():
    vocab = attribute_vocabulary()
    assert "articulation_dof_stiffness" in vocab
    assert "rigid_body_pose" in vocab
    assert "invalid" not in vocab
    assert vocab == sorted(vocab)


def test_resolve_roundtrips_name_and_enum():
    tt = resolve_tensor_type("articulation_dof_stiffness")
    assert tt is TensorType.ARTICULATION_DOF_STIFFNESS
    assert tensor_type_name(tt) == "articulation_dof_stiffness"
    assert resolve_tensor_type("RIGID_BODY_POSE") is TensorType.RIGID_BODY_POSE  # case-insensitive


def test_resolve_unknown_name_raises():
    with pytest.raises(OvPhysxView.UnknownAttribute):
        resolve_tensor_type("not_a_real_attribute")


def test_read_only_names_are_valid_vocabulary():
    # Every read-only name must resolve to a real TensorType, so the hand-maintained
    # set stays coupled to the wheel enum (no dead names).
    from isaaclab_ovphysx.sim.views import ovphysx_view as mod

    assert set(attribute_vocabulary()) >= mod._READ_ONLY_NAMES


def test_read_only_and_cpu_only_classification():
    assert is_read_only("articulation_jacobian")
    assert is_read_only("rigid_body_acceleration")
    assert not is_read_only("articulation_dof_stiffness")
    assert is_cpu_only("articulation_dof_stiffness")
    assert is_cpu_only("rigid_body_mass")
    assert not is_cpu_only("rigid_body_pose")


def test_cpu_only_names_match_canonical_set():
    # The view derives its CPU-only set from tensor_types so the two cannot drift.
    from isaaclab_ovphysx.sim.views import ovphysx_view as mod
    from isaaclab_ovphysx.tensor_types import _CPU_ONLY_TYPES

    assert frozenset(tt.name.lower() for tt in _CPU_ONLY_TYPES) == mod._CPU_ONLY_NAMES


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


def test_requires_exactly_one_of_pattern_or_prim_paths():
    with pytest.raises(ValueError):
        OvPhysxView(_FakePhysX(), pattern="/p", prim_paths=["/p"], device="cpu")
    with pytest.raises(ValueError):
        OvPhysxView(_FakePhysX(), device="cpu")


def test_eager_creates_requested_and_exposes_metadata():
    view = OvPhysxView(
        _FakePhysX(n=5),
        pattern="/World/env_*/body",
        device="cpu",
        tensor_types=[TensorType.RIGID_BODY_POSE, TensorType.RIGID_BODY_MASS],
        eager=True,
    )
    assert view.available_attributes == ["rigid_body_mass", "rigid_body_pose"]
    assert view.count == 5  # metadata works without an explicit get_attribute call


def test_eager_default_sweep_empty_view_raises():
    # Default sweep (no tensor_types) on a pattern that matches nothing -> aggregate raise.
    physx = _FakePhysX(all_unavailable=True)
    with pytest.raises(OvPhysxViewError, match="Could not create any bindings"):
        OvPhysxView(physx, pattern="/no/match", device="cpu", eager=True)


# -----------------------------------------------------------------------------
# get_attribute / read_into
# -----------------------------------------------------------------------------


def test_get_attribute_allocates_fresh_typed_buffer_each_call():
    view = _make_view(n=4)
    buf = view.get_attribute("rigid_body_pose")
    # Pose maps to a structured dtype: an [N] transformf array (== [N, 7] float32).
    assert tuple(buf.shape) == (4,) and buf.dtype == wp.transformf
    binding = view._bindings[TensorType.RIGID_BODY_POSE]
    assert binding.read_calls == 1
    buf2 = view.get_attribute("rigid_body_pose")
    assert buf2 is not buf  # no aliasing of view state
    assert binding.read_calls == 2


def test_get_attribute_no_out_does_not_grow_read_cache():
    # A no-`out` get_attribute allocates a fresh buffer each call, so caching its reinterpret by
    # id() could never hit on a later call and would leak one entry (keeping the buffer alive) per
    # call in a step loop. The structured-dtype path must reinterpret directly, never touching the
    # cache -- the read cache only pays off for a reused `out`/`dst`. Pose is the structured case
    # (transformf), the one that would have been cached before the fix.
    view = _make_view(n=3)
    for _ in range(5):
        view.get_attribute("rigid_body_pose")
    assert view._read_views == {}


def test_get_attribute_types_pose_and_velocity_falls_back_to_float32():
    view = _make_view(n=3)
    assert view.get_attribute("rigid_body_pose").dtype == wp.transformf
    assert view.get_attribute("rigid_body_velocity").dtype == wp.spatial_vectorf
    # An attribute absent from the structured-dtype map stays flat float32.
    mass = view.get_attribute("rigid_body_mass")
    assert mass.dtype == wp.float32 and tuple(mass.shape) == (3,)


def test_read_into_reuses_reinterpret_view_across_calls():
    # The float32 reinterpret of a structured dst is built once and reused so the wheel's
    # object-identity read cache stays warm across steps.
    view = _make_view(n=3)
    dst = wp.zeros((3,), dtype=wp.transformf, device="cpu")
    view.read_into("rigid_body_pose", dst)
    binding = view._bindings[TensorType.RIGID_BODY_POSE]
    first = binding.last_read_obj
    view.read_into("rigid_body_pose", dst)
    assert binding.last_read_obj is first  # same object handed to the wheel both times
    assert first is not dst  # it is the float32 reinterpret, not the transformf buffer


def test_read_into_passthrough_reuses_dst_object():
    # A flat float32 dst is its own stable identity -- passed straight through, not cached.
    view = _make_view(n=3)
    dst = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    view.read_into("rigid_body_pose", dst)
    binding = view._bindings[TensorType.RIGID_BODY_POSE]
    assert binding.last_read_obj is dst
    assert id(dst) not in view._read_views  # float32 dst is not cached


def test_read_into_caches_per_destination_buffer():
    view = _make_view(n=3)
    a = wp.zeros((3,), dtype=wp.transformf, device="cpu")
    b = wp.zeros((3,), dtype=wp.transformf, device="cpu")
    binding = view._bindings.get(TensorType.RIGID_BODY_POSE) or view.binding_for("rigid_body_pose")
    view.read_into("rigid_body_pose", a)
    view_a = binding.last_read_obj
    view.read_into("rigid_body_pose", b)
    view_b = binding.last_read_obj
    assert view_a is not view_b  # distinct reinterprets per destination buffer
    assert view_a.ptr == a.ptr and view_b.ptr == b.ptr


def test_get_attribute_out_param_is_filled_and_returned():
    view = _make_view(n=2)
    out = wp.zeros((2, 7), dtype=wp.float32, device="cpu")
    ret = view.get_attribute("rigid_body_pose", out=out)
    assert ret is out
    assert view._bindings[TensorType.RIGID_BODY_POSE].read_calls == 1


def test_read_into_reinterprets_structured_buffer():
    view = _make_view(n=3)
    dst = wp.zeros((3,), dtype=wp.transformf, device="cpu")  # [N] transformf == [N,7] float32
    view.read_into("rigid_body_pose", dst)
    binding = view._bindings[TensorType.RIGID_BODY_POSE]
    # The binding was handed a float32 view matching its flat shape, not the transformf buffer.
    assert binding.last_read == (wp.float32, (3, 7), "cpu")


def test_read_into_passthrough_when_already_float32():
    view = _make_view(n=3)
    dst = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    view.read_into("rigid_body_pose", dst)
    assert view._bindings[TensorType.RIGID_BODY_POSE].last_read == (wp.float32, (3, 7), "cpu")


def test_read_into_shape_mismatch_raises():
    view = _make_view(n=3)
    wrong = wp.zeros((3, 6), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxView.ShapeMismatch):
        view.read_into("rigid_body_pose", wrong)


# -----------------------------------------------------------------------------
# Device policy (no implicit CPU<->GPU conversion)
# -----------------------------------------------------------------------------


def test_cpu_array_for_device_state_on_gpu_sim_raises():
    # GPU sim, but a CPU buffer is supplied for a device-resident state attribute.
    view = _make_view(n=3, device="cuda:0")
    cpu_buf = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxView.DeviceMismatch, match="cuda:0"):
        view.read_into("rigid_body_pose", cpu_buf)


def test_cpu_only_property_accepts_cpu_buffer_on_gpu_sim():
    # GPU sim, CPU-only property (mass): a CPU buffer is correct and must NOT raise.
    view = _make_view(n=3, device="cuda:0")
    cpu_buf = wp.zeros((3,), dtype=wp.float32, device="cpu")
    view.read_into("rigid_body_mass", cpu_buf)  # no raise
    assert view._bindings[TensorType.RIGID_BODY_MASS].read_calls == 1


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA device to allocate a GPU buffer")
def test_gpu_array_for_cpu_only_property_raises():
    view = _make_view(n=3, device="cuda:0")
    gpu_buf = wp.zeros((3,), dtype=wp.float32, device="cuda:0")
    with pytest.raises(OvPhysxView.DeviceMismatch, match="cpu"):
        view.read_into("rigid_body_mass", gpu_buf)


# -----------------------------------------------------------------------------
# set_attribute
# -----------------------------------------------------------------------------


def test_set_attribute_forwards_indices_and_mask():
    view = _make_view(n=3)
    values = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    idx = wp.array([0, 2], dtype=wp.int32, device="cpu")
    view.set_attribute("rigid_body_pose", values, indices=idx)
    dtype, shape, indices, mask = view._bindings[TensorType.RIGID_BODY_POSE].write_calls[0]
    assert (dtype, shape, indices, mask) == (wp.float32, (3, 7), idx, None)


def test_set_attribute_reinterprets_structured_source():
    view = _make_view(n=3)
    values = wp.zeros((3,), dtype=wp.transformf, device="cpu")
    view.set_attribute("rigid_body_pose", values)
    dtype, shape, _, _ = view._bindings[TensorType.RIGID_BODY_POSE].write_calls[0]
    assert (dtype, shape) == (wp.float32, (3, 7))


def test_set_attribute_read_only_raises_and_does_not_bind():
    view = _make_view(n=3)
    values = wp.zeros((3, 6), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxView.ReadOnlyAttribute, match="read-only"):
        view.set_attribute("rigid_body_acceleration", values)
    assert TensorType.RIGID_BODY_ACCELERATION not in view._bindings


def test_set_attribute_shape_mismatch_raises():
    view = _make_view(n=3)
    wrong = wp.zeros((3, 6), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxView.ShapeMismatch, match="Shape mismatch"):
        view.set_attribute("rigid_body_pose", wrong)


def test_set_attribute_cpu_array_for_state_on_gpu_sim_raises():
    view = _make_view(n=3, device="cuda:0")
    values = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxView.DeviceMismatch):
        view.set_attribute("rigid_body_pose", values)


def test_as_wp_accepts_numpy_float32_and_rejects_float64():
    view = _make_view(n=2)
    # float32 host data is materialized on the native device and written.
    view.set_attribute("rigid_body_pose", np.zeros((2, 7), dtype=np.float32))
    assert len(view._bindings[TensorType.RIGID_BODY_POSE].write_calls) == 1
    # float64 is not float32-bit-equivalent; reject rather than silently reinterpret.
    with pytest.raises(OvPhysxView.DtypeMismatch, match="float32 scalar"):
        view.set_attribute("rigid_body_pose", np.zeros((2, 7), dtype=np.float64))


# -----------------------------------------------------------------------------
# prim_paths + key aliases (RigidObjectCollection fused-binding shape)
# -----------------------------------------------------------------------------


def test_prim_paths_with_key_alias_creates_remapped_type():
    physx = _FakePhysX(n=6)
    view = OvPhysxView(
        physx,
        prim_paths=["/World/env_*/cube", "/World/env_*/sphere"],
        device="cpu",
        key_aliases={TensorType.ARTICULATION_LINK_POSE: TensorType.RIGID_BODY_POSE},
    )
    binding = view.binding_for("articulation_link_pose")
    # Created as RIGID_BODY_POSE via prim_paths, cached under the requested LINK_POSE key.
    created_type, pattern, prim_paths = physx.created[0]
    assert created_type is TensorType.RIGID_BODY_POSE
    assert pattern is None and prim_paths == ["/World/env_*/cube", "/World/env_*/sphere"]
    assert view._bindings[TensorType.ARTICULATION_LINK_POSE] is binding
    # And a write through the aliased key resolves to that same cached binding.
    view.set_attribute("articulation_link_pose", wp.zeros((6, 7), dtype=wp.float32, device="cpu"))
    assert len(binding.write_calls) == 1 and binding is view._bindings[TensorType.ARTICULATION_LINK_POSE]


# -----------------------------------------------------------------------------
# Errors / discoverability / metadata
# -----------------------------------------------------------------------------


def test_unknown_attribute_raises_on_access():
    view = _make_view()
    with pytest.raises(OvPhysxView.UnknownAttribute):
        view.get_attribute("totally_made_up")


def test_unavailable_binding_reports_clear_error():
    view = _make_view(n=3, unavailable={TensorType.RIGID_BODY_VELOCITY})
    with pytest.raises(OvPhysxView.AttributeUnavailable, match="not available"):
        view.get_attribute("rigid_body_velocity")


def test_discoverability_surface():
    view = _make_view()
    assert "rigid_body_pose" in view
    assert view.has_attribute("articulation_dof_stiffness")
    assert not view.has_attribute("nope")
    assert "rigid_body_pose" in view.attribute_names
    assert view.available_attributes == []
    view.get_attribute("rigid_body_pose")
    assert view.available_attributes == ["rigid_body_pose"]


def test_metadata_passthrough_from_sample_binding():
    view = _make_view(n=5)
    with pytest.raises(OvPhysxView.AttributeUnavailable):
        _ = view.count  # metadata before any access raises a clear error
    view.get_attribute("rigid_body_pose")
    assert view.count == 5
    assert len(view.prim_paths) == 5
    assert view.body_names == ["body"]
    assert view.is_fixed_base is True
    assert view.joint_count == 0 and view.fixed_tendon_count == 0


# -----------------------------------------------------------------------------
# dtype safety — the view reinterprets bits, so non-float32 scalars must be rejected
# -----------------------------------------------------------------------------


def test_set_attribute_rejects_same_byte_size_wrong_dtype():
    # int32 has the same 4-byte width as float32: it would pass a byte-count-only guard
    # and get bit-reinterpreted into garbage. It must be rejected, not silently written.
    view = _make_view(n=3)
    int_buf = wp.zeros((3, 7), dtype=wp.int32, device="cpu")
    with pytest.raises(OvPhysxView.DtypeMismatch, match="float32 scalar"):
        view.set_attribute("rigid_body_pose", int_buf)
    assert view._bindings[TensorType.RIGID_BODY_POSE].write_calls == []


def test_set_attribute_rejects_sub_4byte_dtype():
    view = _make_view(n=3)
    half_buf = wp.zeros((3, 7), dtype=wp.float16, device="cpu")
    with pytest.raises(OvPhysxView.DtypeMismatch, match="float32 scalar"):
        view.set_attribute("rigid_body_pose", half_buf)


def test_set_attribute_forwards_both_indices_and_mask():
    # The view forwards both verbatim; the wheel resolves precedence (mask wins).
    view = _make_view(n=3)
    values = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    idx = wp.array([0, 2], dtype=wp.int32, device="cpu")
    mask = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    view.set_attribute("rigid_body_pose", values, indices=idx, mask=mask)
    _, _, fwd_idx, fwd_mask = view._bindings[TensorType.RIGID_BODY_POSE].write_calls[0]
    assert fwd_idx is idx and fwd_mask is mask


def test_get_attribute_out_on_wrong_device_raises():
    # `get_attribute(out=)` has its own device check distinct from read_into's.
    view = _make_view(n=3, device="cuda:0")
    cpu_out = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxView.DeviceMismatch):
        view.get_attribute("rigid_body_pose", out=cpu_out)


def test_resolve_rejects_non_str_non_tensortype():
    view = _make_view()
    with pytest.raises(OvPhysxView.UnknownAttribute):
        view.get_attribute(123)


def test_set_attribute_rejects_non_contiguous_source():
    # A strided slice has the right element count but non-contiguous memory; the ptr-based
    # reinterpret would read the wrong elements, so it must be rejected.
    view = _make_view(n=3)
    base = wp.zeros((3, 14), dtype=wp.float32, device="cpu")
    strided = base[:, :7]
    assert not strided.is_contiguous
    with pytest.raises(OvPhysxView.ShapeMismatch, match="contiguous"):
        view.set_attribute("rigid_body_pose", strided)


# -----------------------------------------------------------------------------
# API hardening — adversarial construction / resolution
# -----------------------------------------------------------------------------


def test_invalid_tensortype_member_rejected():
    view = _make_view()
    with pytest.raises(OvPhysxView.UnknownAttribute):
        view.get_attribute(TensorType.INVALID)


def test_string_keyed_aliases_are_honored():
    # Passing string alias keys/values must be normalized to TensorType, not silently dropped.
    physx = _FakePhysX(n=6)
    view = OvPhysxView(
        physx,
        prim_paths=["/World/env_*/cube"],
        device="cpu",
        key_aliases={"articulation_link_pose": "rigid_body_pose"},
    )
    view.binding_for("articulation_link_pose")
    assert physx.created[0][0] is TensorType.RIGID_BODY_POSE  # alias applied


def test_key_alias_crossing_residency_is_rejected():
    # LINK_POSE is GPU state; RIGID_BODY_MASS is CPU-only -> the device guard would be wrong.
    with pytest.raises(ValueError, match="residency"):
        OvPhysxView(
            _FakePhysX(),
            pattern="/p",
            device="cpu",
            key_aliases={TensorType.ARTICULATION_LINK_POSE: TensorType.RIGID_BODY_MASS},
        )


def test_tensor_types_without_eager_raises():
    with pytest.raises(ValueError, match="eager"):
        OvPhysxView(_FakePhysX(), pattern="/p", device="cpu", tensor_types=[TensorType.RIGID_BODY_POSE])


def test_empty_target_is_rejected():
    with pytest.raises(ValueError):
        OvPhysxView(_FakePhysX(), prim_paths=[], device="cpu")
    with pytest.raises(ValueError):
        OvPhysxView(_FakePhysX(), pattern="", device="cpu")


def test_eager_explicit_unavailable_type_raises_loud():
    # When the caller names exact types, a failing one is surfaced (not silently dropped).
    physx = _FakePhysX(n=3, unavailable={TensorType.RIGID_BODY_VELOCITY})
    with pytest.raises(OvPhysxView.AttributeUnavailable):
        OvPhysxView(
            physx,
            pattern="/World/env_*/body",
            device="cpu",
            tensor_types=[TensorType.RIGID_BODY_POSE, TensorType.RIGID_BODY_VELOCITY],
            eager=True,
        )


def test_get_attribute_cpu_only_property_returns_cpu_buffer_on_gpu_sim():
    # No-out allocation path must use the native device: CPU for a CPU-only property even
    # though the sim device is a GPU. (CPU allocation -> runs without a GPU.)
    view = _make_view(n=3, device="cuda:0")
    buf = view.get_attribute("rigid_body_mass")
    assert str(buf.device) == "cpu"


def test_binding_for_is_idempotent_and_unguarded():
    view = _make_view(n=3)
    # Returns a binding even for a read-only attribute (raw access bypasses the write guard)...
    b1 = view.binding_for("rigid_body_acceleration")
    b2 = view.binding_for("rigid_body_acceleration")
    assert b1 is b2  # cached / created once
    assert len(view._physx.created) == 1


def test_try_binding_for_returns_none_when_unavailable():
    view = _make_view(n=3, unavailable={TensorType.RIGID_BODY_VELOCITY})
    # Available for these prims -> the (cached) binding; unavailable -> None, no raise.
    assert view.try_binding_for("rigid_body_pose") is view._bindings[TensorType.RIGID_BODY_POSE]
    assert view.try_binding_for("rigid_body_velocity") is None
    # An invalid name is still a hard error, not an availability question.
    with pytest.raises(OvPhysxView.UnknownAttribute):
        view.try_binding_for("not_a_real_attribute")


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA device for the cuda:0 buffer")
def test_device_cuda_alias_is_canonicalized():
    # A view built with the bare "cuda" alias must accept a canonical "cuda:0" buffer.
    view = _make_view(n=3, device="cuda")
    gpu_buf = wp.zeros((3, 7), dtype=wp.float32, device="cuda:0")
    view.read_into("rigid_body_pose", gpu_buf)  # must not raise
    assert view._bindings[TensorType.RIGID_BODY_POSE].read_calls == 1
