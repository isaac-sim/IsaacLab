# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the prototype :class:`OvPhysxView` string-keyed binding wrapper.

These exercise the pure-Python name<->enum logic and the view's get/set dispatch
against a fake ``PhysX`` + fake ``TensorBinding`` -- no native simulation required.
Full read/write round-trips on a live sim are covered by the asset integration tests.
"""

from __future__ import annotations

import pytest

# The OVPhysX runtime wheel is optional. ``ovphysx.types`` is pure Python (no native
# dependency), so the import-skip guards only the wheel's presence.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import warp as wp  # noqa: E402
from isaaclab_ovphysx.sim.views.ovphysx_view import (  # noqa: E402
    OvPhysxView,
    OvPhysxViewError,
    attribute_vocabulary,
    is_read_only,
    resolve_tensor_type,
    tensor_type_name,
)
from ovphysx.types import TensorType  # noqa: E402

wp.init()

# Per-type shapes used by the fakes (only the types touched by the tests).
_SHAPES = {
    TensorType.RIGID_BODY_POSE: lambda n: (n, 7),
    TensorType.RIGID_BODY_VELOCITY: lambda n: (n, 6),
    TensorType.RIGID_BODY_MASS: lambda n: (n,),
    TensorType.RIGID_BODY_ACCELERATION: lambda n: (n, 6),
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
        self.dof_count = 0
        self.body_count = 1
        self.read_calls = 0
        self.write_calls: list[tuple] = []

    def read(self, dst) -> None:
        assert tuple(dst.shape) == tuple(self.shape)
        self.read_calls += 1

    def write(self, tensor, indices=None, mask=None) -> None:
        assert tuple(tensor.shape) == tuple(self.shape)
        self.write_calls.append((indices, mask))


class _FakePhysX:
    """Fake ``PhysX`` whose ``create_tensor_binding`` hands back ``_FakeBinding`` instances."""

    def __init__(self, n: int = 3, unavailable: set | None = None):
        self.n = n
        self._unavailable = unavailable or set()
        self.created: list = []

    def create_tensor_binding(self, *, pattern, tensor_type):
        self.created.append(tensor_type)
        if tensor_type in self._unavailable:
            # The wheel returns a 0-count binding when nothing matches.
            b = _FakeBinding(tensor_type, 0)
            return b
        return _FakeBinding(tensor_type, self.n)


def _make_view(n: int = 3, unavailable: set | None = None) -> OvPhysxView:
    return OvPhysxView(_FakePhysX(n=n, unavailable=unavailable), pattern="/World/env_*/body", device="cpu")


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
    # case-insensitive
    assert resolve_tensor_type("RIGID_BODY_POSE") is TensorType.RIGID_BODY_POSE


def test_resolve_unknown_name_raises():
    with pytest.raises(OvPhysxViewError):
        resolve_tensor_type("not_a_real_attribute")


def test_read_only_classification():
    assert is_read_only("articulation_jacobian")
    assert is_read_only("rigid_body_acceleration")
    assert not is_read_only("articulation_dof_stiffness")
    assert not is_read_only("rigid_body_pose")


# -----------------------------------------------------------------------------
# View dispatch (fake physx)
# -----------------------------------------------------------------------------


def test_get_attribute_reads_into_sized_buffer_and_reuses_it():
    view = _make_view(n=4)
    buf = view.get_attribute("rigid_body_pose")
    assert tuple(buf.shape) == (4, 7)
    binding = view._bindings[TensorType.RIGID_BODY_POSE]
    assert binding.read_calls == 1
    # second read reuses the same cached buffer object
    buf2 = view.get_attribute("rigid_body_pose")
    assert buf2 is buf
    assert binding.read_calls == 2


def test_get_attribute_out_param_receives_copy():
    view = _make_view(n=2)
    out = wp.zeros((2, 7), dtype=wp.float32, device="cpu")
    ret = view.get_attribute("rigid_body_pose", out=out)
    assert ret is out


def test_set_attribute_forwards_indices_and_mask():
    view = _make_view(n=3)
    values = wp.zeros((3, 7), dtype=wp.float32, device="cpu")
    idx = wp.array([0, 2], dtype=wp.int32, device="cpu")
    view.set_attribute("rigid_body_pose", values, indices=idx)
    binding = view._bindings[TensorType.RIGID_BODY_POSE]
    assert binding.write_calls == [(idx, None)]


def test_set_attribute_read_only_raises_and_does_not_write():
    view = _make_view(n=3)
    values = wp.zeros((3, 6), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxViewError, match="read-only"):
        view.set_attribute("rigid_body_acceleration", values)
    assert TensorType.RIGID_BODY_ACCELERATION not in view._bindings


def test_set_attribute_shape_mismatch_raises():
    view = _make_view(n=3)
    wrong = wp.zeros((3, 6), dtype=wp.float32, device="cpu")
    with pytest.raises(OvPhysxViewError, match="Shape mismatch"):
        view.set_attribute("rigid_body_pose", wrong)


def test_unknown_attribute_raises_on_access():
    view = _make_view()
    with pytest.raises(OvPhysxViewError):
        view.get_attribute("totally_made_up")


def test_unavailable_binding_reports_clear_error():
    view = _make_view(n=3, unavailable={TensorType.RIGID_BODY_VELOCITY})
    with pytest.raises(OvPhysxViewError, match="not available"):
        view.get_attribute("rigid_body_velocity")


def test_discoverability_surface():
    view = _make_view()
    assert "rigid_body_pose" in view
    assert view.has_attribute("articulation_dof_stiffness")
    assert not view.has_attribute("nope")
    assert "rigid_body_pose" in view.attribute_names
    # available_attributes only lists instantiated bindings
    assert view.available_attributes == []
    view.get_attribute("rigid_body_pose")
    assert view.available_attributes == ["rigid_body_pose"]


def test_metadata_passthrough_from_sample_binding():
    view = _make_view(n=5)
    # metadata before any access raises a clear error
    with pytest.raises(OvPhysxViewError):
        _ = view.count
    view.get_attribute("rigid_body_pose")
    assert view.count == 5
    assert len(view.prim_paths) == 5
