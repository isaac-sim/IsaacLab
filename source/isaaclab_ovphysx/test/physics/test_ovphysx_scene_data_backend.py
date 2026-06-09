# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysxSceneDataBackend (new SceneDataBackend interface, post-#5128)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")


def _make_stub_binding(prim_paths: list[str]) -> SimpleNamespace:
    """Stub an ovphysx ``TensorBinding`` exposing ``shape``, ``count``, ``prim_paths``, and ``read(dst)``."""
    n = len(prim_paths)
    return SimpleNamespace(
        shape=(n, 7),
        count=n,
        prim_paths=list(prim_paths),
        read=lambda dst: None,  # no-op write; transform_count/paths don't trigger reads.
    )


def _bare_backend():
    """Construct an ``OvPhysxSceneDataBackend`` instance bypassing the live-wheel ``__init__``.

    Tests seed ``_rigid_bindings`` and the merged buffer directly, mirroring the
    bypass-init pattern used in ``test_newton_manager_visualization_state.py``.
    """
    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    return object.__new__(OvPhysxSceneDataBackend)


def test_transform_count_sums_across_bindings():
    """``transform_count`` returns the sum of each binding's row count."""
    b = _bare_backend()
    b._rigid_bindings = [
        {
            "pose": _make_stub_binding(["/World/envs/env_0/Cube", "/World/envs/env_1/Cube"]),
            "pose_buf": None,
            "row_offset": 0,
            "row_count": 2,
        },
        {"pose": _make_stub_binding(["/World/envs/env_0/Pole"]), "pose_buf": None, "row_offset": 2, "row_count": 1},
    ]
    assert b.transform_count == 3


def test_transform_paths_concatenates_prim_paths():
    """``transform_paths`` concatenates each binding's ``prim_paths`` in registration order."""
    b = _bare_backend()
    b._rigid_bindings = [
        {
            "pose": _make_stub_binding(["/World/envs/env_0/Cube", "/World/envs/env_1/Cube"]),
            "pose_buf": None,
            "row_offset": 0,
            "row_count": 2,
        },
        {"pose": _make_stub_binding(["/World/envs/env_0/Pole"]), "pose_buf": None, "row_offset": 2, "row_count": 1},
    ]
    assert b.transform_paths == [
        "/World/envs/env_0/Cube",
        "/World/envs/env_1/Cube",
        "/World/envs/env_0/Pole",
    ]


def test_transform_count_zero_when_no_bindings():
    """``transform_count`` returns 0 when the bindings list is empty."""
    b = _bare_backend()
    b._rigid_bindings = []
    assert b.transform_count == 0


def test_transform_paths_empty_when_no_bindings():
    """``transform_paths`` returns an empty list when the bindings list is empty."""
    b = _bare_backend()
    b._rigid_bindings = []
    assert b.transform_paths == []


def test_setup_creates_one_binding_per_distinct_pattern(monkeypatch):
    """``setup(physx, stage, device)`` buckets RigidBodyAPI prims by env-wildcard form.

    For cartpole-shaped scenes (``cart``, ``pole``), expect 2 bindings — one
    per distinct env-relative prim path.
    """
    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()

    # Stage stub: traversal yields four RigidBodyAPI prims (cart/pole across two envs).
    paths = [
        "/World/envs/env_0/Robot/cart",
        "/World/envs/env_0/Robot/pole",
        "/World/envs/env_1/Robot/cart",
        "/World/envs/env_1/Robot/pole",
    ]

    def fake_traverse():
        for p in paths:
            yield SimpleNamespace(
                HasAPI=lambda api: True,
                GetPath=lambda p=p: SimpleNamespace(pathString=p),
            )

    stage = SimpleNamespace(Traverse=fake_traverse)

    created: list[SimpleNamespace] = []

    class FakePhysX:
        def create_tensor_binding(self, pattern, tensor_type):
            shape = (2, 7)  # 2 envs match each pattern
            b = SimpleNamespace(
                pattern=pattern,
                tensor_type=tensor_type,
                shape=shape,
                count=2,
                prim_paths=[],
                read=lambda dst: None,
            )
            created.append(b)
            return b

    # Patch UsdPhysics so HasAPI in the test doesn't depend on the real PXR module.
    import isaaclab_ovphysx.physics.ovphysx_manager as om_mod

    monkeypatch.setattr(om_mod, "UsdPhysics", SimpleNamespace(RigidBodyAPI=object()))

    b.setup(FakePhysX(), stage, "cpu")

    # Cartpole = 2 distinct env-wildcard patterns -> 2 bindings.
    assert len(created) == 2
    assert {c.pattern for c in created} == {
        "/World/envs/env_*/Robot/cart",
        "/World/envs/env_*/Robot/pole",
    }
    # Per-binding row counts sum to 4.
    assert b.transform_count == 4


def test_transforms_reads_each_binding_and_returns_transform_format():
    """``transforms`` writes each binding's poses into the merged buffer at its offset.

    The returned struct is ``SceneDataFormat.Transform`` with ``transforms`` set to
    the merged ``wp.transformf`` array.
    """
    import warp as _wp

    _wp.init()

    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    b._merged_transforms = _wp.zeros((3,), dtype=_wp.transformf, device="cpu")

    # Two bindings: first with 2 rows, second with 1 row.
    buf_a = _wp.zeros((2, 7), dtype=_wp.float32, device="cpu")
    buf_b = _wp.zeros((1, 7), dtype=_wp.float32, device="cpu")

    def fake_read_a(dst):
        # Fill with row-distinct sentinel transforms (pos.x = row index, quat = identity).
        import numpy as np

        host = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        _wp.copy(dst, _wp.from_numpy(host, dtype=_wp.float32, device="cpu").reshape((2, 7)))

    def fake_read_b(dst):
        import numpy as np

        host = np.array([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        _wp.copy(dst, _wp.from_numpy(host, dtype=_wp.float32, device="cpu").reshape((1, 7)))

    # ``pose_buf_transformf`` is the zero-copy transformf view over the float32 staging
    # buffer; production code caches it at setup time. Tests mirror that shape here.
    buf_a_tf = _wp.array(ptr=buf_a.ptr, shape=(2,), dtype=_wp.transformf, device="cpu", copy=False)
    buf_b_tf = _wp.array(ptr=buf_b.ptr, shape=(1,), dtype=_wp.transformf, device="cpu", copy=False)
    b._rigid_bindings = [
        {
            "pattern": "/World/envs/env_*/Cube",
            "pose": SimpleNamespace(read=fake_read_a, prim_paths=["/Cube0", "/Cube1"]),
            "pose_buf": buf_a,
            "pose_buf_transformf": buf_a_tf,
            "row_offset": 0,
            "row_count": 2,
        },
        {
            "pattern": "/World/envs/env_*/Pole",
            "pose": SimpleNamespace(read=fake_read_b, prim_paths=["/Pole"]),
            "pose_buf": buf_b,
            "pose_buf_transformf": buf_b_tf,
            "row_offset": 2,
            "row_count": 1,
        },
    ]

    out = b.transforms
    assert out is b._scene_data
    assert out.transforms is b._merged_transforms

    merged_host = out.transforms.numpy()  # (3,) of transformf -> view as float32 (3, 7) for assertion
    # Each transformf is 7 floats (pos.xyz + quat.xyzw). Verify row 0 / 1 / 2 contents.
    flat = merged_host.view("<f4").reshape((3, 7))
    assert flat[0, 0] == 1.0
    assert flat[1, 0] == 2.0
    assert flat[2, 0] == 3.0


def test_transforms_returns_empty_struct_when_no_bindings():
    """``transforms`` returns the cached struct (transforms None) when nothing is wired."""
    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    out = b.transforms
    assert out is b._scene_data
    assert out.transforms is None


def test_manager_returns_scene_data_backend_instance():
    """``OvPhysxManager.get_scene_data_backend()`` returns the cached singleton."""
    from isaaclab_ovphysx.physics import OvPhysxManager
    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    # Reset class state and inject a fresh backend instance.
    OvPhysxManager._scene_data_backend = OvPhysxSceneDataBackend()
    try:
        out = OvPhysxManager.get_scene_data_backend()
        assert isinstance(out, OvPhysxSceneDataBackend)
        assert out is OvPhysxManager._scene_data_backend
    finally:
        OvPhysxManager._scene_data_backend = None


def test_manager_returns_none_when_backend_uninitialized():
    """Before warmup, ``get_scene_data_backend`` returns the uninitialized ``None``."""
    from isaaclab_ovphysx.physics import OvPhysxManager

    saved = OvPhysxManager._scene_data_backend
    OvPhysxManager._scene_data_backend = None
    try:
        assert OvPhysxManager.get_scene_data_backend() is None
    finally:
        OvPhysxManager._scene_data_backend = saved


def test_setup_continues_when_create_tensor_binding_raises(monkeypatch, caplog):
    """A single failed binding-creation logs a warning and skips that pattern; others proceed."""
    import logging

    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()

    paths = [
        "/World/envs/env_0/Robot/cart",
        "/World/envs/env_0/Robot/pole",
    ]

    def fake_traverse():
        for p in paths:
            yield SimpleNamespace(
                HasAPI=lambda api: True,
                GetPath=lambda p=p: SimpleNamespace(pathString=p),
            )

    stage = SimpleNamespace(Traverse=fake_traverse)

    class FlakyPhysX:
        def create_tensor_binding(self, pattern, tensor_type):
            if pattern.endswith("/cart"):
                raise RuntimeError("simulated wheel-side failure")
            return SimpleNamespace(
                pattern=pattern, tensor_type=tensor_type, shape=(1, 7), count=1, prim_paths=[], read=lambda dst: None
            )

    import isaaclab_ovphysx.physics.ovphysx_manager as om_mod

    monkeypatch.setattr(om_mod, "UsdPhysics", SimpleNamespace(RigidBodyAPI=object()))

    with caplog.at_level(logging.WARNING, logger=om_mod.logger.name):
        b.setup(FlakyPhysX(), stage, "cpu")

    # The pole pattern survived; the cart pattern was logged and skipped.
    assert len(b._rigid_bindings) == 1
    assert b._rigid_bindings[0]["pattern"].endswith("/pole")
    assert any("simulated wheel-side failure" in record.message for record in caplog.records)


def test_transforms_logs_warning_when_a_binding_read_fails(caplog):
    """A failed ``binding.read(dst)`` logs and skips that binding; other bindings still merge."""
    import logging

    import warp as _wp

    _wp.init()

    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    b._merged_transforms = _wp.zeros((2,), dtype=_wp.transformf, device="cpu")

    buf_good = _wp.zeros((1, 7), dtype=_wp.float32, device="cpu")
    buf_bad = _wp.zeros((1, 7), dtype=_wp.float32, device="cpu")
    buf_good_tf = _wp.array(ptr=buf_good.ptr, shape=(1,), dtype=_wp.transformf, device="cpu", copy=False)
    buf_bad_tf = _wp.array(ptr=buf_bad.ptr, shape=(1,), dtype=_wp.transformf, device="cpu", copy=False)

    def good_read(dst):
        import numpy as np

        host = np.array([[7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        _wp.copy(dst, _wp.from_numpy(host, dtype=_wp.float32, device="cpu").reshape((1, 7)))

    def bad_read(dst):
        raise RuntimeError("simulated read failure")

    b._rigid_bindings = [
        {
            "pattern": "/World/envs/env_*/Good",
            "pose": SimpleNamespace(read=good_read, prim_paths=["/Good"]),
            "pose_buf": buf_good,
            "pose_buf_transformf": buf_good_tf,
            "row_offset": 0,
            "row_count": 1,
        },
        {
            "pattern": "/World/envs/env_*/Bad",
            "pose": SimpleNamespace(read=bad_read, prim_paths=["/Bad"]),
            "pose_buf": buf_bad,
            "pose_buf_transformf": buf_bad_tf,
            "row_offset": 1,
            "row_count": 1,
        },
    ]

    import isaaclab_ovphysx.physics.ovphysx_manager as om_mod

    with caplog.at_level(logging.WARNING, logger=om_mod.logger.name):
        out = b.transforms

    assert out is b._scene_data
    assert any("simulated read failure" in record.message for record in caplog.records)
    # Good row was still written; bad row is left at the merged buffer's prior contents (zeros).
    flat = out.transforms.numpy().view("<f4").reshape((2, 7))
    assert flat[0, 0] == 7.0
