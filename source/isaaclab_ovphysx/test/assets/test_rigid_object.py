# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-specific tests for the OVPhysX RigidObject and RigidObjectData."""

from __future__ import annotations

import pytest

# Wheel-gated: skip the entire file if the ovphysx wheel does not
# expose RIGID_BODY_* TensorTypes yet. Use a hasattr check rather
# than chained .attr because importorskip().attr raises AttributeError
# rather than skipping when the module imports but the attribute is
# absent.
_TT_module = pytest.importorskip("isaaclab_ovphysx.tensor_types")
if not hasattr(_TT_module, "RIGID_BODY_ROOT_POSE"):
    pytest.skip(
        "ovphysx wheel does not yet expose RIGID_BODY_* TensorTypes",
        allow_module_level=True,
    )

import warp as wp  # noqa: E402
from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.assets.rigid_object.rigid_object_data import RigidObjectData  # noqa: E402
from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet  # noqa: E402


def _make_data(num_instances: int = 4, device: str = "cuda:0"):
    bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=1,
        asset_kind="rigid_object",
    )
    bindings.set_random_data()
    data = RigidObjectData(bindings.bindings, device)
    data._num_instances = num_instances
    data._num_bodies = 1
    return data, bindings


def test_data_basic_counts():
    data, _ = _make_data(num_instances=8)
    assert data._num_instances == 8
    assert data._num_bodies == 1
    assert data._device.startswith("cuda") or data._device == "cpu"
    assert data.is_primed is False


def test_root_link_pose_reads_from_binding():
    data, bindings = _make_data(num_instances=4)
    data.is_primed = True
    pose_buf = bindings.bindings[TT.RIGID_BODY_ROOT_POSE]._data  # numpy (4, 7)
    pose_buf[0] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    out = data.root_link_pose_w
    out_np = wp.to_torch(out.warp).cpu().numpy()
    assert out_np[0, 0] == 1.0
    assert out_np[0, 6] == 1.0


def test_root_link_quat_slice_matches_pose():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_ROOT_POSE]._data[1, 3:7] = [0.5, 0.5, 0.5, 0.5]
    quat = data.root_link_quat_w
    quat_np = wp.to_torch(quat.warp).cpu().numpy()
    assert quat_np[1, 0] == pytest.approx(0.5)
    assert quat_np[1, 3] == pytest.approx(0.5)


def test_root_lin_vel_w_reads_from_binding():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_ROOT_VELOCITY]._data[0] = [1.5, 2.5, 3.5, 0.1, 0.2, 0.3]
    out = data.root_lin_vel_w
    out_np = wp.to_torch(out.warp).cpu().numpy()
    assert out_np[0, 0] == pytest.approx(1.5)
    assert out_np[0, 2] == pytest.approx(3.5)


def test_invalidate_caches_forces_rebuild_within_same_sim_step():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_ROOT_POSE]._data[0] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    first = wp.to_torch(data.root_link_pose_w.warp).cpu().numpy().copy()
    assert first[0, 0] == pytest.approx(1.0)
    # Mutate the binding storage in-place AND call _invalidate_caches —
    # without bumping _sim_time. The next read must reflect the new
    # binding contents (i.e. invalidation must reset per-buffer timestamps).
    bindings.bindings[TT.RIGID_BODY_ROOT_POSE]._data[0, 0] = 99.0
    data._invalidate_caches()
    second = wp.to_torch(data.root_link_pose_w.warp).cpu().numpy()
    assert second[0, 0] == pytest.approx(99.0)


def test_body_link_pose_w_is_root_with_singleton_body_dim():
    data, bindings = _make_data(num_instances=3)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_ROOT_POSE]._data[2] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    body = data.body_link_pose_w
    body_np = wp.to_torch(body.warp).cpu().numpy()
    assert body_np.shape == (3, 1, 7)
    assert body_np[2, 0, 0] == 1.0


def test_body_acc_w_raises_when_wheel_missing_acceleration_binding():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    # Simulate wheel without RIGID_BODY_ACCELERATION by removing the binding.
    bindings.bindings.pop(TT.RIGID_BODY_ACCELERATION, None)
    data._bindings = bindings.bindings
    with pytest.raises(NotImplementedError, match="RIGID_BODY_ACCELERATION"):
        _ = data.body_link_acc_w


def test_projected_gravity_b_identity_at_world_aligned_orientation():
    data, bindings = _make_data(num_instances=1)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_ROOT_POSE]._data[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    g = data.projected_gravity_b
    g_np = wp.to_torch(g.warp).cpu().numpy()
    # World-frame gravity direction is (0, 0, -1) per BaseRigidObjectData
    # convention (unit direction, not signed-magnitude). Identity rotation
    # leaves it unchanged in the body frame.
    assert g_np[0, 2] == pytest.approx(-1.0, rel=1e-4)
