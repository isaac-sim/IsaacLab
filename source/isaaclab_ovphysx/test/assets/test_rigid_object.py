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
if not hasattr(_TT_module, "RIGID_BODY_POSE"):
    pytest.skip(
        "ovphysx wheel does not yet expose RIGID_BODY_* TensorTypes",
        allow_module_level=True,
    )

import numpy as np  # noqa: E402
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
    pose_buf = bindings.bindings[TT.RIGID_BODY_POSE]._data  # numpy (4, 7)
    pose_buf[0] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    out = data.root_link_pose_w
    out_np = wp.to_torch(out.warp).cpu().numpy()
    assert out_np[0, 0] == 1.0
    assert out_np[0, 6] == 1.0


def test_root_link_quat_slice_matches_pose():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_POSE]._data[1, 3:7] = [0.5, 0.5, 0.5, 0.5]
    quat = data.root_link_quat_w
    quat_np = wp.to_torch(quat.warp).cpu().numpy()
    assert quat_np[1, 0] == pytest.approx(0.5)
    assert quat_np[1, 3] == pytest.approx(0.5)


def test_root_lin_vel_w_reads_from_binding():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_VELOCITY]._data[0] = [1.5, 2.5, 3.5, 0.1, 0.2, 0.3]
    out = data.root_lin_vel_w
    out_np = wp.to_torch(out.warp).cpu().numpy()
    assert out_np[0, 0] == pytest.approx(1.5)
    assert out_np[0, 2] == pytest.approx(3.5)


def test_invalidate_caches_forces_rebuild_within_same_sim_step():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_POSE]._data[0] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    first = wp.to_torch(data.root_link_pose_w.warp).cpu().numpy().copy()
    assert first[0, 0] == pytest.approx(1.0)
    # Mutate the binding storage in-place AND call _invalidate_caches —
    # without bumping _sim_time. The next read must reflect the new
    # binding contents (i.e. invalidation must reset per-buffer timestamps).
    bindings.bindings[TT.RIGID_BODY_POSE]._data[0, 0] = 99.0
    data._invalidate_caches()
    second = wp.to_torch(data.root_link_pose_w.warp).cpu().numpy()
    assert second[0, 0] == pytest.approx(99.0)


def test_body_link_pose_w_is_root_with_singleton_body_dim():
    data, bindings = _make_data(num_instances=3)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_POSE]._data[2] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
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
    bindings.bindings[TT.RIGID_BODY_POSE]._data[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    g = data.projected_gravity_b
    g_np = wp.to_torch(g.warp).cpu().numpy()
    # World-frame gravity direction is (0, 0, -1) per BaseRigidObjectData
    # convention (unit direction, not signed-magnitude). Identity rotation
    # leaves it unchanged in the body frame.
    assert g_np[0, 2] == pytest.approx(-1.0, rel=1e-4)


def test_body_mass_reads_from_cpu_binding():
    data, bindings = _make_data(num_instances=3)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_MASS]._data[:] = [2.5, 3.0, 4.0]
    m = data.body_mass
    m_np = wp.to_torch(m.warp).cpu().numpy()
    assert m_np.shape == (3, 1)
    assert m_np[1, 0] == pytest.approx(3.0)


def test_body_inertia_reads_from_cpu_binding():
    data, bindings = _make_data(num_instances=2)
    data.is_primed = True
    bindings.bindings[TT.RIGID_BODY_INERTIA]._data[0] = [1, 0, 0, 0, 2, 0, 0, 0, 3]
    inertia = data.body_inertia
    inertia_np = wp.to_torch(inertia.warp).cpu().numpy()
    assert inertia_np.shape == (2, 1, 9)
    assert inertia_np[0, 0, 4] == pytest.approx(2.0)


def _make_rigid_object_shell(num_instances: int = 4, device: str = "cuda:0"):
    """Mirrors _make_articulation_shell from test_articulation.py."""
    from unittest.mock import MagicMock

    from isaaclab_ovphysx.assets.rigid_object.rigid_object import RigidObject

    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

    obj = object.__new__(RigidObject)
    obj.cfg = RigidObjectCfg(prim_path="/World/object")
    bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=1,
        asset_kind="rigid_object",
    )
    bindings.set_random_data()
    object.__setattr__(obj, "_device", device)
    object.__setattr__(obj, "_ovphysx", MagicMock())
    object.__setattr__(obj, "_bindings", bindings.bindings)
    object.__setattr__(obj, "_num_instances", num_instances)
    object.__setattr__(obj, "_num_bodies", 1)
    object.__setattr__(obj, "_body_names", ["base_link"])
    object.__setattr__(obj, "_initialize_handle", None)
    object.__setattr__(obj, "_invalidate_initialize_handle", None)
    object.__setattr__(obj, "_prim_deletion_handle", None)
    object.__setattr__(obj, "_debug_vis_handle", None)
    data = RigidObjectData(bindings.bindings, device)
    data._num_instances = num_instances
    data._num_bodies = 1
    object.__setattr__(obj, "_data", data)
    return obj, bindings


def test_rigid_object_count_properties():
    obj, _ = _make_rigid_object_shell(num_instances=8)
    assert obj.num_instances == 8
    assert obj.num_bodies == 1
    assert obj.body_names == ["base_link"]
    assert obj.data is not None


def test_create_buffers_allocates_wrench_buf_and_indices():
    obj, _ = _make_rigid_object_shell(num_instances=5)
    obj._create_buffers()
    assert obj._wrench_buf.shape == (5, 1, 9)
    assert obj._ALL_INDICES.shape == (5,)
    assert obj._ALL_BODY_INDICES.shape == (1,)
    assert obj._instantaneous_wrench_composer is not None
    assert obj._permanent_wrench_composer is not None


# ---------------------------------------------------------------------------
# Task 10 — Root state writers
# ---------------------------------------------------------------------------


def _make_writer_shell(num_instances: int = 4, device: str = "cpu"):
    """Like _make_rigid_object_shell but also calls _create_buffers."""
    obj, bindings = _make_rigid_object_shell(num_instances=num_instances, device=device)
    obj._create_buffers()
    return obj, bindings


def test_write_root_pose_to_sim_index_writes_through_binding():
    """Index variant: pose written via _write_root_state ends up in the mock binding."""
    obj, bindings = _make_writer_shell(num_instances=4, device="cpu")
    pose_binding = bindings.bindings[TT.RIGID_BODY_POSE]
    # Write a known pose for env 0 only.
    pose_data = wp.zeros(4, dtype=wp.transformf, device="cpu")
    pose_np = pose_data.numpy()
    pose_np[0] = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    wp.copy(pose_data, wp.from_numpy(pose_np, dtype=wp.transformf, device="cpu"))

    obj.write_root_pose_to_sim_index(root_pose=pose_data, env_ids=[0])

    stored = pose_binding._data
    assert stored[0, 0] == pytest.approx(1.0)
    assert stored[0, 1] == pytest.approx(2.0)
    assert stored[0, 2] == pytest.approx(3.0)
    assert stored[0, 6] == pytest.approx(1.0)


def test_write_root_velocity_to_sim_index_writes_through_binding():
    """Index variant: velocity written via _write_root_state ends up in the mock binding."""
    obj, bindings = _make_writer_shell(num_instances=4, device="cpu")
    vel_binding = bindings.bindings[TT.RIGID_BODY_VELOCITY]

    vel_data = wp.zeros(4, dtype=wp.spatial_vectorf, device="cpu")
    vel_np = vel_data.numpy()
    vel_np[1] = (4.0, 5.0, 6.0, 0.1, 0.2, 0.3)
    wp.copy(vel_data, wp.from_numpy(vel_np, dtype=wp.spatial_vectorf, device="cpu"))

    obj.write_root_velocity_to_sim_index(root_velocity=vel_data, env_ids=[1])

    stored = vel_binding._data
    assert stored[1, 0] == pytest.approx(4.0)
    assert stored[1, 2] == pytest.approx(6.0)
    assert stored[1, 5] == pytest.approx(0.3)


def test_write_root_pose_to_sim_mask_writes_through_binding():
    """Mask variant: only masked environments receive the write."""
    obj, bindings = _make_writer_shell(num_instances=4, device="cpu")
    pose_binding = bindings.bindings[TT.RIGID_BODY_POSE]
    # Zero out the binding storage so we can detect writes.
    pose_binding._data[:] = 0.0

    pose_data = wp.zeros(4, dtype=wp.transformf, device="cpu")
    pose_np = pose_data.numpy()
    pose_np[:] = (7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0)
    wp.copy(pose_data, wp.from_numpy(pose_np, dtype=wp.transformf, device="cpu"))

    # Only apply to env 2.
    mask_np = np.array([False, False, True, False], dtype=np.uint8)
    mask = wp.from_numpy(mask_np, dtype=wp.uint8, device="cpu")

    obj.write_root_pose_to_sim_mask(root_pose=pose_data, env_mask=mask)

    stored = pose_binding._data
    # Env 2 should have the new values.
    assert stored[2, 0] == pytest.approx(7.0)
    assert stored[2, 1] == pytest.approx(8.0)
    # Env 0 should still be zeros.
    assert stored[0, 0] == pytest.approx(0.0)


def test_write_root_com_pose_to_sim_index_invokes_frame_conversion():
    """COM index variant: binding receives data after frame-conversion kernel."""
    obj, bindings = _make_writer_shell(num_instances=4, device="cpu")
    pose_binding = bindings.bindings[TT.RIGID_BODY_POSE]
    # Set zero COM offset (identity transform) so com_pose == link_pose.
    com_binding = bindings.bindings[TT.RIGID_BODY_COM_POSE]
    # Identity transform: pos=(0,0,0), quat=(0,0,0,1)
    com_binding._data[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    com_pose = wp.zeros(4, dtype=wp.transformf, device="cpu")
    com_np = com_pose.numpy()
    com_np[0] = (5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    wp.copy(com_pose, wp.from_numpy(com_np, dtype=wp.transformf, device="cpu"))

    obj.write_root_com_pose_to_sim_index(root_pose=com_pose, env_ids=[0])

    stored = pose_binding._data
    # With identity COM offset, link_pose should equal com_pose.
    assert stored[0, 0] == pytest.approx(5.0, rel=1e-5)
    assert stored[0, 6] == pytest.approx(1.0, rel=1e-5)


def test_write_root_state_to_sim_deprecated_writes_pose_and_velocity():
    """Deprecated compound writer splits [N,13] into pose and velocity."""
    import warnings

    obj, bindings = _make_writer_shell(num_instances=4, device="cpu")
    pose_binding = bindings.bindings[TT.RIGID_BODY_POSE]
    vel_binding = bindings.bindings[TT.RIGID_BODY_VELOCITY]
    pose_binding._data[:] = 0.0
    vel_binding._data[:] = 0.0

    import torch as _torch

    state = _torch.zeros(4, 13)
    state[2, 0] = 11.0  # x position
    state[2, 6] = 1.0  # w quaternion
    state[2, 7] = 3.0  # vx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj.write_root_state_to_sim(state)
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)

    assert pose_binding._data[2, 0] == pytest.approx(11.0)
    assert vel_binding._data[2, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Task 11 — Body property setters
# ---------------------------------------------------------------------------


def test_set_masses_index_writes_through_cpu_binding():
    obj, bindings = _make_rigid_object_shell(num_instances=3, device="cpu")
    obj._create_buffers()
    masses = wp.from_numpy(np.array([7.5], dtype=np.float32), dtype=wp.float32, device=obj._device)
    import torch as _torch

    env_ids = _torch.tensor([1], dtype=_torch.int32, device=obj._device)
    body_ids = _torch.tensor([0], dtype=_torch.int32, device=obj._device)

    obj.set_masses_index(masses=masses, body_ids=body_ids, env_ids=env_ids)

    # RIGID_BODY_MASS is shape (N,) per Marco's contract
    assert bindings.bindings[TT.RIGID_BODY_MASS]._data[1] == pytest.approx(7.5)


def test_set_inertias_index_writes_through_cpu_binding():
    obj, bindings = _make_rigid_object_shell(num_instances=2, device="cpu")
    obj._create_buffers()
    inertia = wp.from_numpy(
        np.array([[1, 0, 0, 0, 2, 0, 0, 0, 3]], dtype=np.float32),
        dtype=wp.float32,
        device=obj._device,
    )
    import torch as _torch

    env_ids = _torch.tensor([0], dtype=_torch.int32, device=obj._device)
    body_ids = _torch.tensor([0], dtype=_torch.int32, device=obj._device)

    obj.set_inertias_index(inertias=inertia, body_ids=body_ids, env_ids=env_ids)

    assert bindings.bindings[TT.RIGID_BODY_INERTIA]._data[0, 4] == pytest.approx(2.0)


def test_set_coms_index_writes_through_cpu_binding():
    obj, bindings = _make_rigid_object_shell(num_instances=2, device="cpu")
    obj._create_buffers()
    com = wp.from_numpy(
        np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        dtype=wp.float32,
        device=obj._device,
    )
    import torch as _torch

    env_ids = _torch.tensor([1], dtype=_torch.int32, device=obj._device)
    body_ids = _torch.tensor([0], dtype=_torch.int32, device=obj._device)

    obj.set_coms_index(coms=com, body_ids=body_ids, env_ids=env_ids)

    assert bindings.bindings[TT.RIGID_BODY_COM_POSE]._data[1, 0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Task 12 — write_data_to_sim wrench application
# ---------------------------------------------------------------------------


def test_write_data_to_sim_applies_composed_wrench():
    import torch

    obj, bindings = _make_rigid_object_shell(num_instances=2)
    obj._create_buffers()
    obj._data.is_primed = True
    # Identity orientation so body-frame == world-frame (kernel is a passthrough).
    bindings.bindings[TT.RIGID_BODY_POSE]._data[:] = [[0, 0, 0, 0, 0, 0, 1]] * 2
    # Push a body-frame force on env 0 via the instantaneous composer.
    force = torch.tensor([[1.0, 0.0, 0.0]], device=obj._device)
    torque = torch.zeros((1, 3), device=obj._device)
    obj.instantaneous_wrench_composer.add_forces_and_torques_index(
        force,
        torque,
        env_ids=torch.tensor([0], dtype=torch.int32, device=obj._device),
        body_ids=torch.tensor([0], dtype=torch.int32, device=obj._device),
    )

    obj.write_data_to_sim()

    written = bindings.bindings[TT.RIGID_BODY_WRENCH]._data
    assert written.shape == (2, 9)
    assert written[0, 0] == pytest.approx(1.0)
    assert written[1, 0] == 0.0
