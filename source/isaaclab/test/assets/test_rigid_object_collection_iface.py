# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Cross-backend rigid object collection interface contract checks on mocked views."""

from unittest.mock import patch

import pytest
import torch
import warp as wp
from _iface_test_utils import (
    assert_buffers_stale,
    backends_parametrize,
    check_aliases,
    check_data_properties,
    exercise_writer,
    get_rigid_object_collection,
    make_com_data,
    make_data_warp,
    prime_timestamped_properties,
)

from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import BaseRigidObjectCollectionData

pytestmark = pytest.mark.integration

_backends = backends_parametrize()
_production_backends = backends_parametrize("physx", "newton", "ovphysx")
_index_resolution_backends = backends_parametrize("physx", "newton")
_dims = pytest.mark.parametrize("num_instances, num_bodies", [(1, 1), (2, 3)])
_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])


@pytest.fixture
def collection(backend, num_instances, num_bodies, device):
    obj, _ = get_rigid_object_collection(backend, num_instances, num_bodies, device)
    obj.data.update(dt=0.01)
    if backend != "newton":
        yield obj
        return
    # Reading ``body_link_pose_w`` after a write runs ``NewtonManager.forward()`` against a live solver,
    # which the mocked view does not have; the mock supplies the cached poses directly.
    from isaaclab_newton.physics import NewtonManager

    with patch.object(NewtonManager, "forward"):
        yield obj


_BODY_PROPERTIES = {
    "body_link_pose_w": wp.transformf,
    "body_link_vel_w": wp.spatial_vectorf,
    "body_com_pose_w": wp.transformf,
    "body_com_vel_w": wp.spatial_vectorf,
    "body_com_acc_w": wp.spatial_vectorf,
    "body_com_pose_b": wp.transformf,
    "body_link_pos_w": wp.vec3f,
    "body_link_quat_w": wp.quatf,
    "body_link_lin_vel_w": wp.vec3f,
    "body_link_ang_vel_w": wp.vec3f,
    "body_com_pos_w": wp.vec3f,
    "body_com_quat_w": wp.quatf,
    "body_com_lin_vel_w": wp.vec3f,
    "body_com_ang_vel_w": wp.vec3f,
    "body_com_lin_acc_w": wp.vec3f,
    "body_com_ang_acc_w": wp.vec3f,
    "body_com_pos_b": wp.vec3f,
    "body_com_quat_b": wp.quatf,
    "projected_gravity_b": wp.vec3f,
    "heading_w": wp.float32,
    "body_link_lin_vel_b": wp.vec3f,
    "body_link_ang_vel_b": wp.vec3f,
    "body_com_lin_vel_b": wp.vec3f,
    "body_com_ang_vel_b": wp.vec3f,
    "body_mass": wp.float32,
    "default_body_pose": wp.transformf,
    "default_body_vel": wp.spatial_vectorf,
}
_ALIASES = {
    "body_pose_w": "body_link_pose_w",
    "body_pos_w": "body_link_pos_w",
    "body_quat_w": "body_link_quat_w",
    "body_vel_w": "body_com_vel_w",
    "body_lin_vel_w": "body_com_lin_vel_w",
    "body_ang_vel_w": "body_com_ang_vel_w",
    "body_acc_w": "body_com_acc_w",
    "body_lin_acc_w": "body_com_lin_acc_w",
    "body_ang_acc_w": "body_com_ang_acc_w",
    "com_pos_b": "body_com_pos_b",
    "com_quat_b": "body_com_quat_b",
}
# Only the ``body_pose``/``body_velocity`` mask writers accept ``body_mask``; the link/com variants do not.
_POSE_WRITERS = [
    ("write_body_pose_to_sim", "body_poses", wp.transformf, True),
    ("write_body_link_pose_to_sim", "body_poses", wp.transformf, False),
    ("write_body_com_pose_to_sim", "body_poses", wp.transformf, False),
    ("write_body_velocity_to_sim", "body_velocities", wp.spatial_vectorf, True),
    ("write_body_link_velocity_to_sim", "body_velocities", wp.spatial_vectorf, False),
    ("write_body_com_velocity_to_sim", "body_velocities", wp.spatial_vectorf, False),
]
_BODY_WRITERS = [
    ("set_masses", "masses", wp.float32, None),
    ("set_coms", "coms", wp.transformf, None),
    ("set_inertias", "inertias", wp.float32, 9),
]
_BODY_FRAME_VELOCITY_CACHES = [
    ("body_link_lin_vel_b", "_body_link_lin_vel_b"),
    ("body_link_ang_vel_b", "_body_link_ang_vel_b"),
    ("body_com_lin_vel_b", "_body_com_lin_vel_b"),
    ("body_com_ang_vel_b", "_body_com_ang_vel_b"),
]


@_index_resolution_backends
def test_resolve_ids_handle_tensor_views(backend):
    obj, _ = get_rigid_object_collection(backend, num_instances=4, num_bodies=4, device="cpu")
    ids = torch.arange(4, dtype=torch.int32)
    for resolve in (obj._resolve_env_ids, obj._resolve_body_ids):
        assert resolve(ids).shape[0] == 4
        assert resolve(ids[:2]).shape[0] == 2


@_production_backends
@_devices
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reshape_data_to_view_3d(backend, device, dtype):
    num_instances, num_bodies, data_dim = 2, 3, 4
    obj, _ = get_rigid_object_collection(backend, num_instances, num_bodies, device)
    data = torch.arange(num_instances * num_bodies * data_dim, dtype=dtype, device=device).reshape(
        num_instances, num_bodies, data_dim
    )
    expected = data.permute(1, 0, 2).reshape(num_bodies * num_instances, data_dim)

    view = obj.reshape_data_to_view_3d(data, data_dim, device=device)
    assert isinstance(view, torch.Tensor) and view.dtype == dtype and view.device == data.device
    assert view.is_contiguous()
    torch.testing.assert_close(view, expected)

    # A different target device moves the view.
    other_device = "cpu" if device.startswith("cuda") else "cuda:0"
    moved = obj.reshape_data_to_view_3d(data, data_dim, device=other_device)
    assert str(moved.device).startswith(other_device.split(":")[0])
    torch.testing.assert_close(moved.to(data.device), expected)

    if dtype == torch.float32:
        warp_view = obj.reshape_data_to_view_3d(wp.from_torch(data, dtype=wp.float32), data_dim, device=device)
        assert isinstance(warp_view, wp.array)
        assert warp_view.shape == (num_bodies * num_instances, data_dim)
        assert warp_view.dtype == wp.float32
        assert str(warp_view.device) == device
        torch.testing.assert_close(wp.to_torch(warp_view), view)


@_backends
@_dims
@_devices
def test_properties_finders_and_data_layout(backend, num_instances, num_bodies, device, collection):
    obj = collection
    assert isinstance(obj.data, BaseRigidObjectCollectionData)
    assert (obj.num_instances, obj.num_bodies) == (num_instances, num_bodies)
    names = obj.body_names
    assert isinstance(names, list) and len(names) == num_bodies and all(isinstance(n, str) for n in names)
    mask, found = obj.find_bodies(".*")
    assert isinstance(mask, torch.Tensor) and found == names
    assert obj.find_bodies(names[0])[1] == [names[0]]

    expected = {name: ((num_instances, num_bodies), dtype) for name, dtype in _BODY_PROPERTIES.items()}
    expected["body_inertia"] = ((num_instances, num_bodies, 9), wp.float32)
    check_data_properties(obj.data, expected)
    check_aliases(obj.data, _ALIASES)


@_production_backends
def test_find_bodies_returns_legacy_tensor_or_cached_proxy(backend):
    obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
    indices, names = obj.find_bodies(".*")
    proxy, proxy_names = obj.find_bodies(".*", as_proxy=True)
    assert isinstance(indices, torch.Tensor) and indices.dtype == torch.int32
    assert indices.tolist() == proxy.torch.tolist()
    assert names == proxy_names
    assert proxy is obj.find_bodies(".*", as_proxy=True)[0]
    assert proxy.dtype == wp.int32
    assert str(proxy.device) == obj.device
    # The deprecated alias forwards the return mode.
    with pytest.warns(DeprecationWarning):
        alias_proxy, _ = obj.find_objects(".*", as_proxy=True)
    assert alias_proxy is proxy


@_production_backends
def test_pose_write_invalidates_pose_dependent_caches(backend):
    obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
    obj.data.update(dt=0.01)
    pairs = [
        ("body_link_vel_w", "_body_link_vel_w"),
        ("projected_gravity_b", "_projected_gravity_b"),
        ("heading_w", "_heading_w"),
        *_BODY_FRAME_VELOCITY_CACHES,
    ]
    buffers = prime_timestamped_properties(obj.data, pairs)
    obj.write_body_link_pose_to_sim_index(body_poses=make_data_warp((2, 3), "cpu", wp.transformf))
    assert_buffers_stale(obj.data, buffers)


@_production_backends
def test_velocity_write_invalidates_body_frame_caches(backend):
    obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
    obj.data.update(dt=0.01)
    buffers = prime_timestamped_properties(obj.data, _BODY_FRAME_VELOCITY_CACHES)
    obj.write_body_com_velocity_to_sim_index(body_velocities=make_data_warp((2, 3), "cpu", wp.spatial_vectorf))
    assert_buffers_stale(obj.data, buffers)


@_production_backends
@pytest.mark.parametrize("setter_kind", ["index", "mask"])
def test_set_coms_invalidates_same_timestamp_dependents(backend, setter_kind):
    obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
    obj.data.update(dt=0.01)
    pairs = [
        ("body_com_pose_w", "_body_com_pose_w"),
        ("body_link_vel_w", "_body_link_vel_w"),
        *_BODY_FRAME_VELOCITY_CACHES,
        ("body_state_w", "_body_state_w"),
        ("body_link_state_w", "_body_link_state_w"),
        ("body_com_state_w", "_body_com_state_w"),
    ]
    if backend != "newton":
        pairs.append(("body_com_vel_w", "_body_com_vel_w"))
    # Prime public properties before resolving private buffers so Newton allocates its lazy caches.
    buffers = prime_timestamped_properties(obj.data, pairs)
    if backend == "newton":
        buffers += prime_timestamped_properties(obj.data, [("body_com_pose_b", "_body_com_pose_b")])
    coms = make_com_data(backend, (obj.num_instances, obj.num_bodies), "cpu")

    def set_coms() -> None:
        if setter_kind == "index":
            obj.set_coms_index(coms=coms)
        else:
            obj.set_coms_mask(coms=coms)

    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager

        with patch.object(NewtonManager, "add_model_change"):
            set_coms()
    else:
        set_coms()
    assert_buffers_stale(obj.data, buffers)


@_backends
@_dims
@_devices
@pytest.mark.parametrize("name, kwarg, wp_dtype, has_body_mask", _POSE_WRITERS, ids=[w[0] for w in _POSE_WRITERS])
def test_pose_and_velocity_writers(
    backend, num_instances, num_bodies, device, collection, name, kwarg, wp_dtype, has_body_mask
):
    exercise_writer(
        collection,
        name,
        kwarg,
        wp_dtype,
        shape=(num_instances, num_bodies),
        device=device,
        item_axis=("body_ids", "body_mask"),
        mask_item_axis=has_body_mask,
    )


@_backends
@_dims
@_devices
@pytest.mark.parametrize("name, kwarg, wp_dtype, trailing", _BODY_WRITERS, ids=[w[0] for w in _BODY_WRITERS])
def test_body_writers(backend, num_instances, num_bodies, device, collection, name, kwarg, wp_dtype, trailing):
    if backend == "newton" and name == "set_coms":
        pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
    exercise_writer(
        collection,
        name,
        kwarg,
        wp_dtype,
        shape=(num_instances, num_bodies),
        device=device,
        item_axis=("body_ids", "body_mask"),
        trailing=trailing,
    )
