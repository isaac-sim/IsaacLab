# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Cross-backend rigid object interface contract checks on mocked views (no Isaac Sim or physics scene)."""

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
    get_rigid_object,
    make_com_data,
    make_data_warp,
    prime_timestamped_properties,
)

from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData

pytestmark = pytest.mark.integration

_backends = backends_parametrize()
_production_backends = backends_parametrize("physx", "newton", "ovphysx")
_index_resolution_backends = backends_parametrize("physx", "newton")
_dims = pytest.mark.parametrize("num_instances", [1, 2])
_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])


@pytest.fixture
def rigid_object(backend, num_instances, device):
    obj, _ = get_rigid_object(backend, num_instances, device)
    obj.data.update(dt=0.01)
    return obj


_ROOT_PROPERTIES = {
    "root_link_pose_w": wp.transformf,
    "root_link_vel_w": wp.spatial_vectorf,
    "root_com_pose_w": wp.transformf,
    "root_com_vel_w": wp.spatial_vectorf,
    "root_link_pos_w": wp.vec3f,
    "root_link_quat_w": wp.quatf,
    "root_link_lin_vel_w": wp.vec3f,
    "root_link_ang_vel_w": wp.vec3f,
    "root_com_pos_w": wp.vec3f,
    "root_com_quat_w": wp.quatf,
    "root_com_lin_vel_w": wp.vec3f,
    "root_com_ang_vel_w": wp.vec3f,
    "projected_gravity_b": wp.vec3f,
    "heading_w": wp.float32,
    "root_link_lin_vel_b": wp.vec3f,
    "root_link_ang_vel_b": wp.vec3f,
    "root_com_lin_vel_b": wp.vec3f,
    "root_com_ang_vel_b": wp.vec3f,
    "default_root_pose": wp.transformf,
    "default_root_vel": wp.spatial_vectorf,
}
_BODY_PROPERTIES = {
    "body_link_pose_w": wp.transformf,
    "body_link_vel_w": wp.spatial_vectorf,
    "body_com_pose_w": wp.transformf,
    "body_com_vel_w": wp.spatial_vectorf,
    "body_com_acc_w": wp.spatial_vectorf,
    "body_com_pose_b": wp.transformf,
    "body_mass": wp.float32,
    "body_link_pos_w": wp.vec3f,
    "body_link_quat_w": wp.quatf,
    "body_link_lin_vel_w": wp.vec3f,
    "body_link_ang_vel_w": wp.vec3f,
    "body_com_pos_w": wp.vec3f,
    "body_com_quat_w": wp.quatf,
    "body_com_pos_b": wp.vec3f,
    "body_com_quat_b": wp.quatf,
}
_ALIASES = {
    "root_pose_w": "root_link_pose_w",
    "root_pos_w": "root_link_pos_w",
    "root_quat_w": "root_link_quat_w",
    "root_vel_w": "root_com_vel_w",
    "root_lin_vel_w": "root_com_lin_vel_w",
    "root_ang_vel_w": "root_com_ang_vel_w",
    "body_pose_w": "body_link_pose_w",
    "body_pos_w": "body_link_pos_w",
    "body_quat_w": "body_link_quat_w",
    "body_vel_w": "body_com_vel_w",
    "body_lin_vel_w": "body_com_lin_vel_w",
    "body_ang_vel_w": "body_com_ang_vel_w",
}
_ROOT_WRITERS = [
    ("write_root_pose_to_sim", "root_pose", wp.transformf),
    ("write_root_link_pose_to_sim", "root_pose", wp.transformf),
    ("write_root_com_pose_to_sim", "root_pose", wp.transformf),
    ("write_root_velocity_to_sim", "root_velocity", wp.spatial_vectorf),
    ("write_root_link_velocity_to_sim", "root_velocity", wp.spatial_vectorf),
    ("write_root_com_velocity_to_sim", "root_velocity", wp.spatial_vectorf),
]
_BODY_WRITERS = [
    ("set_masses", "masses", wp.float32, None),
    ("set_coms", "coms", wp.transformf, None),
    ("set_inertias", "inertias", wp.float32, 9),
]
_BODY_FRAME_VELOCITY_CACHES = [
    ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
    ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
    ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
    ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
]


@_index_resolution_backends
def test_resolve_env_ids_handles_tensor_views(backend):
    obj, _ = get_rigid_object(backend, num_instances=4, device="cpu")
    env_ids = torch.arange(4, dtype=torch.int32)
    assert obj._resolve_env_ids(env_ids).shape[0] == 4
    assert obj._resolve_env_ids(env_ids[:2]).shape[0] == 2


@_backends
@_dims
@_devices
def test_properties_finders_and_data_layout(backend, num_instances, device, rigid_object):
    obj = rigid_object
    assert isinstance(obj.data, BaseRigidObjectData)
    assert (obj.num_instances, obj.num_bodies) == (num_instances, 1)
    assert isinstance(obj.body_names, list) and len(obj.body_names) == 1 and isinstance(obj.body_names[0], str)
    assert obj.find_bodies(".*") == ([0], obj.body_names)
    assert obj.find_bodies(obj.body_names[0]) == ([0], obj.body_names)

    expected = {name: ((num_instances,), dtype) for name, dtype in _ROOT_PROPERTIES.items()}
    expected |= {name: ((num_instances, 1), dtype) for name, dtype in _BODY_PROPERTIES.items()}
    expected["body_inertia"] = ((num_instances, 1, 9), wp.float32)
    check_data_properties(obj.data, expected)
    check_aliases(obj.data, _ALIASES)


@_production_backends
def test_find_bodies_returns_legacy_list_or_cached_proxy(backend):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
    indices, names = obj.find_bodies(".*")
    proxy, proxy_names = obj.find_bodies(".*", as_proxy=True)
    assert isinstance(indices, list)
    assert indices == proxy.torch.tolist()
    assert names == proxy_names
    assert proxy is obj.find_bodies(".*", as_proxy=True)[0]
    assert proxy.dtype == wp.int32
    assert str(proxy.device) == obj.device


@_production_backends
def test_pose_write_invalidates_pose_dependent_caches(backend):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
    obj.data.update(dt=0.01)
    pairs = [
        ("root_link_vel_w", "_root_link_vel_w"),
        ("projected_gravity_b", "_projected_gravity_b"),
        ("heading_w", "_heading_w"),
        *_BODY_FRAME_VELOCITY_CACHES,
    ]
    buffers = prime_timestamped_properties(obj.data, pairs)
    obj.write_root_link_pose_to_sim_index(root_pose=make_data_warp((2,), "cpu", wp.transformf))
    assert_buffers_stale(obj.data, buffers)


@_production_backends
def test_velocity_write_invalidates_body_frame_caches(backend):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
    obj.data.update(dt=0.01)
    buffers = prime_timestamped_properties(obj.data, _BODY_FRAME_VELOCITY_CACHES)
    obj.write_root_com_velocity_to_sim_index(root_velocity=make_data_warp((2,), "cpu", wp.spatial_vectorf))
    assert_buffers_stale(obj.data, buffers)


@_production_backends
@pytest.mark.parametrize("setter_kind", ["index", "mask"])
def test_set_coms_invalidates_same_timestamp_dependents(backend, setter_kind):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
    obj.data.update(dt=0.01)
    pairs = [
        ("root_com_pose_w", "_root_com_pose_w"),
        ("root_link_vel_w", "_root_link_vel_w"),
        *_BODY_FRAME_VELOCITY_CACHES,
        ("root_state_w", "_root_state_w"),
        ("root_link_state_w", "_root_link_state_w"),
        ("root_com_state_w", "_root_com_state_w"),
    ]
    if backend != "newton":
        pairs.append(("root_com_vel_w", "_root_com_vel_w"))
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
@pytest.mark.parametrize("name, kwarg, wp_dtype", _ROOT_WRITERS, ids=[w[0] for w in _ROOT_WRITERS])
def test_root_writers(backend, num_instances, device, rigid_object, name, kwarg, wp_dtype):
    exercise_writer(rigid_object, name, kwarg, wp_dtype, shape=(num_instances,), device=device)


@_backends
@_dims
@_devices
@pytest.mark.parametrize("name, kwarg, wp_dtype, trailing", _BODY_WRITERS, ids=[w[0] for w in _BODY_WRITERS])
def test_body_writers(backend, num_instances, device, rigid_object, name, kwarg, wp_dtype, trailing):
    if backend == "newton" and name == "set_coms":
        pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
    exercise_writer(
        rigid_object,
        name,
        kwarg,
        wp_dtype,
        shape=(num_instances, 1),
        device=device,
        item_axis=("body_ids", "body_mask"),
        trailing=trailing,
    )
