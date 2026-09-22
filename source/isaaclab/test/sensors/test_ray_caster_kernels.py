# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ray caster Warp kernels.

The kernels in ``sensors/ray_caster/kernels.py`` and ``utils/warp/kernels.py`` are launched directly with
hand-crafted arrays and compared against NumPy reference computations. No simulation is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.sensors.ray_caster.kernels import (
    ALIGNMENT_BASE,
    ALIGNMENT_WORLD,
    ALIGNMENT_YAW,
    apply_z_drift_kernel,
    compute_distance_to_image_plane_to_image_masked_kernel,
    copy_float2d_to_image1_depth_clipped_masked_kernel,
    fill_ray_hits_distance_inf_kernel,
    quat_yaw_only,
    update_ray_caster_kernel,
)
from isaaclab.utils.math import quat_from_euler_xyz, yaw_quat
from isaaclab.utils.warp.kernels import raycast_dynamic_meshes_kernel, raycast_mesh_masked_kernel

pytestmark = pytest.mark.unit

wp.init()
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
ATOL = 1e-5
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


@wp.kernel(enable_backward=False)
def _quat_yaw_only_kernel(q_in: wp.array(dtype=wp.quatf), q_out: wp.array(dtype=wp.quatf)):
    tid = wp.tid()
    q_out[tid] = quat_yaw_only(q_in[tid])


def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Euler angles (intrinsic XYZ) to a quaternion in (x, y, z, w) convention."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _quat_rotate(q_xyzw, v) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q_xyzw``."""
    q_vec = np.asarray(q_xyzw[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(q_vec, v)
    return v + q_xyzw[3] * t + np.cross(q_vec, t)


def _quat_mul(q1, q2) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _yaw_only(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _make_flat_mesh(size: float = 4.0) -> wp.Mesh:
    """Square mesh in the XY plane at z=0, centered at the origin, normal facing +z."""
    half = size / 2.0
    vertices = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=DEVICE),
        indices=wp.array(indices, dtype=wp.int32, device=DEVICE),
    )


def _wp(values, dtype) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device=DEVICE)


"""
update_ray_caster_kernel
"""


def _reference_update(mode, view_pos, view_quat, offset_pos, offset_quat, drift, ray_cast_drift, start, direction):
    """NumPy reference for :func:`update_ray_caster_kernel` for a single environment and ray."""
    quat = _quat_mul(view_quat, offset_quat)
    pos = np.asarray(view_pos) + _quat_rotate(view_quat, offset_pos) + np.asarray(drift)
    if mode == ALIGNMENT_WORLD:
        rot = IDENTITY_QUAT
        direction_w = np.asarray(direction, dtype=np.float64)
    elif mode == ALIGNMENT_YAW:
        rot = _yaw_only(quat)
        direction_w = np.asarray(direction, dtype=np.float64)
    else:
        rot = quat
        direction_w = _quat_rotate(quat, direction)
    rot_drift = _quat_rotate(rot, ray_cast_drift)
    pos_drifted = pos + np.array([rot_drift[0], rot_drift[1], 0.0])
    return pos, quat, _quat_rotate(rot, start) + pos_drifted, direction_w


def _launch_update(
    mode, env_mask, view_pos, view_quat, offset_pos, offset_quat, drift, ray_cast_drift, start, direction
):
    """Launch the kernel for ``len(env_mask)`` envs sharing the same inputs; outputs are pre-filled with 999."""
    num_envs = len(env_mask)
    transforms = _wp([[*view_pos, *view_quat]] * num_envs, wp.transformf)
    outputs = [
        _wp(np.full((num_envs, 3), 999.0), wp.vec3f),
        _wp(np.full((num_envs, 4), 999.0), wp.quatf),
        _wp(np.full((num_envs, 1, 3), 999.0), wp.vec3f),
        _wp(np.full((num_envs, 1, 3), 999.0), wp.vec3f),
    ]
    wp.launch(
        update_ray_caster_kernel,
        dim=(num_envs, 1),
        inputs=[
            transforms,
            wp.array(np.asarray(env_mask, dtype=np.bool_), dtype=wp.bool, device=DEVICE),
            _wp([offset_pos] * num_envs, wp.vec3f),
            _wp([offset_quat] * num_envs, wp.quatf),
            _wp([drift] * num_envs, wp.vec3f),
            _wp([ray_cast_drift] * num_envs, wp.vec3f),
            _wp([[start]] * num_envs, wp.vec3f),
            _wp([[direction]] * num_envs, wp.vec3f),
            int(mode),
        ],
        outputs=outputs,
        device=DEVICE,
    )
    return [out.numpy() for out in outputs]


_UPDATE_SCENARIOS = {
    "identity": dict(view_pos=(0, 0, 0), view_quat=IDENTITY_QUAT, start=(1, 2, 3)),
    "yaw_offset": dict(view_pos=(1, 0, 2), view_quat=_euler_to_quat_xyzw(0, 0, math.pi / 2), offset_pos=(0, 1, 0)),
    "pitch_yaw_drift": dict(
        view_pos=(0, 0, 3),
        view_quat=_euler_to_quat_xyzw(0.2, math.pi / 6, math.pi / 2),
        offset_quat=_euler_to_quat_xyzw(0.1, 0, 0.3),
        drift=(0, 0, 1.5),
        ray_cast_drift=(0.5, 0.3, 0.7),
        start=(1, 0, 0),
    ),
}


@pytest.mark.parametrize("mode", [ALIGNMENT_WORLD, ALIGNMENT_YAW, ALIGNMENT_BASE], ids=["world", "yaw", "base"])
@pytest.mark.parametrize("scenario", list(_UPDATE_SCENARIOS))
def test_update_ray_caster_kernel_matches_reference(mode, scenario):
    """Sensor pose, drift, and per-mode ray alignment match the NumPy reference."""
    kwargs = dict(
        view_pos=(0, 0, 0),
        view_quat=IDENTITY_QUAT,
        offset_pos=(0, 0, 0),
        offset_quat=IDENTITY_QUAT,
        drift=(0, 0, 0),
        ray_cast_drift=(0, 0, 0),
        start=(0, 0, 0),
        direction=(0, 0, -1),
    )
    kwargs.update(_UPDATE_SCENARIOS[scenario])
    pos_w, quat_w, starts_w, dirs_w = _launch_update(mode, [True], **kwargs)
    ref_pos, ref_quat, ref_start, ref_dir = _reference_update(mode, **kwargs)
    np.testing.assert_allclose(pos_w[0], ref_pos, atol=ATOL)
    np.testing.assert_allclose(quat_w[0], ref_quat, atol=ATOL)
    np.testing.assert_allclose(starts_w[0, 0], ref_start, atol=ATOL)
    np.testing.assert_allclose(dirs_w[0, 0], ref_dir, atol=ATOL)


def test_update_ray_caster_kernel_skips_masked_envs():
    """Masked-out environments keep their pre-filled output values."""
    yaw90 = _euler_to_quat_xyzw(0, 0, math.pi / 2)
    pos_w, quat_w, starts_w, dirs_w = _launch_update(
        ALIGNMENT_BASE,
        [False, True],
        view_pos=(0, 0, 2),
        view_quat=yaw90,
        offset_pos=(0, 0, 0),
        offset_quat=IDENTITY_QUAT,
        drift=(0, 0, 0),
        ray_cast_drift=(0, 0, 0),
        start=(1, 0, 0),
        direction=(0, 0, -1),
    )
    for out in (pos_w[0], quat_w[0], starts_w[0, 0], dirs_w[0, 0]):
        assert np.all(out == 999.0)
    np.testing.assert_allclose(pos_w[1], [0, 0, 2], atol=ATOL)
    np.testing.assert_allclose(quat_w[1], yaw90, atol=ATOL)
    np.testing.assert_allclose(starts_w[1, 0], _quat_rotate(yaw90, (1, 0, 0)) + [0, 0, 2], atol=ATOL)
    np.testing.assert_allclose(dirs_w[1, 0], [0, 0, -1], atol=ATOL)


"""
raycast_mesh_masked_kernel
"""


@pytest.mark.parametrize("return_distance, return_normal", [(0, 0), (1, 0), (1, 1)])
def test_raycast_mesh_masked_kernel(return_distance, return_normal):
    """Only the requested outputs are written, and only for active environments."""
    mesh = _make_flat_mesh()
    sentinel = -2.0
    starts = _wp([[[0, 0, 10], [1, 1, 10]]] * 2, wp.vec3f)
    dirs = _wp([[[0, 0, -1], [0, 0, -1]]] * 2, wp.vec3f)
    hits = _wp(np.full((2, 2, 3), np.inf), wp.vec3f)
    dist = _wp(np.full((2, 2), sentinel), wp.float32)
    normals = _wp(np.full((2, 2, 3), sentinel), wp.vec3f)
    env_mask = wp.array([True, False], dtype=wp.bool, device=DEVICE)

    wp.launch(
        raycast_mesh_masked_kernel,
        dim=(2, 2),
        inputs=[mesh.id, env_mask, starts, dirs, 1e6, return_distance, return_normal, hits, dist, normals],
        device=DEVICE,
    )

    np.testing.assert_allclose(hits.numpy()[0], [[0, 0, 0], [1, 1, 0]], atol=ATOL)
    assert np.isinf(hits.numpy()[1]).all(), "masked env hits must remain inf"
    expected_dist = [10.0, 10.0] if return_distance else [sentinel, sentinel]
    np.testing.assert_allclose(dist.numpy()[0], expected_dist, atol=ATOL)
    assert np.all(dist.numpy()[1] == sentinel)
    expected_normal = [[0, 0, 1]] * 2 if return_normal else [[sentinel] * 3] * 2
    np.testing.assert_allclose(normals.numpy()[0], expected_normal, atol=ATOL)
    assert np.all(normals.numpy()[1] == sentinel)


"""
raycast_dynamic_meshes_kernel
"""


def _launch_dynamic(env_mask, mesh_ids, ray_starts, ray_dirs, mesh_pos, mesh_rot, fill=np.inf) -> dict[str, np.ndarray]:
    num_envs, num_rays = np.asarray(ray_starts).shape[:2]
    outputs = {
        "hits": _wp(np.full((num_envs, num_rays, 3), fill), wp.vec3f),
        "distance": _wp(np.full((num_envs, num_rays), fill), wp.float32),
        "normal": _wp(np.full((num_envs, num_rays, 3), fill), wp.vec3f),
        "face_id": wp.array(np.full((num_envs, num_rays), -1, dtype=np.int32), dtype=wp.int32, device=DEVICE),
        "mesh_id": wp.array(np.full((num_envs, num_rays), -1, dtype=np.int16), dtype=wp.int16, device=DEVICE),
    }
    wp.launch(
        raycast_dynamic_meshes_kernel,
        dim=(len(mesh_ids[0]), num_envs, num_rays),
        inputs=[
            wp.array(np.asarray(env_mask, dtype=np.bool_), dtype=wp.bool, device=DEVICE),
            wp.array(np.asarray(mesh_ids, dtype=np.uint64), dtype=wp.uint64, device=DEVICE),
            _wp(ray_starts, wp.vec3f),
            _wp(ray_dirs, wp.vec3f),
            *outputs.values(),
            _wp(mesh_pos, wp.vec3f),
            _wp(mesh_rot, wp.quatf),
            1e6,
            1,  # return_normal
            1,  # return_face_id
            1,  # return_mesh_id
        ],
        device=DEVICE,
    )
    return {name: out.numpy() for name, out in outputs.items()}


def test_raycast_dynamic_meshes_env_mask_skipping():
    """A masked-out environment keeps sentinel values; the active one hits the mesh."""
    mesh = _make_flat_mesh()
    out = _launch_dynamic(
        env_mask=[False, True],
        mesh_ids=[[mesh.id], [mesh.id]],
        ray_starts=[[[0, 0, 10]]] * 2,
        ray_dirs=[[[0, 0, -1]]] * 2,
        mesh_pos=[[[0, 0, 2]]] * 2,
        mesh_rot=[[IDENTITY_QUAT]] * 2,
        fill=999.0,
    )
    np.testing.assert_allclose(out["hits"][0, 0], [999] * 3)
    assert out["distance"][0, 0] == 999.0 and out["face_id"][0, 0] == -1 and out["mesh_id"][0, 0] == -1
    np.testing.assert_allclose(out["hits"][1, 0], [0, 0, 2], atol=ATOL)
    assert out["distance"][1, 0] == pytest.approx(8.0, abs=ATOL)
    assert out["mesh_id"][1, 0] == 0


def test_raycast_dynamic_meshes_closest_hit_and_transform():
    """The closer of two meshes wins, and mesh poses are applied before intersection."""
    mesh_a, mesh_b = _make_flat_mesh(), _make_flat_mesh()
    # mesh_b at z=4 is closer to the ray origin at z=10 than mesh_a at z=2
    out = _launch_dynamic(
        env_mask=[True],
        mesh_ids=[[mesh_a.id, mesh_b.id]],
        ray_starts=[[[0, 0, 10]]],
        ray_dirs=[[[0, 0, -1]]],
        mesh_pos=[[[0, 0, 2], [0, 0, 4]]],
        mesh_rot=[[IDENTITY_QUAT, IDENTITY_QUAT]],
    )
    np.testing.assert_allclose(out["hits"][0, 0], [0, 0, 4], atol=ATOL)
    assert out["distance"][0, 0] == pytest.approx(6.0, abs=ATOL)
    np.testing.assert_allclose(out["normal"][0, 0], [0, 0, 1], atol=ATOL)
    assert out["mesh_id"][0, 0] == 1

    # a 90 deg rotation about Y turns the quad into a vertical plane at x=5 with normal +x
    rot90y = _euler_to_quat_xyzw(0, math.pi / 2, 0)
    out = _launch_dynamic(
        env_mask=[True],
        mesh_ids=[[mesh_a.id]],
        ray_starts=[[[10, 0, 0]]],
        ray_dirs=[[[-1, 0, 0]]],
        mesh_pos=[[[5, 0, 0]]],
        mesh_rot=[[rot90y]],
    )
    np.testing.assert_allclose(out["hits"][0, 0], [5, 0, 0], atol=ATOL)
    assert out["distance"][0, 0] == pytest.approx(5.0, abs=ATOL)
    np.testing.assert_allclose(out["normal"][0, 0], [1, 0, 0], atol=ATOL)


def test_raycast_dynamic_meshes_equidistant_meshes():
    """Equidistant meshes always give the correct hit position; the mesh id may come from either (warp#1058)."""
    mesh_a, mesh_b = _make_flat_mesh(), _make_flat_mesh()
    out = _launch_dynamic(
        env_mask=[True],
        mesh_ids=[[mesh_a.id, mesh_b.id]],
        ray_starts=[[[0, 0, 10]]],
        ray_dirs=[[[0, 0, -1]]],
        mesh_pos=[[[0, 0, 3], [0, 0, 3]]],
        mesh_rot=[[IDENTITY_QUAT, IDENTITY_QUAT]],
    )
    np.testing.assert_allclose(out["hits"][0, 0], [0, 0, 3], atol=ATOL)
    assert out["distance"][0, 0] == pytest.approx(7.0, abs=ATOL)
    assert out["mesh_id"][0, 0] in (0, 1)


"""
Buffer fill / copy kernels
"""


def test_fill_ray_hits_distance_inf_kernel():
    """Active environments are filled with infinity; masked ones keep their values."""
    env_mask = wp.array([False, True], dtype=wp.bool, device=DEVICE)
    hits = _wp(np.arange(12).reshape(2, 2, 3), wp.vec3f)
    distance = _wp([[1.0, 2.0], [3.0, 4.0]], wp.float32)
    normals = _wp(np.ones((2, 2, 3)), wp.vec3f)

    wp.launch(
        fill_ray_hits_distance_inf_kernel,
        dim=(2, 2),
        inputs=[env_mask, True],
        outputs=[hits, distance, normals],
        device=DEVICE,
    )

    np.testing.assert_allclose(hits.numpy()[0], np.arange(6).reshape(2, 3))
    np.testing.assert_allclose(distance.numpy()[0], [1.0, 2.0])
    np.testing.assert_allclose(normals.numpy()[0], np.ones((2, 3)))
    for out in (hits, distance, normals):
        assert np.isinf(out.numpy()[1]).all()


def test_apply_z_drift_kernel():
    """Only the z-component of the drift is applied, and only to active environments."""
    env_mask = wp.array([True, False], dtype=wp.bool, device=DEVICE)
    drift = _wp([[0.5, 0.3, 1.5], [0.5, 0.3, 1.5]], wp.vec3f)
    hits = _wp([[[3.0, 4.0, 5.0]], [[3.0, 4.0, 5.0]]], wp.vec3f)

    wp.launch(apply_z_drift_kernel, dim=(2, 1), inputs=[env_mask, drift], outputs=[hits], device=DEVICE)

    np.testing.assert_allclose(hits.numpy()[0, 0], [3.0, 4.0, 6.5], atol=ATOL)
    np.testing.assert_allclose(hits.numpy()[1, 0], [3.0, 4.0, 5.0], atol=ATOL)


def test_compute_distance_to_image_plane_kernel():
    """Distance-to-image-plane is projected, clipped, reshaped, and masked in one kernel."""
    # env 0 is masked; env 1: ray 0 at 3 m stays, ray 1 at 7 m exceeds max_dist=5 and is filled with 0
    env_mask = wp.array([False, True], dtype=wp.bool, device=DEVICE)
    quat_w = _wp([IDENTITY_QUAT] * 2, wp.quatf)
    ray_distance = _wp([[1.0, 2.0], [3.0, 7.0]], wp.float32)
    ray_directions_w = _wp([[[1, 0, 0], [1, 0, 0]]] * 2, wp.vec3f)
    dst = _wp(np.full((2, 1, 2, 1), -1.0), wp.float32)

    wp.launch(
        compute_distance_to_image_plane_to_image_masked_kernel,
        dim=(2, 2),
        inputs=[env_mask, quat_w, ray_distance, ray_directions_w, 2, True, 5.0, 0.0],
        outputs=[dst],
        device=DEVICE,
    )
    np.testing.assert_allclose(dst.numpy()[..., 0], [[[-1.0, -1.0]], [[3.0, 0.0]]], atol=ATOL)

    # off-axis camera (pitched 45 deg about Y) looking at a ray going world -Z, unclipped; inf is clipped to fill
    quat_w = _wp([_euler_to_quat_xyzw(0, math.pi / 4, 0), IDENTITY_QUAT], wp.quatf)
    ray_distance = _wp([[10.0], [np.inf]], wp.float32)
    ray_directions_w = _wp([[[0, 0, -1]], [[1, 0, 0]]], wp.vec3f)
    dst = _wp(np.full((2, 1, 1, 1), -1.0), wp.float32)
    wp.launch(
        compute_distance_to_image_plane_to_image_masked_kernel,
        dim=(2, 1),
        inputs=[
            wp.array([True, True], dtype=wp.bool, device=DEVICE),
            quat_w,
            ray_distance,
            ray_directions_w,
            1,
            True,
            1e6,
            0.0,
        ],
        outputs=[dst],
        device=DEVICE,
    )
    assert dst.numpy()[0, 0, 0, 0] == pytest.approx(10.0 * math.sin(math.pi / 4), abs=ATOL)
    assert dst.numpy()[1, 0, 0, 0] == pytest.approx(0.0, abs=ATOL)


def test_copy_float2d_to_image_depth_clipped_kernel():
    """Values beyond ``max_dist`` and NaN are replaced by the fill value while copying to image layout."""
    env_mask = wp.array([True], dtype=wp.bool, device=DEVICE)
    src = _wp([[1.0, 7.0, np.nan]], wp.float32)
    dst = _wp(np.full((1, 1, 3, 1), -1.0), wp.float32)

    wp.launch(
        copy_float2d_to_image1_depth_clipped_masked_kernel,
        dim=(1, 3),
        inputs=[env_mask, src, 3, True, 5.0, 0.0],
        outputs=[dst],
        device=DEVICE,
    )
    np.testing.assert_allclose(dst.numpy()[0, :, :, 0], [[1.0, 0.0, 0.0]], atol=ATOL)


"""
quat_yaw_only
"""


def test_quat_yaw_only():
    """Matches :func:`yaw_quat` for combined roll/pitch/yaw and stays finite and unit-norm at gimbal lock."""
    euler = torch.tensor(
        [[0.0, 0.0, 0.5], [0.0, 0.0, math.pi], [0.3, 0.4, 1.2], [-0.2, 0.6, -1.0], [1.0, 1.0, 0.0]],
        device=DEVICE,
    )
    q_in = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    q_out = wp.zeros(len(euler), dtype=wp.quatf, device=DEVICE)
    wp.launch(_quat_yaw_only_kernel, dim=len(euler), inputs=[wp.from_torch(q_in, dtype=wp.quatf), q_out], device=DEVICE)
    torch.testing.assert_close(wp.to_torch(q_out), yaw_quat(q_in), atol=ATOL, rtol=ATOL)

    gimbal = _wp([_euler_to_quat_xyzw(0, math.pi / 2, 0), _euler_to_quat_xyzw(0, -math.pi / 2, 0)], wp.quatf)
    q_out = wp.zeros(2, dtype=wp.quatf, device=DEVICE)
    wp.launch(_quat_yaw_only_kernel, dim=2, inputs=[gimbal, q_out], device=DEVICE)
    result = q_out.numpy()
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, :2], 0.0, atol=ATOL)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=ATOL)
