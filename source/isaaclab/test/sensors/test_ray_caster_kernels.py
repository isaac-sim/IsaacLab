# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ray caster kernels.

Tests for kernels in ``sensors/ray_caster/kernels.py`` and
``utils/warp/kernels.py``.  Exercised directly with hand-crafted warp arrays
and analytically computed expected outputs.  No simulation, no stage, no
AppLauncher -- just warp and numpy on CPU (or CUDA when available).

See ``test_update_ray_caster_kernel.py`` for tests of
:func:`update_ray_caster_kernel`.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest
import warp as wp

# ---------------------------------------------------------------------------
# Import kernel modules directly (avoids Isaac Sim / Omniverse dependencies)
# ---------------------------------------------------------------------------

_SENSOR_KERNEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "isaaclab",
    "sensors",
    "ray_caster",
    "kernels.py",
)
_spec = importlib.util.spec_from_file_location("ray_caster_kernels", os.path.normpath(_SENSOR_KERNEL_PATH))
_sensor_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sensor_mod)

_WARP_KERNEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "isaaclab",
    "utils",
    "warp",
    "kernels.py",
)
_warp_spec = importlib.util.spec_from_file_location("warp_kernels", os.path.normpath(_WARP_KERNEL_PATH))
_warp_mod = importlib.util.module_from_spec(_warp_spec)
_warp_spec.loader.exec_module(_warp_mod)

compute_distance_to_image_plane_to_image_masked_kernel = (
    _sensor_mod.compute_distance_to_image_plane_to_image_masked_kernel
)
apply_z_drift_kernel = _sensor_mod.apply_z_drift_kernel
copy_float2d_to_image1_depth_clipped_masked_kernel = _sensor_mod.copy_float2d_to_image1_depth_clipped_masked_kernel
fill_ray_hits_distance_inf_kernel = _sensor_mod.fill_ray_hits_distance_inf_kernel
quat_yaw_only = _sensor_mod.quat_yaw_only

raycast_dynamic_meshes_kernel = _warp_mod.raycast_dynamic_meshes_kernel

# ---------------------------------------------------------------------------
# Constants & setup
# ---------------------------------------------------------------------------

wp.init()
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Wrapper kernel for quat_yaw_only (@wp.func cannot be launched directly)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _quat_yaw_only_test_kernel(
    q_in: wp.array(dtype=wp.quatf),
    q_out: wp.array(dtype=wp.quatf),
):
    tid = wp.tid()
    q_out[tid] = quat_yaw_only(q_in[tid])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Euler angles (intrinsic XYZ) to quaternion in xyzw convention."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return (qx, qy, qz, qw)


def _make_flat_mesh(size: float = 4.0) -> wp.Mesh:
    """Create a flat square mesh in the XY plane at z=0, centered at origin."""
    half = size / 2.0
    vertices = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=DEVICE),
        indices=wp.array(indices, dtype=wp.int32, device=DEVICE),
    )


def _to_numpy(a: wp.array) -> np.ndarray:
    """Convert a warp array to numpy, handling GPU arrays transparently."""
    return a.numpy()


# ---------------------------------------------------------------------------
# Tests: fill_ray_hits_distance_inf_kernel
# ---------------------------------------------------------------------------


class TestFillRayHitsDistanceInfKernel:
    """Tests for :func:`fill_ray_hits_distance_inf_kernel`."""

    def test_active_envs_are_filled_and_masked_envs_are_preserved(self):
        """Active environments are filled with infinity while masked environments retain prior values."""
        env_mask = wp.array(np.array([False, True], dtype=np.bool_), dtype=wp.bool, device=DEVICE)
        ray_hits = wp.array(
            np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            dtype=wp.vec3f,
            device=DEVICE,
        )
        ray_distance = wp.array(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            dtype=wp.float32,
            device=DEVICE,
        )
        ray_normals = wp.array(
            np.array(
                [
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            dtype=wp.vec3f,
            device=DEVICE,
        )

        wp.launch(
            fill_ray_hits_distance_inf_kernel,
            dim=(2, 2),
            inputs=[env_mask, True],
            outputs=[ray_hits, ray_distance, ray_normals],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        hits_np = _to_numpy(ray_hits)
        distance_np = _to_numpy(ray_distance)
        normals_np = _to_numpy(ray_normals)

        np.testing.assert_allclose(hits_np[0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], atol=ATOL)
        np.testing.assert_allclose(distance_np[0], [1.0, 2.0], atol=ATOL)
        np.testing.assert_allclose(normals_np[0], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], atol=ATOL)
        assert np.isinf(hits_np[1]).all()
        assert np.isinf(distance_np[1]).all()
        assert np.isinf(normals_np[1]).all()


# ---------------------------------------------------------------------------
# Tests: raycast_dynamic_meshes_kernel
# ---------------------------------------------------------------------------


class TestRaycastDynamicMeshesKernel:
    """Tests for :func:`raycast_dynamic_meshes_kernel` from ``utils/warp/kernels.py``.

    Each test creates trivial warp meshes (flat quads) and verifies raycasting
    results against analytical expectations.
    """

    IDENT_Q = [0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def _launch(
        num_envs: int,
        num_meshes: int,
        num_rays: int,
        env_mask: np.ndarray,
        mesh_ids: np.ndarray,
        ray_starts: np.ndarray,
        ray_dirs: np.ndarray,
        mesh_pos: np.ndarray,
        mesh_rot: np.ndarray,
        max_dist: float = 1e6,
        sentinel: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Build warp arrays, launch kernel, return outputs as numpy dicts."""
        env_mask_wp = wp.array(env_mask.astype(np.bool_), dtype=wp.bool, device=DEVICE)
        mesh_wp = wp.array(mesh_ids, dtype=wp.uint64, device=DEVICE)
        starts_wp = wp.array(ray_starts, dtype=wp.vec3f, device=DEVICE)
        dirs_wp = wp.array(ray_dirs, dtype=wp.vec3f, device=DEVICE)
        mpos_wp = wp.array(mesh_pos, dtype=wp.vec3f, device=DEVICE)
        mrot_wp = wp.array(mesh_rot, dtype=wp.quatf, device=DEVICE)

        fill = sentinel if sentinel is not None else float("inf")

        hits_np = np.full((num_envs, num_rays, 3), fill, dtype=np.float32)
        ray_hits = wp.array(hits_np, dtype=wp.vec3f, device=DEVICE)

        dist_np = np.full((num_envs, num_rays), fill, dtype=np.float32)
        ray_distance = wp.array(dist_np, dtype=wp.float32, device=DEVICE)

        normal_np = np.full((num_envs, num_rays, 3), fill, dtype=np.float32)
        ray_normal = wp.array(normal_np, dtype=wp.vec3f, device=DEVICE)

        face_np = np.full((num_envs, num_rays), -1, dtype=np.int32)
        ray_face_id = wp.array(face_np, dtype=wp.int32, device=DEVICE)

        mesh_id_np = np.full((num_envs, num_rays), -1, dtype=np.int16)
        ray_mesh_id = wp.array(mesh_id_np, dtype=wp.int16, device=DEVICE)

        wp.launch(
            raycast_dynamic_meshes_kernel,
            dim=(num_meshes, num_envs, num_rays),
            inputs=[
                env_mask_wp,
                mesh_wp,
                starts_wp,
                dirs_wp,
                ray_hits,
                ray_distance,
                ray_normal,
                ray_face_id,
                ray_mesh_id,
                mpos_wp,
                mrot_wp,
                max_dist,
                1,  # return_normal
                1,  # return_face_id
                1,  # return_mesh_id
            ],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        return {
            "hits": _to_numpy(ray_hits),
            "distance": _to_numpy(ray_distance),
            "normal": _to_numpy(ray_normal),
            "face_id": _to_numpy(ray_face_id),
            "mesh_id": _to_numpy(ray_mesh_id),
        }

    def test_env_mask_skipping(self):
        """Env 0 masked out -- verify output buffers retain sentinel values."""
        mesh = _make_flat_mesh()
        iq = self.IDENT_Q
        out = self._launch(
            num_envs=2,
            num_meshes=1,
            num_rays=1,
            env_mask=np.array([False, True]),
            mesh_ids=np.array([[mesh.id], [mesh.id]], dtype=np.uint64),
            ray_starts=np.array([[[0, 0, 10]], [[0, 0, 10]]], dtype=np.float32),
            ray_dirs=np.array([[[0, 0, -1]], [[0, 0, -1]]], dtype=np.float32),
            mesh_pos=np.array([[[0, 0, 2]], [[0, 0, 2]]], dtype=np.float32),
            mesh_rot=np.array([[iq], [iq]], dtype=np.float32),
            sentinel=999.0,
        )

        # Env 0 (masked): all outputs retain sentinel / initial fill
        np.testing.assert_allclose(out["hits"][0, 0], [999, 999, 999], atol=ATOL)
        assert out["distance"][0, 0] == pytest.approx(999.0, abs=ATOL)
        np.testing.assert_allclose(out["normal"][0, 0], [999, 999, 999], atol=ATOL)
        assert out["face_id"][0, 0] == -1
        assert out["mesh_id"][0, 0] == -1

        # Env 1 (active): should have hit the mesh at z=2, distance 8
        np.testing.assert_allclose(out["hits"][1, 0], [0, 0, 2], atol=ATOL)
        assert out["distance"][1, 0] == pytest.approx(8.0, abs=ATOL)
        assert out["mesh_id"][1, 0] == 0

    def test_closest_hit_overlapping_meshes(self):
        """Two meshes at different distances -- closer hit wins.

        Mesh A at z=2 (farther), Mesh B at z=4 (closer to ray origin at z=10).
        Ray from (0,0,10) going (0,0,-1).  Expected: hit Mesh B at distance 6.
        """
        mesh_a = _make_flat_mesh()
        mesh_b = _make_flat_mesh()
        iq = self.IDENT_Q

        out = self._launch(
            num_envs=1,
            num_meshes=2,
            num_rays=1,
            env_mask=np.array([True]),
            mesh_ids=np.array([[mesh_a.id, mesh_b.id]], dtype=np.uint64),
            ray_starts=np.array([[[0, 0, 10]]], dtype=np.float32),
            ray_dirs=np.array([[[0, 0, -1]]], dtype=np.float32),
            mesh_pos=np.array([[[0, 0, 2], [0, 0, 4]]], dtype=np.float32),
            mesh_rot=np.array([[iq, iq]], dtype=np.float32),
        )

        np.testing.assert_allclose(out["hits"][0, 0], [0, 0, 4], atol=ATOL)
        assert out["distance"][0, 0] == pytest.approx(6.0, abs=ATOL)
        np.testing.assert_allclose(out["normal"][0, 0], [0, 0, 1], atol=ATOL)
        assert out["mesh_id"][0, 0] == 1  # mesh_b is closer

    def test_mesh_transform_application(self):
        """Mesh translated/rotated -- verify hits in correct world-space coordinates.

        Mesh: flat XY quad at z=0 (local), placed at world (5,0,0) with 90 deg
        Y rotation.  This turns it into a vertical plane at x=5.
        Ray from (10,0,0) going (-1,0,0) should hit at (5,0,0), distance=5.
        World-space normal: local (0,0,1) rotated by 90 deg Y = (1,0,0).
        """
        mesh = _make_flat_mesh()
        rot90y = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]

        out = self._launch(
            num_envs=1,
            num_meshes=1,
            num_rays=1,
            env_mask=np.array([True]),
            mesh_ids=np.array([[mesh.id]], dtype=np.uint64),
            ray_starts=np.array([[[10, 0, 0]]], dtype=np.float32),
            ray_dirs=np.array([[[-1, 0, 0]]], dtype=np.float32),
            mesh_pos=np.array([[[5, 0, 0]]], dtype=np.float32),
            mesh_rot=np.array([[rot90y]], dtype=np.float32),
        )

        np.testing.assert_allclose(out["hits"][0, 0], [5, 0, 0], atol=ATOL)
        assert out["distance"][0, 0] == pytest.approx(5.0, abs=ATOL)
        np.testing.assert_allclose(out["normal"][0, 0], [1, 0, 0], atol=ATOL)

    def test_equidistant_meshes(self):
        """Two meshes at exact same distance -- hit position is always correct.

        Known limitation (warp#1058): when two meshes are equidistant, the
        ``atomic_min`` + equality-check pattern is not fully thread-safe.
        Normals, face_ids, and mesh_ids may come from either mesh.  The hit
        *position* is always correct because both threads compute the same
        world-space point.
        """
        mesh_a = _make_flat_mesh()
        mesh_b = _make_flat_mesh()
        iq = self.IDENT_Q

        out = self._launch(
            num_envs=1,
            num_meshes=2,
            num_rays=1,
            env_mask=np.array([True]),
            mesh_ids=np.array([[mesh_a.id, mesh_b.id]], dtype=np.uint64),
            ray_starts=np.array([[[0, 0, 10]]], dtype=np.float32),
            ray_dirs=np.array([[[0, 0, -1]]], dtype=np.float32),
            mesh_pos=np.array([[[0, 0, 3], [0, 0, 3]]], dtype=np.float32),
            mesh_rot=np.array([[iq, iq]], dtype=np.float32),
        )

        # Position and distance are always correct, even under the race
        np.testing.assert_allclose(out["hits"][0, 0], [0, 0, 3], atol=ATOL)
        assert out["distance"][0, 0] == pytest.approx(7.0, abs=ATOL)
        # mesh_id can be 0 or 1 -- both are valid under the race condition
        assert out["mesh_id"][0, 0] in (0, 1)


class TestComputeDistanceToImagePlaneToImageMaskedKernel:
    """Tests for :func:`compute_distance_to_image_plane_to_image_masked_kernel`."""

    def test_compute_clip_and_copy_to_image(self):
        """Distance-to-image-plane is computed, clipped, reshaped, and masked in one kernel."""
        env_mask = wp.array(np.array([False, True], dtype=np.bool_), dtype=wp.bool, device=DEVICE)
        quat_w = wp.array(np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32), dtype=wp.quatf, device=DEVICE)
        ray_distance = wp.array(np.array([[1.0, 2.0], [3.0, 7.0]], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        ray_directions_w = wp.array(
            np.array(
                [
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            dtype=wp.vec3f,
            device=DEVICE,
        )
        dst = wp.array(np.full((2, 1, 2, 1), -1.0, dtype=np.float32), dtype=wp.float32, device=DEVICE)

        wp.launch(
            compute_distance_to_image_plane_to_image_masked_kernel,
            dim=(2, 2),
            inputs=[env_mask, quat_w, ray_distance, ray_directions_w, 2, True, 5.0, 0.0],
            outputs=[dst],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        dst_np = _to_numpy(dst)
        np.testing.assert_allclose(dst_np[0, :, :, 0], [[-1.0, -1.0]], atol=ATOL)
        np.testing.assert_allclose(dst_np[1, :, :, 0], [[3.0, 0.0]], atol=ATOL)

    def test_off_axis_camera_without_clipping(self):
        """Camera pitched 45 deg around Y, ray going world -Z."""
        env_mask = wp.array(np.array([True], dtype=np.bool_), dtype=wp.bool, device=DEVICE)
        quat_w = wp.array(
            np.array([_euler_to_quat_xyzw(0, math.pi / 4, 0)], dtype=np.float32), dtype=wp.quatf, device=DEVICE
        )
        ray_distance = wp.array(np.array([[10.0]], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        ray_directions_w = wp.array(np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32), dtype=wp.vec3f, device=DEVICE)
        dst = wp.array(np.full((1, 1, 1, 1), -1.0, dtype=np.float32), dtype=wp.float32, device=DEVICE)

        wp.launch(
            compute_distance_to_image_plane_to_image_masked_kernel,
            dim=(1, 1),
            inputs=[env_mask, quat_w, ray_distance, ray_directions_w, 1, False, 5.0, 0.0],
            outputs=[dst],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        expected = 10.0 * math.sin(math.pi / 4)
        assert _to_numpy(dst)[0, 0, 0, 0] == pytest.approx(expected, abs=ATOL)

    def test_inf_distance_is_clipped(self):
        """Inf distance produces NaN through projection and is replaced by fill value."""
        env_mask = wp.array(np.array([True], dtype=np.bool_), dtype=wp.bool, device=DEVICE)
        quat_w = wp.array(np.array([[0, 0, 0, 1]], dtype=np.float32), dtype=wp.quatf, device=DEVICE)
        ray_distance = wp.array(np.array([[float("inf")]], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        ray_directions_w = wp.array(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), dtype=wp.vec3f, device=DEVICE)
        dst = wp.array(np.full((1, 1, 1, 1), -1.0, dtype=np.float32), dtype=wp.float32, device=DEVICE)

        wp.launch(
            compute_distance_to_image_plane_to_image_masked_kernel,
            dim=(1, 1),
            inputs=[env_mask, quat_w, ray_distance, ray_directions_w, 1, True, 5.0, 0.0],
            outputs=[dst],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        assert _to_numpy(dst)[0, 0, 0, 0] == pytest.approx(0.0, abs=ATOL)


class TestCopyFloat2dToImage1DepthClippedMaskedKernel:
    """Tests for :func:`copy_float2d_to_image1_depth_clipped_masked_kernel`."""

    def test_clip_and_copy_to_image(self):
        """Flat distances are optionally clipped while copying to image layout."""
        env_mask = wp.array(np.array([True], dtype=np.bool_), dtype=wp.bool, device=DEVICE)
        src = wp.array(np.array([[1.0, 7.0, np.nan]], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        dst = wp.array(np.full((1, 1, 3, 1), -1.0, dtype=np.float32), dtype=wp.float32, device=DEVICE)

        wp.launch(
            copy_float2d_to_image1_depth_clipped_masked_kernel,
            dim=(1, 3),
            inputs=[env_mask, src, 3, True, 5.0, 0.0],
            outputs=[dst],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(_to_numpy(dst)[0, :, :, 0], [[1.0, 0.0, 0.0]], atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: apply_z_drift_kernel
# ---------------------------------------------------------------------------


class TestApplyZDriftKernel:
    """Tests for :func:`apply_z_drift_kernel`."""

    @staticmethod
    def _launch(
        hits: list[list[list[float]]],
        drift: list[list[float]],
        env_mask: list[bool] | None = None,
    ) -> np.ndarray:
        """Launch kernel and return modified ray_hits as numpy."""
        num_envs = len(hits)
        num_rays = len(hits[0])
        if env_mask is None:
            env_mask = [True] * num_envs

        mask_wp = wp.array(np.array(env_mask, dtype=np.bool_), dtype=wp.bool, device=DEVICE)
        drift_wp = wp.array(np.array(drift, dtype=np.float32), dtype=wp.vec3f, device=DEVICE)
        hits_wp = wp.array(np.array(hits, dtype=np.float32), dtype=wp.vec3f, device=DEVICE)

        wp.launch(
            apply_z_drift_kernel,
            dim=(num_envs, num_rays),
            inputs=[mask_wp, drift_wp],
            outputs=[hits_wp],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        return _to_numpy(hits_wp)

    def test_known_drift(self):
        """ray_cast_drift = (0, 0, 1.5) shifts ray hit z by exactly 1.5."""
        result = self._launch(
            hits=[[[3.0, 4.0, 5.0]]],
            drift=[[0.0, 0.0, 1.5]],
        )
        np.testing.assert_allclose(result[0, 0], [3.0, 4.0, 6.5], atol=ATOL)

    def test_only_z_component(self):
        """Only z-component of drift is applied; x and y are unchanged."""
        result = self._launch(
            hits=[[[3.0, 4.0, 5.0]]],
            drift=[[0.5, 0.3, 1.0]],
        )
        np.testing.assert_allclose(result[0, 0], [3.0, 4.0, 6.0], atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: quat_yaw_only
# ---------------------------------------------------------------------------


class TestQuatYawOnly:
    """Tests for :func:`quat_yaw_only` (a ``@wp.func`` tested via wrapper kernel)."""

    def test_gimbal_lock(self):
        """At pitch = +/-pi/2, atan2 is near-degenerate but should produce a
        finite, unit-norm, pure-yaw quaternion (only z and w components).
        """
        q_down = _euler_to_quat_xyzw(0, math.pi / 2, 0)  # pitch = +pi/2
        q_up = _euler_to_quat_xyzw(0, -math.pi / 2, 0)  # pitch = -pi/2

        q_in_np = np.array([list(q_down), list(q_up)], dtype=np.float32)
        q_in = wp.array(q_in_np, dtype=wp.quatf, device=DEVICE)
        q_out = wp.zeros(2, dtype=wp.quatf, device=DEVICE)

        wp.launch(
            _quat_yaw_only_test_kernel,
            dim=2,
            inputs=[q_in],
            outputs=[q_out],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        result = _to_numpy(q_out)

        for i in range(2):
            qx, qy, qz, qw = result[i]
            # Must be finite (no NaN / inf)
            assert np.isfinite(result[i]).all(), f"Non-finite output at index {i}: {result[i]}"
            # Must be a pure-yaw quaternion: x ~ 0, y ~ 0
            assert abs(qx) < ATOL, f"x-component should be ~0 at gimbal lock, got {qx}"
            assert abs(qy) < ATOL, f"y-component should be ~0 at gimbal lock, got {qy}"
            # Must be unit-norm
            norm = math.sqrt(float(qx) ** 2 + float(qy) ** 2 + float(qz) ** 2 + float(qw) ** 2)
            assert norm == pytest.approx(1.0, abs=ATOL), f"Non-unit quaternion at index {i}: norm={norm}"
