# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`update_ray_caster_kernel`.

These tests exercise the kernel directly with hand-crafted warp arrays and
analytically computed expected outputs.  No simulation, no stage, no AppLauncher
— just warp on CPU (or CUDA when available).
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest
import torch
import warp as wp

# Import the kernel module directly to avoid pulling in the full isaaclab package
# (which requires Isaac Sim / Omniverse dependencies).  The kernel file itself only
# depends on warp.
_KERNEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "isaaclab",
    "sensors",
    "ray_caster",
    "kernels.py",
)
_spec = importlib.util.spec_from_file_location("ray_caster_kernels", os.path.normpath(_KERNEL_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

update_ray_caster_kernel = _mod.update_ray_caster_kernel
ALIGNMENT_WORLD = _mod.ALIGNMENT_WORLD
ALIGNMENT_YAW = _mod.ALIGNMENT_YAW
ALIGNMENT_BASE = _mod.ALIGNMENT_BASE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wp.init()
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
TORCH_DEVICE = torch.device(DEVICE)
ATOL = 1e-5


def _make_transform(pos: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]) -> wp.array:
    """Create a warp transformf array (1,) from position and xyzw quaternion."""
    t = torch.tensor([[pos[0], pos[1], pos[2], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]], device=DEVICE)
    return wp.from_torch(t.contiguous()).view(wp.transformf)


def _identity_quat() -> tuple[float, float, float, float]:
    """Return identity quaternion in xyzw."""
    return (0.0, 0.0, 0.0, 1.0)


def _yaw_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """Pure yaw quaternion in xyzw."""
    return (0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2))


def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Euler angles (intrinsic XYZ) to quaternion in xyzw convention."""
    q = torch.zeros(1, 4)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # xyzw
    q[0, 0] = sr * cp * cy - cr * sp * sy
    q[0, 1] = cr * sp * cy + sr * cp * sy
    q[0, 2] = cr * cp * sy - sr * sp * cy
    q[0, 3] = cr * cp * cy + sr * sp * sy
    return tuple(q[0].tolist())


def _quat_rotate(q_xyzw: tuple, v: tuple) -> np.ndarray:
    """Rotate vector v by quaternion q (xyzw) using numpy."""
    qx, qy, qz, qw = q_xyzw
    # quaternion rotation: v' = q * v * q^-1
    # Using the formula: v' = v + 2*w*(w×v) + 2*(q_vec × (q_vec × v + w*v))
    # Simpler: v' = v + 2w(q×v) + 2(q×(q×v))
    q_vec = np.array([qx, qy, qz])
    v = np.array(v)
    t = 2.0 * np.cross(q_vec, v)
    return v + qw * t + np.cross(q_vec, t)


def _launch_kernel(
    transforms: wp.array,
    env_mask: wp.array,
    offset_pos: wp.array,
    offset_quat: wp.array,
    drift: wp.array,
    ray_cast_drift: wp.array,
    ray_starts_local: wp.array,
    ray_directions_local: wp.array,
    alignment_mode: int,
    num_envs: int,
    num_rays: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Launch the kernel and return (pos_w, quat_w, ray_starts_w, ray_directions_w) as numpy arrays."""
    pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=DEVICE)
    quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=DEVICE)
    ray_starts_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=DEVICE)
    ray_directions_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=DEVICE)

    wp.launch(
        update_ray_caster_kernel,
        dim=(num_envs, num_rays),
        inputs=[
            transforms,
            env_mask,
            offset_pos,
            offset_quat,
            drift,
            ray_cast_drift,
            ray_starts_local,
            ray_directions_local,
            alignment_mode,
        ],
        outputs=[pos_w, quat_w, ray_starts_w, ray_directions_w],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)

    return (
        wp.to_torch(pos_w).cpu().numpy(),
        wp.to_torch(quat_w).cpu().numpy(),
        wp.to_torch(ray_starts_w).cpu().numpy(),
        wp.to_torch(ray_directions_w).cpu().numpy(),
    )


def _make_inputs(
    view_pos=(0.0, 0.0, 0.0),
    view_quat=None,
    offset_pos=(0.0, 0.0, 0.0),
    offset_quat=None,
    drift=(0.0, 0.0, 0.0),
    ray_cast_drift=(0.0, 0.0, 0.0),
    ray_start=(0.0, 0.0, 0.0),
    ray_dir=(0.0, 0.0, -1.0),
    num_envs=1,
):
    """Build all kernel input arrays for a single-ray, single (or multi)-env scenario."""
    if view_quat is None:
        view_quat = _identity_quat()
    if offset_quat is None:
        offset_quat = _identity_quat()

    transforms = _make_transform(view_pos, view_quat)
    if num_envs > 1:
        # Replicate the same transform for all envs
        t_torch = wp.to_torch(transforms).repeat(num_envs, 1)
        transforms = wp.from_torch(t_torch.contiguous()).view(wp.transformf)

    mask_t = torch.ones(num_envs, dtype=torch.bool, device=TORCH_DEVICE)
    env_mask = wp.from_torch(mask_t)

    op = torch.tensor(
        [[offset_pos[0], offset_pos[1], offset_pos[2]]] * num_envs, dtype=torch.float32, device=TORCH_DEVICE
    )
    offset_pos_wp = wp.from_torch(op.contiguous(), dtype=wp.vec3f)

    oq = torch.tensor(
        [[offset_quat[0], offset_quat[1], offset_quat[2], offset_quat[3]]] * num_envs,
        dtype=torch.float32,
        device=TORCH_DEVICE,
    )
    offset_quat_wp = wp.from_torch(oq.contiguous(), dtype=wp.quatf)

    d = torch.tensor([[drift[0], drift[1], drift[2]]] * num_envs, dtype=torch.float32, device=TORCH_DEVICE)
    drift_wp = wp.from_torch(d.contiguous(), dtype=wp.vec3f)

    rcd = torch.tensor(
        [[ray_cast_drift[0], ray_cast_drift[1], ray_cast_drift[2]]] * num_envs, dtype=torch.float32, device=TORCH_DEVICE
    )
    rcd_wp = wp.from_torch(rcd.contiguous(), dtype=wp.vec3f)

    rs = torch.tensor(
        [[[ray_start[0], ray_start[1], ray_start[2]]]] * num_envs, dtype=torch.float32, device=TORCH_DEVICE
    )
    rs_wp = wp.from_torch(rs.contiguous(), dtype=wp.vec3f)

    rd = torch.tensor([[[ray_dir[0], ray_dir[1], ray_dir[2]]]] * num_envs, dtype=torch.float32, device=TORCH_DEVICE)
    rd_wp = wp.from_torch(rd.contiguous(), dtype=wp.vec3f)

    return transforms, env_mask, offset_pos_wp, offset_quat_wp, drift_wp, rcd_wp, rs_wp, rd_wp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateRayCasterKernel:
    """Unit tests for update_ray_caster_kernel launched directly with warp arrays."""

    def test_identity_passthrough(self):
        """All identity/zero inputs → pos_w = origin, quat_w = identity, rays unchanged."""
        inputs = _make_inputs(ray_start=(1.0, 2.0, 3.0), ray_dir=(0.0, 0.0, -1.0))
        pos_w, quat_w, starts_w, dirs_w = _launch_kernel(*inputs, alignment_mode=0, num_envs=1, num_rays=1)

        np.testing.assert_allclose(pos_w[0], [0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(quat_w[0], [0, 0, 0, 1], atol=ATOL)
        # World mode, identity: ray_start_w = local_start + pos (= local_start + origin)
        np.testing.assert_allclose(starts_w[0, 0], [1, 2, 3], atol=ATOL)
        np.testing.assert_allclose(dirs_w[0, 0], [0, 0, -1], atol=ATOL)

        # Same for yaw and base — all should agree at identity
        for mode in [1, 2]:
            inputs = _make_inputs(ray_start=(1.0, 2.0, 3.0), ray_dir=(0.0, 0.0, -1.0))
            _, _, starts_w2, dirs_w2 = _launch_kernel(*inputs, alignment_mode=mode, num_envs=1, num_rays=1)
            np.testing.assert_allclose(starts_w2[0, 0], [1, 2, 3], atol=ATOL)
            np.testing.assert_allclose(dirs_w2[0, 0], [0, 0, -1], atol=ATOL)

    def test_offset_composition(self):
        """View at (1,0,2) yawed 90° + offset (0,1,0) → combined_pos = (1,-1,2).

        90° yaw: quat = (0, 0, sin(45°), cos(45°)) = (0, 0, 0.7071, 0.7071)
        quat_rotate(90°yaw, (0,1,0)) = (-1, 0, 0)  [Y axis maps to -X]
        combined_pos = (1,0,2) + (-1,0,0) = (0,0,2)
        combined_quat = yaw90 * identity = yaw90
        """
        yaw90 = _yaw_quat(math.pi / 2)
        inputs = _make_inputs(
            view_pos=(1.0, 0.0, 2.0),
            view_quat=yaw90,
            offset_pos=(0.0, 1.0, 0.0),
        )
        pos_w, quat_w, _, _ = _launch_kernel(*inputs, alignment_mode=2, num_envs=1, num_rays=1)

        expected_offset_rotated = _quat_rotate(yaw90, (0, 1, 0))  # (-1, 0, 0)
        expected_pos = np.array([1, 0, 2]) + expected_offset_rotated
        np.testing.assert_allclose(pos_w[0], expected_pos, atol=ATOL)
        np.testing.assert_allclose(quat_w[0], list(yaw90), atol=ATOL)

    def test_world_alignment_ignores_rotation(self):
        """World mode: ray starts = local_start + combined_pos, directions unchanged.

        Sensor at (0,0,5), pitched 45° around Y. Local ray at (+1,0,0), direction (0,0,-1).
        World mode should NOT rotate the ray start or direction.
        """
        pitch45 = _euler_to_quat_xyzw(0, math.pi / 4, 0)
        inputs = _make_inputs(
            view_pos=(0.0, 0.0, 5.0),
            view_quat=pitch45,
            ray_start=(1.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        pos_w, _, starts_w, dirs_w = _launch_kernel(*inputs, alignment_mode=0, num_envs=1, num_rays=1)

        # ray_start_w = local_start + combined_pos = (1,0,0) + (0,0,5) = (1,0,5)
        np.testing.assert_allclose(starts_w[0, 0], [1, 0, 5], atol=ATOL)
        # direction unchanged
        np.testing.assert_allclose(dirs_w[0, 0], [0, 0, -1], atol=ATOL)

    def test_yaw_alignment_rotates_starts_only(self):
        """Yaw mode: ray starts rotated by yaw-only quaternion, directions unchanged.

        Sensor yawed 90° + pitched 30°. Local ray start at (+1, 0, 0).
        Yaw-only extracts 90° yaw → rotates (+1,0,0) to (0,+1,0).
        Direction (0,0,-1) is NOT rotated in yaw mode.
        """
        q = _euler_to_quat_xyzw(0, math.pi / 6, math.pi / 2)  # pitch 30°, yaw 90°
        inputs = _make_inputs(
            view_pos=(0.0, 0.0, 3.0),
            view_quat=q,
            ray_start=(1.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        pos_w, _, starts_w, dirs_w = _launch_kernel(*inputs, alignment_mode=1, num_envs=1, num_rays=1)

        # yaw-only of 90° yaw + 30° pitch → pure 90° yaw
        yaw_only = _yaw_quat(math.pi / 2)
        rotated_start = _quat_rotate(yaw_only, (1, 0, 0))  # (0, 1, 0)
        expected_start = rotated_start + np.array([0, 0, 3])  # + combined_pos
        np.testing.assert_allclose(starts_w[0, 0], expected_start, atol=ATOL)

        # direction unchanged in yaw mode
        np.testing.assert_allclose(dirs_w[0, 0], [0, 0, -1], atol=ATOL)

    def test_base_alignment_rotates_starts_and_directions(self):
        """Base mode: both ray starts and directions rotated by full combined quaternion.

        Sensor yawed 90°. Local ray at (+1, 0, 0), direction (0, 0, -1).
        90° yaw rotates:
          (+1,0,0) → (0,+1,0)
          (0,0,-1) → (0,0,-1) [yaw doesn't affect Z-down]
        """
        yaw90 = _yaw_quat(math.pi / 2)
        inputs = _make_inputs(
            view_pos=(0.0, 0.0, 4.0),
            view_quat=yaw90,
            ray_start=(1.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        _, _, starts_w, dirs_w = _launch_kernel(*inputs, alignment_mode=2, num_envs=1, num_rays=1)

        rotated_start = _quat_rotate(yaw90, (1, 0, 0))  # (0, 1, 0)
        expected_start = rotated_start + np.array([0, 0, 4])
        np.testing.assert_allclose(starts_w[0, 0], expected_start, atol=ATOL)

        rotated_dir = _quat_rotate(yaw90, (0, 0, -1))  # (0, 0, -1) — Z unaffected by yaw
        np.testing.assert_allclose(dirs_w[0, 0], rotated_dir, atol=ATOL)

    def test_base_alignment_with_pitch_rotates_direction(self):
        """Base mode with pitch: direction is rotated by the full orientation.

        Sensor pitched 90° around Y (looking forward instead of down).
        Direction (0,0,-1) rotated by 90° pitch around Y → (-1,0,0).
        """
        pitch90 = _euler_to_quat_xyzw(0, math.pi / 2, 0)
        inputs = _make_inputs(
            view_pos=(0.0, 0.0, 2.0),
            view_quat=pitch90,
            ray_start=(0.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        _, _, _, dirs_w = _launch_kernel(*inputs, alignment_mode=2, num_envs=1, num_rays=1)

        rotated_dir = _quat_rotate(pitch90, (0, 0, -1))  # (-1, 0, 0)
        np.testing.assert_allclose(dirs_w[0, 0], rotated_dir, atol=ATOL)

    def test_ray_cast_drift_world_mode(self):
        """World mode: ray_cast_drift XY is added raw to position, Z is NOT applied.

        drift = (0.5, 0.3, 0.7). In world mode:
        pos_drifted = (combined_pos.x + 0.5, combined_pos.y + 0.3, combined_pos.z)
        Note: Z component of ray_cast_drift is NOT added to position in any mode.
        """
        inputs = _make_inputs(
            view_pos=(1.0, 2.0, 3.0),
            ray_cast_drift=(0.5, 0.3, 0.7),
            ray_start=(0.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        _, _, starts_w, dirs_w = _launch_kernel(*inputs, alignment_mode=0, num_envs=1, num_rays=1)

        # World mode: pos_drifted = (1+0.5, 2+0.3, 3) = (1.5, 2.3, 3)
        # ray_start_w = local_start + pos_drifted = (0,0,0) + (1.5, 2.3, 3)
        np.testing.assert_allclose(starts_w[0, 0], [1.5, 2.3, 3.0], atol=ATOL)
        np.testing.assert_allclose(dirs_w[0, 0], [0, 0, -1], atol=ATOL)

    def test_ray_cast_drift_yaw_mode(self):
        """Yaw mode: ray_cast_drift XY is rotated by yaw-only quat, Z is NOT applied.

        Sensor yawed 90°, drift = (1.0, 0.0, 0.5).
        yaw-rotated drift = quat_rotate(yaw90, (1,0,0.5)) — but only XY of the result
        is used for pos_drifted. Actually looking at the kernel:
          rot_drift = quat_rotate(yaw_q, rcd)  # full rotation of the drift vector
          pos_drifted = (combined_pos.x + rot_drift.x, combined_pos.y + rot_drift.y, combined_pos.z)
        So the drift vector is fully rotated, but only XY of the result is added.
        """
        yaw90 = _yaw_quat(math.pi / 2)
        inputs = _make_inputs(
            view_pos=(0.0, 0.0, 5.0),
            view_quat=yaw90,
            ray_cast_drift=(1.0, 0.0, 0.5),
            ray_start=(0.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        _, _, starts_w, _ = _launch_kernel(*inputs, alignment_mode=1, num_envs=1, num_rays=1)

        # yaw90 rotates (1, 0, 0.5) → (0, 1, 0.5) [X→Y under 90° yaw, Z unchanged]
        rot_drift = _quat_rotate(yaw90, (1, 0, 0.5))
        # pos_drifted = (0 + rot_drift.x, 0 + rot_drift.y, 5) — Z from combined_pos
        expected_start = np.array([rot_drift[0], rot_drift[1], 5.0])
        # local_start = (0,0,0), rotated by yaw_q → still (0,0,0)
        np.testing.assert_allclose(starts_w[0, 0], expected_start, atol=ATOL)

    def test_ray_cast_drift_base_mode(self):
        """Base mode: ray_cast_drift XY is rotated by full combined_quat, Z is NOT applied.

        Sensor pitched 90° around Y, drift = (1.0, 0.0, 0.0).
        Full rotation of (1,0,0) by 90° pitch around Y → (0, 0, -1).
        pos_drifted = (combined_pos.x + 0, combined_pos.y + 0, combined_pos.z) — both XY of
        rotated drift happen to be 0 in this case.
        """
        pitch90 = _euler_to_quat_xyzw(0, math.pi / 2, 0)
        inputs = _make_inputs(
            view_pos=(0.0, 0.0, 5.0),
            view_quat=pitch90,
            ray_cast_drift=(1.0, 0.0, 0.0),
            ray_start=(0.0, 0.0, 0.0),
            ray_dir=(0.0, 0.0, -1.0),
        )
        _, _, starts_w, _ = _launch_kernel(*inputs, alignment_mode=2, num_envs=1, num_rays=1)

        rot_drift = _quat_rotate(pitch90, (1, 0, 0))  # (0, 0, -1)
        # pos_drifted = (0 + rot_drift.x, 0 + rot_drift.y, 5) = (0, 0, 5)
        # local_start (0,0,0) rotated by pitch90 → still (0,0,0)
        expected_start = np.array([rot_drift[0], rot_drift[1], 5.0])
        np.testing.assert_allclose(starts_w[0, 0], expected_start, atol=ATOL)

    def test_env_mask_skips_masked_envs(self):
        """Masked-out environments retain sentinel values in output buffers.

        2 envs, env 0 masked out (False), env 1 active (True).
        Output buffers are pre-filled with sentinel (999). After kernel launch,
        env 0 should still have 999, env 1 should have computed values.
        """
        yaw90 = _yaw_quat(math.pi / 2)

        # Build transforms for 2 envs: both at (0,0,2) with yaw90
        t_single = torch.tensor(
            [[0, 0, 2, yaw90[0], yaw90[1], yaw90[2], yaw90[3]]],
            dtype=torch.float32,
            device=DEVICE,
        )
        t_both = t_single.repeat(2, 1).contiguous()
        transforms = wp.from_torch(t_both).view(wp.transformf)

        # Mask: env 0 = False, env 1 = True
        mask_t = torch.tensor([False, True], dtype=torch.bool, device=TORCH_DEVICE)
        env_mask = wp.from_torch(mask_t)

        # Zero offsets and drifts for both envs
        zero3 = torch.zeros(2, 3, dtype=torch.float32, device=TORCH_DEVICE)
        offset_pos_wp = wp.from_torch(zero3.clone().contiguous(), dtype=wp.vec3f)
        iq = torch.tensor([[0, 0, 0, 1]] * 2, dtype=torch.float32, device=TORCH_DEVICE)
        offset_quat_wp = wp.from_torch(iq.contiguous(), dtype=wp.quatf)
        drift_wp = wp.from_torch(zero3.clone().contiguous(), dtype=wp.vec3f)
        rcd_wp = wp.from_torch(zero3.clone().contiguous(), dtype=wp.vec3f)

        # Single ray per env
        rs = torch.tensor([[[1, 0, 0]]] * 2, dtype=torch.float32, device=TORCH_DEVICE)
        rs_wp = wp.from_torch(rs.contiguous(), dtype=wp.vec3f)
        rd = torch.tensor([[[0, 0, -1]]] * 2, dtype=torch.float32, device=TORCH_DEVICE)
        rd_wp = wp.from_torch(rd.contiguous(), dtype=wp.vec3f)

        # Pre-fill outputs with sentinel
        sentinel = 999.0
        pos_w_t = torch.full((2, 3), sentinel, dtype=torch.float32, device=TORCH_DEVICE)
        pos_w = wp.from_torch(pos_w_t.contiguous(), dtype=wp.vec3f)
        quat_w_t = torch.full((2, 4), sentinel, dtype=torch.float32, device=TORCH_DEVICE)
        quat_w = wp.from_torch(quat_w_t.contiguous(), dtype=wp.quatf)
        starts_w_t = torch.full((2, 1, 3), sentinel, dtype=torch.float32, device=TORCH_DEVICE)
        starts_w = wp.from_torch(starts_w_t.contiguous(), dtype=wp.vec3f)
        dirs_w_t = torch.full((2, 1, 3), sentinel, dtype=torch.float32, device=TORCH_DEVICE)
        dirs_w = wp.from_torch(dirs_w_t.contiguous(), dtype=wp.vec3f)

        wp.launch(
            update_ray_caster_kernel,
            dim=(2, 1),
            inputs=[transforms, env_mask, offset_pos_wp, offset_quat_wp, drift_wp, rcd_wp, rs_wp, rd_wp, 2],
            outputs=[pos_w, quat_w, starts_w, dirs_w],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        pos_np = wp.to_torch(pos_w).cpu().numpy()
        quat_np = wp.to_torch(quat_w).cpu().numpy()
        starts_np = wp.to_torch(starts_w).cpu().numpy()
        dirs_np = wp.to_torch(dirs_w).cpu().numpy()

        # Env 0 (masked): all outputs should still be sentinel
        np.testing.assert_allclose(pos_np[0], [sentinel] * 3, atol=ATOL)
        np.testing.assert_allclose(quat_np[0], [sentinel] * 4, atol=ATOL)
        np.testing.assert_allclose(starts_np[0, 0], [sentinel] * 3, atol=ATOL)
        np.testing.assert_allclose(dirs_np[0, 0], [sentinel] * 3, atol=ATOL)

        # Env 1 (active): should have computed values
        np.testing.assert_allclose(pos_np[1], [0, 0, 2], atol=ATOL)
        np.testing.assert_allclose(quat_np[1], list(yaw90), atol=ATOL)
        # Base mode: (1,0,0) rotated by yaw90 = (0,1,0), + pos (0,0,2)
        expected_start = _quat_rotate(yaw90, (1, 0, 0)) + np.array([0, 0, 2])
        np.testing.assert_allclose(starts_np[1, 0], expected_start, atol=ATOL)
        expected_dir = _quat_rotate(yaw90, (0, 0, -1))  # (0, 0, -1) unaffected by yaw
        np.testing.assert_allclose(dirs_np[1, 0], expected_dir, atol=ATOL)

    def test_positional_drift_added_before_alignment(self):
        """The `drift` parameter is added to combined_pos before ray transformation.

        Verify that drift shifts the sensor position (and therefore ray starts)
        equally across all alignment modes.
        """
        drift_val = (0.0, 0.0, 1.5)  # shift up 1.5m
        results = {}
        for mode_name, mode_int in [("world", 0), ("yaw", 1), ("base", 2)]:
            inputs = _make_inputs(
                view_pos=(0.0, 0.0, 3.0),
                drift=drift_val,
                ray_start=(0.0, 0.0, 0.0),
                ray_dir=(0.0, 0.0, -1.0),
            )
            pos_w, _, starts_w, _ = _launch_kernel(*inputs, alignment_mode=mode_int, num_envs=1, num_rays=1)
            results[mode_name] = (pos_w, starts_w)

        # All modes: pos_w should be (0, 0, 4.5) = view_pos + drift
        for mode_name in ["world", "yaw", "base"]:
            np.testing.assert_allclose(
                results[mode_name][0][0],
                [0, 0, 4.5],
                atol=ATOL,
                err_msg=f"{mode_name} mode: pos_w should include drift",
            )
            # ray_start_w Z should also reflect the drifted position
            assert results[mode_name][1][0, 0, 2] == pytest.approx(4.5, abs=ATOL), (
                f"{mode_name} mode: ray start Z should be 4.5"
            )
