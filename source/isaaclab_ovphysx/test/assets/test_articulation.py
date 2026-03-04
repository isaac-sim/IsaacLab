# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the ovphysx articulation backend.

These tests run a real ovphysx simulation using a test USD file and verify
that the articulation lifecycle (init, step, read state, write state, reset)
works end-to-end through the OvPhysxManager and the TensorBindingsAPI.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import warp as wp

import ovphysx

wp.init()

DEVICE = "cuda:0"


def gpu_read(binding) -> np.ndarray:
    """Read a binding into a GPU warp array, return as numpy for assertions."""
    buf = wp.zeros(binding.shape, dtype=wp.float32, device=DEVICE)
    binding.read(buf)
    return buf.numpy()


def gpu_write(binding, np_data: np.ndarray):
    """Write numpy data through a GPU warp array into a binding."""
    wp_buf = wp.from_numpy(np_data.astype(np.float32), dtype=wp.float32, device=DEVICE)
    binding.write(wp_buf)

TWO_ARTICULATIONS_USD = os.path.join(
    os.path.expanduser("~"), "physics_backup", "omni", "ovphysx", "tests", "data", "two_articulations.usda"
)


@pytest.fixture(scope="module")
def physx_cpu():
    """Create a GPU ovphysx instance loaded with the two-articulations scene.

    Named 'physx_cpu' for historical reasons; uses GPU so all tests in a
    single pytest process share the same device mode (ovphysx locks device
    mode process-wide on first create_instance call).
    """
    px = ovphysx.PhysX(device="gpu", gpu_index=0)
    usd_h, op = px.add_usd(TWO_ARTICULATIONS_USD)
    px.wait_op(op)
    px.warmup_gpu()
    yield px
    px.release()


class TestTensorBindingsSmoke:
    """Smoke tests that tensor bindings work for articulations."""

    def test_create_root_pose_binding(self, physx_cpu):
        b = physx_cpu.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        )
        assert b.count == 2, "Expected 2 articulations matching the pattern"
        assert b.shape == (2, 7)
        buf = gpu_read(b)
        assert not np.allclose(buf, 0.0), "Root poses should be non-zero after load"
        b.destroy()

    def test_create_dof_position_binding(self, physx_cpu):
        b = physx_cpu.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
        )
        assert b.dof_count == 2, "Each articulation has 2 revolute joints"
        assert b.shape == (2, 2)
        b.destroy()

    def test_step_and_read(self, physx_cpu):
        pose_b = physx_cpu.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        )
        buf_before = gpu_read(pose_b)

        for _ in range(10):
            op = physx_cpu.step(dt=1.0 / 60.0, sim_time=0.0)
            physx_cpu.wait_op(op)

        buf_after = gpu_read(pose_b)

        np.testing.assert_allclose(buf_before, buf_after, atol=1e-3,
                                   err_msg="Fixed-base root poses should not change significantly")
        pose_b.destroy()

    def test_write_dof_position_target(self, physx_cpu):
        tgt_b = physx_cpu.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_TARGET_F32,
        )
        gpu_write(tgt_b, np.full(tgt_b.shape, 0.5, dtype=np.float32))

        # Step so the target takes effect
        for _ in range(60):
            op = physx_cpu.step(dt=1.0 / 60.0, sim_time=0.0)
            physx_cpu.wait_op(op)

        pos_b = physx_cpu.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
        )
        pos = gpu_read(pos_b)
        assert np.any(np.abs(pos) > 0.01), "Joints should have moved toward the position target"
        tgt_b.destroy()
        pos_b.destroy()

    def test_body_metadata(self, physx_cpu):
        b = physx_cpu.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_LINK_POSE_F32,
        )
        assert b.body_count == 3, "articulation has 3 links"
        assert b.count == 1, "Only one articulation matches this exact pattern"
        assert len(b.body_names) == 3
        assert len(b.dof_names) == 2
        b.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
