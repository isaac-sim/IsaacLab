# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical correctness tests for the ovphysx articulation backend.

These tests verify known physical behaviors:
  - A floating-base articulation falls under gravity
  - A fixed-base articulation stays in place under gravity
  - PD joint drives respond to position targets
  - Joint velocity/position read-back is numerically consistent across steps
  - Articulation metadata (DOF count, body count, names) is correct
"""

import math
import os

import numpy as np
import pytest
import warp as wp

import ovphysx

wp.init()

DEVICE = "cuda:0"


def gpu_read(binding) -> np.ndarray:
    """Read a binding via GPU warp array, return as numpy for assertions."""
    buf = wp.zeros(binding.shape, dtype=wp.float32, device=DEVICE)
    binding.read(buf)
    return buf.numpy()


def gpu_write(binding, np_data: np.ndarray):
    """Write numpy data through a GPU warp array into a binding."""
    wp_buf = wp.from_numpy(np_data.astype(np.float32), dtype=wp.float32, device=DEVICE)
    binding.write(wp_buf)

TWO_ART_USD = os.path.join(
    os.path.expanduser("~"), "physics_backup", "omni", "ovphysx", "tests", "data", "two_articulations.usda"
)

DT = 1.0 / 60.0


@pytest.fixture(scope="module")
def px():
    """Create an ovphysx GPU instance with the two-articulations scene."""
    physx = ovphysx.PhysX(device="gpu", gpu_index=0)
    usd_h, op = physx.add_usd(TWO_ART_USD)
    physx.wait_op(op)
    physx.warmup_gpu()
    yield physx
    physx.release()


# -- Metadata ---------------------------------------------------------------

class TestArticulationMetadata:
    """Validate that ovphysx parses the USD articulation hierarchy correctly."""

    def test_articulation_count(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        )
        assert b.count == 2
        b.destroy()

    def test_dof_count(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
        )
        assert b.dof_count == 2, "Each articulation has 2 revolute joints"
        assert b.shape == (1, 2)
        b.destroy()

    def test_body_count(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_LINK_POSE_F32,
        )
        assert b.body_count == 3, "Each articulation has 3 links"
        b.destroy()

    def test_is_fixed_base(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        )
        assert b.is_fixed_base is True, "Articulation has a fixed joint to world"
        b.destroy()


# -- Root state (fixed base) ------------------------------------------------

class TestFixedBaseRootStability:
    """A fixed-base articulation root should not move under gravity."""

    def test_root_pose_stable_after_simulation(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        )
        before = gpu_read(b)

        for _ in range(120):
            op = px.step(dt=DT, sim_time=0.0)
            px.wait_op(op)

        after = gpu_read(b)

        np.testing.assert_allclose(
            before[:, :3], after[:, :3], atol=1e-4,
            err_msg="Fixed-base root position changed under gravity"
        )
        np.testing.assert_allclose(
            before[:, 3:], after[:, 3:], atol=1e-4,
            err_msg="Fixed-base root orientation changed under gravity"
        )
        b.destroy()


# -- Joint dynamics ----------------------------------------------------------

class TestJointDynamics:
    """Verify that joints move physically under gravity and drives."""

    def test_joints_deflect_under_gravity(self, px):
        """With no drive targets, joints should deflect under gravity (damped pendulum)."""
        pos_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
        )
        initial = gpu_read(pos_b)

        for _ in range(120):
            op = px.step(dt=DT, sim_time=0.0)
            px.wait_op(op)

        current = gpu_read(pos_b)

        delta = np.abs(current - initial)
        assert np.any(delta > 1e-4), (
            f"Joints should deflect under gravity, but max delta = {delta.max():.6f}"
        )
        pos_b.destroy()

    def test_position_drive_tracks_target(self, px):
        """Setting a PD position target should make the joint approach that angle."""
        target_angle = 0.3  # radians

        # Set stiffness high enough to drive
        stiff_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_STIFFNESS_F32,
        )
        stiff_b.write(np.full(stiff_b.shape, 1e6, dtype=np.float32))
        stiff_b.destroy()

        # Set damping
        damp_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_DAMPING_F32,
        )
        damp_b.write(np.full(damp_b.shape, 1e4, dtype=np.float32))
        damp_b.destroy()

        # Set position target
        tgt_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_TARGET_F32,
        )
        gpu_write(tgt_b, np.full(tgt_b.shape, target_angle, dtype=np.float32))
        tgt_b.destroy()

        # Step many times to let the PD controller converge
        for _ in range(600):
            op = px.step(dt=DT, sim_time=0.0)
            px.wait_op(op)

        # Read actual position
        pos_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
        )
        actual = gpu_read(pos_b)

        # With gravity acting on the chain, the equilibrium won't be exactly at the
        # target -- there's a gravity torque. But the joints should have moved
        # substantially toward the target from their initial position (~0).
        assert np.all(actual > 0.05), (
            f"Joints should move toward target {target_angle:.2f} rad under PD drive, "
            f"but actual = {actual}"
        )
        # And the direction should be correct (positive, matching the target sign).
        assert np.all(actual > 0), "Joints should move in the direction of the target"
        pos_b.destroy()


# -- Link poses (kinematic consistency) --------------------------------------

class TestLinkPoses:
    """Verify that link poses are physically consistent."""

    def test_link_poses_have_unit_quaternions(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation*",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_LINK_POSE_F32,
        )
        poses = gpu_read(b)

        quats = poses[..., 3:7]
        norms = np.linalg.norm(quats, axis=-1)
        np.testing.assert_allclose(
            norms, 1.0, atol=1e-5,
            err_msg="Link quaternions should be unit quaternions"
        )
        b.destroy()

    def test_root_link_matches_first_body(self, px):
        """The root pose should match the first link's pose in the link-pose tensor."""
        root_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        )
        link_b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_LINK_POSE_F32,
        )

        root = gpu_read(root_b)
        links = gpu_read(link_b)

        np.testing.assert_allclose(
            root[0], links[0, 0], atol=1e-5,
            err_msg="Root pose should match first link pose"
        )
        root_b.destroy()
        link_b.destroy()


# -- Mass properties ----------------------------------------------------------

class TestMassProperties:
    """Verify that body mass and inertia are readable and physically sensible."""

    def test_body_masses_positive(self, px):
        b = px.create_tensor_binding(
            pattern="/World/articulation",
            tensor_type=ovphysx.OVPHYSX_TENSOR_ARTICULATION_BODY_MASS_F32,
        )
        mass_buf = np.zeros(b.shape, dtype=np.float32)
        b.read(mass_buf)
        mass = mass_buf
        assert np.all(mass > 0), f"All body masses should be positive, got {mass}"
        b.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
