# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.test.mock_interfaces.assets import MockRigidObjectCollection
from isaaclab.utils.wrench_composer import WrenchComposer


def create_mock_asset(
    num_envs: int,
    num_bodies: int,
    device: str,
    link_pos: torch.Tensor | None = None,
    link_quat: torch.Tensor | None = None,
) -> MockRigidObjectCollection:
    """Create a MockRigidObjectCollection with optional custom link poses.

    Args:
        num_envs: Number of environments.
        num_bodies: Number of bodies.
        device: Device to use.
        link_pos: Optional link positions (num_envs, num_bodies, 3). Defaults to zeros.
        link_quat: Optional link quaternions in (x, y, z, w) format (num_envs, num_bodies, 4).
                   Defaults to identity quaternion.

    Returns:
        MockRigidObjectCollection with body_link_pose_w set.
    """
    mock = MockRigidObjectCollection(num_instances=num_envs, num_bodies=num_bodies, device=device)

    # Build combined pose (N, B, 7) = pos(3) + quat_xyzw(4) matching wp.transformf layout
    if link_pos is None:
        pos = torch.zeros(num_envs, num_bodies, 3, dtype=torch.float32)
    else:
        pos = link_pos.float()

    if link_quat is None:
        # Identity quaternion in (x, y, z, w) format = (0, 0, 0, 1)
        quat = torch.zeros(num_envs, num_bodies, 4, dtype=torch.float32)
        quat[..., 3] = 1.0
    else:
        quat = link_quat.float()

    pose = torch.cat([pos, quat], dim=-1)  # (N, B, 7)
    mock.data.set_body_link_pose_w(pose)
    return mock


# --- Helper functions for quaternion math ---


def quat_rotate_inv_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion (numpy).

    Args:
        quat_xyzw: Quaternion in (x, y, z, w) format. Shape: (..., 4)
        vec: Vector to rotate. Shape: (..., 3)

    Returns:
        Rotated vector. Shape: (..., 3)
    """
    # Extract components
    xyz = quat_xyzw[..., 0:3]
    w = quat_xyzw[..., 3:4]

    # For inverse rotation, we conjugate the quaternion (negate xyz)
    # q^-1 * v * q = q_conj * v * q_conj^-1 for unit quaternion
    # Using the formula: v' = v + 2*w*(xyz x v) + 2*(xyz x (xyz x v))
    # But for inverse: use -xyz

    # Cross product: xyz x vec
    t = 2.0 * np.cross(-xyz, vec, axis=-1)
    # Result: vec + w*t + xyz x t
    return vec + w * t + np.cross(-xyz, t, axis=-1)


def random_unit_quaternion_np(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """Generate random unit quaternions in (x, y, z, w) format.

    Args:
        rng: Random number generator.
        shape: Output shape, e.g. (num_envs, num_bodies).

    Returns:
        Random unit quaternions. Shape: (*shape, 4)
    """
    # Generate random quaternion components
    q = rng.standard_normal(shape + (4,)).astype(np.float32)
    # Normalize to unit quaternion
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_force(device: str, num_envs: int, num_bodies: int):
    # Initialize random number generator
    rng = np.random.default_rng(seed=0)

    for _ in range(10):
        mock_asset = create_mock_asset(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed force
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            # Get random number of envs and bodies and their indices
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            # Convert to warp arrays
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            # Get random forces
            forces_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            # Add forces to wrench composer
            wrench_composer.add_forces_and_torques_index(forces=forces, body_ids=body_ids, env_ids=env_ids)
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.out_force_b.warp.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_torque(device: str, num_envs: int, num_bodies: int):
    # Initialize random number generator
    rng = np.random.default_rng(seed=1)

    for _ in range(10):
        mock_asset = create_mock_asset(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed torque
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            # Get random number of envs and bodies and their indices
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            # Convert to warp arrays
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            # Get random torques
            torques_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            # Add torques to wrench composer
            wrench_composer.add_forces_and_torques_index(torques=torques, body_ids=body_ids, env_ids=env_ids)
            # Add torques to hand-calculated composed torque
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_at_positions(device: str, num_envs: int, num_bodies: int):
    """Test adding forces at local positions (offset from link frame)."""
    rng = np.random.default_rng(seed=2)

    for _ in range(10):
        # Initialize wrench composer
        mock_asset = create_mock_asset(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed force
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        # Initialize hand-calculated composed torque
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            # Get random number of envs and bodies and their indices
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            # Convert to warp arrays
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            # Get random forces
            forces_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            positions_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
            # Add forces at positions to wrench composer
            wrench_composer.add_forces_and_torques_index(
                forces=forces, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            # Add torques to hand-calculated composed torque: torque = cross(position, force)
            torques_from_forces = np.cross(positions_np, forces_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += torques_from_forces[i, j, :]

        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.out_force_b.warp.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_torques_at_position(device: str, num_envs: int, num_bodies: int):
    rng = np.random.default_rng(seed=3)

    for _ in range(10):
        mock_asset = create_mock_asset(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed torque
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            # Get random number of envs and bodies and their indices
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            # Convert to warp arrays
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            # Get random torques
            torques_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            positions_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
            # Add torques at positions to wrench composer
            wrench_composer.add_forces_and_torques_index(
                torques=torques, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            # Add torques to hand-calculated composed torque
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_and_torques_at_position(device: str, num_envs: int, num_bodies: int):
    """Test adding forces and torques at local positions."""
    rng = np.random.default_rng(seed=4)

    for _ in range(10):
        mock_asset = create_mock_asset(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed force and torque
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            # Get random number of envs and bodies and their indices
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            # Convert to warp arrays
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            # Get random forces and torques
            forces_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            torques_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            positions_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
            # Add forces and torques at positions to wrench composer
            wrench_composer.add_forces_and_torques_index(
                forces=forces, torques=torques, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            # Add torques to hand-calculated composed torque: torque = cross(position, force) + torque
            torques_from_forces = np.cross(positions_np, forces_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += torques_from_forces[i, j, :]
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.out_force_b.warp.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_reset(device: str, num_envs: int, num_bodies: int):
    rng = np.random.default_rng(seed=5)
    for _ in range(10):
        mock_asset = create_mock_asset(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Get random number of envs and bodies and their indices
        num_envs_np = rng.integers(1, num_envs, endpoint=True)
        num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
        env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
        body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
        # Convert to warp arrays
        env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
        body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
        # Get random forces and torques
        forces_np = (
            np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
            .reshape(num_envs_np, num_bodies_np, 3)
            .astype(np.float32)
        )
        torques_np = (
            np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
            .reshape(num_envs_np, num_bodies_np, 3)
            .astype(np.float32)
        )
        forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
        torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
        # Add forces and torques to wrench composer
        wrench_composer.add_forces_and_torques_index(forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids)
        # Reset wrench composer
        wrench_composer.reset()
        # Check all 7 buffers are zero (5 input + 2 output)
        zeros = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        assert np.allclose(wrench_composer.global_force_w.numpy(), zeros, atol=1, rtol=1e-7)
        assert np.allclose(wrench_composer.global_torque_w.numpy(), zeros, atol=1, rtol=1e-7)
        assert np.allclose(wrench_composer.global_force_at_com_w.numpy(), zeros, atol=1, rtol=1e-7)
        assert np.allclose(wrench_composer.local_force_b.numpy(), zeros, atol=1, rtol=1e-7)
        assert np.allclose(wrench_composer.local_torque_b.numpy(), zeros, atol=1, rtol=1e-7)
        assert np.allclose(wrench_composer.out_force_b.warp.numpy(), zeros, atol=1, rtol=1e-7)
        assert np.allclose(wrench_composer.out_torque_b.warp.numpy(), zeros, atol=1, rtol=1e-7)


# ============================================================================
# Global Frame Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_global_forces_with_rotation(device: str, num_envs: int, num_bodies: int):
    """Test that global forces are correctly rotated to the local frame."""
    rng = np.random.default_rng(seed=10)

    for _ in range(5):
        # Create random link quaternions
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_quat_torch = torch.from_numpy(link_quat_np)

        # Create mock asset with custom quaternions
        mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global forces for all envs and bodies
        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

        # Apply global forces
        wrench_composer.add_forces_and_torques_index(forces=forces_global, is_global=True)

        # Compute expected local forces by rotating global forces by inverse quaternion
        expected_forces_local = quat_rotate_inv_np(link_quat_np, forces_global_np)

        # Check raw global buffer has the global forces
        global_force_np = wrench_composer.global_force_at_com_w.numpy()
        assert np.allclose(global_force_np, forces_global_np, atol=1e-4, rtol=1e-5), (
            f"Global force buffer mismatch.\nExpected:\n{forces_global_np}\nGot:\n{global_force_np}"
        )

        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()

        # Verify
        composed_force_np = wrench_composer.out_force_b.warp.numpy()
        assert np.allclose(composed_force_np, expected_forces_local, atol=1e-4, rtol=1e-5), (
            f"Global force rotation failed.\nExpected:\n{expected_forces_local}\nGot:\n{composed_force_np}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_global_torques_with_rotation(device: str, num_envs: int, num_bodies: int):
    """Test that global torques are correctly rotated to the local frame."""
    rng = np.random.default_rng(seed=11)

    for _ in range(5):
        # Create random link quaternions
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_quat_torch = torch.from_numpy(link_quat_np)

        # Create mock asset with custom quaternions
        mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global torques
        torques_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        torques_global = wp.from_numpy(torques_global_np, dtype=wp.vec3f, device=device)

        # Apply global torques
        wrench_composer.add_forces_and_torques_index(torques=torques_global, is_global=True)

        # Compute expected local torques
        expected_torques_local = quat_rotate_inv_np(link_quat_np, torques_global_np)

        # Check raw global buffer has the global torques
        global_torque_np = wrench_composer.global_torque_w.numpy()
        assert np.allclose(global_torque_np, torques_global_np, atol=1e-4, rtol=1e-5), (
            f"Global torque buffer mismatch.\nExpected:\n{torques_global_np}\nGot:\n{global_torque_np}"
        )

        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()

        # Verify
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()
        assert np.allclose(composed_torque_np, expected_torques_local, atol=1e-4, rtol=1e-5), (
            f"Global torque rotation failed.\nExpected:\n{expected_torques_local}\nGot:\n{composed_torque_np}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_global_forces_at_global_position(device: str, num_envs: int, num_bodies: int):
    """Test global forces at global positions with full coordinate transformation."""
    rng = np.random.default_rng(seed=12)

    for _ in range(5):
        # Create random link poses
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        # Create mock asset
        mock_asset = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global forces and positions
        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_global_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)
        positions_global = wp.from_numpy(positions_global_np, dtype=wp.vec3f, device=device)

        # Apply global forces at global positions
        wrench_composer.add_forces_and_torques_index(forces=forces_global, positions=positions_global, is_global=True)

        # Compute expected results:
        # 1. Force in local frame = quat_rotate_inv(link_quat, global_force)
        expected_forces_local = quat_rotate_inv_np(link_quat_np, forces_global_np)

        # 2. Torque about CoM in world frame = cross(P_global - link_pos, F_global)
        #    Then rotate to body frame
        position_offset_global = positions_global_np - link_pos_np
        expected_torques_local = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for i in range(num_envs):
            for j in range(num_bodies):
                torque_w = np.cross(position_offset_global[i, j], forces_global_np[i, j])
                expected_torques_local[i, j] = quat_rotate_inv_np(
                    link_quat_np[i : i + 1, j : j + 1], torque_w.reshape(1, 1, 3)
                )[0, 0]

        # Check raw global force buffer has the global forces
        global_force_np = wrench_composer.global_force_w.numpy()
        assert np.allclose(global_force_np, forces_global_np, atol=1e-4, rtol=1e-5), (
            f"Global force buffer mismatch.\nExpected:\n{forces_global_np}\nGot:\n{global_force_np}"
        )

        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()

        # Verify forces
        composed_force_np = wrench_composer.out_force_b.warp.numpy()
        assert np.allclose(composed_force_np, expected_forces_local, atol=1e-3, rtol=1e-4), (
            f"Global force at position failed.\nExpected forces:\n{expected_forces_local}\nGot:\n{composed_force_np}"
        )

        # Verify torques
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()
        assert np.allclose(composed_torque_np, expected_torques_local, atol=1e-3, rtol=1e-4), (
            f"Global force at position failed.\nExpected torques:\n{expected_torques_local}\nGot:\n{composed_torque_np}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_local_vs_global_identity_quaternion(device: str):
    """Test that local and global give same result with identity quaternion and zero position."""
    rng = np.random.default_rng(seed=13)
    num_envs, num_bodies = 10, 5

    # Create mock with identity pose (default)
    mock_asset_local = create_mock_asset(num_envs, num_bodies, device)
    mock_asset_global = create_mock_asset(num_envs, num_bodies, device)

    wrench_composer_local = WrenchComposer(mock_asset_local)
    wrench_composer_global = WrenchComposer(mock_asset_global)

    # Generate random forces and torques
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

    # Apply as local
    wrench_composer_local.add_forces_and_torques_index(forces=forces, torques=torques, is_global=False)

    # Apply as global (should be same with identity quaternion)
    wrench_composer_global.add_forces_and_torques_index(forces=forces, torques=torques, is_global=True)

    # Compose to body frame before checking output
    wrench_composer_local.compose_to_body_frame()
    wrench_composer_global.compose_to_body_frame()

    # Results should be identical
    assert np.allclose(
        wrench_composer_local.out_force_b.warp.numpy(),
        wrench_composer_global.out_force_b.warp.numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        wrench_composer_local.out_torque_b.warp.numpy(),
        wrench_composer_global.out_torque_b.warp.numpy(),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_90_degree_rotation_global_force(device: str):
    """Test global force with a known 90-degree rotation for easy verification."""
    num_envs, num_bodies = 1, 1

    # 90-degree rotation around Z-axis: (x, y, z, w) = (0, 0, sin(45°), cos(45°))
    # This rotates X -> Y, Y -> -X
    angle = np.pi / 2
    link_quat_np = np.array([[[[0, 0, np.sin(angle / 2), np.cos(angle / 2)]]]], dtype=np.float32).reshape(1, 1, 4)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Apply force in global +X direction
    force_global = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    force_wp = wp.from_numpy(force_global, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques_index(forces=force_wp, is_global=True)

    # Expected: After inverse rotation (rotate by -90° around Z), X becomes -Y
    # Actually, inverse rotation of +90° around Z applied to (1,0,0) gives (0,-1,0)
    expected_force_local = np.array([[[0.0, -1.0, 0.0]]], dtype=np.float32)

    # Compose to body frame before checking output
    wrench_composer.compose_to_body_frame()

    composed_force_np = wrench_composer.out_force_b.warp.numpy()
    assert np.allclose(composed_force_np, expected_force_local, atol=1e-5), (
        f"90-degree rotation test failed.\nExpected:\n{expected_force_local}\nGot:\n{composed_force_np}"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composition_mixed_local_and_global(device: str):
    """Test that local and global forces can be composed together correctly."""
    rng = np.random.default_rng(seed=14)
    num_envs, num_bodies = 5, 3

    # Create random link quaternions
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Generate random local and global forces
    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

    forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    # Add local forces first
    wrench_composer.add_forces_and_torques_index(forces=forces_local, is_global=False)

    # Add global forces
    wrench_composer.add_forces_and_torques_index(forces=forces_global, is_global=True)

    # Expected: local forces stay as-is, global forces get rotated, then sum
    global_forces_in_local = quat_rotate_inv_np(link_quat_np, forces_global_np)
    expected_total = forces_local_np + global_forces_in_local

    # Check raw buffer properties
    local_force_np = wrench_composer.local_force_b.numpy()
    assert np.allclose(local_force_np, forces_local_np, atol=1e-4, rtol=1e-5)
    global_force_at_com_np = wrench_composer.global_force_at_com_w.numpy()
    assert np.allclose(global_force_at_com_np, forces_global_np, atol=1e-4, rtol=1e-5)

    # Compose to body frame before checking output
    wrench_composer.compose_to_body_frame()

    composed_force_np = wrench_composer.out_force_b.warp.numpy()
    assert np.allclose(composed_force_np, expected_total, atol=1e-4, rtol=1e-5), (
        f"Mixed local/global composition failed.\nExpected:\n{expected_total}\nGot:\n{composed_force_np}"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_local_forces_at_local_position(device: str, num_envs: int, num_bodies: int):
    """Test local forces at local positions (offset from link frame)."""
    rng = np.random.default_rng(seed=15)

    for _ in range(5):
        # Create random link poses (shouldn't affect local frame calculations)
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random local forces and local positions (offsets)
        forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_local_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
        positions_local = wp.from_numpy(positions_local_np, dtype=wp.vec3f, device=device)

        # Apply local forces at local positions
        wrench_composer.add_forces_and_torques_index(forces=forces_local, positions=positions_local, is_global=False)

        # Expected: forces stay as-is, torque = cross(position, force)
        expected_forces = forces_local_np
        expected_torques = np.cross(positions_local_np, forces_local_np)

        # Check raw local buffer
        local_force_np = wrench_composer.local_force_b.numpy()
        assert np.allclose(local_force_np, expected_forces, atol=1e-4, rtol=1e-5)

        # Compose to body frame before checking output
        wrench_composer.compose_to_body_frame()

        # Verify
        composed_force_np = wrench_composer.out_force_b.warp.numpy()
        composed_torque_np = wrench_composer.out_torque_b.warp.numpy()

        assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
        assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_at_link_origin_no_torque(device: str):
    """Test that a global force applied at the link origin produces no torque."""
    rng = np.random.default_rng(seed=16)
    num_envs, num_bodies = 5, 3

    # Create random link poses
    link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch = torch.from_numpy(link_pos_np)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Generate random global forces
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    # Position = link position (so offset is zero)
    positions_at_link = wp.from_numpy(link_pos_np, dtype=wp.vec3f, device=device)

    # Apply global forces at link origin
    wrench_composer.add_forces_and_torques_index(forces=forces_global, positions=positions_at_link, is_global=True)

    # Expected: force rotated to local, torque = 0 (since position offset is zero)
    expected_forces = quat_rotate_inv_np(link_quat_np, forces_global_np)
    expected_torques = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

    # Check raw global force buffer
    global_force_np = wrench_composer.global_force_w.numpy()
    assert np.allclose(global_force_np, forces_global_np, atol=1e-4, rtol=1e-5)

    # Compose to body frame before checking output
    wrench_composer.compose_to_body_frame()

    composed_force_np = wrench_composer.out_force_b.warp.numpy()
    composed_torque_np = wrench_composer.out_torque_b.warp.numpy()

    assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
    assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)


# ============================================================================
# add_raw_buffers_from Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_add_raw_buffers_from(device: str, num_envs: int, num_bodies: int):
    """Test that add_raw_buffers_from merges all five input buffers correctly."""
    rng = np.random.default_rng(seed=20)

    # Create two composers with random link poses
    link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch = torch.from_numpy(link_pos_np)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_a = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    mock_b = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)

    composer_a = WrenchComposer(mock_a)
    composer_b = WrenchComposer(mock_b)

    # Populate composer_a with local forces at positions
    forces_local_a_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_local_a_np = rng.uniform(-5.0, 5.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_a.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_local_a_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_local_a_np, dtype=wp.vec3f, device=device),
        is_global=False,
    )

    # Populate composer_b with global forces at global positions
    forces_global_b_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_global_b_np = rng.uniform(-5.0, 5.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_b.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_global_b_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_global_b_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Merge b into a
    composer_a.add_raw_buffers_from(composer_b)

    # Build a reference composer that receives both calls directly
    mock_ref = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    composer_ref = WrenchComposer(mock_ref)
    composer_ref.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_local_a_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_local_a_np, dtype=wp.vec3f, device=device),
        is_global=False,
    )
    composer_ref.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_global_b_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_global_b_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Compose both and compare
    composer_a.compose_to_body_frame()
    composer_ref.compose_to_body_frame()

    assert np.allclose(
        composer_a.out_force_b.warp.numpy(), composer_ref.out_force_b.warp.numpy(), atol=1e-4, rtol=1e-5
    ), "add_raw_buffers_from force mismatch vs direct accumulation"
    assert np.allclose(
        composer_a.out_torque_b.warp.numpy(), composer_ref.out_torque_b.warp.numpy(), atol=1e-4, rtol=1e-5
    ), "add_raw_buffers_from torque mismatch vs direct accumulation"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from_inactive_is_noop(device: str):
    """Test that add_raw_buffers_from is a no-op when the source composer is inactive."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=21)

    mock_a = create_mock_asset(num_envs, num_bodies, device)
    mock_b = create_mock_asset(num_envs, num_bodies, device)
    composer_a = WrenchComposer(mock_a)
    composer_b = WrenchComposer(mock_b)

    # Populate composer_a with some forces
    forces_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_a.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
    )

    # composer_b is inactive (never written to)
    assert not composer_b.active

    # Snapshot composer_a's local buffer before merge
    local_force_before = composer_a.local_force_b.numpy().copy()

    # Merge inactive composer_b into composer_a -- should be a no-op
    composer_a.add_raw_buffers_from(composer_b)

    assert np.allclose(composer_a.local_force_b.numpy(), local_force_before, atol=1e-7)


# ============================================================================
# Mask-based API Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_add_forces_mask(device: str, num_envs: int, num_bodies: int):
    """Test that add_forces_and_torques_mask produces the same result as the index variant."""
    rng = np.random.default_rng(seed=30)

    for _ in range(5):
        # Random subset selection
        env_select = rng.choice([True, False], size=num_envs, replace=True)
        body_select = rng.choice([True, False], size=num_bodies, replace=True)
        # Ensure at least one env and body are selected
        env_select[0] = True
        body_select[0] = True

        env_ids_np = np.where(env_select)[0].astype(np.int32)
        body_ids_np = np.where(body_select)[0].astype(np.int32)
        env_mask_np = env_select.astype(np.bool_)
        body_mask_np = body_select.astype(np.bool_)

        # Random forces for the full grid (mask variant takes full-sized arrays)
        forces_full_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

        # Index-based composer
        mock_idx = create_mock_asset(num_envs, num_bodies, device)
        composer_idx = WrenchComposer(mock_idx)
        # Extract the subset for index API
        forces_subset_np = forces_full_np[env_ids_np[:, None], body_ids_np[None, :], :]
        composer_idx.add_forces_and_torques_index(
            forces=wp.from_numpy(forces_subset_np, dtype=wp.vec3f, device=device),
            env_ids=wp.from_numpy(env_ids_np, dtype=wp.int32, device=device),
            body_ids=wp.from_numpy(body_ids_np, dtype=wp.int32, device=device),
        )

        # Mask-based composer
        mock_mask = create_mock_asset(num_envs, num_bodies, device)
        composer_mask = WrenchComposer(mock_mask)
        composer_mask.add_forces_and_torques_mask(
            forces=wp.from_numpy(forces_full_np, dtype=wp.vec3f, device=device),
            env_mask=wp.from_numpy(env_mask_np, dtype=wp.bool, device=device),
            body_mask=wp.from_numpy(body_mask_np, dtype=wp.bool, device=device),
        )

        # Compose both
        composer_idx.compose_to_body_frame()
        composer_mask.compose_to_body_frame()

        assert np.allclose(
            composer_idx.out_force_b.warp.numpy(), composer_mask.out_force_b.warp.numpy(), atol=1e-4, rtol=1e-5
        ), f"Mask vs index force mismatch (envs={num_envs}, bodies={num_bodies})"
        assert np.allclose(
            composer_idx.out_torque_b.warp.numpy(), composer_mask.out_torque_b.warp.numpy(), atol=1e-4, rtol=1e-5
        ), f"Mask vs index torque mismatch (envs={num_envs}, bodies={num_bodies})"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_add_forces_mask_global(device: str, num_envs: int, num_bodies: int):
    """Test mask-based API with global forces and positions."""
    rng = np.random.default_rng(seed=31)

    # Random link poses
    link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch = torch.from_numpy(link_pos_np)
    link_quat_torch = torch.from_numpy(link_quat_np)

    # Select all envs and bodies to keep comparison simple
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)

    # Index-based
    mock_idx = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    composer_idx = WrenchComposer(mock_idx)
    composer_idx.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Mask-based (all-True masks)
    mock_mask = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    composer_mask = WrenchComposer(mock_mask)
    env_mask = wp.from_numpy(np.ones(num_envs, dtype=np.bool_), dtype=wp.bool, device=device)
    body_mask = wp.from_numpy(np.ones(num_bodies, dtype=np.bool_), dtype=wp.bool, device=device)
    composer_mask.add_forces_and_torques_mask(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        env_mask=env_mask,
        body_mask=body_mask,
        is_global=True,
    )

    composer_idx.compose_to_body_frame()
    composer_mask.compose_to_body_frame()

    assert np.allclose(
        composer_idx.out_force_b.warp.numpy(), composer_mask.out_force_b.warp.numpy(), atol=1e-4, rtol=1e-5
    ), "Mask vs index global force mismatch"
    assert np.allclose(
        composer_idx.out_torque_b.warp.numpy(), composer_mask.out_torque_b.warp.numpy(), atol=1e-4, rtol=1e-5
    ), "Mask vs index global torque mismatch"


# ============================================================================
# set_forces_and_torques_index Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_forces_overwrites_previous_add(device: str):
    """Test that set_forces_and_torques_index clears previously accumulated values."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=40)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # First accumulate some forces via add
    forces_a_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_a_np, dtype=wp.vec3f, device=device),
    )

    # Now set new forces -- should replace, not accumulate
    forces_b_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.set_forces_and_torques_index(
        forces=wp.from_numpy(forces_b_np, dtype=wp.vec3f, device=device),
    )

    composer.compose_to_body_frame()

    # Output should match forces_b only (forces_a should be gone)
    assert np.allclose(composer.out_force_b.warp.numpy(), forces_b_np, atol=1e-4, rtol=1e-5), (
        "set_forces did not clear previous add"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_forces_clears_targeted_envs_only(device: str):
    """Test that set_forces_and_torques_index clears only the targeted environments."""
    num_envs, num_bodies = 4, 3
    rng = np.random.default_rng(seed=41)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # Add global forces at positions (populates global_force_w and global_torque_w)
    forces_global_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-5.0, 5.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Also add local torques (populates local_torque_b)
    torques_local_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        torques=wp.from_numpy(torques_local_np, dtype=wp.vec3f, device=device),
        is_global=False,
    )

    # Now set local forces for envs [0, 2] -- should clear only envs 0, 2
    env_ids_np = np.array([0, 2], dtype=np.int32)
    kept_env_ids = np.array([1, 3], dtype=np.int32)
    forces_new_np = rng.uniform(-50.0, 50.0, (2, num_bodies, 3)).astype(np.float32)
    composer.set_forces_and_torques_index(
        forces=wp.from_numpy(forces_new_np, dtype=wp.vec3f, device=device),
        env_ids=wp.from_numpy(env_ids_np, dtype=wp.int32, device=device),
        is_global=False,
    )

    zeros = np.zeros((num_bodies, 3), dtype=np.float32)

    # Targeted envs [0, 2]: all buffers cleared, then local_force_b written
    for eid in env_ids_np:
        assert np.allclose(composer.global_force_w.numpy()[eid], zeros, atol=1e-7), (
            f"global_force_w not cleared for targeted env {eid}"
        )
        assert np.allclose(composer.global_torque_w.numpy()[eid], zeros, atol=1e-7), (
            f"global_torque_w not cleared for targeted env {eid}"
        )
        assert np.allclose(composer.local_torque_b.numpy()[eid], zeros, atol=1e-7), (
            f"local_torque_b not cleared for targeted env {eid}"
        )

    # Non-targeted envs [1, 3]: should retain original values
    for eid in kept_env_ids:
        assert np.allclose(composer.global_force_w.numpy()[eid], forces_global_np[eid], atol=1e-4, rtol=1e-5), (
            f"global_force_w changed for non-targeted env {eid}"
        )
        assert np.allclose(composer.local_torque_b.numpy()[eid], torques_local_np[eid], atol=1e-4, rtol=1e-5), (
            f"local_torque_b changed for non-targeted env {eid}"
        )

    # local_force_b should have new values at env_ids [0, 2], zeros at [1, 3]
    expected_local_force = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    expected_local_force[env_ids_np] = forces_new_np
    assert np.allclose(composer.local_force_b.numpy(), expected_local_force, atol=1e-4, rtol=1e-5), (
        "local_force_b has wrong values after set"
    )


# ============================================================================
# Partial Reset Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_partial_reset_zeros_only_specified_envs(device: str):
    """Test that partial reset zeros only the specified environments and leaves others intact."""
    num_envs, num_bodies = 8, 3
    rng = np.random.default_rng(seed=50)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # Populate all envs with local forces
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
    )

    # Also add global forces to populate more buffers
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Partial reset: only envs [1, 3, 5]
    reset_env_ids = np.array([1, 3, 5], dtype=np.int32)
    kept_env_ids = np.array([0, 2, 4, 6, 7], dtype=np.int32)
    composer.reset(env_ids=wp.from_numpy(reset_env_ids, dtype=wp.int32, device=device))

    # Reset envs should be zeroed across all input buffers
    zeros = np.zeros((num_bodies, 3), dtype=np.float32)
    local_force = composer.local_force_b.numpy()
    global_force_at_com = composer.global_force_at_com_w.numpy()
    for eid in reset_env_ids:
        assert np.allclose(local_force[eid], zeros, atol=1e-7), f"local_force_b not zeroed for env {eid}"
        assert np.allclose(global_force_at_com[eid], zeros, atol=1e-7), (
            f"global_force_at_com_w not zeroed for env {eid}"
        )

    # Kept envs should retain their values
    for eid in kept_env_ids:
        assert np.allclose(local_force[eid], forces_np[eid], atol=1e-4, rtol=1e-5), (
            f"local_force_b changed for non-reset env {eid}"
        )
        assert np.allclose(global_force_at_com[eid], forces_global_np[eid], atol=1e-4, rtol=1e-5), (
            f"global_force_at_com_w changed for non-reset env {eid}"
        )

    # Flags: _active should still be True, _dirty should be True
    assert composer.active
    assert composer._dirty


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_full_reset_clears_active_flag(device: str):
    """Test that full reset (no args) clears the _active flag."""
    num_envs, num_bodies = 4, 2

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    forces_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
    )
    assert composer.active

    composer.reset()
    assert not composer.active
    assert not composer._dirty


# ============================================================================
# Deprecated API Backward-Compatibility Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composed_force_emits_deprecation_warning(device: str):
    """Test that accessing composed_force emits a DeprecationWarning."""
    num_envs, num_bodies = 2, 1

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    forces_np = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
    )

    with pytest.warns(DeprecationWarning, match="composed_force.*is deprecated"):
        result = composer.composed_force

    # Should return the same data as out_force_b
    assert np.allclose(result.warp.numpy(), composer.out_force_b.warp.numpy(), atol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composed_torque_emits_deprecation_warning(device: str):
    """Test that accessing composed_torque emits a DeprecationWarning."""
    num_envs, num_bodies = 2, 1

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    torques_np = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    composer.add_forces_and_torques_index(
        torques=wp.from_numpy(torques_np, dtype=wp.vec3f, device=device),
    )

    with pytest.warns(DeprecationWarning, match="composed_torque.*is deprecated"):
        result = composer.composed_torque

    assert np.allclose(result.warp.numpy(), composer.out_torque_b.warp.numpy(), atol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deprecated_add_forces_and_torques_emits_warning(device: str):
    """Test that the deprecated add_forces_and_torques wrapper emits a warning and works."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=52)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    forces_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)

    with pytest.warns(DeprecationWarning, match="add_forces_and_torques.*is deprecated"):
        composer.add_forces_and_torques(
            forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        )

    composer.compose_to_body_frame()
    assert np.allclose(composer.out_force_b.warp.numpy(), forces_np, atol=1e-4, rtol=1e-5)


# ============================================================================
# set_forces_and_torques_mask Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_forces_mask_overwrites_previous_add(device: str):
    """Test that set_forces_and_torques_mask clears previously accumulated values."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=60)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # Accumulate some forces via add
    forces_a_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_a_np, dtype=wp.vec3f, device=device),
    )

    # Now set new forces via mask -- should replace, not accumulate
    forces_b_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.set_forces_and_torques_mask(
        forces=wp.from_numpy(forces_b_np, dtype=wp.vec3f, device=device),
    )

    composer.compose_to_body_frame()

    # Output should match forces_b only (forces_a should be gone)
    assert np.allclose(composer.out_force_b.warp.numpy(), forces_b_np, atol=1e-4, rtol=1e-5), (
        "set_forces_and_torques_mask did not clear previous add"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_forces_mask_clears_targeted_envs_only(device: str):
    """Test that set_forces_and_torques_mask clears only the masked environments."""
    num_envs, num_bodies = 4, 3
    rng = np.random.default_rng(seed=61)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # Populate global buffers for all envs
    forces_global_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-5.0, 5.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Also add local torques for all envs
    torques_local_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        torques=wp.from_numpy(torques_local_np, dtype=wp.vec3f, device=device),
        is_global=False,
    )

    # Set local forces via mask for envs [0, 2] -- should clear only masked envs
    env_mask_np = np.array([True, False, True, False], dtype=np.bool_)
    body_mask_np = np.array([True, True, False], dtype=np.bool_)
    forces_new_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.set_forces_and_torques_mask(
        forces=wp.from_numpy(forces_new_np, dtype=wp.vec3f, device=device),
        env_mask=wp.from_numpy(env_mask_np, dtype=wp.bool, device=device),
        body_mask=wp.from_numpy(body_mask_np, dtype=wp.bool, device=device),
        is_global=False,
    )

    zeros = np.zeros((num_bodies, 3), dtype=np.float32)

    # Masked envs [0, 2]: all buffers cleared by reset, then local_force_b written where body_mask is True
    for eid in [0, 2]:
        assert np.allclose(composer.global_force_w.numpy()[eid], zeros, atol=1e-7), (
            f"global_force_w not cleared for masked env {eid}"
        )
        assert np.allclose(composer.global_torque_w.numpy()[eid], zeros, atol=1e-7), (
            f"global_torque_w not cleared for masked env {eid}"
        )
        assert np.allclose(composer.local_torque_b.numpy()[eid], zeros, atol=1e-7), (
            f"local_torque_b not cleared for masked env {eid}"
        )

    # Non-masked envs [1, 3]: should retain original values
    for eid in [1, 3]:
        assert np.allclose(composer.global_force_w.numpy()[eid], forces_global_np[eid], atol=1e-4, rtol=1e-5), (
            f"global_force_w changed for non-masked env {eid}"
        )
        assert np.allclose(composer.local_torque_b.numpy()[eid], torques_local_np[eid], atol=1e-4, rtol=1e-5), (
            f"local_torque_b changed for non-masked env {eid}"
        )

    # local_force_b should have new values where both masks are True, zeros for masked envs otherwise
    expected_local_force = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    for e in range(num_envs):
        for b in range(num_bodies):
            if env_mask_np[e] and body_mask_np[b]:
                expected_local_force[e, b] = forces_new_np[e, b]
    assert np.allclose(composer.local_force_b.numpy(), expected_local_force, atol=1e-4, rtol=1e-5), (
        "local_force_b has wrong values after mask set"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_forces_mask_matches_set_forces_index(device: str):
    """Test that set_forces_and_torques_mask produces the same result as the index variant."""
    num_envs, num_bodies = 6, 3
    rng = np.random.default_rng(seed=62)

    # Random link poses
    link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch = torch.from_numpy(link_pos_np)
    link_quat_torch = torch.from_numpy(link_quat_np)

    # Use all envs/bodies to compare
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)

    # Index-based
    mock_idx = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    composer_idx = WrenchComposer(mock_idx)
    composer_idx.set_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Mask-based (all-True)
    mock_mask = create_mock_asset(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    composer_mask = WrenchComposer(mock_mask)
    composer_mask.set_forces_and_torques_mask(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    composer_idx.compose_to_body_frame()
    composer_mask.compose_to_body_frame()

    assert np.allclose(
        composer_idx.out_force_b.warp.numpy(), composer_mask.out_force_b.warp.numpy(), atol=1e-4, rtol=1e-5
    ), "set mask vs index force mismatch"
    assert np.allclose(
        composer_idx.out_torque_b.warp.numpy(), composer_mask.out_torque_b.warp.numpy(), atol=1e-4, rtol=1e-5
    ), "set mask vs index torque mismatch"


# ============================================================================
# Lazy Composition (_ensure_composed) Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_out_force_b_triggers_lazy_composition(device: str):
    """Test that accessing out_force_b without explicit compose_to_body_frame still returns correct results."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=70)

    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
    composer = WrenchComposer(mock_asset)

    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Do NOT call compose_to_body_frame -- rely on lazy composition
    expected_forces_local = quat_rotate_inv_np(link_quat_np, forces_global_np)
    composed_force_np = composer.out_force_b.warp.numpy()

    assert np.allclose(composed_force_np, expected_forces_local, atol=1e-4, rtol=1e-5), (
        "Lazy composition via out_force_b failed"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_out_torque_b_triggers_lazy_composition(device: str):
    """Test that accessing out_torque_b without explicit compose_to_body_frame still returns correct results."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=71)

    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
    composer = WrenchComposer(mock_asset)

    torques_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        torques=wp.from_numpy(torques_global_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Do NOT call compose_to_body_frame -- rely on lazy composition
    expected_torques_local = quat_rotate_inv_np(link_quat_np, torques_global_np)
    composed_torque_np = composer.out_torque_b.warp.numpy()

    assert np.allclose(composed_torque_np, expected_torques_local, atol=1e-4, rtol=1e-5), (
        "Lazy composition via out_torque_b failed"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lazy_composition_tracks_dirty_flag(device: str):
    """Test that the dirty flag is correctly managed through add/compose/add cycles."""
    num_envs, num_bodies = 2, 1

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # Initially clean
    assert not composer._dirty

    # After add, dirty
    forces_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
    )
    assert composer._dirty

    # After accessing out_force_b, clean (lazy compose happened)
    _ = composer.out_force_b
    assert not composer._dirty

    # After another add, dirty again
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
    )
    assert composer._dirty

    # Accessing out_torque_b also triggers composition
    _ = composer.out_torque_b
    assert not composer._dirty

    # Verify accumulated result (2x forces)
    expected = 2.0 * forces_np
    assert np.allclose(composer.out_force_b.warp.numpy(), expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_is_idempotent(device: str):
    """Calling compose_to_body_frame twice without intervening writes produces the same result."""
    rng = np.random.default_rng(seed=456)
    num_envs, num_bodies = 4, 3

    # Non-trivial link pose so the rotation path is exercised
    link_pos_np = rng.uniform(-2, 2, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = rng.standard_normal((num_envs, num_bodies, 4)).astype(np.float32)
    link_quat_np /= np.linalg.norm(link_quat_np, axis=-1, keepdims=True)

    mock_asset = create_mock_asset(
        num_envs,
        num_bodies,
        device,
        link_pos=torch.from_numpy(link_pos_np),
        link_quat=torch.from_numpy(link_quat_np),
    )
    composer = WrenchComposer(mock_asset)

    # Add global forces with positions (exercises cross-product torque path)
    forces_np = rng.uniform(-5, 5, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-1, 1, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-3, 3, (num_envs, num_bodies, 3)).astype(np.float32)

    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        torques=wp.from_numpy(torques_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # First compose
    composer.compose_to_body_frame()
    force_first = composer.out_force_b.warp.numpy().copy()
    torque_first = composer.out_torque_b.warp.numpy().copy()

    # Second compose (no writes in between)
    composer.compose_to_body_frame()
    force_second = composer.out_force_b.warp.numpy()
    torque_second = composer.out_torque_b.warp.numpy()

    np.testing.assert_array_equal(force_first, force_second)
    np.testing.assert_array_equal(torque_first, torque_second)


# ============================================================================
# CoM Offset from Link Origin Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_with_com_offset(device: str):
    """Test that torque correction uses CoM position, not link position, when they differ."""
    num_envs, num_bodies = 2, 1

    # Link at origin, CoM offset by [1, 0, 0]
    link_pos_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    link_quat_np = np.zeros((num_envs, num_bodies, 4), dtype=np.float32)
    link_quat_np[..., 3] = 1.0  # identity quaternion (xyzw)

    com_pos_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    com_pos_np[..., 0] = 1.0  # CoM at [1, 0, 0]

    mock_asset = create_mock_asset(
        num_envs,
        num_bodies,
        device,
        link_pos=torch.from_numpy(link_pos_np),
        link_quat=torch.from_numpy(link_quat_np),
    )
    # Set CoM pose separately (pos=[1,0,0], quat=identity)
    com_pose = torch.cat([torch.from_numpy(com_pos_np), torch.from_numpy(link_quat_np)], dim=-1)
    mock_asset.data.set_body_com_pose_w(com_pose)

    composer = WrenchComposer(mock_asset)

    # Apply global force [0, 0, 10] at position [0, 0, 0] (world origin)
    forces_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    forces_np[..., 2] = 10.0
    positions_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    composer.compose_to_body_frame()

    # With identity quaternion:
    #   torque_w = cross(P, F) - cross(com, F) = cross([0,0,0], [0,0,10]) - cross([1,0,0], [0,0,10])
    #            = [0,0,0] - [0*10-0*0, 0*0-1*10, 1*0-0*0] = [0,0,0] - [0, -10, 0] = [0, 10, 0]
    # In body frame (identity rotation): [0, 10, 0]
    expected_torque = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    expected_torque[..., 1] = 10.0

    actual_torque = composer.out_torque_b.warp.numpy()
    assert np.allclose(actual_torque, expected_torque, atol=1e-4, rtol=1e-5), (
        f"CoM offset torque correction failed.\nExpected:\n{expected_torque}\nGot:\n{actual_torque}"
    )

    # Force should be unchanged (identity rotation)
    assert np.allclose(composer.out_force_b.warp.numpy(), forces_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_at_com_no_torque_with_com_offset(device: str):
    """Test that a global force at CoM position produces zero torque even with CoM offset."""
    num_envs, num_bodies = 2, 1

    # Link at origin, CoM offset by [2, 3, 0]
    link_pos_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    link_quat_np = np.zeros((num_envs, num_bodies, 4), dtype=np.float32)
    link_quat_np[..., 3] = 1.0

    com_pos_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    com_pos_np[..., 0] = 2.0
    com_pos_np[..., 1] = 3.0

    mock_asset = create_mock_asset(
        num_envs,
        num_bodies,
        device,
        link_pos=torch.from_numpy(link_pos_np),
        link_quat=torch.from_numpy(link_quat_np),
    )
    com_pose = torch.cat([torch.from_numpy(com_pos_np), torch.from_numpy(link_quat_np)], dim=-1)
    mock_asset.data.set_body_com_pose_w(com_pose)

    composer = WrenchComposer(mock_asset)

    # Apply global force at the CoM position
    forces_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    forces_np[..., 2] = 50.0
    positions_np = com_pos_np.copy()

    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    composer.compose_to_body_frame()

    # Torque = cross(com, F) - cross(com, F) = 0
    expected_torque = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    assert np.allclose(composer.out_torque_b.warp.numpy(), expected_torque, atol=1e-4, rtol=1e-5), (
        "Force at CoM should produce zero torque regardless of CoM offset"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_com_offset_with_rotation(device: str):
    """Test torque correction with both CoM offset and non-identity rotation."""
    num_envs, num_bodies = 1, 1
    rng = np.random.default_rng(seed=73)

    # Random rotation
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_np = rng.uniform(-5.0, 5.0, (num_envs, num_bodies, 3)).astype(np.float32)

    # CoM offset from link
    com_offset_np = rng.uniform(0.5, 2.0, (num_envs, num_bodies, 3)).astype(np.float32)
    com_pos_np = link_pos_np + com_offset_np  # simple world-frame offset for test clarity

    mock_asset = create_mock_asset(
        num_envs,
        num_bodies,
        device,
        link_pos=torch.from_numpy(link_pos_np),
        link_quat=torch.from_numpy(link_quat_np),
    )
    com_pose = torch.cat([torch.from_numpy(com_pos_np), torch.from_numpy(link_quat_np)], dim=-1)
    mock_asset.data.set_body_com_pose_w(com_pose)

    composer = WrenchComposer(mock_asset)

    # Apply global force at a random world position
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)

    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    composer.compose_to_body_frame()

    # Expected: torque_w = cross(P, F) - cross(com, F) = cross(P - com, F)
    lever_arm = positions_np - com_pos_np
    torque_w = np.cross(lever_arm, forces_np)
    expected_torque_b = quat_rotate_inv_np(link_quat_np, torque_w)
    expected_force_b = quat_rotate_inv_np(link_quat_np, forces_np)

    assert np.allclose(composer.out_force_b.warp.numpy(), expected_force_b, atol=1e-3, rtol=1e-4), (
        "Force mismatch with CoM offset + rotation"
    )
    assert np.allclose(composer.out_torque_b.warp.numpy(), expected_torque_b, atol=1e-3, rtol=1e-4), (
        f"Torque mismatch with CoM offset + rotation.\n"
        f"Expected:\n{expected_torque_b}\nGot:\n{composer.out_torque_b.warp.numpy()}"
    )


# ============================================================================
# Deprecated set_forces_and_torques Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deprecated_set_forces_and_torques_emits_warning(device: str):
    """Test that the deprecated set_forces_and_torques wrapper emits a warning and works."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=80)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    forces_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)

    with pytest.warns(DeprecationWarning, match="set_forces_and_torques.*is deprecated"):
        composer.set_forces_and_torques(
            forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        )

    composer.compose_to_body_frame()
    assert np.allclose(composer.out_force_b.warp.numpy(), forces_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deprecated_set_forces_and_torques_clears_previous(device: str):
    """Test that deprecated set_forces_and_torques actually replaces previous values."""
    num_envs, num_bodies = 4, 2
    rng = np.random.default_rng(seed=81)

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    composer = WrenchComposer(mock_asset)

    # First add some forces
    forces_a_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(forces_a_np, dtype=wp.vec3f, device=device),
    )

    # Then set via deprecated method -- should replace
    forces_b_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    with pytest.warns(DeprecationWarning):
        composer.set_forces_and_torques(
            forces=wp.from_numpy(forces_b_np, dtype=wp.vec3f, device=device),
        )

    composer.compose_to_body_frame()
    assert np.allclose(composer.out_force_b.warp.numpy(), forces_b_np, atol=1e-4, rtol=1e-5), (
        "Deprecated set_forces_and_torques did not replace previous values"
    )
