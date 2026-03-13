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

from isaaclab.assets import RigidObject
from isaaclab.utils.wrench_composer import WrenchComposer


class MockAssetData:
    """Mock data class that provides body link positions and quaternions."""

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
        link_pos: torch.Tensor | None = None,
        link_quat: torch.Tensor | None = None,
    ):
        """Initialize mock asset data.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            link_pos: Optional link positions (num_envs, num_bodies, 3). Defaults to zeros.
            link_quat: Optional link quaternions in (w, x, y, z) format (num_envs, num_bodies, 4).
                       Defaults to identity quaternion.
        """
        if link_pos is not None:
            self.body_link_pos_w = link_pos.to(device=device, dtype=torch.float32)
        else:
            self.body_link_pos_w = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32, device=device)

        if link_quat is not None:
            self.body_link_quat_w = link_quat.to(device=device, dtype=torch.float32)
        else:
            # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
            self.body_link_quat_w = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32, device=device)
            self.body_link_quat_w[..., 0] = 1.0


class MockRigidObject:
    """Mock RigidObject that provides the minimal interface required by WrenchComposer.

    This mock enables testing WrenchComposer in isolation without requiring a full simulation setup.
    It passes isinstance checks by registering as a virtual subclass of RigidObject.
    """

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
        link_pos: torch.Tensor | None = None,
        link_quat: torch.Tensor | None = None,
    ):
        """Initialize mock rigid object.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            link_pos: Optional link positions (num_envs, num_bodies, 3).
            link_quat: Optional link quaternions in (w, x, y, z) format (num_envs, num_bodies, 4).
        """
        self.num_instances = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.data = MockAssetData(num_envs, num_bodies, device, link_pos, link_quat)


# --- Helper functions for quaternion math ---


def quat_rotate_inv_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion (numpy).

    Args:
        quat_wxyz: Quaternion in (w, x, y, z) format. Shape: (..., 4)
        vec: Vector to rotate. Shape: (..., 3)

    Returns:
        Rotated vector. Shape: (..., 3)
    """
    # Extract components
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]

    # For inverse rotation, we conjugate the quaternion (negate xyz)
    # q^-1 * v * q = q_conj * v * q_conj^-1 for unit quaternion
    # Using the formula: v' = v + 2*w*(xyz x v) + 2*(xyz x (xyz x v))
    # But for inverse: use -xyz

    # Cross product: xyz x vec
    t = 2.0 * np.cross(-xyz, vec, axis=-1)
    # Result: vec + w*t + xyz x t
    return vec + w * t + np.cross(-xyz, t, axis=-1)


def random_unit_quaternion_np(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """Generate random unit quaternions in (w, x, y, z) format.

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


# Register MockRigidObject as a virtual subclass of RigidObject
# This allows isinstance(mock, RigidObject) to return True
RigidObject.register(MockRigidObject)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_force(device: str, num_envs: int, num_bodies: int):
    # Initialize random number generator
    rng = np.random.default_rng(seed=0)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(forces=forces, body_ids=body_ids, env_ids=env_ids)
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_torque(device: str, num_envs: int, num_bodies: int):
    # Initialize random number generator
    rng = np.random.default_rng(seed=1)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(torques=torques, body_ids=body_ids, env_ids=env_ids)
            # Add torques to hand-calculated composed torque
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_at_positons(device: str, num_envs: int, num_bodies: int):
    """Test adding forces at local positions (offset from link frame)."""
    rng = np.random.default_rng(seed=2)

    for _ in range(10):
        # Initialize wrench composer
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(
                forces=forces, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            # Add torques to hand-calculated composed torque: torque = cross(position, force)
            torques_from_forces = np.cross(positions_np, forces_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += torques_from_forces[i, j, :]

        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_torques_at_position(device: str, num_envs: int, num_bodies: int):
    rng = np.random.default_rng(seed=3)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(
                torques=torques, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            # Add torques to hand-calculated composed torque
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_and_torques_at_position(device: str, num_envs: int, num_bodies: int):
    """Test adding forces and torques at local positions."""
    rng = np.random.default_rng(seed=4)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(
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
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_reset(device: str, num_envs: int, num_bodies: int):
    rng = np.random.default_rng(seed=5)
    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
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
        wrench_composer.add_forces_and_torques(forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids)
        # Reset wrench composer
        wrench_composer.reset()
        # Get composed force and torque from wrench composer
        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, np.zeros((num_envs, num_bodies, 3)), atol=1, rtol=1e-7)
        assert np.allclose(composed_torque_np, np.zeros((num_envs, num_bodies, 3)), atol=1, rtol=1e-7)


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
        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global forces for all envs and bodies
        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

        # Apply global forces
        wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

        # Compute expected local forces by rotating global forces by inverse quaternion
        expected_forces_local = quat_rotate_inv_np(link_quat_np, forces_global_np)

        # Verify
        composed_force_np = wrench_composer.composed_force.numpy()
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
        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global torques
        torques_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        torques_global = wp.from_numpy(torques_global_np, dtype=wp.vec3f, device=device)

        # Apply global torques
        wrench_composer.add_forces_and_torques(torques=torques_global, is_global=True)

        # Compute expected local torques
        expected_torques_local = quat_rotate_inv_np(link_quat_np, torques_global_np)

        # Verify
        composed_torque_np = wrench_composer.composed_torque.numpy()
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
        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global forces and positions
        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_global_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)
        positions_global = wp.from_numpy(positions_global_np, dtype=wp.vec3f, device=device)

        # Apply global forces at global positions
        wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_global, is_global=True)

        # Compute expected results:
        # 1. Force in local frame = quat_rotate_inv(link_quat, global_force)
        expected_forces_local = quat_rotate_inv_np(link_quat_np, forces_global_np)

        # 2. Position offset in local frame = global_position - link_position (then used for torque)
        position_offset_global = positions_global_np - link_pos_np

        # 3. Torque = skew(position_offset_global) @ force_global, then rotate to local
        expected_torques_local = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for i in range(num_envs):
            for j in range(num_bodies):
                pos_offset = position_offset_global[i, j]  # global frame offset
                force_local = expected_forces_local[i, j]  # local frame force
                # skew(pos_offset) @ force_local
                expected_torques_local[i, j] = np.cross(pos_offset, force_local)

        # Verify forces
        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(composed_force_np, expected_forces_local, atol=1e-3, rtol=1e-4), (
            f"Global force at position failed.\nExpected forces:\n{expected_forces_local}\nGot:\n{composed_force_np}"
        )

        # Verify torques
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, expected_torques_local, atol=1e-3, rtol=1e-4), (
            f"Global force at position failed.\nExpected torques:\n{expected_torques_local}\nGot:\n{composed_torque_np}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_local_vs_global_identity_quaternion(device: str):
    """Test that local and global give same result with identity quaternion and zero position."""
    rng = np.random.default_rng(seed=13)
    num_envs, num_bodies = 10, 5

    # Create mock with identity pose (default)
    mock_asset_local = MockRigidObject(num_envs, num_bodies, device)
    mock_asset_global = MockRigidObject(num_envs, num_bodies, device)

    wrench_composer_local = WrenchComposer(mock_asset_local)
    wrench_composer_global = WrenchComposer(mock_asset_global)

    # Generate random forces and torques
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

    # Apply as local
    wrench_composer_local.add_forces_and_torques(forces=forces, torques=torques, is_global=False)

    # Apply as global (should be same with identity quaternion)
    wrench_composer_global.add_forces_and_torques(forces=forces, torques=torques, is_global=True)

    # Results should be identical
    assert np.allclose(
        wrench_composer_local.composed_force.numpy(),
        wrench_composer_global.composed_force.numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        wrench_composer_local.composed_torque.numpy(),
        wrench_composer_global.composed_torque.numpy(),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_90_degree_rotation_global_force(device: str):
    """Test global force with a known 90-degree rotation for easy verification."""
    num_envs, num_bodies = 1, 1

    # 90-degree rotation around Z-axis: (w, x, y, z) = (cos(45°), 0, 0, sin(45°))
    # This rotates X -> Y, Y -> -X
    angle = np.pi / 2
    link_quat_np = np.array([[[[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]]]], dtype=np.float32).reshape(1, 1, 4)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Apply force in global +X direction
    force_global = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    force_wp = wp.from_numpy(force_global, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques(forces=force_wp, is_global=True)

    # Expected: After inverse rotation (rotate by -90° around Z), X becomes -Y
    # Actually, inverse rotation of +90° around Z applied to (1,0,0) gives (0,-1,0)
    expected_force_local = np.array([[[0.0, -1.0, 0.0]]], dtype=np.float32)

    composed_force_np = wrench_composer.composed_force.numpy()
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

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Generate random local and global forces
    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

    forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    # Add local forces first
    wrench_composer.add_forces_and_torques(forces=forces_local, is_global=False)

    # Add global forces
    wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

    # Expected: local forces stay as-is, global forces get rotated, then sum
    global_forces_in_local = quat_rotate_inv_np(link_quat_np, forces_global_np)
    expected_total = forces_local_np + global_forces_in_local

    composed_force_np = wrench_composer.composed_force.numpy()
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

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random local forces and local positions (offsets)
        forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_local_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
        positions_local = wp.from_numpy(positions_local_np, dtype=wp.vec3f, device=device)

        # Apply local forces at local positions
        wrench_composer.add_forces_and_torques(forces=forces_local, positions=positions_local, is_global=False)

        # Expected: forces stay as-is, torque = cross(position, force)
        expected_forces = forces_local_np
        expected_torques = np.cross(positions_local_np, forces_local_np)

        # Verify
        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()

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

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Generate random global forces
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    # Position = link position (so offset is zero)
    positions_at_link = wp.from_numpy(link_pos_np, dtype=wp.vec3f, device=device)

    # Apply global forces at link origin
    wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_at_link, is_global=True)

    # Expected: force rotated to local, torque = 0 (since position offset is zero)
    expected_forces = quat_rotate_inv_np(link_quat_np, forces_global_np)
    expected_torques = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

    composed_force_np = wrench_composer.composed_force.numpy()
    composed_torque_np = wrench_composer.composed_torque.numpy()

    assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
    assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)
