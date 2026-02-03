# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import numpy as np
import torch

import pytest
import warp as wp

from isaaclab.utils.wrench_composer import WrenchComposer


class MockAssetData:
    """Mock data class that provides body com poses (position + quaternion as transform)."""

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
        com_pos: torch.Tensor | None = None,
        com_quat: torch.Tensor | None = None,
    ):
        """Initialize mock asset data.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            com_pos: Optional COM positions (num_envs, num_bodies, 3). Defaults to zeros.
            com_quat: Optional COM quaternions in (w, x, y, z) format (num_envs, num_bodies, 4).
                      Defaults to identity quaternion.
        """
        # Build the COM poses as transforms (7-element: pos + quat)
        if com_pos is None:
            com_pos = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32, device=device)
        else:
            com_pos = com_pos.to(device=device, dtype=torch.float32)

        if com_quat is None:
            # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
            com_quat = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32, device=device)
            com_quat[..., 0] = 1.0
        else:
            com_quat = com_quat.to(device=device, dtype=torch.float32)

        # Create transform tensor: (num_envs, num_bodies, 7) -> (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
        com_pose = torch.cat([com_pos, com_quat[..., 1:], com_quat[..., 0].unsqueeze(-1)], dim=-1)
        self.body_com_pose_w = wp.from_torch(com_pose, dtype=wp.transformf)


class MockRigidObject:
    """Mock RigidObject that provides the minimal interface required by WrenchComposer.

    This mock enables testing WrenchComposer in isolation without requiring a full simulation setup.
    WrenchComposer uses hasattr() checks for duck typing, so this mock just needs the right attributes.
    """

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
        com_pos: torch.Tensor | None = None,
        com_quat: torch.Tensor | None = None,
    ):
        """Initialize mock rigid object.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            com_pos: Optional COM positions (num_envs, num_bodies, 3).
            com_quat: Optional COM quaternions in (w, x, y, z) format (num_envs, num_bodies, 4).
        """
        self.num_instances = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.data = MockAssetData(num_envs, num_bodies, device, com_pos, com_quat)


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


# Note: WrenchComposer uses hasattr() checks rather than isinstance() checks,
# so we don't need to register MockRigidObject as a virtual subclass of RigidObject.


# ============================================================================
# WARP CODE PATH TESTS (using warp arrays and masks)
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_warp_add_force(device: str, num_envs: int, num_bodies: int):
    """Test adding forces using warp arrays and masks."""
    rng = np.random.default_rng(seed=0)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed force
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            # Create random masks
            env_mask_np = rng.choice([True, False], size=num_envs).astype(bool)
            body_mask_np = rng.choice([True, False], size=num_bodies).astype(bool)
            # Ensure at least one True in each mask
            env_mask_np[rng.integers(0, num_envs)] = True
            body_mask_np[rng.integers(0, num_bodies)] = True

            # Convert to warp arrays
            env_mask = wp.from_numpy(env_mask_np, dtype=wp.bool, device=device)
            body_mask = wp.from_numpy(body_mask_np, dtype=wp.bool, device=device)

            # Get random forces for all envs/bodies (complete data)
            forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)

            # Add forces to wrench composer
            wrench_composer.add_forces_and_torques(forces=forces, body_mask=body_mask, env_mask=env_mask)

            # Add forces to hand-calculated composed force (only where masks are True)
            # Use vectorized numpy operations instead of nested loops
            combined_mask = np.outer(env_mask_np, body_mask_np)[..., np.newaxis]  # (num_envs, num_bodies, 1)
            hand_calculated_composed_force_np += forces_np * combined_mask

        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_warp_add_torque(device: str, num_envs: int, num_bodies: int):
    """Test adding torques using warp arrays and masks."""
    rng = np.random.default_rng(seed=1)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        # Initialize hand-calculated composed torque
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            # Create random masks
            env_mask_np = rng.choice([True, False], size=num_envs).astype(bool)
            body_mask_np = rng.choice([True, False], size=num_bodies).astype(bool)
            env_mask_np[rng.integers(0, num_envs)] = True
            body_mask_np[rng.integers(0, num_bodies)] = True

            env_mask = wp.from_numpy(env_mask_np, dtype=wp.bool, device=device)
            body_mask = wp.from_numpy(body_mask_np, dtype=wp.bool, device=device)

            # Get random torques
            torques_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

            # Add torques to wrench composer
            wrench_composer.add_forces_and_torques(torques=torques, body_mask=body_mask, env_mask=env_mask)

            # Add torques to hand-calculated composed torque
            combined_mask = np.outer(env_mask_np, body_mask_np)[..., np.newaxis]
            hand_calculated_composed_torque_np += torques_np * combined_mask

        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_warp_add_forces_at_positions(device: str, num_envs: int, num_bodies: int):
    """Test adding forces at local positions (offset from COM frame) using warp arrays."""
    rng = np.random.default_rng(seed=2)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            # Create random masks
            env_mask_np = rng.choice([True, False], size=num_envs).astype(bool)
            body_mask_np = rng.choice([True, False], size=num_bodies).astype(bool)
            env_mask_np[rng.integers(0, num_envs)] = True
            body_mask_np[rng.integers(0, num_bodies)] = True

            env_mask = wp.from_numpy(env_mask_np, dtype=wp.bool, device=device)
            body_mask = wp.from_numpy(body_mask_np, dtype=wp.bool, device=device)

            # Get random forces and positions
            forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
            positions_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)

            # Add forces at positions
            wrench_composer.add_forces_and_torques(
                forces=forces, positions=positions, body_mask=body_mask, env_mask=env_mask
            )

            # Calculate expected: force stays, torque = cross(position, force)
            combined_mask = np.outer(env_mask_np, body_mask_np)[..., np.newaxis]
            hand_calculated_composed_force_np += forces_np * combined_mask
            hand_calculated_composed_torque_np += np.cross(positions_np, forces_np) * combined_mask

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1e-2, rtol=1e-5)
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_warp_add_forces_and_torques_at_position(device: str, num_envs: int, num_bodies: int):
    """Test adding forces and torques at local positions using warp arrays."""
    rng = np.random.default_rng(seed=4)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            env_mask_np = rng.choice([True, False], size=num_envs).astype(bool)
            body_mask_np = rng.choice([True, False], size=num_bodies).astype(bool)
            env_mask_np[rng.integers(0, num_envs)] = True
            body_mask_np[rng.integers(0, num_bodies)] = True

            env_mask = wp.from_numpy(env_mask_np, dtype=wp.bool, device=device)
            body_mask = wp.from_numpy(body_mask_np, dtype=wp.bool, device=device)

            forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
            torques_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
            positions_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)

            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)

            wrench_composer.add_forces_and_torques(
                forces=forces, torques=torques, positions=positions, body_mask=body_mask, env_mask=env_mask
            )

            combined_mask = np.outer(env_mask_np, body_mask_np)[..., np.newaxis]
            hand_calculated_composed_force_np += forces_np * combined_mask
            hand_calculated_composed_torque_np += np.cross(positions_np, forces_np) * combined_mask
            hand_calculated_composed_torque_np += torques_np * combined_mask

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1e-2, rtol=1e-5)
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_warp_reset(device: str, num_envs: int, num_bodies: int):
    """Test reset functionality using warp arrays."""
    rng = np.random.default_rng(seed=5)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)

        # Create random masks
        env_mask_np = rng.choice([True, False], size=num_envs).astype(bool)
        body_mask_np = rng.choice([True, False], size=num_bodies).astype(bool)
        env_mask_np[rng.integers(0, num_envs)] = True
        body_mask_np[rng.integers(0, num_bodies)] = True

        env_mask = wp.from_numpy(env_mask_np, dtype=wp.bool, device=device)
        body_mask = wp.from_numpy(body_mask_np, dtype=wp.bool, device=device)

        forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
        torques_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)

        forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
        torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(forces=forces, torques=torques, body_mask=body_mask, env_mask=env_mask)
        wrench_composer.reset()

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, np.zeros((num_envs, num_bodies, 3)), atol=1e-6)
        assert np.allclose(composed_torque_np, np.zeros((num_envs, num_bodies, 3)), atol=1e-6)


# ============================================================================
# TORCH CODE PATH TESTS (using torch tensors and indices)
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_torch_add_force(device: str, num_envs: int, num_bodies: int):
    """Test adding forces using torch tensors and indices."""
    rng = np.random.default_rng(seed=0)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            # Get random number of envs and bodies and their indices
            num_envs_selected = rng.integers(1, num_envs, endpoint=True)
            num_bodies_selected = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_selected, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_selected, replace=False)

            # Convert to torch tensors
            env_ids = torch.from_numpy(env_ids_np).to(device=device, dtype=torch.int64)
            body_ids = torch.from_numpy(body_ids_np).to(device=device, dtype=torch.int64)

            # Get random forces (partial data shape)
            forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )
            forces = torch.from_numpy(forces_np).to(device=device, dtype=torch.float32)

            # Add forces to wrench composer
            wrench_composer.add_forces_and_torques(forces=forces, body_ids=body_ids, env_ids=env_ids)

            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np

        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_torch_add_torque(device: str, num_envs: int, num_bodies: int):
    """Test adding torques using torch tensors and indices."""
    rng = np.random.default_rng(seed=1)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            num_envs_selected = rng.integers(1, num_envs, endpoint=True)
            num_bodies_selected = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_selected, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_selected, replace=False)

            env_ids = torch.from_numpy(env_ids_np).to(device=device, dtype=torch.int64)
            body_ids = torch.from_numpy(body_ids_np).to(device=device, dtype=torch.int64)

            torques_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )
            torques = torch.from_numpy(torques_np).to(device=device, dtype=torch.float32)

            wrench_composer.add_forces_and_torques(torques=torques, body_ids=body_ids, env_ids=env_ids)

            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np

        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_torch_add_forces_at_positions(device: str, num_envs: int, num_bodies: int):
    """Test adding forces at local positions using torch tensors and indices."""
    rng = np.random.default_rng(seed=2)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            num_envs_selected = rng.integers(1, num_envs, endpoint=True)
            num_bodies_selected = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_selected, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_selected, replace=False)

            env_ids = torch.from_numpy(env_ids_np).to(device=device, dtype=torch.int64)
            body_ids = torch.from_numpy(body_ids_np).to(device=device, dtype=torch.int64)

            forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )
            positions_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )

            forces = torch.from_numpy(forces_np).to(device=device, dtype=torch.float32)
            positions = torch.from_numpy(positions_np).to(device=device, dtype=torch.float32)

            wrench_composer.add_forces_and_torques(
                forces=forces, positions=positions, body_ids=body_ids, env_ids=env_ids
            )

            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            torques_from_forces = np.cross(positions_np, forces_np)
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_from_forces

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1e-2, rtol=1e-5)
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_torch_add_forces_and_torques_at_position(device: str, num_envs: int, num_bodies: int):
    """Test adding forces and torques at local positions using torch tensors."""
    rng = np.random.default_rng(seed=4)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

        for _ in range(10):
            num_envs_selected = rng.integers(1, num_envs, endpoint=True)
            num_bodies_selected = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_selected, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_selected, replace=False)

            env_ids = torch.from_numpy(env_ids_np).to(device=device, dtype=torch.int64)
            body_ids = torch.from_numpy(body_ids_np).to(device=device, dtype=torch.int64)

            forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )
            torques_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )
            positions_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
                np.float32
            )

            forces = torch.from_numpy(forces_np).to(device=device, dtype=torch.float32)
            torques = torch.from_numpy(torques_np).to(device=device, dtype=torch.float32)
            positions = torch.from_numpy(positions_np).to(device=device, dtype=torch.float32)

            wrench_composer.add_forces_and_torques(
                forces=forces, torques=torques, positions=positions, body_ids=body_ids, env_ids=env_ids
            )

            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            torques_from_forces = np.cross(positions_np, forces_np)
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_from_forces
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1e-2, rtol=1e-5)
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_torch_reset(device: str, num_envs: int, num_bodies: int):
    """Test reset functionality using torch tensors."""
    rng = np.random.default_rng(seed=5)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)

        num_envs_selected = rng.integers(1, num_envs, endpoint=True)
        num_bodies_selected = rng.integers(1, num_bodies, endpoint=True)
        env_ids_np = rng.choice(num_envs, size=num_envs_selected, replace=False)
        body_ids_np = rng.choice(num_bodies, size=num_bodies_selected, replace=False)

        env_ids = torch.from_numpy(env_ids_np).to(device=device, dtype=torch.int64)
        body_ids = torch.from_numpy(body_ids_np).to(device=device, dtype=torch.int64)

        forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
            np.float32
        )
        torques_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs_selected, num_bodies_selected, 3)).astype(
            np.float32
        )

        forces = torch.from_numpy(forces_np).to(device=device, dtype=torch.float32)
        torques = torch.from_numpy(torques_np).to(device=device, dtype=torch.float32)

        wrench_composer.add_forces_and_torques(forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids)
        wrench_composer.reset()

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(composed_force_np, np.zeros((num_envs, num_bodies, 3)), atol=1e-6)
        assert np.allclose(composed_torque_np, np.zeros((num_envs, num_bodies, 3)), atol=1e-6)


# ============================================================================
# Global Frame Tests (COM Frame Transformation) - WARP CODE PATH
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_warp_global_forces_with_rotation(device: str, num_envs: int, num_bodies: int):
    """Test that global forces are correctly rotated to the COM frame using warp."""
    rng = np.random.default_rng(seed=10)

    for _ in range(5):
        # Create random COM quaternions
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global forces
        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

        # Apply global forces (no mask = all)
        wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

        # Compute expected local forces by rotating global forces by inverse quaternion
        expected_forces_local = quat_rotate_inv_np(com_quat_np, forces_global_np)

        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(
            composed_force_np, expected_forces_local, atol=1e-4, rtol=1e-5
        ), f"Global force rotation failed.\nExpected:\n{expected_forces_local}\nGot:\n{composed_force_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_warp_global_torques_with_rotation(device: str, num_envs: int, num_bodies: int):
    """Test that global torques are correctly rotated to the COM frame using warp."""
    rng = np.random.default_rng(seed=11)

    for _ in range(5):
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        torques_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        torques_global = wp.from_numpy(torques_global_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(torques=torques_global, is_global=True)

        expected_torques_local = quat_rotate_inv_np(com_quat_np, torques_global_np)

        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(
            composed_torque_np, expected_torques_local, atol=1e-4, rtol=1e-5
        ), f"Global torque rotation failed.\nExpected:\n{expected_torques_local}\nGot:\n{composed_torque_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_warp_global_forces_at_global_position(device: str, num_envs: int, num_bodies: int):
    """Test global forces at global positions with full coordinate transformation using warp."""
    rng = np.random.default_rng(seed=12)

    for _ in range(5):
        # Create random COM poses
        com_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_pos_torch = torch.from_numpy(com_pos_np)
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_pos=com_pos_torch, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Generate random global forces and positions
        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_global_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)
        positions_global = wp.from_numpy(positions_global_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_global, is_global=True)

        # Compute expected results:
        # 1. Force in COM frame = quat_rotate_inv(com_quat, global_force)
        expected_forces_local = quat_rotate_inv_np(com_quat_np, forces_global_np)

        # 2. Position offset in global frame = global_position - com_position
        position_offset_global = positions_global_np - com_pos_np

        # 3. Torque = skew(position_offset_global) @ force_local
        expected_torques_local = np.cross(position_offset_global, expected_forces_local)

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()

        assert np.allclose(
            composed_force_np, expected_forces_local, atol=1e-3, rtol=1e-4
        ), f"Global force at position failed.\nExpected forces:\n{expected_forces_local}\nGot:\n{composed_force_np}"
        assert np.allclose(
            composed_torque_np, expected_torques_local, atol=1e-3, rtol=1e-4
        ), f"Global force at position failed.\nExpected torques:\n{expected_torques_local}\nGot:\n{composed_torque_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_warp_local_vs_global_identity_quaternion(device: str):
    """Test that local and global give same result with identity quaternion and zero position (warp)."""
    rng = np.random.default_rng(seed=13)
    num_envs, num_bodies = 10, 5

    mock_asset_local = MockRigidObject(num_envs, num_bodies, device)
    mock_asset_global = MockRigidObject(num_envs, num_bodies, device)

    wrench_composer_local = WrenchComposer(mock_asset_local)
    wrench_composer_global = WrenchComposer(mock_asset_global)

    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

    wrench_composer_local.add_forces_and_torques(forces=forces, torques=torques, is_global=False)
    wrench_composer_global.add_forces_and_torques(forces=forces, torques=torques, is_global=True)

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
def test_warp_90_degree_rotation_global_force(device: str):
    """Test global force with a known 90-degree rotation for easy verification (warp)."""
    num_envs, num_bodies = 1, 1

    # 90-degree rotation around Z-axis: (w, x, y, z) = (cos(45°), 0, 0, sin(45°))
    angle = np.pi / 2
    com_quat_np = np.array([[[[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]]]], dtype=np.float32).reshape(1, 1, 4)
    com_quat_torch = torch.from_numpy(com_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    force_global = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    force_wp = wp.from_numpy(force_global, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques(forces=force_wp, is_global=True)

    # Expected: After inverse rotation (rotate by -90° around Z), X becomes -Y
    expected_force_local = np.array([[[0.0, -1.0, 0.0]]], dtype=np.float32)

    composed_force_np = wrench_composer.composed_force.numpy()
    assert np.allclose(
        composed_force_np, expected_force_local, atol=1e-5
    ), f"90-degree rotation test failed.\nExpected:\n{expected_force_local}\nGot:\n{composed_force_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_warp_composition_mixed_local_and_global(device: str):
    """Test that local and global forces can be composed together correctly (warp)."""
    rng = np.random.default_rng(seed=14)
    num_envs, num_bodies = 5, 3

    com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    com_quat_torch = torch.from_numpy(com_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

    forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques(forces=forces_local, is_global=False)
    wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

    global_forces_in_local = quat_rotate_inv_np(com_quat_np, forces_global_np)
    expected_total = forces_local_np + global_forces_in_local

    composed_force_np = wrench_composer.composed_force.numpy()
    assert np.allclose(
        composed_force_np, expected_total, atol=1e-4, rtol=1e-5
    ), f"Mixed local/global composition failed.\nExpected:\n{expected_total}\nGot:\n{composed_force_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_warp_local_forces_at_local_position(device: str, num_envs: int, num_bodies: int):
    """Test local forces at local positions using warp."""
    rng = np.random.default_rng(seed=15)

    for _ in range(5):
        # Random COM poses (shouldn't affect local frame calculations except for masking)
        com_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_pos_torch = torch.from_numpy(com_pos_np)
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_pos=com_pos_torch, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_local_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
        positions_local = wp.from_numpy(positions_local_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(forces=forces_local, positions=positions_local, is_global=False)

        expected_forces = forces_local_np
        expected_torques = np.cross(positions_local_np, forces_local_np)

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()

        assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
        assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_warp_global_force_at_com_origin_no_torque(device: str):
    """Test that a global force applied at the COM origin produces no torque (warp)."""
    rng = np.random.default_rng(seed=16)
    num_envs, num_bodies = 5, 3

    com_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    com_pos_torch = torch.from_numpy(com_pos_np)
    com_quat_torch = torch.from_numpy(com_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, com_pos=com_pos_torch, com_quat=com_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    # Position = COM position (so offset is zero)
    positions_at_com = wp.from_numpy(com_pos_np, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_at_com, is_global=True)

    expected_forces = quat_rotate_inv_np(com_quat_np, forces_global_np)
    expected_torques = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

    composed_force_np = wrench_composer.composed_force.numpy()
    composed_torque_np = wrench_composer.composed_torque.numpy()

    assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
    assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)


# ============================================================================
# Global Frame Tests (COM Frame Transformation) - TORCH CODE PATH
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_torch_global_forces_with_rotation(device: str, num_envs: int, num_bodies: int):
    """Test that global forces are correctly rotated to the COM frame using torch."""
    rng = np.random.default_rng(seed=10)

    for _ in range(5):
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = torch.from_numpy(forces_global_np).to(device=device, dtype=torch.float32)

        wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

        expected_forces_local = quat_rotate_inv_np(com_quat_np, forces_global_np)

        composed_force_np = wrench_composer.composed_force.numpy()
        assert np.allclose(
            composed_force_np, expected_forces_local, atol=1e-4, rtol=1e-5
        ), f"Global force rotation failed.\nExpected:\n{expected_forces_local}\nGot:\n{composed_force_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_torch_global_torques_with_rotation(device: str, num_envs: int, num_bodies: int):
    """Test that global torques are correctly rotated to the COM frame using torch."""
    rng = np.random.default_rng(seed=11)

    for _ in range(5):
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        torques_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        torques_global = torch.from_numpy(torques_global_np).to(device=device, dtype=torch.float32)

        wrench_composer.add_forces_and_torques(torques=torques_global, is_global=True)

        expected_torques_local = quat_rotate_inv_np(com_quat_np, torques_global_np)

        composed_torque_np = wrench_composer.composed_torque.numpy()
        assert np.allclose(
            composed_torque_np, expected_torques_local, atol=1e-4, rtol=1e-5
        ), f"Global torque rotation failed.\nExpected:\n{expected_torques_local}\nGot:\n{composed_torque_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_torch_global_forces_at_global_position(device: str, num_envs: int, num_bodies: int):
    """Test global forces at global positions with full coordinate transformation using torch."""
    rng = np.random.default_rng(seed=12)

    for _ in range(5):
        com_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_pos_torch = torch.from_numpy(com_pos_np)
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_pos=com_pos_torch, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_global_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = torch.from_numpy(forces_global_np).to(device=device, dtype=torch.float32)
        positions_global = torch.from_numpy(positions_global_np).to(device=device, dtype=torch.float32)

        wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_global, is_global=True)

        expected_forces_local = quat_rotate_inv_np(com_quat_np, forces_global_np)
        position_offset_global = positions_global_np - com_pos_np
        expected_torques_local = np.cross(position_offset_global, expected_forces_local)

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()

        assert np.allclose(
            composed_force_np, expected_forces_local, atol=1e-3, rtol=1e-4
        ), f"Global force at position failed.\nExpected forces:\n{expected_forces_local}\nGot:\n{composed_force_np}"
        assert np.allclose(
            composed_torque_np, expected_torques_local, atol=1e-3, rtol=1e-4
        ), f"Global force at position failed.\nExpected torques:\n{expected_torques_local}\nGot:\n{composed_torque_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_torch_local_vs_global_identity_quaternion(device: str):
    """Test that local and global give same result with identity quaternion and zero position (torch)."""
    rng = np.random.default_rng(seed=13)
    num_envs, num_bodies = 10, 5

    mock_asset_local = MockRigidObject(num_envs, num_bodies, device)
    mock_asset_global = MockRigidObject(num_envs, num_bodies, device)

    wrench_composer_local = WrenchComposer(mock_asset_local)
    wrench_composer_global = WrenchComposer(mock_asset_global)

    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = torch.from_numpy(forces_np).to(device=device, dtype=torch.float32)
    torques = torch.from_numpy(torques_np).to(device=device, dtype=torch.float32)

    wrench_composer_local.add_forces_and_torques(forces=forces, torques=torques, is_global=False)
    wrench_composer_global.add_forces_and_torques(forces=forces, torques=torques, is_global=True)

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
def test_torch_90_degree_rotation_global_force(device: str):
    """Test global force with a known 90-degree rotation for easy verification (torch)."""
    num_envs, num_bodies = 1, 1

    angle = np.pi / 2
    com_quat_np = np.array([[[[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]]]], dtype=np.float32).reshape(1, 1, 4)
    com_quat_torch = torch.from_numpy(com_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    force_global = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32, device=device)

    wrench_composer.add_forces_and_torques(forces=force_global, is_global=True)

    expected_force_local = np.array([[[0.0, -1.0, 0.0]]], dtype=np.float32)

    composed_force_np = wrench_composer.composed_force.numpy()
    assert np.allclose(
        composed_force_np, expected_force_local, atol=1e-5
    ), f"90-degree rotation test failed.\nExpected:\n{expected_force_local}\nGot:\n{composed_force_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_torch_composition_mixed_local_and_global(device: str):
    """Test that local and global forces can be composed together correctly (torch)."""
    rng = np.random.default_rng(seed=14)
    num_envs, num_bodies = 5, 3

    com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    com_quat_torch = torch.from_numpy(com_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, com_quat=com_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

    forces_local = torch.from_numpy(forces_local_np).to(device=device, dtype=torch.float32)
    forces_global = torch.from_numpy(forces_global_np).to(device=device, dtype=torch.float32)

    wrench_composer.add_forces_and_torques(forces=forces_local, is_global=False)
    wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

    global_forces_in_local = quat_rotate_inv_np(com_quat_np, forces_global_np)
    expected_total = forces_local_np + global_forces_in_local

    composed_force_np = wrench_composer.composed_force.numpy()
    assert np.allclose(
        composed_force_np, expected_total, atol=1e-4, rtol=1e-5
    ), f"Mixed local/global composition failed.\nExpected:\n{expected_total}\nGot:\n{composed_force_np}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_torch_local_forces_at_local_position(device: str, num_envs: int, num_bodies: int):
    """Test local forces at local positions using torch."""
    rng = np.random.default_rng(seed=15)

    for _ in range(5):
        com_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        com_pos_torch = torch.from_numpy(com_pos_np)
        com_quat_torch = torch.from_numpy(com_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, com_pos=com_pos_torch, com_quat=com_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_local_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_local = torch.from_numpy(forces_local_np).to(device=device, dtype=torch.float32)
        positions_local = torch.from_numpy(positions_local_np).to(device=device, dtype=torch.float32)

        wrench_composer.add_forces_and_torques(forces=forces_local, positions=positions_local, is_global=False)

        expected_forces = forces_local_np
        expected_torques = np.cross(positions_local_np, forces_local_np)

        composed_force_np = wrench_composer.composed_force.numpy()
        composed_torque_np = wrench_composer.composed_torque.numpy()

        assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
        assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_torch_global_force_at_com_origin_no_torque(device: str):
    """Test that a global force applied at the COM origin produces no torque (torch)."""
    rng = np.random.default_rng(seed=16)
    num_envs, num_bodies = 5, 3

    com_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    com_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    com_pos_torch = torch.from_numpy(com_pos_np)
    com_quat_torch = torch.from_numpy(com_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, com_pos=com_pos_torch, com_quat=com_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = torch.from_numpy(forces_global_np).to(device=device, dtype=torch.float32)

    positions_at_com = torch.from_numpy(com_pos_np).to(device=device, dtype=torch.float32)

    wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_at_com, is_global=True)

    expected_forces = quat_rotate_inv_np(com_quat_np, forces_global_np)
    expected_torques = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

    composed_force_np = wrench_composer.composed_force.numpy()
    composed_torque_np = wrench_composer.composed_torque.numpy()

    assert np.allclose(composed_force_np, expected_forces, atol=1e-4, rtol=1e-5)
    assert np.allclose(composed_torque_np, expected_torques, atol=1e-4, rtol=1e-5)


# ============================================================================
# Set Forces and Torques Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_warp_set_forces_and_torques(device: str, num_envs: int, num_bodies: int):
    """Test setting forces and torques (overwrites rather than adds) using warp."""
    rng = np.random.default_rng(seed=20)

    for _ in range(5):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)

        # First add some forces
        forces1_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
        forces1 = wp.from_numpy(forces1_np, dtype=wp.vec3f, device=device)
        wrench_composer.add_forces_and_torques(forces=forces1)

        # Now set new forces (should overwrite, not add)
        forces2_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
        forces2 = wp.from_numpy(forces2_np, dtype=wp.vec3f, device=device)
        wrench_composer.set_forces_and_torques(forces=forces2)

        composed_force_np = wrench_composer.composed_force.numpy()
        # Should be equal to forces2, not forces1 + forces2
        assert np.allclose(composed_force_np, forces2_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_torch_set_forces_and_torques(device: str, num_envs: int, num_bodies: int):
    """Test setting forces and torques (overwrites rather than adds) using torch."""
    rng = np.random.default_rng(seed=20)

    for _ in range(5):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)

        # First add some forces
        forces1_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
        forces1 = torch.from_numpy(forces1_np).to(device=device, dtype=torch.float32)
        wrench_composer.add_forces_and_torques(forces=forces1)

        # Now set new forces (should overwrite, not add)
        forces2_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
        forces2 = torch.from_numpy(forces2_np).to(device=device, dtype=torch.float32)
        wrench_composer.set_forces_and_torques(forces=forces2)

        composed_force_np = wrench_composer.composed_force.numpy()
        # Should be equal to forces2, not forces1 + forces2
        assert np.allclose(composed_force_np, forces2_np, atol=1e-4, rtol=1e-5)


# ============================================================================
# Partial Reset Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_torch_partial_reset(device: str):
    """Test partial reset (reset only specific environments) using torch."""
    rng = np.random.default_rng(seed=21)
    num_envs, num_bodies = 10, 5

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    # Add forces to all envs
    forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
    forces = torch.from_numpy(forces_np).to(device=device, dtype=torch.float32)
    wrench_composer.add_forces_and_torques(forces=forces)

    # Partial reset - only reset first half of envs
    reset_env_ids = torch.arange(num_envs // 2, device=device)
    wrench_composer.reset(env_ids=reset_env_ids)

    composed_force_np = wrench_composer.composed_force.numpy()

    # First half should be zero
    assert np.allclose(composed_force_np[: num_envs // 2], np.zeros((num_envs // 2, num_bodies, 3)), atol=1e-6)
    # Second half should still have the forces
    assert np.allclose(composed_force_np[num_envs // 2 :], forces_np[num_envs // 2 :], atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_warp_partial_reset_with_mask(device: str):
    """Test partial reset using warp mask."""
    rng = np.random.default_rng(seed=22)
    num_envs, num_bodies = 10, 5

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    # Add forces to all envs
    forces_np = rng.uniform(low=-100.0, high=100.0, size=(num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces)

    # Partial reset using mask - reset odd indices
    env_mask_np = np.array([i % 2 == 1 for i in range(num_envs)], dtype=bool)
    env_mask = wp.from_numpy(env_mask_np, dtype=wp.bool, device=device)
    wrench_composer.reset(env_mask=env_mask)

    composed_force_np = wrench_composer.composed_force.numpy()

    # Check that odd envs are zero, even envs still have forces
    for i in range(num_envs):
        if i % 2 == 1:
            assert np.allclose(composed_force_np[i], np.zeros((num_bodies, 3)), atol=1e-6)
        else:
            assert np.allclose(composed_force_np[i], forces_np[i], atol=1e-4)
