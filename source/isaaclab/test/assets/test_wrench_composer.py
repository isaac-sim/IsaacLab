# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import numpy as np

import pytest
import warp as wp

from isaaclab.assets.utils.wrench_composer import WrenchComposer


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_force(device: str, num_envs: int, num_bodies: int):
    # Initialize random number generator
    rng = np.random.default_rng(seed=0)

    for _ in range(10):
        wrench_composer = WrenchComposer(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(env_ids, body_ids, forces=forces)
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force_as_numpy
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_torque(device: str, num_envs: int, num_bodies: int):
    # Initialize random number generator
    rng = np.random.default_rng(seed=1)

    for _ in range(10):
        wrench_composer = WrenchComposer(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(env_ids, body_ids, torques=torques)
            # Add torques to hand-calculated composed torque
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque_as_numpy
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_at_positons(device: str, num_envs: int, num_bodies: int):

    def skew(vector):
        skew = np.zeros((vector.shape[0], vector.shape[1], 3, 3), dtype=np.float32)
        skew[:, :, 0, 1] = -vector[:, :, 2]
        skew[:, :, 0, 2] = vector[:, :, 1]
        skew[:, :, 1, 0] = vector[:, :, 2]
        skew[:, :, 1, 2] = -vector[:, :, 0]
        skew[:, :, 2, 0] = -vector[:, :, 1]
        skew[:, :, 2, 1] = vector[:, :, 0]
        return skew

    rng = np.random.default_rng(seed=2)

    for _ in range(10):
        # Initialize wrench composer
        wrench_composer = WrenchComposer(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(env_ids, body_ids, forces=forces, positions=positions)
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            # Add torques to hand-calculated composed torque
            skew_matrices = skew(positions_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += (
                        skew_matrices[i, j, :, :] @ forces_np[i, j, :]
                    )
            # hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += np.einsum('EBij,EBi->EBi', forces_np, skew_matrices)

        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force_as_numpy
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque_as_numpy
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_torques_at_position(device: str, num_envs: int, num_bodies: int):
    rng = np.random.default_rng(seed=3)

    for _ in range(10):
        wrench_composer = WrenchComposer(num_envs, num_bodies, device)
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
            wrench_composer.add_forces_and_torques(env_ids, body_ids, torques=torques, positions=positions)
            # Add torques to hand-calculated composed torque
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque_as_numpy
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_and_torques_at_position(device: str, num_envs: int, num_bodies: int):

    def skew(vector):
        skew = np.zeros((vector.shape[0], vector.shape[1], 3, 3), dtype=np.float32)
        skew[:, :, 0, 1] = -vector[:, :, 2]
        skew[:, :, 0, 2] = vector[:, :, 1]
        skew[:, :, 1, 0] = vector[:, :, 2]
        skew[:, :, 1, 2] = -vector[:, :, 0]
        skew[:, :, 2, 0] = -vector[:, :, 1]
        skew[:, :, 2, 1] = vector[:, :, 0]
        return skew

    rng = np.random.default_rng(seed=4)

    for _ in range(10):
        wrench_composer = WrenchComposer(num_envs, num_bodies, device)
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
                env_ids, body_ids, forces=forces, torques=torques, positions=positions
            )
            # Add forces to hand-calculated composed force
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            # Add torques to hand-calculated composed torque
            skew_matrices = skew(positions_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += (
                        skew_matrices[i, j, :, :] @ forces_np[i, j, :]
                    )
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        # Get composed force from wrench composer
        composed_force_np = wrench_composer.composed_force_as_numpy
        assert np.allclose(composed_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        # Get composed torque from wrench composer
        composed_torque_np = wrench_composer.composed_torque_as_numpy
        assert np.allclose(composed_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_reset(device: str, num_envs: int, num_bodies: int):
    rng = np.random.default_rng(seed=5)
    for _ in range(10):
        wrench_composer = WrenchComposer(num_envs, num_bodies, device)
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
        wrench_composer.add_forces_and_torques(env_ids, body_ids, forces=forces, torques=torques)
        # Reset wrench composer
        wrench_composer.reset()
        # Get composed force and torque from wrench composer
        composed_force_np = wrench_composer.composed_force_as_numpy
        composed_torque_np = wrench_composer.composed_torque_as_numpy
        assert np.allclose(composed_force_np, np.zeros((num_envs, num_bodies, 3)), atol=1, rtol=1e-7)
        assert np.allclose(composed_torque_np, np.zeros((num_envs, num_bodies, 3)), atol=1, rtol=1e-7)
