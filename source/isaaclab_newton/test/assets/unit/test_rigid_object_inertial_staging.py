# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused kernel tests for Newton rigid-body inertial staging."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets import kernels

pytestmark = pytest.mark.unit


def _diagonal_inertias(values: list[tuple[float, float, float]]) -> wp.array:
    data = np.zeros((2, 2, 9), dtype=np.float32)
    for body_index, diagonal in enumerate(values):
        data[0, body_index] = np.diag(diagonal).reshape(9)
        data[1, body_index] = np.diag(diagonal).reshape(9)
    return wp.array(data, dtype=wp.float32, device="cpu")


def test_masked_mass_staging_updates_only_selected_inverse_properties() -> None:
    """A selected positive-to-static transition zeros both Newton inverse arrays."""
    masses = wp.array([[1.0, 1.0], [1.0, 0.0]], dtype=wp.float32, device="cpu")
    body_mass = wp.ones((2, 2), dtype=wp.float32, device="cpu")
    body_inertia = _diagonal_inertias([(2.0, 3.0, 4.0), (5.0, 6.0, 8.0)])
    body_inv_mass = wp.ones((2, 2), dtype=wp.float32, device="cpu")
    body_inv_inertia = wp.ones((2, 2), dtype=wp.mat33f, device="cpu")
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    body_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    body_indices = wp.array([0, 1], dtype=wp.int32, device="cpu")

    wp.launch(
        kernels.write_body_mass_and_inverse_mask,
        dim=(2, 2),
        inputs=[masses, env_mask, body_mask, body_indices, False, body_inertia],
        outputs=[body_mass, body_mass, body_inv_mass, body_inv_inertia],
        device="cpu",
    )

    np.testing.assert_allclose(body_mass.numpy(), [[1.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(body_inv_mass.numpy(), [[1.0, 1.0], [1.0, 0.0]])
    expected_inverse = np.ones((2, 2, 3, 3), dtype=np.float32)
    expected_inverse[1, 1] = 0.0
    np.testing.assert_allclose(body_inv_inertia.numpy(), expected_inverse)


def test_masked_inertia_staging_inverts_selected_matrix_without_changing_mass() -> None:
    """A selected inertia write updates its inverse while preserving inverse mass."""
    inertias = _diagonal_inertias([(2.0, 4.0, 8.0), (3.0, 5.0, 10.0)])
    body_mass = wp.full((2, 2), value=2.0, dtype=wp.float32, device="cpu")
    body_inertia = _diagonal_inertias([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
    body_inv_mass = wp.full((2, 2), value=0.5, dtype=wp.float32, device="cpu")
    body_inv_inertia = wp.ones((2, 2), dtype=wp.mat33f, device="cpu")
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    body_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    body_indices = wp.array([0, 1], dtype=wp.int32, device="cpu")

    wp.launch(
        kernels.write_body_inertia_and_inverse_mask,
        dim=(2, 2),
        inputs=[inertias, env_mask, body_mask, body_indices, False, body_mass],
        outputs=[body_inertia, body_inertia, body_inv_mass, body_inv_inertia],
        device="cpu",
    )

    expected_inertia = _diagonal_inertias([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]).numpy()
    expected_inertia[0, 1] = np.diag([3.0, 5.0, 10.0]).reshape(9)
    np.testing.assert_allclose(body_inertia.numpy(), expected_inertia)
    np.testing.assert_allclose(body_inv_mass.numpy(), np.full((2, 2), 0.5, dtype=np.float32))
    expected_inverse = np.ones((2, 2, 3, 3), dtype=np.float32)
    expected_inverse[0, 1] = np.diag([1.0 / 3.0, 0.2, 0.1])
    np.testing.assert_allclose(body_inv_inertia.numpy(), expected_inverse, atol=1e-6)
