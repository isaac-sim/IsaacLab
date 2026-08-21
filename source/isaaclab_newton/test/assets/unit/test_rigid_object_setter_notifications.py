# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused notification tests for Newton rigid-asset inertial setters."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags


def _diagonal_inertias(num_bodies: int, diagonal: tuple[float, float, float]) -> wp.array:
    data = np.zeros((2, num_bodies, 9), dtype=np.float32)
    data[:] = np.diag(diagonal).reshape(9)
    return wp.array(data, dtype=wp.float32, device="cpu")


def _minimal_asset(asset_type, num_bodies: int):
    """Create a tiny array-backed asset for invoking production setters."""
    asset = object.__new__(asset_type)
    asset._device = "cpu"
    asset._check_shapes = False
    asset._ALL_BODY_INDICES = wp.array(np.arange(num_bodies), dtype=wp.int32, device="cpu")
    if asset_type in (Articulation, RigidObject):
        asset._ALL_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
    else:
        asset._ALL_ENV_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
    staged_inertia = _diagonal_inertias(num_bodies, (2.0, 3.0, 4.0))
    data = SimpleNamespace(
        has_body_ordering=False,
        body_ordering=None,
        _body_mass_user=None,
        _body_inertia_user=None,
        _sim_bind_body_mass=wp.ones((2, num_bodies), dtype=wp.float32, device="cpu"),
        _sim_bind_body_inv_mass=wp.ones((2, num_bodies), dtype=wp.float32, device="cpu"),
        _sim_bind_body_inv_inertia=wp.ones((2, num_bodies), dtype=wp.mat33f, device="cpu"),
        _sim_bind_body_inertia=staged_inertia,
        _body_inertia=staged_inertia,
    )
    asset._data = data
    return asset, staged_inertia


@pytest.mark.parametrize(
    ("asset_type", "num_bodies"),
    [(Articulation, 2), (RigidObject, 1), (RigidObjectCollection, 2)],
)
def test_mass_setter_stages_only_selection_and_notifies_inertial_change(
    asset_type, num_bodies: int, monkeypatch
) -> None:
    """Production mass setters stage one selection and emit exactly one inertial notification."""
    asset, _ = _minimal_asset(asset_type, num_bodies)
    notifications = []
    monkeypatch.setattr(
        SimulationManager,
        "add_model_change",
        classmethod(lambda cls, flag: notifications.append(flag)),
    )
    env_ids = wp.array([1], dtype=wp.int32, device="cpu")
    body_ids = wp.array([num_bodies - 1], dtype=wp.int32, device="cpu")

    asset.set_masses_index(
        masses=wp.array([[4.0]], dtype=wp.float32, device="cpu"),
        env_ids=env_ids,
        body_ids=body_ids,
    )

    expected_mass = np.ones((2, num_bodies), dtype=np.float32)
    expected_mass[1, -1] = 4.0
    expected_inv_mass = np.ones((2, num_bodies), dtype=np.float32)
    expected_inv_mass[1, -1] = 0.25
    np.testing.assert_allclose(asset.data._sim_bind_body_mass.numpy(), expected_mass)
    np.testing.assert_allclose(asset.data._sim_bind_body_inv_mass.numpy(), expected_inv_mass)
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]


@pytest.mark.parametrize(
    ("asset_type", "num_bodies"),
    [(Articulation, 2), (RigidObject, 1), (RigidObjectCollection, 2)],
)
def test_inertia_setter_stages_only_selection_and_notifies_inertial_change(
    asset_type, num_bodies: int, monkeypatch
) -> None:
    """Production inertia setters stage one selection and emit exactly one inertial notification."""
    asset, staged_inertia = _minimal_asset(asset_type, num_bodies)
    notifications = []
    monkeypatch.setattr(
        SimulationManager,
        "add_model_change",
        classmethod(lambda cls, flag: notifications.append(flag)),
    )
    env_ids = wp.array([1], dtype=wp.int32, device="cpu")
    body_ids = wp.array([num_bodies - 1], dtype=wp.int32, device="cpu")
    inertias = np.zeros((1, 1, 9), dtype=np.float32)
    inertias[0, 0] = np.diag([5.0, 10.0, 20.0]).reshape(9)

    asset.set_inertias_index(
        inertias=wp.array(inertias, dtype=wp.float32, device="cpu"),
        env_ids=env_ids,
        body_ids=body_ids,
    )

    expected_inertia = _diagonal_inertias(num_bodies, (2.0, 3.0, 4.0)).numpy()
    expected_inertia[1, -1] = inertias[0, 0]
    np.testing.assert_allclose(staged_inertia.numpy(), expected_inertia)
    expected_inv_inertia = np.ones((2, num_bodies, 3, 3), dtype=np.float32)
    expected_inv_inertia[1, -1] = np.diag([0.2, 0.1, 0.05])
    np.testing.assert_allclose(asset.data._sim_bind_body_inv_inertia.numpy(), expected_inv_inertia, atol=1e-6)
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]
