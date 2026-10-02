# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-manager and tensor-view doubles shared by the mocked contract factories."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp


def patch_physx_manager(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give PhysX data classes gravity and a scene-data backend without creating a physics scene."""
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    physics_sim_view = MagicMock()
    physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)
    monkeypatch.setattr(PhysxManager, "get_physics_sim_view", MagicMock(return_value=physics_sim_view), raising=False)
    # Writers bump the scene-data transform version that ``initialize()`` would normally create.
    monkeypatch.setattr(PhysxManager, "_scene_data_backend", PhysxSceneDataBackend(), raising=False)


def patch_ovphysx_manager(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create the OVPhysX scene-data backend whose transform version the writers bump."""
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager, OvPhysxSceneDataBackend

    monkeypatch.setattr(OvPhysxManager, "_scene_data_backend", OvPhysxSceneDataBackend(), raising=False)


def install_physx_recording_setters(mock_view, storage_by_method: dict[str, str]) -> None:
    """Replace PhysX view setters with TensorAPI-faithful ones that apply the selected rows to view storage.

    The PhysX TensorAPI receives full-size data and applies only the rows named by ``indices``. The fixture view
    setters either drop writes or expect compact data, so the contract installs setters that require full-size data
    and leave every unselected row untouched.

    Args:
        mock_view: PhysX fixture view whose setters are replaced.
        storage_by_method: Setter name to the view attribute that stores its values.
    """

    def make_setter(storage_name: str) -> Callable[[wp.array, wp.array | None], None]:
        def setter(values: wp.array, indices: wp.array | None = None) -> None:
            values_np = values.numpy()
            stored = getattr(mock_view, storage_name, None)
            if stored is None:
                stored = wp.zeros((mock_view._count, *values_np.shape[1:]), dtype=wp.float32, device=values.device)
                setattr(mock_view, storage_name, stored)
            stored_np = stored.numpy()
            if indices is None:
                stored_np[...] = values_np.reshape(stored_np.shape)
            else:
                if values_np.size != stored_np.size:
                    raise ValueError(f"{storage_name}: expected full-size data of shape {stored_np.shape}")
                rows = indices.numpy().astype(np.int64)
                stored_np[rows] = values_np.reshape(stored_np.shape)[rows]
            if stored.device.is_cuda:
                stored.assign(wp.array(stored_np, dtype=stored.dtype, device=stored.device))

        return setter

    mock_view._noop_setters = False
    for method_name, storage_name in storage_by_method.items():
        setattr(mock_view, method_name, make_setter(storage_name))


def read_backend_joint_state(backend: str, art, raw_backend) -> tuple[np.ndarray, np.ndarray]:
    """Return joint position and velocity from backend-order storage."""
    if backend == "physx":
        return raw_backend._dof_positions.numpy().copy(), raw_backend._dof_velocities.numpy().copy()
    if backend == "ovphysx":
        from isaaclab_ov import tensor_types as TT

        position = np.asarray(raw_backend.bindings[TT.DOF_POSITION]._data).copy()
        velocity = np.asarray(raw_backend.bindings[TT.DOF_VELOCITY]._data).copy()
        return position, velocity
    if backend == "newton":
        return art.data._sim_bind_joint_pos.numpy().copy(), art.data._sim_bind_joint_vel.numpy().copy()
    raise AssertionError(f"Unsupported backend for joint-state parity test: {backend}")
