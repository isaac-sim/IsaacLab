# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-manager and tensor-view doubles shared by the mocked contract factories."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import warp as wp

from ._manager_patch_scope import patch_contract_manager


def patch_physx_manager() -> None:
    """Give PhysX data classes gravity and a scene-data backend without creating a physics scene."""
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    physics_sim_view = MagicMock()
    physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)
    patch_contract_manager(PhysxManager, "get_physics_sim_view", MagicMock(return_value=physics_sim_view))
    # Writers bump the scene-data transform version that ``initialize()`` would normally create.
    patch_contract_manager(PhysxManager, "_scene_data_backend", PhysxSceneDataBackend())


def patch_ovphysx_manager() -> None:
    """Create the OVPhysX scene-data backend whose transform version the writers bump."""
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager, OvPhysxSceneDataBackend

    patch_contract_manager(OvPhysxManager, "_scene_data_backend", OvPhysxSceneDataBackend())


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
