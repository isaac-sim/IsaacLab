# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend bootstrap, availability, and manager doubles shared by the asset contract tests.

Importing this module launches Kit when the process runs inside Isaac Sim. Otherwise it stubs the Kit-only modules
that the PhysX and OVPhysX asset classes import, so the real asset and data classes run against mocked views.
"""

import os
import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

if "ovphysx" not in os.environ.get("LD_PRELOAD", "") and (os.environ.get("LD_PRELOAD") or "EXP_PATH" in os.environ):
    from isaaclab.test.utils import launch_test_simulation

    launch_test_simulation()
else:
    import omni  # noqa: F401  # real namespace package; the stubs below become its attributes

    for _name, _is_package in (
        ("carb", False),
        ("usdrt", True),
        ("omni.kit", True),
        ("omni.kit.app", False),
        ("omni.physics", True),
        ("omni.physics.tensors", False),
        ("omni.physx", False),
        ("omni.timeline", False),
        ("omni.usd", False),
        ("isaacsim", True),
        ("isaacsim.core", True),
        ("isaacsim.core.simulation_manager", False),
    ):
        if _name in sys.modules:
            continue
        _stub = MagicMock()
        _stub.__spec__ = ModuleSpec(_name, loader=None, is_package=_is_package)
        if _is_package:
            _stub.__path__ = []
        sys.modules[_name] = _stub
        if "." in _name:
            _parent, _attribute = _name.rsplit(".", 1)
            setattr(sys.modules[_parent], _attribute, _stub)
    sys.modules["omni.kit.app"].get_app.return_value = None


def _unavailable_reason(*modules: str, cuda: bool = False) -> str | None:
    """Return why a backend cannot run in this process, or None when it can."""
    for module in modules:
        if find_spec(module) is None:
            return f"missing module: {module}"
    if cuda and not wp.is_cuda_available():
        # The mocked OVPhysX bindings allocate pinned host staging buffers even for CPU tensors.
        return "requires a CUDA runtime"
    return None


UNAVAILABLE = {
    name: reason
    for name, reason in {
        "physx": _unavailable_reason("isaaclab_physx"),
        "newton": _unavailable_reason("isaaclab_newton"),
        "ovphysx": _unavailable_reason("ovphysx", "isaaclab_ov", cuda=True),
    }.items()
    if reason is not None
}
"""Backend name to the reason it is skipped in this process."""

AVAILABLE = [name for name in ("physx", "newton", "ovphysx") if name not in UNAVAILABLE]
"""Backends that run in this process."""


def backends(*names: str) -> list:
    """Return pytest parameters for the named backends (default: all), skipping the unavailable ones."""
    return [
        pytest.param(name, id=name, marks=pytest.mark.skipif(name in UNAVAILABLE, reason=UNAVAILABLE.get(name, "")))
        for name in names or ("physx", "newton", "ovphysx")
    ]


def requires(name: str) -> pytest.MarkDecorator:
    """Skip a backend-specific test when that backend is unavailable."""
    return pytest.mark.skipif(name in UNAVAILABLE, reason=f"{name}: {UNAVAILABLE.get(name)}")


def patch_physx_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give PhysX data classes gravity and a scene-data backend without creating a physics scene."""
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    physics_sim_view = MagicMock()
    physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)
    monkeypatch.setattr(PhysxManager, "get_physics_sim_view", MagicMock(return_value=physics_sim_view), raising=False)
    # Writers bump the scene-data transform version that ``initialize()`` would normally create.
    monkeypatch.setattr(PhysxManager, "_scene_data_backend", PhysxSceneDataBackend(), raising=False)


def patch_ovphysx_manager(monkeypatch: pytest.MonkeyPatch) -> None:
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
    return art.data._sim_bind_joint_pos.numpy().copy(), art.data._sim_bind_joint_vel.numpy().copy()
