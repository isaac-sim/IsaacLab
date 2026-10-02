# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared asset API contract: counts, names, finders, index resolution, and the classified public surface."""

import inspect
from importlib.util import find_spec

import pytest

from .articulation_factory import get_articulation
from .capabilities import BACKEND_STATUSES, available_backends
from .public_surface import BASE_SURFACE_CLASSES, PUBLIC_SURFACE_CONTRACTS, public_surface_mismatches
from .rigid_object_collection_factory import get_rigid_object_collection
from .rigid_object_factory import get_rigid_object

pytestmark = pytest.mark.integration

_MISSING = object()


def test_installed_backend_packages_run_the_contract() -> None:
    """An installed backend package must not silently drop out of the contract matrix."""
    unavailable = {
        status.declaration.name: status.reason
        for status in BACKEND_STATUSES
        if not status.available
        and status.reason != "CUDA runtime unavailable"
        and find_spec(status.declaration.required_modules[-1]) is not None
    }

    assert unavailable == {}


def _manager_bindings() -> dict[str, object]:
    """Return the backend-manager bindings that the mocked factories replace during a test."""
    bindings = {}
    backends = available_backends()
    if "physx" in backends:
        from isaaclab_physx.physics import PhysxManager

        for name in ("get_physics_sim_view", "_scene_data_backend"):
            bindings[f"physx.{name}"] = inspect.getattr_static(PhysxManager, name, _MISSING)
    if "ovphysx" in backends:
        from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

        bindings["ovphysx._scene_data_backend"] = inspect.getattr_static(
            OvPhysxManager, "_scene_data_backend", _MISSING
        )
    if "newton" in backends:
        import isaaclab_newton.assets.articulation.articulation_data as articulation_data
        import isaaclab_newton.assets.rigid_object.rigid_object_data as rigid_object_data
        import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection as collection
        import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data as collection_data

        for module in (articulation_data, rigid_object_data, collection, collection_data):
            bindings[f"newton.{module.__name__}"] = module.SimulationManager
    return bindings


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_contract_factories_scope_manager_patches_to_one_test(reverse: bool) -> None:
    """Factories patch backend managers only inside a test scope and restore the production bindings after it."""
    original = _manager_bindings()
    # Importing the contract modules must not patch a manager: production bindings precede every factory call.
    if "physx" in available_backends():
        assert isinstance(original["physx.get_physics_sim_view"], classmethod)
    if "newton" in available_backends():
        from isaaclab_newton.physics import NewtonManager

        assert all(manager is NewtonManager for name, manager in original.items() if name.startswith("newton."))

    backends = available_backends()[::-1] if reverse else available_backends()
    patches = pytest.MonkeyPatch()
    try:
        for backend in backends:
            rigid_object, _ = get_rigid_object(backend, device="cpu", monkeypatch=patches)
            collection, _ = get_rigid_object_collection(backend, device="cpu", monkeypatch=patches)
            articulation, _ = get_articulation(backend, device="cpu", monkeypatch=patches)
            # Later factories must not break the patched managers the earlier families read through.
            assert rigid_object.data.root_link_pose_w.shape == (2,)
            assert collection.data.body_link_pose_w.shape == (2, 3)
            assert articulation.data.root_link_pose_w.shape == (2,)

    finally:
        patches.undo()

    restored = _manager_bindings()
    assert restored.keys() == original.keys()
    assert all(restored[name] is binding for name, binding in original.items())


def test_base_public_surface_has_an_explicit_contract_classification() -> None:
    """Every public member of the asset base classes is classified, and every classified member still exists."""
    assert public_surface_mismatches(BASE_SURFACE_CLASSES, PUBLIC_SURFACE_CONTRACTS) == {
        "missing": [],
        "stale": [],
        "unreasoned": [],
    }
