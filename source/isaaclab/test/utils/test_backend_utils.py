# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend module resolution."""

import pytest

from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.backend_utils import FactoryBase


@pytest.mark.parametrize(
    "backend, package",
    [("ovphysx", "isaaclab_ov"), ("physx", "isaaclab_physx"), ("newton", "isaaclab_newton")],
)
def test_get_module_name_routes_backend_to_package(monkeypatch, backend, package):
    """OVPhysX backend modules resolve from the consolidated package; others use their backend-named packages."""
    monkeypatch.setattr(FactoryBase, "_module_subpath", "assets.articulation", raising=False)

    assert FactoryBase._get_package_name(backend) == package
    assert FactoryBase._get_module_name(backend) == f"{package}.assets.articulation"


def test_factory_backend_falls_back_to_newton_without_simulation_context(monkeypatch):
    """Backend resolution uses Newton before a simulation context exists."""
    monkeypatch.setattr(SimulationContext, "_instance", None)

    assert FactoryBase._get_backend() == "newton"


def test_factory_and_visualizer_use_inherited_backend_identifier(monkeypatch):
    """Renaming a solver manager must not change backend dispatch or visualizer capabilities."""
    from types import SimpleNamespace

    from isaaclab_newton.physics import NewtonManager

    from isaaclab.sim import SimulationContext
    from isaaclab.visualizers import BaseVisualizer

    class CustomSolver(NewtonManager):
        pass

    sim = SimpleNamespace(physics_manager=CustomSolver)
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: sim))
    assert FactoryBase._get_backend() == "newton"
    assert BaseVisualizer.physics_backend.fget(SimpleNamespace(_sim=sim)) == "newton"
