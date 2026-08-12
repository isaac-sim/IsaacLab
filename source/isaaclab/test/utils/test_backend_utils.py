# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend module resolution."""

from isaaclab.utils.backend_utils import FactoryBase


def test_get_module_name_routes_ovphysx_to_isaaclab_ov(monkeypatch):
    """OVPhysX backend modules resolve from the consolidated package."""
    monkeypatch.setattr(FactoryBase, "_module_subpath", "assets.articulation", raising=False)

    assert FactoryBase._get_package_name("ovphysx") == "isaaclab_ov"
    assert FactoryBase._get_module_name("ovphysx") == "isaaclab_ov.assets.articulation"


def test_get_module_name_preserves_other_backend_conventions(monkeypatch):
    """Other backend modules continue to use their backend-named packages."""
    monkeypatch.setattr(FactoryBase, "_module_subpath", "assets.articulation", raising=False)

    assert FactoryBase._get_package_name("physx") == "isaaclab_physx"
    assert FactoryBase._get_package_name("newton") == "isaaclab_newton"
    assert FactoryBase._get_module_name("physx") == "isaaclab_physx.assets.articulation"
    assert FactoryBase._get_module_name("newton") == "isaaclab_newton.assets.articulation"
