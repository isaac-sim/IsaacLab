# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless coverage for PhysX tendon fragments using OVPhysX's codeless USD schemas."""

import pytest

ovphysx = pytest.importorskip("ovphysx", reason="ovphysx wheel not installed")

from pxr import Plug, Usd


def _register_codeless_physx_schemas():
    registry = Plug.Registry()
    registered_names = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
    schema_paths = [
        str(path) for path in ovphysx.codeless_schema_paths() if path.parent.name.casefold() not in registered_names
    ]
    if schema_paths:
        registry.RegisterPlugins(schema_paths)


def test_tendon_fragments_use_codeless_schema_names_and_types():
    from isaaclab_physx.sim.schemas import (
        PhysxTendonAttachmentRootCfg,
        PhysxTendonAxisCfg,
        PhysxTendonAxisRootCfg,
    )

    from isaaclab.sim.schemas import apply_fixed_tendon_properties, apply_spatial_tendon_properties

    _register_codeless_physx_schemas()
    stage = Usd.Stage.CreateInMemory()
    fixed = stage.DefinePrim("/Fixed", "Xform")
    fixed.AddAppliedSchema("PhysxTendonAxisRootAPI:index")
    fixed.AddAppliedSchema("PhysxTendonAxisRootAPI:middle")
    spatial = stage.DefinePrim("/Spatial", "Xform")
    spatial.AddAppliedSchema("PhysxTendonAttachmentRootAPI:cable")

    assert apply_fixed_tendon_properties(
        "/Fixed",
        [
            PhysxTendonAxisRootCfg(instance_names="index", stiffness=3.0),
            PhysxTendonAxisCfg(instance_names="index", gearing=[-0.5], joint_axis=["rotX"]),
        ],
        stage,
    )
    assert apply_spatial_tendon_properties("/Spatial", [PhysxTendonAttachmentRootCfg(stiffness=4.0)], stage)

    assert fixed.GetAttribute("physxTendon:index:stiffness").Get() == pytest.approx(3.0)
    assert not fixed.GetAttribute("physxTendon:middle:stiffness").HasAuthoredValue()
    assert list(fixed.GetAttribute("physxTendon:index:gearing").Get()) == pytest.approx([-0.5])
    assert list(fixed.GetAttribute("physxTendon:index:jointAxis").Get()) == ["rotX"]
    assert spatial.GetAttribute("physxTendon:cable:stiffness").Get() == pytest.approx(4.0)
