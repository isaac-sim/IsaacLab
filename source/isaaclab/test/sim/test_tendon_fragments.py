# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from pxr import PhysxSchema, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _make_xform(stage, path="/World/Tendon"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


def _make_fixed_tendon_prim(stage, path, instance="default"):
    """Create a prim with a multi-instance PhysxTendonAxisRootAPI applied."""
    prim = _make_xform(stage, path)
    PhysxSchema.PhysxTendonAxisRootAPI.Apply(prim, instance)
    return prim


def _make_spatial_tendon_prim(stage, path, instance="default"):
    """Create a prim with a multi-instance PhysxTendonAttachmentRootAPI applied."""
    prim = _make_xform(stage, path)
    PhysxSchema.PhysxTendonAttachmentRootAPI.Apply(prim, instance)
    return prim


def _tendon_attr_prefix(prim, schema_substr):
    """Return the applied-schema name used by the writer as the authored-attribute prefix.

    The legacy writer authors ``f"{schema_name}:{camelCase(field)}"`` where ``schema_name`` is
    the entry returned by ``prim.GetAppliedSchemas()`` (e.g. ``PhysxTendonAxisRootAPI:t0``).
    """
    for schema_name in prim.GetAppliedSchemas():
        if schema_substr in schema_name:
            return schema_name
    raise AssertionError(f"no applied schema containing {schema_substr!r} on {prim.GetPath()}")


# -------------------------------------------------------------------------------------
# Fixed-tendon marker + metadata defaults
# -------------------------------------------------------------------------------------


def test_fixed_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import FixedTendonFragment, SchemaFragment

    cfg = PhysxFixedTendonCfg(stiffness=1.0)
    assert isinstance(cfg, FixedTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_fixed_tendon"
    assert cfg.stiffness == 1.0 and cfg.damping is None


def test_spatial_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import SchemaFragment, SpatialTendonFragment

    cfg = PhysxSpatialTendonCfg(stiffness=2.0)
    assert isinstance(cfg, SpatialTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_spatial_tendon"
    assert cfg.stiffness == 2.0 and cfg.damping is None


# -------------------------------------------------------------------------------------
# PhysxFixedTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_physx_fixed_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_fixed_tendon_prim(stage, "/World/FT", instance="t0")
    apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=3.0, damping=0.5), "/World/FT", stage)
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 3.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.5) < 1e-6
    # the ``func`` plumbing field must not be authored as an attribute
    assert not prim.HasAttribute(f"{prefix}:func")


# -------------------------------------------------------------------------------------
# PhysxSpatialTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_physx_spatial_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, apply_spatial_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_spatial_tendon_prim(stage, "/World/ST", instance="s0")
    apply_spatial_tendon(PhysxSpatialTendonCfg(stiffness=4.0, limit_stiffness=0.25), "/World/ST", stage)
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAttachmentRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 4.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:limitStiffness").Get() - 0.25) < 1e-6


# -------------------------------------------------------------------------------------
# apply_fixed_tendon_properties dispatch (tune-not-apply, multi-fragment)
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_properties_dispatches_fragments():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_fixed_tendon_prim(stage, "/World/FT2", instance="t0")
    apply_fixed_tendon_properties(
        "/World/FT2",
        [PhysxFixedTendonCfg(stiffness=5.0), PhysxFixedTendonCfg(damping=0.75)],
        stage,
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 5.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.75) < 1e-6


def test_apply_spatial_tendon_properties_dispatches_fragments():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_spatial_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_spatial_tendon_prim(stage, "/World/ST2", instance="s0")
    apply_spatial_tendon_properties(
        "/World/ST2",
        [PhysxSpatialTendonCfg(stiffness=6.0), PhysxSpatialTendonCfg(offset=0.1)],
        stage,
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAttachmentRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 6.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:offset").Get() - 0.1) < 1e-6


# -------------------------------------------------------------------------------------
# Public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_physx.sim.schemas import (  # noqa: F401
        PhysxFixedTendonCfg,
        PhysxSpatialTendonCfg,
        apply_fixed_tendon,
        apply_spatial_tendon,
    )

    from isaaclab.sim.schemas import (  # noqa: F401
        FixedTendonFragment,
        SpatialTendonFragment,
        apply_fixed_tendon_properties,
        apply_spatial_tendon_properties,
    )
