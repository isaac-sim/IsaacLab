# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from pxr import UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

# -------------------------------------------------------------------------------------
# RigidBodyMaterialFragment marker + metadata
# -------------------------------------------------------------------------------------


def test_rigid_body_material_fragment_metadata_defaults():
    from isaaclab.sim.schemas import SchemaFragment
    from isaaclab.sim.spawners.materials.physics_materials_cfg import (
        RigidBodyMaterialFragment,
        UsdPhysicsRigidBodyMaterialCfg,
    )

    cfg = UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7)
    assert isinstance(cfg, RigidBodyMaterialFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physics"
    assert type(cfg)._usd_applied_schema is None  # MaterialAPI applied by the family writer
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.static_friction == 0.7 and cfg.dynamic_friction is None


def test_physx_material_fragment_metadata_defaults():
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialFragment

    cfg = PhysxMaterialCfg(compliant_contact_stiffness=100.0)
    assert isinstance(cfg, RigidBodyMaterialFragment)
    assert type(cfg)._usd_namespace == "physxMaterial"
    assert type(cfg)._usd_applied_schema == "PhysxMaterialAPI"
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"


# -------------------------------------------------------------------------------------
# spawn_rigid_body_material_from_fragments: spawn prim + anchor + multi-namespace compose
# -------------------------------------------------------------------------------------


def test_spawn_rigid_body_material_from_fragments_composes_namespaces():
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat",
        [
            UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.1),
            PhysxMaterialCfg(compliant_contact_stiffness=100.0, friction_combine_mode="max"),
        ],
        stage,
    )
    assert prim.IsA(UsdShade.Material)
    assert bool(UsdPhysics.MaterialAPI(prim))  # neutral anchor applied by the writer
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert prim.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(0.6)
    assert prim.GetAttribute("physics:restitution").Get() == pytest.approx(0.1)
    # the PhysX fragment applied its own schema and namespace
    assert "PhysxMaterialAPI" in prim.GetAppliedSchemas()
    assert prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(100.0)
    assert prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"


def test_spawn_rigid_body_material_from_fragments_accepts_single_fragment():
    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat2", UsdPhysicsRigidBodyMaterialCfg(static_friction=0.3), stage
    )
    assert prim.IsA(UsdShade.Material)
    assert bool(UsdPhysics.MaterialAPI(prim))
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.3)


def test_spawn_physics_material_dispatches_fragments_and_legacy():
    """The shared slot dispatcher handles both forms a spawner ``physics_material`` slot accepts: a
    rigid-body fragment list, and a legacy material cfg carrying its own ``func``."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))

    # fragment-list form
    frag_prim = spawn_physics_material("/World/MaterialA", [UsdPhysicsRigidBodyMaterialCfg(static_friction=0.4)])
    assert bool(UsdPhysics.MaterialAPI(frag_prim))
    assert frag_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.4)

    # legacy single-cfg form (RigidBodyMaterialBaseCfg subclass with its own spawner func)
    legacy_prim = spawn_physics_material("/World/MaterialB", PhysxRigidBodyMaterialCfg(static_friction=0.9))
    assert bool(UsdPhysics.MaterialAPI(legacy_prim))
    assert legacy_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.9)


def test_spawn_physics_material_rejects_empty_and_mixed_lists():
    """A malformed fragment list surfaces a clear error rather than an opaque AttributeError."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))

    with pytest.raises(ValueError):
        spawn_physics_material("/World/MatEmpty", [])
    # a list mixing a fragment with a legacy cfg is not a valid fragment list
    with pytest.raises(TypeError):
        spawn_physics_material(
            "/World/MatMixed",
            [UsdPhysicsRigidBodyMaterialCfg(static_friction=0.4), PhysxRigidBodyMaterialCfg(static_friction=0.9)],
        )


def test_spawn_rigid_body_material_from_fragments_leaves_none_fields_unwritten():
    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat3", [UsdPhysicsRigidBodyMaterialCfg(static_friction=0.5)], stage
    )
    # only the authored field is written; None fields are left unauthored (partial update)
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.5)
    assert not prim.GetAttribute("physics:dynamicFriction").HasAuthoredValue()
