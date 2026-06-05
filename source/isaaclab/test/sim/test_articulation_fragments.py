# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _make_xform(stage, path="/World/Art"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# ArticulationRootFragment marker + metadata
# -------------------------------------------------------------------------------------


def test_articulation_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import ArticulationRootFragment, SchemaFragment

    cfg = PhysxArticulationCfg(articulation_enabled=True)
    assert isinstance(cfg, ArticulationRootFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physxArticulation"
    assert type(cfg)._usd_applied_schema == "PhysxArticulationAPI"
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.articulation_enabled is True and cfg.enabled_self_collisions is None


# -------------------------------------------------------------------------------------
# PhysxArticulationCfg writes physxArticulation:* namespace
# -------------------------------------------------------------------------------------


def test_physx_articulation_fragment_writes_physx_namespace():
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/A1")
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    apply_namespaced(
        PhysxArticulationCfg(articulation_enabled=True, enabled_self_collisions=False, sleep_threshold=0.1),
        "/World/A1",
        stage,
    )
    assert prim.GetAttribute("physxArticulation:articulationEnabled").Get() is True
    assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is False
    assert abs(prim.GetAttribute("physxArticulation:sleepThreshold").Get() - 0.1) < 1e-6


# -------------------------------------------------------------------------------------
# NewtonArticulationCfg writes newton:* namespace
# -------------------------------------------------------------------------------------


def test_newton_articulation_fragment_writes_newton_namespace():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/A2")
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    apply_namespaced(NewtonArticulationCfg(self_collision_enabled=True), "/World/A2", stage)
    assert prim.GetAttribute("newton:selfCollisionEnabled").Get() is True


# -------------------------------------------------------------------------------------
# apply_articulation_root_properties dispatch (anchor + multi-namespace composition)
# -------------------------------------------------------------------------------------


def test_apply_articulation_root_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/A3")
    apply_articulation_root_properties(
        "/World/A3",
        [
            PhysxArticulationCfg(enabled_self_collisions=True, solver_position_iteration_count=8),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        stage,
    )
    prim = stage.GetPrimAtPath("/World/A3")
    assert bool(UsdPhysics.ArticulationRootAPI(prim))  # presence-gated anchor applied
    assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert prim.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    assert prim.GetAttribute("newton:selfCollisionEnabled").Get() is True


# -------------------------------------------------------------------------------------
# fix_root_link spawner-level flag: toggles an existing fixed joint
# -------------------------------------------------------------------------------------


def test_apply_articulation_root_properties_toggles_existing_fixed_joint():
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    # root prim with a rigid body and an articulation root
    root = _make_xform(stage, "/World/A4")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # author an existing global fixed joint between the world and the root link
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/A4/FixedJoint")
    joint.CreateBody1Rel().SetTargets(["/World/A4"])
    joint.CreateJointEnabledAttr(True)

    apply_articulation_root_properties(
        "/World/A4",
        [PhysxArticulationCfg(articulation_enabled=True)],
        stage,
        fix_root_link=False,
    )
    assert joint.GetJointEnabledAttr().Get() is False


# -------------------------------------------------------------------------------------
# public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg  # noqa: F401
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg  # noqa: F401

    from isaaclab.sim.schemas import (  # noqa: F401
        ArticulationRootFragment,
        SchemaFragment,
        apply_articulation_root_properties,
    )
