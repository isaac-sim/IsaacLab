# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Group exclusions survive asset authoring, Newton import, and environment replication."""

import pytest
from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg, VBDSolverCfg

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Asset, AssetBaseCfg
from isaaclab.cloner import make_clone_plan, replicate
from isaaclab.sim import MultiAssetSpawnerCfg, SimulationCfg, SimulationContext, SpawnerCfg, build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _spawn_robot(prim_path: str, _cfg: SpawnerCfg | None = None, **_kwargs) -> Usd.Prim:
    """Author a star articulation whose three leaves can collide with one another."""
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for index, name in enumerate(("base", "left", "right", "spare")):
        body_path = f"{prim_path}/{name}"
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set((float(index), 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        for shape_name in ("a", "b") if name == "left" else ("a",):
            shape = UsdGeom.Cube.Define(stage, f"{body_path}/{shape_name}")
            shape.CreateSizeAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if name != "base":
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/joint_{name}")
            joint.CreateBody0Rel().SetTargets([f"{prim_path}/base"])
            joint.CreateBody1Rel().SetTargets([body_path])
            joint.CreateCollisionEnabledAttr(False)

    # Keep one asset-authored shape exclusion alongside the configured body-group exclusion.
    existing = stage.GetPrimAtPath(f"{prim_path}/left/a")
    UsdPhysics.FilteredPairsAPI.Apply(existing).CreateFilteredPairsRel().SetTargets([f"{prim_path}/spare/a"])
    return root


def _spawn_scene(prim_path: str, _cfg: SpawnerCfg, **_kwargs) -> Usd.Prim:
    """Author selected and unselected articulations, including a pre-existing body exclusion."""
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    for name in ("Robot", "Nested", "Sibling"):
        _spawn_robot(f"{prim_path}/{name}")
    left = stage.GetPrimAtPath(f"{prim_path}/Robot/left")
    UsdPhysics.FilteredPairsAPI.Apply(left).CreateFilteredPairsRel().AddTarget(f"{prim_path}/Robot/spare")
    return root


def _finalize_contact_pairs(sim: SimulationContext) -> set[frozenset[str]]:
    """Finalize the CPU builder and read the eligible collision pairs."""
    model = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics)).finalize(device="cpu")
    return {
        frozenset((model.shape_label[first], model.shape_label[second]))
        for first, second in model.shape_contact_pairs.numpy()
    }


def test_body_group_exclusions_survive_newton_replication():
    """Every collider of the selected bodies is excluded in each clone, with sibling contacts preserved."""
    physics_cfg = NewtonCfg(solver_cfg=VBDSolverCfg(), load_visual_shapes=False)
    with build_simulation_context(sim_cfg=SimulationCfg(device="cpu", physics=physics_cfg)) as sim:
        source = "/World/envs/env_0/Robot"
        sibling = "/World/envs/env_0/Sibling"
        _spawn_robot(sibling)
        cfg = ArticulationCfg(
            prim_path="/World/envs/env_[^/]+/Robot",
            spawn=SpawnerCfg(func=_spawn_robot, spawn_path=source),
            actuators={},
            self_collision_filter_pairs=[(["left.*"], "right")],
        )
        Asset(cfg)
        sibling_cfg = ArticulationCfg(prim_path="/World/envs/env_[^/]+/Sibling", actuators={})
        plan = make_clone_plan((cfg, sibling_cfg), ((0, 1),), 2)
        sim.set_clone_plan(plan)
        replicate(plan)
        allowed_pairs = _finalize_contact_pairs(sim)
        for env in range(2):
            robot_path = f"/World/envs/env_{env}/Robot"
            sibling_path = f"/World/envs/env_{env}/Sibling"
            for path, expected in (
                (robot_path, {("left/b", "spare/a"), ("right/a", "spare/a")}),
                (
                    sibling_path,
                    {
                        ("left/a", "right/a"),
                        ("left/b", "right/a"),
                        ("left/b", "spare/a"),
                        ("right/a", "spare/a"),
                    },
                ),
            ):
                internal_pairs = {
                    frozenset(label.removeprefix(path + "/") for label in pair)
                    for pair in allowed_pairs
                    if all(label.startswith(path + "/") for label in pair)
                }
                assert internal_pairs == {frozenset(pair) for pair in expected}
            assert frozenset((f"{robot_path}/left/a", f"{sibling_path}/right/a")) in allowed_pairs


def test_group_exclusions_follow_multiple_prototype_sources_and_nested_views():
    """Overlapping groups preserve exclusions and stay within the selected graphs of every source."""
    physics_cfg = NewtonCfg(solver_cfg=VBDSolverCfg(), load_visual_shapes=False)
    with build_simulation_context(sim_cfg=SimulationCfg(device="cpu", physics=physics_cfg)) as sim:
        sources = ["/World/prototypes/First", "/World/prototypes/Second"]
        group = ["left", "(left|right)"]
        cfg = ArticulationCfg(
            prim_path="/World/envs/env_[^/]+/Scene",
            articulation_root_prim_path="/Robot",
            spawn=MultiAssetSpawnerCfg(
                assets_cfg=[SpawnerCfg(func=_spawn_scene), SpawnerCfg(func=_spawn_scene)],
                spawn_paths=sources,
                random_choice=False,
            ),
            actuators={},
            self_collision_filter_pairs=[(group, group)],
        )
        # No destination prims or clone plan exist yet: the multi-spawner owns both sources.
        Asset(cfg)
        prototypes = tuple(
            AssetBaseCfg(prim_path=cfg.prim_path, spawn=SpawnerCfg(spawn_path=source)) for source in sources
        )
        plan = make_clone_plan(prototypes, ((0,), (1,)), 2)
        sim.set_clone_plan(plan)
        # This view has no spawner; its parent scene's plan supplies both authored source paths.
        nested_cfg = ArticulationCfg(
            prim_path=cfg.prim_path + "/Nested",
            actuators={},
            self_collision_filter_pairs=[("left", "right")],
        )
        Asset(nested_cfg)
        Asset(nested_cfg)
        assert all(
            target != prim.GetPath()
            for prim in sim.stage.Traverse()
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
        )
        replicate(plan)
        allowed_pairs = _finalize_contact_pairs(sim)
        for env in range(2):
            for name in ("Robot", "Nested"):
                path = f"/World/envs/env_{env}/Scene/{name}"
                assert frozenset((f"{path}/left/a", f"{path}/right/a")) not in allowed_pairs
            scene = f"/World/envs/env_{env}/Scene"
            assert frozenset((f"{scene}/Robot/left/b", f"{scene}/Robot/spare/a")) not in allowed_pairs
            assert frozenset((f"{scene}/Robot/right/a", f"{scene}/Robot/spare/a")) in allowed_pairs
            assert frozenset((f"{scene}/Nested/left/b", f"{scene}/Nested/spare/a")) in allowed_pairs
            assert frozenset((f"{scene}/Sibling/left/a", f"{scene}/Sibling/right/a")) in allowed_pairs
            assert frozenset((f"{scene}/Sibling/left/b", f"{scene}/Sibling/spare/a")) in allowed_pairs
