# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-collision group filtering through USD import and cloning."""

import numpy as np
import pytest
from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg, VBDSolverCfg
from newton_test_utils import local_usd

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Asset
from isaaclab.cloner import make_clone_plan, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import replace

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _spawn_with_colliders(path: str, cfg: sim_utils.UsdFileCfg, **kwargs) -> Usd.Prim:
    """Add only the colliders and existing exclusion needed by the chain regression."""
    parent = sim_utils.create_prim(path)
    path += "/Robot"
    prim = sim_utils.spawn_from_usd(path, cfg, **kwargs)
    stage = prim.GetStage()
    for body, shape in (("Link_0", "a"), ("Link_0", "b"), ("Link_2", "a"), ("Link_4", "a")):
        collider = UsdGeom.Cube.Define(stage, f"{path}/{body}/{shape}")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    existing = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{path}/Link_0"))
    existing.CreateFilteredPairsRel().AddTarget(f"{path}/Link_4")
    return parent


def test_self_collision_groups_survive_import_and_cloning() -> None:
    """Exclude every shape of nonadjacent bodies, preserving existing and unrelated contacts."""
    sim_cfg = SimulationCfg(device="cpu", physics=NewtonCfg(solver_cfg=VBDSolverCfg(), load_visual_shapes=False))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        cfg = ArticulationCfg(
            prim_path="/World/envs/env_[^/]+",
            articulation_root_prim_path="/Robot.*",
            spawn=local_usd("fixed_spatial_chain.usda", func=_spawn_with_colliders, spawn_path="/World/envs/env_1"),
            actuators={},
            self_collision_filter_pairs=[("Link_0", ["Link_2"])],
        )
        other_cfg = replace(
            cfg,
            spawn=replace(cfg.spawn, spawn_path="/World/envs/env_0"),
            self_collision_filter_pairs=[("Link_2", "Link_4")],
        )
        plan = make_clone_plan(
            (cfg, other_cfg),
            ((0,), (1,)),
            4,
            clone_strategy=lambda weights, count: np.array([1, 0, 1, 0]),
        )
        sim.set_clone_plan(plan)
        Asset(cfg)
        Asset(other_cfg)
        invalid_cfg = replace(
            cfg, spawn=None, self_collision_filter_pairs=[("Link_2", "Link_4"), ("missing", "Link_0")]
        )
        with pytest.raises(ValueError, match="missing"):
            Asset(invalid_cfg)
        replicate(plan)
        model = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim_cfg.physics)).finalize(device="cpu")
        allowed = {
            frozenset((model.shape_label[a], model.shape_label[b])) for a, b in model.shape_contact_pairs.numpy()
        }
        assert allowed == {
            frozenset((f"/World/envs/env_{env}/Robot/Link_2/a", f"/World/envs/env_{env}/Robot/Link_4/a"))
            for env in (1, 3)
        } | {
            frozenset((f"/World/envs/env_{env}/Robot/Link_0/{shape}", f"/World/envs/env_{env}/Robot/Link_2/a"))
            for env in (0, 2)
            for shape in ("a", "b")
        }
