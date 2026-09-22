# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def _shape_cfgs(**kwargs) -> list[sim_utils.SpawnerCfg]:
    return [
        sim_utils.ConeCfg(radius=0.3, height=0.6, **kwargs),
        sim_utils.CuboidCfg(size=(0.3, 0.3, 0.3), **kwargs),
        sim_utils.SphereCfg(radius=0.3, **kwargs),
    ]


def test_spawn_multiple_shapes_with_global_settings(stage):
    """Wrapper-level physics settings override the per-asset ones on every clone."""
    num_envs = 3
    for env_idx in range(num_envs):
        sim_utils.create_prim(f"/World/env_{env_idx}/Cone", "Xform")
    assets_cfg = _shape_cfgs(mass_props=sim_utils.MassCfg(mass=100.0))
    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=assets_cfg,
        rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )

    prim = cfg.func("/World/env_.*/Cone/asset_.*", cfg)

    assert prim.GetPath() == "/World/env_0/Cone/asset_0"
    prim_paths = sim_utils.find_matching_prim_paths("/World/env_[^/]+/Cone/asset_[^/]*")
    assert len(prim_paths) == num_envs * len(assets_cfg)
    for env_idx in range(num_envs):
        for asset_idx in range(len(assets_cfg)):
            path = f"/World/env_{env_idx}/Cone/asset_{asset_idx}"
            assert path in prim_paths
            assert stage.GetPrimAtPath(path).GetAttribute("physics:mass").Get() == 1.0
            assert stage.GetPrimAtPath(path).GetAttribute("physxRigidBody:solverPositionIterationCount").Get() == 4


def test_spawn_multiple_shapes_with_individual_settings(stage):
    sim_utils.create_prim("/World/template", "Xform")
    mass_variations = [2.0, 3.0, 4.0]
    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            asset_cfg.replace(mass_props=sim_utils.MassCfg(mass=mass), rigid_props=sim_utils.UsdPhysicsRigidBodyCfg())
            for asset_cfg, mass in zip(_shape_cfgs(), mass_variations)
        ]
    )

    prim = cfg.func("/World/template/Cone/asset_.*", cfg)

    assert prim.GetPath() == "/World/template/Cone/asset_0"
    prim_paths = sim_utils.find_matching_prim_paths("/World/template/Cone/asset_[^/]*")
    assert [stage.GetPrimAtPath(path).GetAttribute("physics:mass").Get() for path in prim_paths] == mass_variations


def test_spawn_multiple_shapes_with_explicit_spawn_paths(stage):
    """Planned per-variant source paths take precedence over the prim path pattern."""
    sim_utils.create_prim("/World/planned", "Xform")
    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=_shape_cfgs(),
        spawn_paths=["/World/planned/apple", None, "/World/planned/banana"],
        mass_props=sim_utils.MassCfg(mass=1.0),
    )

    prim = cfg.func("/World/ignored_without_regex", cfg)

    assert prim.GetPath() == "/World/planned/apple"
    assert stage.GetPrimAtPath("/World/planned/banana").IsValid()
    assert not stage.GetPrimAtPath("/World/planned/ignored").IsValid()
    assert prim.GetAttribute("physics:mass").Get() == 1.0


def test_spawn_multiple_shapes_rejects_invalid_paths(stage):
    cfg = sim_utils.MultiAssetSpawnerCfg(assets_cfg=_shape_cfgs()[:2], spawn_paths=["/World/planned/apple"])
    with pytest.raises(ValueError, match="spawn_paths"):
        cfg.func("/World/ignored_without_regex", cfg)

    cfg = sim_utils.MultiAssetSpawnerCfg(assets_cfg=_shape_cfgs())
    with pytest.raises(ValueError, match="segment wildcard"):
        cfg.func("/World/template/asset", cfg)


def test_spawn_multiple_files_with_global_settings(stage):
    sim_utils.create_prim("/World/template", "Xform")
    cfg = sim_utils.MultiUsdFileCfg(
        usd_path=[
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        ],
        rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=1.0),
        articulation_props=[
            PhysxArticulationCfg(enabled_self_collisions=True, solver_position_iteration_count=4),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        activate_contact_sensors=True,
    )

    prim = cfg.func("/World/template/Robot/asset_.*", cfg)

    assert prim.GetPath() == "/World/template/Robot/asset_0"
    prim_paths = sim_utils.find_matching_prim_paths("/World/template/Robot/asset_[^/]*")
    assert prim_paths == ["/World/template/Robot/asset_0", "/World/template/Robot/asset_1"]
    # articulation settings and contact sensors are authored on every file
    for path in prim_paths:
        assert sim_utils.get_first_matching_child_prim(
            path, lambda p: p.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
        )
        assert sim_utils.get_first_matching_child_prim(
            path, lambda p: "PhysxContactReportAPI" in p.GetPrimTypeInfo().GetAppliedAPISchemas()
        )
