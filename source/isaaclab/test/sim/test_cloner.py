# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cloner utilities and InteractiveScene cloning behavior."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
import warp as wp
from isaaclab_physx.cloner import physx_replicate

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import TemplateCloneCfg, clone_from_template, sequential, usd_replicate
from isaaclab.sim import build_simulation_context


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


def test_usd_replicate_with_positions_and_mask(sim):
    """Replicate sources to selected envs and author translate ops from positions."""
    # Prepare sources under /World/template
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/template/B", "Xform")

    # Prepare destination env namespaces
    num_envs = 3
    env_ids = torch.arange(num_envs, dtype=torch.long)
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    # Map A -> env 0 and 2; B -> env 1 only
    mask = torch.zeros((2, num_envs), dtype=torch.bool)
    mask[0, [0, 2]] = True
    mask[1, [1]] = True

    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/A", "/World/template/B"],
        destinations=["/World/envs/env_{}/Object/A", "/World/envs/env_{}/Object/B"],
        env_ids=env_ids,
        mask=mask,
    )

    # Validate replication and translate op
    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Object/A").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Object/B").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_1/Object/B").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Object/A").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_2/Object/A").IsValid()

    # Check xformOp:translate authored for env_2/A
    prim = stage.GetPrimAtPath("/World/envs/env_2/Object/A")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    assert any(op.GetOpType() == UsdGeom.XformOp.TypeTranslate for op in ops)


def test_usd_replicate_depth_order_parent_child(sim):
    """Replicate parent and child when provided out of order; parent should exist before child."""
    # Prepare sources
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/Parent", "Xform")
    sim_utils.create_prim("/World/template/Parent/Child", "Xform")

    # Destinations (single env)
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    # Provide child first, then parent; depth sort should handle this
    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/Parent/Child", "/World/template/Parent"],
        destinations=["/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent"],
        env_ids=env_ids,
    )

    stage = sim_utils.get_current_stage()
    for i in range(2):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent/Child").IsValid()


def test_physx_replicate_no_error(sim):
    """PhysX replicator call runs without raising exceptions for simple mapping."""
    # Prepare sources and envs
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")

    num_envs = 2
    env_ids = torch.arange(num_envs, dtype=torch.long)
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    mapping = torch.ones((1, num_envs), dtype=torch.bool)

    # Should not raise
    physx_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/A"],
        destinations=["/World/envs/env_{}/A"],
        env_ids=env_ids,
        mapping=mapping,
    )


def test_clone_from_template(sim):
    """Clone prototypes via TemplateCloneCfg and clone_from_template and exercise both USD and PhysX.

    Steps:
    - Create /World/template and /World/envs/env_0..env_31
    - Spawn three prototypes under /World/template/Object/proto_asset_.*
    - Clone using TemplateCloneCfg with random_heterogeneous_cloning=False (modulo mapping)
    - Verify modulo placement exists; then call sim.reset(), and create PhysX view
    """
    num_clones = 32
    clone_cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential)
    sim_utils.create_prim(clone_cfg.template_root, "Xform")
    sim_utils.create_prim(f"{clone_cfg.template_root}/Object", "Xform")  # Parent for prototypes
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    # Spawn prototypes under template
    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            ),
            sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            sim_utils.SphereCfg(
                radius=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
        ],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    prim = cfg.func(f"{clone_cfg.template_root}/Object/{clone_cfg.template_prototype_identifier}_.*", cfg)
    assert prim.IsValid()

    stage = sim_utils.get_current_stage()
    clone_from_template(stage, num_clones=num_clones, template_clone_cfg=clone_cfg)

    primitive_prims = sim_utils.get_all_matching_child_prims(
        "/World/envs", predicate=lambda prim: prim.GetTypeName() in ["Cone", "Cube", "Sphere"]
    )

    for i, primitive_prim in enumerate(primitive_prims):
        modulus = i % 3
        if modulus == 0:
            assert primitive_prim.GetTypeName() == "Cone"
        elif modulus == 1:
            assert primitive_prim.GetTypeName() == "Cube"
        else:
            assert primitive_prim.GetTypeName() == "Sphere"

    # Exercise PhysX initialization; should not raise error
    sim.reset()
    object_view_regex = f"{clone_cfg.clone_regex}/Object".replace(".*", "*")
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view(object_view_regex)
    assert physx_view is not None


def _run_colocation_collision_filter(sim, asset_cfg, expected_types, assert_count=False):
    """Shared harness for colocated collision filter checks across devices."""
    num_clones = 32
    clone_cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential)
    sim_utils.create_prim(clone_cfg.template_root, "Xform")
    sim_utils.create_prim(f"{clone_cfg.template_root}/Object", "Xform")  # Parent for prototypes
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    # Use _.*  pattern - the @clone decorator replaces .* with 0 for single-asset spawners
    prim = asset_cfg.func(f"{clone_cfg.template_root}/Object/{clone_cfg.template_prototype_identifier}_.*", asset_cfg)
    assert prim.IsValid()

    stage = sim_utils.get_current_stage()
    clone_from_template(stage, num_clones=num_clones, template_clone_cfg=clone_cfg)

    primitive_prims = sim_utils.get_all_matching_child_prims(
        "/World/envs", predicate=lambda prim: prim.GetTypeName() in expected_types
    )

    if assert_count:
        assert len(primitive_prims) == num_clones

    for i, primitive_prim in enumerate(primitive_prims):
        assert primitive_prim.GetTypeName() == expected_types[i % len(expected_types)]

    sim.reset()
    object_view_regex = f"{clone_cfg.clone_regex}/Object".replace(".*", "*")
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view(object_view_regex)
    for _ in range(100):
        sim.step()
    transforms = wp.to_torch(physx_view.get_transforms())
    distance_from_origin = torch.linalg.norm(transforms[:, :2], dim=-1)
    assert torch.all(distance_from_origin < 0.1)


def test_colocation_collision_filter_homogeneous(sim):
    """Verify colocated clones of a single prototype stay stable after PhysX cloning.

    All clones are spawned at exactly the same pose; if the collision filter is wrong the pile
    explodes on reset. This asserts the filter keeps the colocated objects stable while stepping
    across CPU and CUDA backends.
    """
    _run_colocation_collision_filter(
        sim,
        sim_utils.ConeCfg(
            radius=0.3,
            height=0.6,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        expected_types=["Cone"],
        assert_count=True,
    )


@pytest.mark.xfail(reason="Heterogeneous cloning with collision filtering not yet fully supported")
def test_colocation_collision_filter_heterogeneous(sim):
    """Verify colocated clones of multiple prototypes retain modulo ordering and remain stable.

    The cone, cube, and sphere are all spawned in the identical pose for every clone; an incorrect
    collision filter would blow up the simulation on reset. This guards both modulo ordering and
    that the colocated set stays stable through PhysX steps across CPU and CUDA.
    """
    _run_colocation_collision_filter(
        sim,
        sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        expected_types=["Cone", "Cube", "Sphere"],
    )
